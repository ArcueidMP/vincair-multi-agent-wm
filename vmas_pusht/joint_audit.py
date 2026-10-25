"""Independent replay, model arithmetic, candidate and paired-task auditing."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from vmas_pusht.boundary import _plain, _rotate
from vmas_pusht.interface_numerics import audit_file
from vmas_pusht.joint_experiment import verify_inputs
from vmas_pusht.joint_interface import HELPERS, JointActionInterface, features, joint_table
from vmas_pusht.joint_models import FrozenJointPredictor, execute, validate
from vmas_pusht.option_models import normalization, sha
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.snapshot import restore


def numpy_prediction(entry, x):
    norm, weights = entry["normalization"], entry["state_dict"]
    x = np.asarray(x, dtype=np.float32)
    x = (x-norm["x_mean"].numpy())/norm["x_scale"].numpy()
    for layer in (0, 2, 4):
        x = x@weights[f"net.{layer}.weight"].numpy().T + weights[f"net.{layer}.bias"].numpy()
        if layer != 4:
            x = x/(1.+np.exp(-x))
    x = x*norm["y_scale"].numpy()+norm["y_mean"].numpy()
    x[:, 2] = (x[:, 2]+np.pi) % (2*np.pi)-np.pi
    x[:, 3] = np.maximum(x[:, 3], 0.)
    x[:, 4] = np.clip(x[:, 4], 0., 1.)
    return x


def audit_models(root):
    """Recompute training moments and forward arithmetic without Torch layers."""
    pack = torch.load(root/"local"/"models.pt", weights_only=False)
    parts = []
    for path, digest in pack["data_sha256"].items():
        if sha(path) != digest:
            raise AssertionError("Local data changed")
        data = torch.load(path, weights_only=False)
        validate(data); parts.append(data)
    x = torch.cat([p["X"] for p in parts]); y = torch.cat([p["Y"] for p in parts])
    rows = [r for p in parts for r in p["rows"]]
    train = torch.tensor([r["split"] == "train" for r in rows])
    expected = normalization(x[train], y[train])
    # Prespecified evenly spaced rows, including every local data partition.
    indices = torch.linspace(0, len(x)-1, min(4096, len(x))).long()
    errors = {}
    for seed, entry in pack["checkpoints"].items():
        if not all(torch.equal(expected[k], entry["normalization"][k]) for k in expected):
            raise AssertionError("Training-only normalization does not reproduce")
        actual = FrozenJointPredictor(entry)(x[indices]).numpy()
        independent = numpy_prediction(entry, x[indices].numpy())
        error = float(np.max(np.abs(actual-independent)))
        if error > 1e-5:
            raise AssertionError(f"Independent prediction mismatch: {error}")
        errors[seed] = error
    return dict(rows=len(rows), train_rows=int(train.sum()), training_moments_exact=True,
        split_counts=dict(Counter(r["split"] for r in rows)),
        independent_numpy_rows_per_seed=len(indices), max_numpy_errors=errors,
        data_hashes_verified=True, tolerance=1e-5)


@torch.no_grad()
def replay_local(path, chunk_size=3072):
    """Replay first and last complete collection batches, retaining batch layout."""
    torch.set_num_threads(1)
    data = torch.load(path, weights_only=False)
    last = ((len(data["rows"])-1)//chunk_size)*chunk_size
    reports = []
    for begin in sorted({0, last}):
        end = min(begin+chunk_size, len(data["rows"]))
        rows = data["rows"][begin:end]
        indices = torch.tensor([r["source_index"] for r in rows])
        env = make_profile("dry_light", num_envs=len(rows), substeps=160, seed=92617)
        restore(env, data["source_snapshot"], indices)
        anchor = data["prepared"]["record"]["anchor"][indices]
        interface = JointActionInterface(env, rows[0]["support"], anchor)
        interface.reset_contact(torch.ones(len(rows), dtype=torch.bool),
                                mode=torch.tensor([r["current_mode"] for r in rows]))
        options = data["options"][begin:end]
        x = features(env, interface, options)
        angle = env.scenario.bar.state.rot[:, 0].clone()
        result = execute(env, interface, options)
        body_y = result["Y"].clone()
        body_y[:, :2] = _rotate(body_y[:, :2], -angle)
        exact = (torch.equal(x, data["X"][begin:end])
            and torch.equal(result["physical_start"], data["physical_start"][begin:end])
            and torch.equal(result["physical_end"], data["physical_end"][begin:end])
            and torch.equal(result["Y"], data["world_Y"][begin:end])
            and torch.equal(body_y, data["Y"][begin:end]))
        if not exact:
            raise AssertionError("Local option replay did not reproduce exactly")
        reports.append(dict(begin=begin, end=end, exact=exact))
    return dict(path=str(path), batches=reports, cases=sum(r["end"]-r["begin"] for r in reports))


def audit_queries(group):
    checked, evaluated, active_steps, padded_steps, seconds = 0, 0, 0, 0, 0.
    helper_histogram = Counter()
    for query in group["queries"]:
        diagnostics = query["diagnostics"]
        tables = diagnostics["candidate_table"]
        mask, chosen = query["query_mask"], diagnostics["chosen_index"]
        for i in mask.nonzero().flatten().tolist():
            row = group["rows"][i]
            direct = joint_table(planned_helper=row["planned_helper"])
            expanded = joint_table(planned_helper=row["planned_helper"], expanded=True, allocation=row["allocation"])
            if not (torch.equal(tables[i], direct) or torch.equal(tables[i], expanded)):
                raise AssertionError("Unexpected candidate allocation")
            if not torch.equal(tables[i, :, 1:], direct[:, 1:]):
                raise AssertionError("Force/duration/helper multiset changed")
            if not query["allow_routes"][i] and not tables[i, :, 0].eq(0).all():
                raise AssertionError("First stage received routed choices")
            index = int(chosen[i])
            if index >= 0 and not torch.equal(query["options"][i], tables[i, index]):
                raise AssertionError("Executed selection differs from scored candidate")
            if index < 0 and not torch.equal(query["options"][i], torch.tensor([0., 0., 0., 5., 0.])):
                raise AssertionError("Unexpected fallback")
            helper_histogram[HELPERS[int(query["options"][i, 4])]] += 1
            checked += 1
        evaluated += int(diagnostics["evaluated_candidates"].sum())
        active_steps += int(diagnostics["active_simulator_steps"].sum())
        padded_steps += int(diagnostics["padded_simulator_steps"])
        seconds += float(diagnostics["query_wall_seconds"])
    if group["rows"][0]["predictor"] == "learned" and (active_steps or padded_steps):
        raise AssertionError("Learned planning performed simulator work")
    return dict(decisions=checked, evaluated_candidates=evaluated, active_simulator_steps=active_steps,
        padded_simulator_steps=padded_steps, selector_seconds=seconds, helper_choices=dict(helper_histogram))


def paired(direct, expanded):
    """Only route expansion is intervened on: verify all pre-final trajectories."""
    if len(direct["rows"]) != len(expanded["rows"]):
        raise AssertionError("Paired batches differ")
    rows = []
    for i, (a, b) in enumerate(zip(direct["rows"], expanded["rows"])):
        keys = ("initial_angle_deg", "nuisance_id", "support", "goal_pair", "model_seed", "planned_helper")
        if any(a[key] != b[key] for key in keys):
            raise AssertionError("Paired initial conditions differ")
        end = a["phase_end_steps"]["recovery"]
        for phase in ("rotate_first", "retreat", "recovery"):
            if a["phase_end_steps"][phase] != b["phase_end_steps"][phase]:
                raise AssertionError("Paired pre-final clocks differ")
        state_error = float((direct["trajectory"][i, :end+1]-expanded["trajectory"][i, :end+1]).abs().max())
        action_error = float((direct["actions"][i, :end]-expanded["actions"][i, :end]).abs().max())
        if max(state_error, action_error) > 1e-6 or not torch.equal(direct["prelude_trajectory"][i], expanded["prelude_trajectory"][i]):
            raise AssertionError("Route ablation changed the shared physical prefix")
        if a["first_goal_success_at_phase_boundary"] != b["first_goal_success_at_phase_boundary"]:
            raise AssertionError("Paired first-stage scoring differs")
        rows.append(dict(row=i, predictor=a["predictor"], model_seed=a["model_seed"], support=a["support"],
            initial_angle_deg=a["initial_angle_deg"], nuisance_id=a["nuisance_id"],
            goal_pair=a["goal_pair"], allocation=b["allocation"], planned_helper=a["planned_helper"],
            state_error=state_error, action_error=action_error, direct=a["ordered_goal_success"],
            expanded=b["ordered_goal_success"], first=a["first_goal_success_at_phase_boundary"]))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    root = args.output
    torch.set_num_threads(1)
    manifest = json.loads((root/"manifest.json").read_text())
    verify_inputs(manifest["frozen"])
    model_check = root/"model_verification.json"
    pack = torch.load(root/"local"/"models.pt", weights_only=False)
    for path, digest in pack["data_sha256"].items():
        if sha(path) != digest:
            raise AssertionError("Local data changed after freezing")
    if model_check.exists():
        model_report = json.loads(model_check.read_text())
        if model_report["model_sha256"] != sha(root/"local"/"models.pt"):
            raise AssertionError("Cached model audit belongs to different weights")
    else:
        model_report = audit_models(root)
    cached_local = root/"local_verification.json"
    local_reports = None
    if cached_local.exists():
        cached = json.loads(cached_local.read_text())
        if cached["data_sha256"] != pack["data_sha256"]:
            raise AssertionError("Cached local replay belongs to different data")
        local_reports = cached["reports"]
    numeric = root/"numerics"; numeric.mkdir(exist_ok=True)
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        local_futures = ([] if local_reports is not None else
            [pool.submit(replay_local, root/"local"/(support+".pt"))
             for support in ("passive", "world_anchor", "preloaded_anchor")])
        futures = []
        for item in manifest["groups"]:
            path = numeric/Path(item["path"]).stem
            if path.with_suffix(".json").exists():
                saved = json.loads(path.with_suffix(".json").read_text())
                if saved["input_sha256"] != item["sha256"] or sha(saved["raw_path"]) != saved["raw_sha256"]:
                    raise AssertionError("Existing numerical replay changed")
                reports.append(saved)
            else:
                futures.append(pool.submit(audit_file, item["path"], item["sha256"], 320, path))
        for future in as_completed(futures):
            path = future.result()
            report = json.loads(Path(path).read_text()); reports.append(report)
            print(json.dumps(dict(numerical_file=path, completed=len(reports))), flush=True)
        if local_reports is None:
            local_reports = [f.result() for f in local_futures]
    if not all(result["source_replay_bitwise"] and result["source_scoring_reproduced"]
               for report in reports for result in report["results"]):
        raise AssertionError("Nominal replay or original task scoring did not reproduce")
    # Load at most two groups for prefix comparisons, instead of retaining all
    # high-frequency measurement arrays in memory.
    decision_check = root/"decision_verification.json"
    if decision_check.exists():
        cached = json.loads(decision_check.read_text())
        expected = {g["path"]: g["sha256"] for g in manifest["groups"]}
        if cached["group_sha256"] != expected or any(sha(p) != h for p, h in expected.items()):
            raise AssertionError("Cached decision audit belongs to different executions")
        queries, pairs = cached["queries"], cached["paired_rows"]
    else:
        by_key = {tuple(item["job"]): item for item in manifest["groups"]}
        queries, pairs = [], []
        for job, item in by_key.items():
            group = torch.load(item["path"], weights_only=False)["groups"][0]
            queries.append(dict(job=job, **audit_queries(group)))
            predictor, seed, goal, planned, expanded, shift = job
            if expanded:
                other = by_key[(predictor, seed, goal, planned, False, 0)]
                direct = torch.load(other["path"], weights_only=False)["groups"][0]
                pairs.extend(paired(direct, group))
    verification = dict(models=model_report, local_replays=local_reports, queries=queries, paired_rows=pairs,
        audit_source_sha256={str(Path(__file__).relative_to(Path.cwd())): sha(__file__)},
        all_prefixes_match=True, max_prefix_state_error=max(r["state_error"] for r in pairs),
        max_prefix_action_error=max(r["action_error"] for r in pairs), numerical_reports=reports)
    verify_inputs(manifest["frozen"])
    (root/"verification.json").write_text(json.dumps(_plain(verification), indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
