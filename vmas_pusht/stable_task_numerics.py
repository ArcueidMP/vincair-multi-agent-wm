"""Same-action refinement audit on the complete frozen composition trials.

The source's 160-substep joint forces are replayed unchanged at 160 and 320,
starting from the same full post-prelude snapshot in the same batch/order.
No model, planner, helper feedback, or phase decision is queried during replay.
All errors use a common physical clock. Source-active outcomes and post-done
zero-command padding are reported separately; event times are never aligned.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import torch

from vmas_pusht.abstraction_convergence import _weld_residual, restore_across_substeps
from vmas_pusht.boundary import _at_wall, _finite
from vmas_pusht.physical_abstraction import physical_state
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.scenario import wrap
from vmas_pusht.stable_composition import snapshot_from_record


MAX_XY = .001
MAX_ANGLE = .003


def validate_source(group):
    trajectory, actions, active, phases = (group[k] for k in ("trajectory", "actions", "active", "phases"))
    n, horizon = actions.shape[:2]
    if actions.shape != (n, horizon, 2, 2) or trajectory.shape != (n, horizon+1, 24):
        raise ValueError("Composition actions and complete states are misaligned")
    if active.shape != (n, horizon) or phases.shape != (n, horizon) or len(group["rows"]) != n:
        raise ValueError("Source rows, activity and phases must retain the complete original batch")
    if not torch.isfinite(actions).all() or float(actions.norm(dim=-1).max()) > 1.000001:
        raise ValueError("Source commands must be finite and norm bounded")
    if (active[:, 1:] & ~active[:, :-1]).any():
        raise ValueError("Source activity must be a monotone prefix")
    if (actions[~active].abs() > 0).any():
        raise ValueError("Post-done padding was expected to contain only zero commands")
    record = group["counterfactual_snapshots"][0]
    if len(record["rows"]) != n or record["snapshot"]["batch_size"] != n:
        raise ValueError("Must restore the entire source batch, never select a worst-case subset")
    if [r["composition_row"] for r in record["rows"]] != list(range(n)):
        raise ValueError("Post-prelude snapshot does not preserve source batch order")
    if any(r["snapshot_stage"] != "post_prelude" for r in record["rows"]):
        raise ValueError("The common start must precede the first task action")
    return snapshot_from_record(record)


@torch.no_grad()
def replay_group(group, substeps):
    snapshot = validate_source(group)
    n, horizon = group["actions"].shape[:2]
    env = make_profile(group["physics_profile"], num_envs=n, substeps=substeps, seed=101)
    restore_across_substeps(env, snapshot)
    if not torch.equal(physical_state(env), group["trajectory"][:, 0]):
        raise AssertionError("Full post-prelude physical state was not copied exactly")
    states = [physical_state(env).clone()]
    finite, walls = [_finite(env).clone()], [_at_wall(env).clone()]
    for command in group["actions"].unbind(1):
        # clone prevents any simulator implementation from aliasing the saved data.
        env.step([command[:, 0].clone(), command[:, 1].clone()])
        states.append(physical_state(env).clone())
        finite.append(_finite(env).clone())
        walls.append(_at_wall(env).clone())
    trajectory = torch.stack(states, 1)
    weld_xy, weld_angle = _weld_residual(trajectory)
    return dict(trajectory=trajectory, finite=torch.stack(finite, 1), touched_wall=torch.stack(walls, 1),
        weld_xy=weld_xy, weld_angle=weld_angle, substeps=substeps,
        time=torch.arange(horizon+1) * env.world.dt, dt=float(env.world.dt),
        full_batch_size=n, exact_source_joint_actions=True)


def _location(values, mask, group, *, first_exceedance=None):
    if not bool(mask.any()):
        return None
    if first_exceedance is None:
        index = int(torch.where(mask, values, -torch.inf).flatten().argmax())
        row_index, state_step = divmod(index, values.shape[1])
    else:
        breaches = (mask & (values > first_exceedance)).nonzero()
        if not len(breaches):
            return None
        first_step = int(breaches[:, 1].min())
        at_first = breaches[breaches[:, 1] == first_step]
        best = values[at_first[:, 0], at_first[:, 1]].argmax()
        row_index, state_step = map(int, at_first[best])
    phase_id = -1 if state_step == 0 else int(group["phases"][row_index, state_step-1])
    row = group["rows"][row_index]
    return dict(row_index=row_index, state_step=state_step, seconds_after_post_prelude=state_step*.1,
        phase_at_preceding_action="initial" if phase_id < 0 else group["phase_names"][phase_id],
        source_was_active=state_step == 0 or bool(group["active"][row_index, state_step-1]),
        support=row["support"], planner=row["planner"], initial_angle_deg=row["initial_angle_deg"],
        nuisance_id=row["nuisance_id"], goal_pair=row["goal_pair"], value=float(values[row_index, state_step]))


def error_summary(left, right, mask, group):
    if left.shape != right.shape or mask.shape != left.shape[:2] or left.shape[-1] != 24:
        raise ValueError("Errors require aligned full states and a physical-time mask")
    xy = (left[..., 12:14]-right[..., 12:14]).norm(dim=-1)
    angle = wrap(left[..., 16]-right[..., 16]).abs()
    if not bool(mask.any()):
        return dict(samples=0, cases=0, passes=None)
    return dict(samples=int(mask.sum()), cases=int(mask.any(1).sum()),
        max_xy_units=float(xy[mask].max()), rms_xy_units=float(xy[mask].square().mean().sqrt()),
        max_angle_rad=float(angle[mask].max()), rms_angle_rad=float(angle[mask].square().mean().sqrt()),
        max_absolute_complete_state_error=float((left-right).abs()[mask].max()),
        max_xy_location=_location(xy, mask, group), max_angle_location=_location(angle, mask, group),
        first_xy_breach=_location(xy, mask, group, first_exceedance=MAX_XY),
        first_angle_breach=_location(angle, mask, group, first_exceedance=MAX_ANGLE),
        passes=bool((xy[mask] <= MAX_XY).all() and (angle[mask] <= MAX_ANGLE).all()))


def summarize_group(group, replay160, replay320):
    source = group["trajectory"]
    active = torch.cat((torch.ones(len(source), 1, dtype=torch.bool), group["active"]), 1)
    whole = torch.ones_like(active)
    a, b = replay160["trajectory"], replay320["trajectory"]
    if not torch.equal(replay160["time"], replay320["time"]) or not torch.equal(a[:, 0], b[:, 0]):
        raise ValueError("Refinement must use identical complete starts and timestamps")
    row_nonzero = torch.tensor([r["planner"] != "zero_push" for r in group["rows"]])
    masks = dict(source_active=active, whole_saved_prefix=whole, post_done_zero_command_padding=~active,
                 source_active_excluding_zero_push=active & row_nonzero[:, None])
    refined = {key: error_summary(a, b, mask, group) for key, mask in masks.items()}
    phase_errors = {}
    for phase in range(4):
        mask = torch.cat((torch.zeros(len(source), 1, dtype=torch.bool),
                          (group["phases"] == phase) & group["active"]), 1)
        phase_errors[group["phase_names"][phase]] = error_summary(a, b, mask, group)
    planner_errors = {}
    for planner in dict.fromkeys(row["planner"] for row in group["rows"]):
        selected = torch.tensor([row["planner"] == planner for row in group["rows"]])
        planner_errors[planner] = error_summary(a, b, active & selected[:, None], group)
    active_reproduction = error_summary(source, a, active, group)
    whole_reproduction = error_summary(source, a, whole, group)
    exact_reproduction = bool(torch.equal(source, a))
    summary = dict(goal_pair=group["rows"][0]["goal_pair"], profile=group["physics_profile"],
        cases=len(source), source_substeps=160, comparison_substeps=320, horizon_steps=source.shape[1]-1,
        horizon_seconds=float(replay160["time"][-1]), physical_dt=.1,
        initial_full_state_identical=True, same_batch_size_and_order=True,
        controller_and_model_queries=0, exact_joint_action_replay=True,
        maximum_source_action_norm=float(group["actions"].norm(dim=-1).max()),
        source_nonzero_primary_command_samples=int((group["actions"][:, :, 0].norm(dim=-1) > 1e-8).sum()),
        source_zero_push_rows=int((~row_nonzero).sum()),
        source_reproduction_bit_equal=exact_reproduction,
        source_reproduction_active=active_reproduction, source_reproduction_whole_prefix=whole_reproduction,
        tolerances=dict(max_xy_units=MAX_XY, max_angle_rad=MAX_ANGLE), refinement=refined,
        refinement_by_source_phase=phase_errors, refinement_by_planner=planner_errors,
        physical_diagnostics={str(ss): dict(all_finite=bool(run["finite"].all()),
            active_wall_cases=int((run["touched_wall"] & active).any(1).sum()),
            max_active_weld_xy_units=float(run["weld_xy"][active].max()),
            max_active_weld_angle_rad=float(run["weld_angle"][active].max()))
            for ss, run in ((160, replay160), (320, replay320))})
    if not exact_reproduction:
        summary["interpretation"] = "Original 160 trajectory was not bit reproduced; inspect replay validity before attributing refinement differences."
    elif refined["source_active_excluding_zero_push"]["passes"]:
        summary["interpretation"] = "The exact active action sequences pass this long-task numerical gate; remaining closed-loop discrepancies may arise from changed preparations, feedback and event decisions."
    else:
        summary["interpretation"] = "Identical active joint actions fail the long-task numerical gate; local-grid convergence does not establish composition-wide stability and closed-loop differences cannot be assigned solely to event decisions."
    return summary


def provenance():
    paths = [Path(__file__), Path(__file__).with_name("physics_profiles.py"),
             Path(__file__).with_name("abstraction_convergence.py"),
             Path(__file__).with_name("stable_composition.py"), Path(__file__).with_name("snapshot.py")]
    return dict(git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        versions={key: importlib.metadata.version(key) for key in ("vmas", "torch", "numpy")},
        torch_threads=torch.get_num_threads(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/novelty/stable_composition_selected_matched.pt"))
    parser.add_argument("--out", type=Path, default=Path("data/novelty/stable_task_numerics"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = torch.load(args.source, weights_only=False)
    groups = [g for g in source["groups"] if g["rows"][0]["substeps"] == 160]
    report = dict(provenance=provenance(), source_path=str(args.source),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(), groups=[],
        protocol="Exact 160-origin joint-action replay from identical post-prelude full states at common physical time; no event alignment.")
    outputs = []
    started = time.time()
    for index, group in enumerate(groups):
        a, b = replay_group(group, 160), replay_group(group, 320)
        summary = summarize_group(group, a, b)
        report["groups"].append(summary)
        outputs.append(dict(rows=group["rows"], source_actions=group["actions"], source_active=group["active"],
            source_phases=group["phases"], initial_snapshot=group["counterfactual_snapshots"][0],
            replay160=a, replay320=b))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        raw_path = args.out.with_suffix(".pt")
        torch.save(dict(groups=outputs), raw_path)
        report["raw_path"] = str(raw_path)
        report["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        report["wall_seconds"] = time.time()-started
        args.out.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(dict(group=index, goal_pair=summary["goal_pair"], exact160=summary["source_reproduction_bit_equal"],
            active=summary["refinement"]["source_active"], interpretation=summary["interpretation"],
            wall_seconds=report["wall_seconds"])), flush=True)


if __name__ == "__main__":
    main()
