"""Paired and budget audits for the frozen predictor by action-interface matrix.

Pairing uses physical condition and fit seed, never row order or final success.
Numerical prefixes are compared with an explicit absolute tolerance; discrete
phase clocks and ordered-goal labels must still agree exactly. This module does
not train models, query future physics, or modify experimental artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

import torch


PREFIX_ATOL = 1e-6
PREFIX_PHASES = ("rotate_first", "retreat", "recovery")
ALL_PHASES = PREFIX_PHASES + ("rotate_final",)
PREDICTORS = ("wm_full", "wm_large", "true_option")
BASE_ARMS = tuple(arm for predictor in PREDICTORS for arm in (predictor+"_direct", predictor))
HISTORICAL_ARMS = ("wm_full_direct", "wm_full", "wm_large", "true_option")
IDENTITY_FIELDS = ("model_seed", "support", "initial_angle_deg", "nuisance_id",
                   "first_goal_rad", "relative_goal_rad", "physics_profile", "substeps")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_key(row):
    """Seeds are repeated fits, not new physical samples."""
    return tuple(row[key] for key in IDENTITY_FIELDS)


def row_index(rows):
    result = {}
    for i, row in enumerate(rows):
        key = case_key(row)
        if key in result:
            raise ValueError("Duplicate paired physical condition: "+repr(key))
        expected = (row["first_goal_success_at_phase_boundary"] and row["final_goal_dwell_success"]
            and row["terminal_success"] and row["finite"] and not row["wall"])
        if bool(row["ordered_goal_success"]) != bool(expected):
            raise ValueError("Ordered success must retain failed first stages and safety failures")
        result[key] = i
    return result


def _tensor_difference(left, right):
    left, right = torch.as_tensor(left), torch.as_tensor(right)
    same_shape = left.shape == right.shape
    if not same_shape:
        return dict(same_shape=False, bitwise=False, max_absolute_difference=None)
    if left.is_floating_point() or right.is_floating_point():
        finite = torch.isfinite(left) & torch.isfinite(right)
        if not finite.all():
            return dict(same_shape=True, bitwise=False, max_absolute_difference=None)
        maximum = float((left[finite].double()-right[finite].double()).abs().max()) if finite.any() else 0.
        exact = bool(torch.equal(left[finite], right[finite]))
    else:
        maximum = float((left.to(torch.float64)-right.to(torch.float64)).abs().max()) if left.numel() else 0.
        exact = torch.equal(left, right)
    return dict(same_shape=True, bitwise=bool(exact), max_absolute_difference=maximum)


def _tree_comparison(left, right, atol):
    differences, structural = [], []
    def visit(a, b, path):
        if torch.is_tensor(a) or torch.is_tensor(b):
            result = _tensor_difference(a, b)
            differences.append(dict(path=path, **result))
        elif isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b): structural.append(path+": keys differ")
            for key in sorted(set(a) & set(b)): visit(a[key], b[key], path+"/"+str(key))
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b): structural.append(path+": length differs")
            for i, (x, y) in enumerate(zip(a, b)): visit(x, y, path+"/"+str(i))
        elif a != b:
            structural.append(path+": value differs")
    visit(left, right, "root")
    valid = not structural and all(d["same_shape"] and d["max_absolute_difference"] is not None
        and d["max_absolute_difference"] <= atol for d in differences)
    return dict(matches=valid, bitwise=not structural and all(d["bitwise"] for d in differences),
        max_absolute_difference=max((d["max_absolute_difference"] or 0. for d in differences), default=0.),
        structural_differences=structural,
        differing_tensors=[d for d in differences if not d["bitwise"]])


def _snapshot_for_row(group, row, stage):
    for record in group["counterfactual_snapshots"]:
        for index, metadata in enumerate(record["rows"]):
            if metadata["composition_row"] == row and metadata["snapshot_stage"] == stage:
                snapshot = record["snapshot"]
                n = int(snapshot["batch_size"])
                def select(value):
                    if torch.is_tensor(value):
                        return value[index] if value.ndim and value.shape[0] == n else value
                    if isinstance(value, dict): return {k: select(v) for k, v in value.items() if k != "batch_size"}
                    if isinstance(value, tuple): return tuple(select(v) for v in value)
                    if isinstance(value, list): return [select(v) for v in value]
                    return value
                return dict(snapshot=select(snapshot), anchor=record["anchor"][index],
                            history=record["history"][index])
    raise ValueError(f"Missing complete {stage} snapshot for row {row}")


def compare_groups(left, right, *, full_task=False, atol=PREFIX_ATOL):
    """Pair rows without assuming their order or dropping failed prefixes."""
    a, b = row_index(left["rows"]), row_index(right["rows"])
    if set(a) != set(b):
        raise ValueError("Paired groups do not contain exactly the same physical conditions")
    results = []
    for key in sorted(a):
        i, j = a[key], b[key]
        x, y = left["rows"][i], right["rows"][j]
        phases = ALL_PHASES if full_task else PREFIX_PHASES
        ends_a = [x["phase_end_steps"][p] for p in phases]
        ends_b = [y["phase_end_steps"][p] for p in phases]
        prefix_steps = min(ends_a[-1], ends_b[-1])
        state = _tensor_difference(left["trajectory"][i, :prefix_steps+1], right["trajectory"][j, :prefix_steps+1])
        action = _tensor_difference(left["actions"][i, :prefix_steps], right["actions"][j, :prefix_steps])
        prelude = _tree_comparison(
            (left["prelude_trajectory"][i], left["prelude_actions"][i], left["initial_pose"][i]),
            (right["prelude_trajectory"][j], right["prelude_actions"][j], right["initial_pose"][j]), atol)
        start = _tree_comparison(_snapshot_for_row(left, i, "post_prelude"),
                                 _snapshot_for_row(right, j, "post_prelude"), atol)
        handoff = _tree_comparison(_snapshot_for_row(left, i, "post_recovery"),
                                   _snapshot_for_row(right, j, "post_recovery"), atol)
        first_matches = x["first_goal_success_at_phase_boundary"] == y["first_goal_success_at_phase_boundary"]
        trajectory_match = all(item["same_shape"] and item["max_absolute_difference"] is not None
            and item["max_absolute_difference"] <= atol for item in (state, action))
        comparable = (ends_a == ends_b and first_matches and trajectory_match
            and prelude["matches"] and start["matches"] and handoff["matches"])
        labels = ("first_goal_success_at_phase_boundary", "final_goal_dwell_success", "terminal_success",
                  "ordered_goal_success", "finite", "wall")
        s0, s1 = bool(x["ordered_goal_success"]), bool(y["ordered_goal_success"])
        results.append(dict(zip(IDENTITY_FIELDS, key),
            direct_arm=x["planner"], compared_arm=y["planner"], direct_row=i, compared_row=j,
            direct_success=s0, compared_success=s1, improved=not s0 and s1,
            worsened=s0 and not s1, both_success=s0 and s1, both_failure=not s0 and not s1,
            first_goal_matches=first_matches, shared_first_goal_success=bool(first_matches and x[labels[0]]),
            discrete_prefix_clocks_match=ends_a == ends_b, direct_phase_endpoints=ends_a,
            compared_phase_endpoints=ends_b, prefix_steps_compared=prefix_steps,
            full_task_compared=full_task, max_state_difference=state["max_absolute_difference"],
            max_action_difference=action["max_absolute_difference"],
            state_bitwise=state["bitwise"], actions_bitwise=action["bitwise"],
            prelude=prelude, initial_complete_snapshot=start, handoff_complete_snapshot=handoff,
            prefix_matches=comparable, all_scoring_labels_match=all(x[k] == y[k] for k in labels)))
    return results


def summarize_pairs(rows):
    return dict(cases=len(rows), direct_success=sum(r["direct_success"] for r in rows),
        compared_success=sum(r["compared_success"] for r in rows),
        improved=sum(r["improved"] for r in rows), worsened=sum(r["worsened"] for r in rows),
        both_success=sum(r["both_success"] for r in rows), both_failure=sum(r["both_failure"] for r in rows),
        shared_first_goal_success=sum(r["shared_first_goal_success"] for r in rows),
        direct_success_given_shared_first=sum(r["shared_first_goal_success"] and r["direct_success"] for r in rows),
        compared_success_given_shared_first=sum(r["shared_first_goal_success"] and r["compared_success"] for r in rows),
        first_goal_matches=all(r["first_goal_matches"] for r in rows),
        discrete_prefix_clocks_match=all(r["discrete_prefix_clocks_match"] for r in rows),
        prefix_matches=all(r["prefix_matches"] for r in rows),
        scoring_labels_match=all(r["all_scoring_labels_match"] for r in rows),
        all_state_bitwise=all(r["state_bitwise"] for r in rows),
        all_actions_bitwise=all(r["actions_bitwise"] for r in rows),
        unmeasurable_state_pairs=sum(r["max_state_difference"] is None for r in rows),
        unmeasurable_action_pairs=sum(r["max_action_difference"] is None for r in rows),
        max_state_difference=max((r["max_state_difference"] or 0. for r in rows), default=0.),
        max_action_difference=max((r["max_action_difference"] or 0. for r in rows), default=0.))


def canonical_direct_table():
    pushes = [(0., 0.)] + list(itertools.product((-20., 0., 20.), (.15, .3, .6, .9)))
    return torch.tensor([(0., angle, strength, duration) for duration in (5, 10, 20)
                         for angle, strength in pushes], dtype=torch.float32)


def expected_allocation(shift):
    table = canonical_direct_table()
    for duration_index in range(3):
        for force_index in range(12):
            table[duration_index*13+force_index+1, 0] = (duration_index+force_index+shift) % 3
    return table


def _multiset(values):
    return Counter(tuple(row) for row in values.detach().cpu().tolist())


def audit_queries(group):
    """Account selector work and check the declared equal-budget construction."""
    rows = group["rows"]
    options = {}
    for record in group["option_records"]:
        key = (record["row"], record["start_step"])
        if key in options: raise ValueError("More than one option starts at a row/step")
        options[key] = record
    violations, decisions = [], []
    total_wall, oracle_calls = 0., 0
    direct = canonical_direct_table()
    for query in group["queries"]:
        diag, step = query["diagnostics"], int(query["step"])
        table = torch.as_tensor(diag["candidate_table"])
        if table.ndim != 3 or table.shape[0] != len(rows) or table.shape[-1] != 4:
            raise ValueError("Candidate diagnostics must preserve the B by K by 4 table")
        total_wall += float(diag["query_wall_seconds"])
        oracle_calls += int(diag["oracle_calls"])
        for i in query["query_mask"].nonzero().flatten().tolist():
            row = rows[i]
            arm = row["planner"]
            errors = []
            option = options.get((i, step))
            if option is None: raise ValueError("Query has no matching option execution record")
            budget = int(diag["candidate_budget"][i])
            evaluated = int(diag["evaluated_candidates"][i])
            legal = int(diag["candidate_count"][i])
            k = table.shape[1]
            if budget != k or not 0 <= evaluated <= budget or not 0 <= legal <= budget:
                errors.append("invalid declared/evaluated/legal candidate counts")
            if not bool(diag["queried_mask"][i]): errors.append("active query missing prediction accounting")
            if len(_multiset(table[i])) != k: errors.append("duplicate candidates consume the declared budget")
            phase, mode = int(query["phase"][i]), int(option["start_contact_mode"])
            policy = row.get("candidate_policy", "")
            if "budget39_s" in arm or policy.startswith("budget39_s"):
                shift = int((policy or arm).rsplit("s", 1)[1])
                if shift not in (0, 1, 2): errors.append("undeclared allocation shift")
                if budget != 39: errors.append("matched candidate budget differs from 39")
                if _multiset(table[i, :, 1:]) != _multiset(direct[:, 1:]):
                    errors.append("force/direction/duration marginal multiset differs from direct39")
                all_direct = bool(table[i, :, 0].eq(0).all())
                if phase == 0 or mode != 3 or not bool(query["allow_routes"][i]):
                    if not all_direct: errors.append("routes offered before final phase or after acquiring another face")
                if not all_direct and not torch.equal(table[i], expected_allocation(shift)):
                    errors.append("route allocation differs from the prespecified shift")
                if diag["source"][i] == "world_model" and evaluated != 39:
                    errors.append("world model did not evaluate all 39 declared candidates")
            if arm.endswith("_direct") and table[i, :, 0].ne(0).any():
                errors.append("direct-only arm offered a route")
            simulator_steps = int(diag["simulated_candidate_steps"][i])
            active_steps = int(diag["active_simulated_candidate_steps"][i])
            padded_steps = int(diag["padded_simulator_steps"][i])
            padded_substeps = int(diag["padded_simulator_substeps"][i])
            if simulator_steps < 0: errors.append("negative simulated candidate steps")
            if active_steps != simulator_steps or padded_steps < active_steps:
                errors.append("active and padded simulator accounting is inconsistent")
            if padded_substeps != padded_steps*int(row["substeps"]):
                errors.append("padded simulator substeps differ from declared physics resolution")
            if diag["source"][i] == "world_model" and (simulator_steps or padded_steps):
                errors.append("learned selector used future simulator steps")
            decisions.append(dict(row=i, step=step, planner=arm, model_seed=row["model_seed"],
                phase=phase, contact_mode=mode, candidate_budget=budget, evaluated_candidates=evaluated,
                legal_candidates=legal, simulated_candidate_steps=simulator_steps,
                padded_simulator_steps=padded_steps, padded_simulator_substeps=padded_substeps,
                source=diag["source"][i], fallback=bool(diag["budget_fallback"][i])))
            if errors: violations.append(dict(row=i, step=step, planner=arm, reasons=errors))
    per_episode = []
    for i, row in enumerate(rows):
        own = [entry for entry in decisions if entry["row"] == i]
        if len(own) != row["option_queries"]:
            violations.append(dict(row=i, planner=row["planner"], reasons=["query count differs from recorded executed options"]))
        per_episode.append(dict(row=i, **{key: row[key] for key in IDENTITY_FIELDS}, planner=row["planner"],
            decisions=len(own), candidate_budget_total=sum(x["candidate_budget"] for x in own),
            evaluated_candidates=sum(x["evaluated_candidates"] for x in own),
            simulated_candidate_steps=sum(x["simulated_candidate_steps"] for x in own),
            padded_simulator_steps=sum(x["padded_simulator_steps"] for x in own),
            padded_simulator_substeps=sum(x["padded_simulator_substeps"] for x in own),
            total_steps=row["total_steps"], ordered_goal_success=row["ordered_goal_success"]))
    return dict(violations=violations, decisions=decisions, per_episode=per_episode,
                batched_selector_wall_seconds=total_wall, oracle_calls=oracle_calls)


def aggregate_cases(rows):
    def one(selected):
        result = dict(cases=len(selected), first=sum(r["first_goal_success_at_phase_boundary"] for r in selected),
            ordered=sum(r["ordered_goal_success"] for r in selected),
            final=sum(r["final_goal_dwell_success"] for r in selected),
            interface=sum(r["interface_completion_success"] for r in selected),
            finite_all=all(r["finite"] for r in selected), walls=sum(r["wall"] for r in selected),
            mean_total_steps=sum(r["total_steps"] for r in selected)/len(selected))
        for field in ("actuator_absolute_work_by_agent", "norm_force_integral_by_agent", "agent_travel_units"):
            if all(field in r for r in selected):
                result["mean_"+field] = torch.tensor([r[field] for r in selected]).double().mean(0).tolist()
        return result
    aggregate, seeds, supports = [], [], []
    for arm in sorted({r["planner"] for r in rows}):
        subset = [r for r in rows if r["planner"] == arm]
        aggregate.append(dict(planner=arm, **one(subset)))
        for seed in sorted({r["model_seed"] for r in subset}):
            seeds.append(dict(planner=arm, model_seed=seed, **one([r for r in subset if r["model_seed"] == seed])))
        for support in sorted({r["support"] for r in subset}):
            supports.append(dict(planner=arm, support=support, **one([r for r in subset if r["support"] == support])))
    return aggregate, seeds, supports


@lru_cache(maxsize=4)
def load_group(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if len(raw["groups"]) != 1:
        raise ValueError("The factorial audit expects one complete original batch per source file")
    return raw["groups"][0]


def load_manifest(path, *, verify_source=True):
    manifest = json.loads(Path(path).read_text())
    references = {str(path): sha(path)}
    for group in manifest["groups"]:
        for filename, digest in ((group["raw_path"], group["raw_sha256"]),
                                 (group["report_path"], group["report_sha256"])):
            if sha(filename) != digest: raise AssertionError("Manifest artifact digest mismatch: "+filename)
            references[filename] = digest
    if manifest.get("model_path") and manifest.get("model_sha256"):
        references[manifest["model_path"]] = manifest["model_sha256"]
    if verify_source:
        references.update(manifest.get("source_sha256", {}))
        references.update(manifest.get("old_model_sha256", {}))
        if manifest.get("registration_path"):
            references[manifest["registration_path"]] = manifest["registration_sha256"]
    for filename, digest in references.items():
        if sha(filename) != digest: raise AssertionError("Input changed: "+filename)
    index, physical, actual_rows = {}, set(), {}
    for entry in manifest["groups"]:
        key = (entry["planner"], entry["model_seed"], entry["goal"])
        if key in index: raise ValueError("Duplicate planner/seed/goal source group")
        index[key] = entry
        group = load_group(entry["raw_path"])
        row_index(group["rows"])
        for row in group["rows"]:
            if row["planner"] != entry["planner"] or row["model_seed"] != entry["model_seed"]:
                raise ValueError("Source row identity differs from manifest group")
            identity = (row["planner"], case_key(row))
            if identity in physical: raise ValueError("Physical case duplicated across source groups")
            physical.add(identity)
            actual_rows[identity] = row
    if "total_cases" in manifest and len(physical) != manifest["total_cases"]:
        raise ValueError("Raw cases do not match the declared experiment size")
    if "rows" in manifest:
        declared = {(row["planner"], case_key(row)): row for row in manifest["rows"]}
        if len(declared) != len(manifest["rows"]) or set(declared) != physical:
            raise ValueError("Manifest rows differ from the unique raw physical cases")
        for identity, row in declared.items():
            for label in ("ordered_goal_success", "first_goal_success_at_phase_boundary",
                          "final_goal_dwell_success", "terminal_success", "finite", "wall"):
                if row[label] != actual_rows[identity][label]:
                    raise ValueError("Manifest scoring labels differ from raw case labels")
    return manifest, index, references


def run_audit(manifest_path, previous_path=None, *, atol=PREFIX_ATOL):
    manifest, groups, references = load_manifest(manifest_path)
    rows, pairs, audits = [], [], []
    for (arm, seed, goal), entry in sorted(groups.items()):
        group = load_group(entry["raw_path"])
        rows.extend(group["rows"])
        audit = audit_queries(group)
        audits.append(dict(planner=arm, model_seed=seed, goal=goal, source=entry["raw_path"], **audit))
        if arm.endswith("_direct"): continue
        base = arm.split("_budget39_s", 1)[0]
        direct_key = (base+"_direct", seed, goal)
        if direct_key not in groups: raise ValueError("Missing paired direct baseline: "+repr(direct_key))
        direct = load_group(groups[direct_key]["raw_path"])
        pairs.extend(compare_groups(direct, group, atol=atol))
    summaries, seed_summary, support_summary = aggregate_cases(rows)
    pair_summary, pair_support, pair_seed = [], [], []
    for arm in sorted({r["compared_arm"] for r in pairs}):
        own = [r for r in pairs if r["compared_arm"] == arm]
        pair_summary.append(dict(planner=arm, **summarize_pairs(own)))
        for seed in sorted({r["model_seed"] for r in own}):
            pair_seed.append(dict(planner=arm, model_seed=seed,
                **summarize_pairs([r for r in own if r["model_seed"] == seed])))
        for support in sorted({r["support"] for r in own}):
            pair_support.append(dict(planner=arm, support=support,
                **summarize_pairs([r for r in own if r["support"] == support])))
    historical = []
    if previous_path is not None:
        _, previous, old_references = load_manifest(previous_path, verify_source=False)
        references.update(old_references)
        for (arm, seed, goal), entry in sorted(groups.items()):
            if arm not in HISTORICAL_ARMS: continue
            key = (arm, seed, goal)
            if key not in previous: raise ValueError("Missing requested historical comparator")
            historical.extend(compare_groups(load_group(previous[key]["raw_path"]),
                load_group(entry["raw_path"]), full_task=True, atol=atol))
    compute = []
    for arm in sorted({r["planner"] for r in rows}):
        own = [x for x in audits if x["planner"] == arm]
        decisions = [row for x in own for row in x["decisions"]]
        episodes = [row for x in own for row in x["per_episode"]]
        compute.append(dict(planner=arm, episodes=len(episodes), decisions=len(decisions),
            candidate_budget_histogram=dict(sorted(Counter(x["candidate_budget"] for x in decisions).items())),
            evaluated_candidate_histogram=dict(sorted(Counter(x["evaluated_candidates"] for x in decisions).items())),
            candidate_budget_total=sum(x["candidate_budget"] for x in decisions),
            evaluated_candidates=sum(x["evaluated_candidates"] for x in decisions),
            simulated_candidate_steps=sum(x["simulated_candidate_steps"] for x in decisions),
            padded_simulator_steps=sum(x["padded_simulator_steps"] for x in decisions),
            padded_simulator_substeps=sum(x["padded_simulator_substeps"] for x in decisions),
            oracle_calls=sum(x["oracle_calls"] for x in own),
            decisions_per_episode_min=min(x["decisions"] for x in episodes),
            decisions_per_episode_max=max(x["decisions"] for x in episodes),
            mean_evaluated_candidates_per_episode=sum(x["evaluated_candidates"] for x in episodes)/len(episodes),
            batched_selector_wall_seconds=sum(x["batched_selector_wall_seconds"] for x in own)))
    references[str(Path(__file__))] = sha(__file__)
    for filename, digest in references.items():
        if sha(filename) != digest: raise AssertionError("Input changed during audit: "+filename)
    violations = [dict(source=x["source"], **row) for x in audits for row in x["violations"]]
    return dict(manifest_path=str(manifest_path), previous_path=None if previous_path is None else str(previous_path),
        prefix_absolute_tolerance=atol, cases=len(rows), six_cell_table=[r for r in summaries if r["planner"] in BASE_ARMS],
        all_arms=summaries, per_seed=seed_summary, per_support=support_summary,
        paired_summary=pair_summary, paired_support=pair_support, paired_seed=pair_seed, paired_rows=pairs,
        historical_replication=summarize_pairs(historical) if historical else None,
        historical_rows=historical, compute=compute, query_audits=audits,
        budget_violations=violations, budget_valid=not violations,
        all_paired_prefixes_match=all(r["prefix_matches"] for r in pairs),
        input_sha256=references, verified_input_files=len(references),
        interpretation="Candidate budgets are per decision. Total replans, candidate simulations, rollout steps, wall time and physical execution costs are not matched. Fit seeds and allocation shifts repeat the same physical cases; no population confidence intervals are inferred.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("data/novelty/interface_complete_v1.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix-atol", type=float, default=PREFIX_ATOL)
    args = parser.parse_args()
    if args.prefix_atol < 0: raise ValueError("Prefix tolerance must be nonnegative")
    output = args.out.with_suffix(".json")
    if output.exists(): raise FileExistsError("Preserve the existing audit with a fresh output prefix")
    torch.set_num_threads(1)
    result = run_audit(args.manifest, args.previous, atol=args.prefix_atol)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(output), cases=result["cases"],
        table=result["six_cell_table"], paired=result["paired_summary"],
        budget_valid=result["budget_valid"], historical=result["historical_replication"])), flush=True)
    if not result["budget_valid"]:
        raise AssertionError("Candidate-budget audit failed; diagnostic output was preserved")


if __name__ == "__main__":
    main()
