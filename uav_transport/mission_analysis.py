"""Read-only verification and reporting of composed UAV transport missions.

Delivery, operational constraints and diagnostic cable loads are reported
separately. Learned decisions are reconstructed from saved observations and
controller memory using the frozen predictors, without simulator rollouts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from functools import lru_cache
import json
from pathlib import Path
import pickle

import numpy as np

from .collect import candidate_bank
from .local_model import features, load_model, prediction_metrics, TARGET_NAMES
from .mission import LEARNED, MissionConfig, model_kind, outcome_scores
from .selection_audit import margin_coverage, sha, verify_references


PAIRS = (("wm_small_direct", "wm_small"), ("wm_large_direct", "wm_large"),
         ("true_option_direct", "true_option"), ("rule_fixed", "rule_reconfigure"))
SCORE_FIELDS = ("delivery_completion", "constrained_completion")


def equal_tree(left, right):
    if is_dataclass(left): left = asdict(left)
    if is_dataclass(right): right = asdict(right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def max_difference(left, right):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    return float(np.max(np.abs(a-b))) if a.size else 0.


def mission_key(row):
    return row["planner"], int(row["model_seed"]), row["case_id"]


def rescore_delivery(stages, operations, config, *, finite, collision, warnings):
    """A later successful leg cannot replace an earlier failed waypoint."""
    by_name = {s["name"]: s for s in stages}
    if len(by_name) != len(stages): raise ValueError("Mission stages must have unique names")
    waypoint = [bool(by_name.get(f"leg_{i}", {}).get("success", False)) for i in (1, 2, 3)]
    completion = [bool(operations[i]["summary"]["completion"]) if i < len(operations) else False for i in range(2)]
    slack = [bool(operations[i]["summary"]["donor_slack_seconds"]+1e-10 >= config.required_slack_seconds)
             if i < len(operations) else False for i in range(2)]
    delivery = bool(all(waypoint) and all(completion) and all(slack) and finite and not collision and warnings == 0)
    constrained = delivery and all(op["summary"]["peak_payload_excursion"] <= config.payload_excursion_limit
        and op["summary"]["peak_donor_tension"] <= config.donor_tension_limit for op in operations)
    return dict(delivery_completion=bool(delivery), constrained_completion=bool(constrained),
                waypoint_successes=waypoint, option_completions=completion, actual_slack_successes=slack)


def verify_execution(run):
    row, trace = run["summary"], run["trace"]
    config = MissionConfig(**run["mission_config"])
    steps, dt = row["control_steps"], run["plant_config"]["dt"]
    control_dt = run["plant_config"]["control_dt"]
    issues = []
    if trace["actions"].shape != (steps, 3, 4) or trace["physical_states"].shape != (steps+1, 64):
        issues.append("physical action/state clock differs from the recorded mission length")
    if len(trace["observations"]) != steps or len(trace["stages"]) != steps:
        issues.append("per-step observation or stage trace is incomplete")
    if not np.array_equal(run["initial_snapshot"].state, trace["integration_states"][0]):
        issues.append("initial complete snapshot differs from the integration trace")
    if not np.array_equal(run["final_snapshot"].state, trace["integration_states"][-1]):
        issues.append("terminal complete snapshot differs from the integration trace")
    if not np.isclose(row["total_seconds"], steps*control_dt, atol=1e-12, rtol=0):
        issues.append("reported duration differs from the physical control clock")
    sub = trace["substeps"]
    if len(sub["time"]) != row["physics_steps"] or row["physics_steps"] != steps*round(control_dt/dt):
        issues.append("physics substeps are missing from the complete mission")
    end = 0
    goal_checks = []
    for stage in run["stages"]:
        begin, stop = stage["begin_step"], stage["end_step"]
        if begin != end or stop <= begin:
            issues.append("physical stage intervals are not a contiguous ordered sequence")
        end = stop
        if trace["stages"][begin:stop] != [stage["name"]]*(stop-begin):
            issues.append("stage labels differ from the declared physical interval")
        if not stage["name"].startswith("leg_"): continue
        dwell = 0
        for local, observation in enumerate(trace["observations"][begin:stop]):
            eligible = (local+1)*control_dt+1e-10 >= config.move_seconds
            okay = (np.linalg.norm(observation["payload_pos"]-stage["target"]) <= config.position_tolerance
                and np.linalg.norm(observation["payload_vel"]) <= config.speed_tolerance)
            dwell = dwell+1 if eligible and okay else 0
        required = max(1, int(np.ceil(config.goal_dwell_seconds/control_dt-1e-10)))
        matches = bool(stage["success"]) == bool(dwell >= required)
        goal_checks.append(dict(stage=stage["name"], final_dwell_steps=dwell, required=required, matches=matches))
        if not matches: issues.append("waypoint dwell score differs from saved actual observations: "+stage["name"])
    if end != steps: issues.append("recorded stages do not cover the full mission")
    for operation in run["operations"]:
        label = np.asarray([float(operation["summary"][name]) for name in TARGET_NAMES])
        if not np.array_equal(label, operation["label"]):
            issues.append("executed operation label differs from its metric summary")
    finite = bool(np.isfinite(trace["physical_states"]).all() and np.asarray(sub["finite"]).all())
    collision = bool(np.asarray(sub["collision"]).any())
    warnings = int(np.asarray(run["final_snapshot"].warning_number).sum())
    scores = rescore_delivery(run["stages"], run["operations"], config,
        finite=finite, collision=collision, warnings=warnings)
    for name, value in scores.items():
        if row[name] != value: issues.append("mission score differs from its complete ordered execution: "+name)
    if finite != row["finite"] or collision != row["collision"] or warnings != row["warning_count"]:
        issues.append("safety summary differs from the recorded physical execution")
    tensions = np.asarray(sub["cable_tensions"])
    observed_tensions = np.asarray([obs["cable_tensions"] for obs in trace["observations"]])
    maximum_cables = np.maximum(tensions.max(0), observed_tensions.max(0))
    non_donor_max = []
    for op in run["operations"]:
        donor = int(op["donor"]); other = [i for i in range(3) if i != donor]
        force = np.asarray(op["trace"]["substeps"]["cable_tensions"])
        observed = np.asarray([o["cable_tensions"] for o in op["trace"]["observations"]])
        non_donor_max.append(float(max(force[:, other].max(), observed[:, other].max())))
    motor_limit = float(run["plant_config"]["rotor_max"])
    actuator = trace["physical_states"][:, -12:]
    motor_violation = bool((actuator < -1e-12).any() or (actuator > motor_limit+1e-12).any()
        or row["maximum_rotor_force"] > motor_limit+1e-12)
    return dict(case_id=row["case_id"], planner=row["planner"], model_seed=row["model_seed"],
        issues=issues, scores_reproduced=not issues, goal_dwell_checks=goal_checks,
        maximum_operation_excursion=max(row["option_peak_excursions"], default=None),
        maximum_operation_donor_tension=max(row["option_peak_donor_tensions"], default=None),
        maximum_all_cable_tension=float(maximum_cables.max()), maximum_tension_by_cable=maximum_cables.tolist(),
        maximum_non_donor_tension_by_operation=non_donor_max,
        any_cable_above_registered_donor_limit_seconds=float(np.any(tensions > config.donor_tension_limit, axis=1).sum()*dt),
        per_cable_above_registered_donor_limit_seconds=((tensions > config.donor_tension_limit).sum(0)*dt).tolist(),
        rotor_bound_violation=motor_violation,
        all_cable_diagnostic_note="All-cable and non-donor maxima are diagnostics; the registered constrained score remains donor-based.")


def compare_prefix(left, right):
    a, b = left["summary"], right["summary"]
    if (a["case_id"], a["model_seed"]) != (b["case_id"], b["model_seed"]):
        raise ValueError("Paired missions must have the same case and model-fit seed")
    if not equal_tree(left["case"], right["case"]): raise ValueError("Paired case declarations differ")
    if not left["decisions"] or not right["decisions"]:
        raise ValueError("Pair is missing its first operation decision")
    lstop, rstop = (int(run["decisions"][0]["mission_step"]) for run in (left, right))
    stop = min(lstop, rstop)
    prefix_checks = {}
    for name, upto in (("physical_states", stop+1), ("integration_states", stop+1),
                       ("actions", stop), ("external_forces", stop)):
        lvalue, rvalue = left["trace"][name][:upto], right["trace"][name][:upto]
        prefix_checks[name] = dict(bitwise=np.array_equal(lvalue, rvalue),
                                  max_absolute_difference=max_difference(lvalue, rvalue))
    lfirst = next(s for s in left["stages"] if s["name"] == "leg_1")
    rfirst = next(s for s in right["stages"] if s["name"] == "leg_1")
    leg_fields = ("begin_step", "end_step", "success", "goal_dwell_steps", "required_dwell_steps")
    complete_initial_equal = equal_tree(left["initial_snapshot"], right["initial_snapshot"])
    complete_handoff_equal = equal_tree(left["decisions"][0]["start_snapshot"], right["decisions"][0]["start_snapshot"])
    controller_equal = equal_tree(left["initial_controller_memory"], right["initial_controller_memory"])
    handoff_controller_equal = equal_tree(left["decisions"][0]["controller_memory"], right["decisions"][0]["controller_memory"])
    first_leg_equal = all(equal_tree(lfirst[k], rfirst[k]) for k in leg_fields)
    matches = (lstop == rstop and complete_initial_equal and complete_handoff_equal and controller_equal
        and handoff_controller_equal and first_leg_equal and all(x["bitwise"] for x in prefix_checks.values()))
    scores = {name: dict(direct=bool(a[name]), expanded=bool(b[name]),
        improved=not a[name] and b[name], worsened=a[name] and not b[name]) for name in SCORE_FIELDS}
    return dict(direct_planner=a["planner"], expanded_planner=b["planner"], case_id=a["case_id"],
        model_seed=a["model_seed"], prefix_steps=stop, operation_start_steps=[lstop, rstop],
        first_leg_clock_and_score_equal=first_leg_equal, initial_snapshot_equal=complete_initial_equal,
        operation_start_snapshot_equal=complete_handoff_equal, initial_controller_equal=controller_equal,
        operation_start_controller_equal=handoff_controller_equal, traces=prefix_checks,
        prefix_matches=bool(matches), scores=scores)


def audit_decisions(run, models, model_manifest):
    planner, seed = run["summary"]["planner"], run["summary"]["model_seed"]
    cfg = MissionConfig(**run["mission_config"])
    output = []
    for i, record in enumerate(run["decisions"]):
        issues = []
        operation = run["operations"][i]
        if operation["mission_begin_step"] != record["mission_step"]:
            issues.append("decision and executed operation begin at different physical steps")
        if operation["reconfigure"] != record["selected_reconfigure"] or not equal_tree(operation["recovery_config"], record["selected_recovery_config"]):
            issues.append("executed option differs from the selected option")
        source = record["prediction_source"]
        learned = planner in LEARNED
        oracle = planner.startswith("true_option")
        rebuilt = None
        prediction_error, prediction_bitwise, oracle_error = None, None, None
        selected_matches, scores_match = True, True
        if learned or oracle:
            bank = candidate_bank(record["slack_distance"], expanded=not planner.endswith("_direct"))
            expected_bank = [dict(reconfigure=mode, recovery_config=asdict(rc)) for mode, rc in bank]
            if not equal_tree(expected_bank, record["options"]):
                issues.append("saved candidate bank differs from the registered interface")
            if record["candidate_count"] != len(bank) or record["predictor_evaluations"] != len(bank):
                issues.append("prediction count differs from declared candidate bank")
            if learned:
                key = f"{model_kind(planner)}/seed{seed}"
                expected_margins = np.asarray(model_manifest["models"][key]["margins"])
                if source != "frozen_local_model" or record["simulated_candidate_control_steps"] != 0 or record["simulated_candidate_physics_steps"] != 0:
                    issues.append("learned decision source or future-simulator accounting is invalid")
                x = np.stack([features(record["observation"], run["plant_config"], record["donor"], rc,
                    mode, record["controller_memory"]) for mode, rc in bank])
                rebuilt = models[key].predict(x)
                prediction_error = max_difference(rebuilt, record["predictions"])
                prediction_bitwise = np.array_equal(rebuilt, record["predictions"])
                if not prediction_bitwise: issues.append("fresh frozen inference differs from recorded predictions")
            else:
                expected_margins = np.zeros(2)
                if source != "true_option_simulator": issues.append("exact predictor source is invalid")
                rebuilt = np.asarray(record["predictions"])
                oracle_error = max_difference(record["selected_prediction"], operation["label"])
                if oracle_error is None or oracle_error > 1e-9:
                    issues.append("exact selected prediction differs from its executed eight-outcome label")
            if not np.array_equal(expected_margins, record["validation_margins"]):
                issues.append("decision uses different margins from the frozen model manifest")
            scores, feasible, fallback = outcome_scores(rebuilt, cfg, expected_margins)
            selected_matches = int(np.argmin(scores)) == record["selected_index"]
            scores_match = np.array_equal(scores, record["ranking_scores"])
            if not selected_matches or not scores_match:
                issues.append("chosen argmin or criterion differs from the frozen decision rule")
            if not np.array_equal(feasible, record["predicted_feasible"]) or int(feasible.sum()) != record["feasible_count"]:
                issues.append("predicted-feasibility bookkeeping differs from the declared criterion")
            if fallback != record["minimum_risk_fallback"] or fallback != record["feasibility_fallback"]:
                issues.append("fallback bookkeeping differs from the declared criterion")
            if not np.array_equal(rebuilt[record["selected_index"]], record["selected_prediction"]):
                issues.append("selected prediction does not match its bank row")
        else:
            if source != "fixed_feedback_rule" or record["predictor_evaluations"] != 0 or record["simulated_candidate_control_steps"] != 0:
                issues.append("fixed feedback rule unexpectedly queried a predictor")
            if record["selected_reconfigure"] != (planner == "rule_reconfigure"):
                issues.append("fixed feedback rule chose the wrong geometry policy")
        output.append(dict(case_id=run["summary"]["case_id"], planner=planner, model_seed=seed,
            operation_index=i, issues=issues, verified=not issues, learned=learned, oracle=oracle,
            prediction_source=source, predictions_bitwise=prediction_bitwise,
            prediction_max_absolute_difference=prediction_error, oracle_actual_max_absolute_difference=oracle_error,
            selected_argmin_matches=selected_matches, ranking_scores_match=scores_match,
            selected_reconfigure=bool(record["selected_reconfigure"]),
            selected_durations=[record["selected_recovery_config"][key] for key in ("rearrange_seconds", "retension_seconds")],
            candidate_count=int(record["candidate_count"]), predictor_evaluations=int(record["predictor_evaluations"]),
            predicted_feasible_candidates=int(record["feasible_count"]), fallback=bool(record["minimum_risk_fallback"]),
            simulated_candidate_control_steps=int(record["simulated_candidate_control_steps"]),
            simulated_candidate_physics_steps=int(record["simulated_candidate_physics_steps"]),
            decision_seconds=float(record["decision_seconds"]),
            decision_exceeds_control_period=bool(record["decision_seconds"] > run["plant_config"]["control_dt"]),
            actual_local_outcome={name: float(value) for name, value in zip(TARGET_NAMES, operation["label"])}))
    return output


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values): return dict(count=0, minimum=None, median=None, mean=None, p95=None, maximum=None)
    return dict(count=len(values), minimum=float(values.min()), median=float(np.median(values)),
        mean=float(values.mean()), p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def summarize(rows, execution, decisions, *, by_seed=False):
    keys = sorted({(r["planner"], r["model_seed"] if by_seed else None) for r in rows})
    output = []
    for planner, seed in keys:
        selected = [r for r in rows if r["planner"] == planner and (seed is None or r["model_seed"] == seed)]
        checks = [r for r in execution if r["planner"] == planner and (seed is None or r["model_seed"] == seed)]
        choices = [r for r in decisions if r["planner"] == planner and (seed is None or r["model_seed"] == seed)]
        result = dict(planner=planner, model_seed=seed, missions=len(selected),
            delivery_completion=sum(r["delivery_completion"] for r in selected),
            constrained_completion=sum(r["constrained_completion"] for r in selected),
            continuous_operation_excursion=distribution([r["maximum_operation_excursion"] for r in checks]),
            continuous_operation_donor_tension=distribution([r["maximum_operation_donor_tension"] for r in checks]),
            whole_mission_all_cable_tension=distribution([r["maximum_all_cable_tension"] for r in checks]),
            maximum_non_donor_tension=max((max(r["maximum_non_donor_tension_by_operation"], default=0.) for r in checks), default=0.),
            mission_seconds=distribution([r["total_seconds"] for r in selected]),
            actuator_work=distribution([r["absolute_actuator_work"] for r in selected]),
            squared_thrust_integral=distribution([r["total_squared_thrust_integral"] for r in selected]),
            flight_tracking_rmse=distribution([r["flight_tracking_rmse"] for r in selected if r["flight_tracking_rmse"] is not None]),
            maximum_rotor_force=max(r["maximum_rotor_force"] for r in selected),
            rotor_bound_violations=sum(r["rotor_bound_violation"] for r in checks),
            motor_bound_seconds=sum(r["motor_bound_seconds"] for r in selected),
            minimum_uav_separation=min(r["minimum_uav_separation"] for r in selected),
            maximum_uav_tilt=max(r["peak_uav_tilt"] for r in selected),
            all_finite=all(r["finite"] for r in selected), collisions=sum(r["collision"] for r in selected),
            warnings=sum(r["warning_count"] for r in selected), decisions=len(choices),
            selection_latency_seconds=distribution([r["decision_seconds"] for r in choices]),
            decisions_exceeding_control_period=sum(r["decision_exceeds_control_period"] for r in choices),
            fallbacks=sum(r["fallback"] for r in choices),
            fallback_fraction=float(np.mean([r["fallback"] for r in choices])) if choices else None,
            candidate_slots=sum(r["candidate_count"] for r in choices),
            predictor_evaluations=sum(r["predictor_evaluations"] for r in choices),
            simulated_candidate_control_steps=sum(r["simulated_candidate_control_steps"] for r in choices),
            simulated_candidate_physics_steps=sum(r["simulated_candidate_physics_steps"] for r in choices),
            geometry_reconfigurations_selected=sum(r["selected_reconfigure"] for r in choices),
            all_decisions_verified=all(r["verified"] for r in choices),
            all_execution_scores_reproduced=all(r["scores_reproduced"] for r in checks))
        output.append(result)
    return output


@lru_cache(maxsize=2)
def load_run(path):
    with Path(path).open("rb") as stream: return pickle.load(stream)


def run_audit(manifest_path, models_path):
    manifest_path, models_path = Path(manifest_path), Path(models_path)
    manifest, model_manifest = json.loads(manifest_path.read_text()), json.loads(models_path.read_text())
    references = {str(manifest_path): sha(manifest_path), str(models_path): sha(models_path)}
    references.update(manifest["source_sha256"]); references.update(manifest["model_sha256"])
    references.update(model_manifest["source_sha256"])
    references[model_manifest["dataset_path"]] = model_manifest["dataset_sha256"]
    references[model_manifest["training_registration_path"]] = model_manifest["training_registration_sha256"]
    references[manifest["registration_path"]] = manifest["registration_sha256"]
    for source in (Path(__file__), Path("uav_transport/selection_audit.py")):
        references[str(source)] = sha(source)
    index = {}
    for item in manifest["results"]:
        key = mission_key(item["summary"])
        if key in index: raise ValueError("Duplicate mission result")
        index[key] = item; references[item["raw_path"]] = item["raw_sha256"]
    expected = {(planner, seed, case["case_id"]) for planner in manifest["planners"]
        for seed in (manifest["model_seeds"] if planner in LEARNED else [0]) for case in manifest["cases"]}
    if set(index) != expected or len(index) != manifest["mission_count"]:
        raise ValueError("Mission matrix is incomplete or contains undeclared cases")
    verify_references(references)
    models = {key: load_model(entry["prefix"]) for key, entry in model_manifest["models"].items()}
    rows, execution, decisions, pairs = [], [], [], []
    oracle_x, oracle_y = [], []
    for key, item in sorted(index.items()):
        run = load_run(item["raw_path"])
        if not equal_tree(run["summary"], item["summary"]):
            raise ValueError("Raw mission summary differs from the manifest")
        rows.append(run["summary"]); execution.append(verify_execution(run))
        decisions.extend(audit_decisions(run, models, model_manifest))
        direct = next((a for a, b in PAIRS if b == key[0]), None)
        if direct is not None:
            other_key = (direct, key[1], key[2])
            if other_key not in index: raise ValueError("Missing paired direct mission")
            pairs.append(compare_prefix(load_run(index[other_key]["raw_path"]), run))
        if key[0] == "true_option":
            for record in run["decisions"]:
                bank = candidate_bank(record["slack_distance"], expanded=True)
                oracle_x.append(np.stack([features(record["observation"], run["plant_config"], record["donor"],
                    cfg, mode, record["controller_memory"]) for mode, cfg in bank]))
                oracle_y.append(np.asarray(record["predictions"]).copy())
    pair_summary = []
    for a, b in PAIRS:
        own = [row for row in pairs if row["direct_planner"] == a]
        pair_summary.append(dict(direct=a, expanded=b, pairs=len(own),
            all_prefixes_bitwise=all(x["prefix_matches"] for x in own),
            scores={name: dict(direct=sum(x["scores"][name]["direct"] for x in own),
                expanded=sum(x["scores"][name]["expanded"] for x in own),
                improved=sum(x["scores"][name]["improved"] for x in own),
                worsened=sum(x["scores"][name]["worsened"] for x in own)) for name in SCORE_FIELDS}))
    common = []
    if oracle_x:
        x, y = np.concatenate(oracle_x), np.concatenate(oracle_y)
        for key, model in models.items():
            prediction = model.predict(x)
            common.append(dict(predictor=key, examples=len(y), metrics=prediction_metrics(prediction, y),
                margin_coverage=margin_coverage(prediction, y, model_manifest["models"][key]["margins"])))
    verify_references(references)
    return dict(manifest_path=str(manifest_path), models_path=str(models_path), missions=len(rows),
        summary=summarize(rows, execution, decisions), per_seed=summarize(rows, execution, decisions, by_seed=True),
        paired_summary=pair_summary, paired_rows=pairs, execution_checks=execution, decision_checks=decisions,
        all_execution_scores_reproduced=all(x["scores_reproduced"] for x in execution),
        all_decisions_verified=all(x["verified"] for x in decisions), all_paired_prefixes_bitwise=all(x["prefix_matches"] for x in pairs),
        validation_margins={key: entry["margins"] for key, entry in model_manifest["models"].items()},
        common_state_predictions=common, common_state_source="All saved candidate outcomes at true_option mission decision states; diagnostic use only",
        input_sha256=references, verified_input_files=len(references),
        latency_note="Recorded decision wall times under the original parallel workload. Simulation time advances through physical control steps; selection wall time is reported separately.",
        outcome_note="Delivery requires all waypoint, local-operation, actual-slack and physical safety conditions. Constrained completion adds the registered per-operation excursion and donor-tension limits.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/uav_transport/mission_v1.json"))
    parser.add_argument("--models", type=Path, default=Path("data/uav_transport/models_v1_manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("data/uav_transport/mission_v1_analysis"))
    args = parser.parse_args(); output = Path(str(args.out)+".json")
    if output.exists(): raise FileExistsError("Preserve completed mission audits with a fresh output prefix")
    report = run_audit(args.manifest, args.models)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(output), missions=report["missions"],
        all_execution_scores_reproduced=report["all_execution_scores_reproduced"],
        all_decisions_verified=report["all_decisions_verified"],
        all_paired_prefixes_bitwise=report["all_paired_prefixes_bitwise"], paired=report["paired_summary"])), flush=True)
    if not all(report[name] for name in ("all_execution_scores_reproduced", "all_decisions_verified", "all_paired_prefixes_bitwise")):
        raise AssertionError("Mission audit has retained verification failures; inspect the saved report")


if __name__ == "__main__": main()
