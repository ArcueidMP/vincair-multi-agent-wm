"""Publish compact evidence and portable frozen joint-option weights."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import torch

from vmas_pusht.boundary import _plain
from vmas_pusht.option_models import sha


def export_models(source, destination):
    pack = torch.load(source, weights_only=False)
    output = {k: v for k, v in pack.items() if k != "checkpoints"}
    output["checkpoints"] = {}
    for seed, entry in pack["checkpoints"].items():
        output["checkpoints"][seed] = {k: v for k, v in entry.items() if k not in ("state_dict", "normalization")}
        for name in ("state_dict", "normalization"):
            output["checkpoints"][seed][name] = {k: v.detach().cpu().tolist() for k, v in entry[name].items()}
    output["original_torch_file_sha256"] = sha(source)
    Path(destination).write_text(json.dumps(_plain(output), separators=(",", ":"), allow_nan=False)+"\n")


def restore_models(source, destination):
    pack = json.loads(Path(source).read_text())
    pack["checkpoints"] = {int(k): v for k, v in pack["checkpoints"].items()}
    for entry in pack["checkpoints"].values():
        for name in ("state_dict", "normalization"):
            entry[name] = {k: torch.tensor(v, dtype=torch.float32) for k, v in entry[name].items()}
    if Path(destination).exists():
        raise FileExistsError("Do not replace existing frozen models")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, destination)


def summarize(rows, fine):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["planner"]].append(row)
    output = []
    for name, values in sorted(grouped.items()):
        finer = [fine[(name, r["model_seed"], tuple(r["goal_pair"]), r["support"],
                       r["initial_angle_deg"], r["nuisance_id"])] for r in values]
        output.append(dict(planner=name, predictor=values[0]["predictor"],
            planned_helper=values[0]["planned_helper"], candidate_slots=values[0]["candidate_slots"],
            cases=len(values), first=sum(r["first_goal_success_at_phase_boundary"] for r in values),
            ordered=sum(r["ordered_goal_success"] for r in values),
            fine_ordered=sum(r["replay_fine"]["ordered_goal_success"] for r in finer),
            refinement_passes=sum(r["active_prefix_passes"] for r in finer),
            finite_all=all(r["finite"] for r in values), wall_cases=sum(r["wall"] for r in values),
            failed_options=sum(r["failed_options"] for r in values),
            helper_nonfixed_options=sum(r["helper_nonfixed_options"] for r in values),
            episodes_with_helper_switches=sum(r["helper_switches"] > 0 for r in values),
            helper_choices=dict(sum((Counter(r["helper_choices"]) for r in values), Counter())),
            mean_steps=sum(r["total_steps"] for r in values)/len(values),
            mean_actuator_absolute_work=[sum(r["actuator_absolute_work_by_agent"][j] for r in values)/len(values) for j in (0, 1)]))
    return output


def publish(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/"manifest.json").read_text())
    verification = json.loads((root/"verification.json").read_text())
    rows = [row for item in manifest["groups"] for row in item["rows"]]
    fine, numerical = {}, []
    by_path = {item["path"]: item for item in manifest["groups"]}
    for report in verification["numerical_reports"]:
        sources = by_path[report["source_path"]]["rows"]
        for result in report["results"]:
            numerical.append(result)
            for source, row in zip(sources, result["rows"]):
                key = (source["planner"], source["model_seed"], tuple(source["goal_pair"]),
                       source["support"], source["initial_angle_deg"], source["nuisance_id"])
                fine[key] = row
    summary = summarize(rows, fine)
    paired_groups = defaultdict(list)
    for row in verification["paired_rows"]:
        stem = row["predictor"] + ("_joint" if row["planned_helper"] else "_fixed")
        key = (row["model_seed"], tuple(row["goal_pair"]), row["support"], row["initial_angle_deg"], row["nuisance_id"])
        a = fine[(stem+"_direct", *key)]["replay_fine"]
        b = fine[(stem+f"_expanded_s{row['allocation']}", *key)]["replay_fine"]
        row.update(fine_direct=a["ordered_goal_success"], fine_expanded=b["ordered_goal_success"],
                   fine_first_matches=a["first_goal_dwell_success"] == b["first_goal_dwell_success"])
        paired_groups[(row["predictor"], row["planned_helper"], row["allocation"])].append(row)
    pairs = [dict(predictor=k[0], planned_helper=k[1], allocation=k[2], cases=len(v),
        improved=sum(not r["direct"] and r["expanded"] for r in v),
        regressed=sum(r["direct"] and not r["expanded"] for r in v),
        fine_improved=sum(not r["fine_direct"] and r["fine_expanded"] for r in v),
        fine_regressed=sum(r["fine_direct"] and not r["fine_expanded"] for r in v),
        fine_first_mismatches=sum(not r["fine_first_matches"] for r in v),
        unchanged=sum(r["direct"] == r["expanded"] for r in v)) for k, v in sorted(paired_groups.items())]
    validation = dict(cases=len(rows), model_audit=verification["models"],
        local_replay_cases=sum(r["cases"] for r in verification["local_replays"]),
        local_replays_exact=all(b["exact"] for r in verification["local_replays"] for b in r["batches"]),
        all_nominal_replays_bitwise=all(r["source_replay_bitwise"] for r in numerical),
        all_nominal_scores_reproduced=all(r["source_scoring_reproduced"] for r in numerical),
        active_refinement_passes=sum(s["refinement_passes"] for s in summary),
        max_position_discrepancy=max(r["refinement"]["active_prefix"]["max_xy_units"] for r in numerical),
        max_angle_discrepancy=max(r["refinement"]["active_prefix"]["max_angle_rad"] for r in numerical),
        ordered_lost=sum(r["stop_time_outcomes"]["ordered_goal_success"]["lost"] for r in numerical),
        ordered_gained=sum(r["stop_time_outcomes"]["ordered_goal_success"]["gained"] for r in numerical),
        all_prefixes_match=verification["all_prefixes_match"],
        max_prefix_state_error=verification["max_prefix_state_error"],
        max_prefix_action_error=verification["max_prefix_action_error"])
    finer = root/"finer_diagnostics.json"
    if finer.exists():
        diagnostic = json.loads(finer.read_text())
        (destination/"finer_diagnostics.json").write_text(json.dumps(diagnostic, indent=2)+"\n")
        validation["post_hoc_finer_groups"] = len(diagnostic["reports"])
        validation["post_hoc_320_640_all_pass"] = all(row["passes_320_640"]
            for report in diagnostic["reports"] for row in report["rows"])
        validation["post_hoc_320_640_ordered_changes"] = sum(row["ordered_320"] != row["ordered_640"]
            for report in diagnostic["reports"] for row in report["rows"])
    for name, value in (("protocol.json", manifest["protocol"]), ("summary.json", summary),
                        ("paired_results.json", pairs), ("verification.json", validation),
                        ("compute.json", verification["queries"]),
                        ("training.json", json.loads((root/"local"/"models.json").read_text()))):
        (destination/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    fields = ["planner", "model_seed", "support", "initial_angle_deg", "nuisance_id", "first_goal_rad", "relative_goal_rad",
              "candidate_slots", "first_goal_success_at_phase_boundary", "ordered_goal_success", "total_steps", "finite", "wall",
              "failed_options", "helper_nonfixed_options", "helper_distinct_options", "helper_switches",
              "final_angle_error_rad", "final_bar_translation_units"]
    with (destination/"episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields+["fine_ordered_success", "refinement_passes"], lineterminator="\n")
        writer.writeheader()
        for r in rows:
            f = fine[(r["planner"], r["model_seed"], tuple(r["goal_pair"]), r["support"], r["initial_angle_deg"], r["nuisance_id"])]
            writer.writerow(dict({k: r[k] for k in fields}, fine_ordered_success=f["replay_fine"]["ordered_goal_success"],
                                 refinement_passes=f["active_prefix_passes"]))
    export_models(root/"local"/"models.pt", destination/"frozen_models.json")
    provenance = dict(frozen=manifest["frozen"],
        audit_source_sha256=verification["audit_source_sha256"],
        report_source_sha256={str(Path(__file__).relative_to(Path.cwd())): sha(__file__)},
        raw_manifest=dict(path=str(root/"manifest.json"), sha256=sha(root/"manifest.json")),
        raw_verification=dict(path=str(root/"verification.json"), sha256=sha(root/"verification.json")),
        raw_groups=[{k: item[k] for k in ("job", "path", "sha256")} for item in manifest["groups"]],
        raw_numerical=[{k: item[k] for k in ("source_path", "raw_path", "raw_sha256")} for item in verification["numerical_reports"]],
        raw_data_location="gitignored data/novelty/planned_helper_v1; not included in the compact evidence",
        published_sha256={p.name: sha(p) for p in destination.iterdir() if p.is_file() and p.name != "provenance.json"})
    archive = root/"collection_source_archive.json"
    if archive.exists():
        provenance["collection_source_archive"] = json.loads(archive.read_text())
    (destination/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(dict(summary=summary, verification=validation), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restore-model", type=Path)
    args = parser.parse_args()
    if args.restore_model:
        restore_models(args.restore_model, args.output)
    elif args.run:
        publish(args.run, args.output)
    else:
        parser.error("Use --run or --restore-model")


if __name__ == "__main__":
    main()
