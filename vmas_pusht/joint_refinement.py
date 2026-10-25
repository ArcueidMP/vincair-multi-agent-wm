"""Post-hoc 320/640 replay of every group failing the registered 160/320 check.

This diagnostic never replaces the primary audit, changes a threshold, calls a
planner, or changes stopping times. Complete original batches are preserved.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import torch

from vmas_pusht.boundary import _plain
from vmas_pusht.interface_numerics import score_original_endpoints
from vmas_pusht.option_models import sha
from vmas_pusht.scenario import wrap
from vmas_pusht.stable_task_numerics import replay_group


def diagnose(report_path, output):
    torch.set_num_threads(1)
    report = json.loads(Path(report_path).read_text())
    if sha(report["source_path"]) != report["input_sha256"] or sha(report["raw_path"]) != report["raw_sha256"]:
        raise AssertionError("The primary audit inputs changed")
    group = torch.load(report["source_path"], weights_only=False)["groups"][0]
    old = torch.load(report["raw_path"], weights_only=False)["results"][0]["replay_fine"]
    new = replay_group(group, 640)
    if not torch.equal(old["trajectory"][:, 0], new["trajectory"][:, 0]) or not torch.equal(old["time"], new["time"]):
        raise AssertionError("The original complete start and physical clock must match")
    a, b = score_original_endpoints(group, old), score_original_endpoints(group, new)
    xy = (old["trajectory"][..., 12:14]-new["trajectory"][..., 12:14]).norm(dim=-1)
    angle = wrap(old["trajectory"][..., 16]-new["trajectory"][..., 16]).abs()
    mask = torch.cat((torch.ones(len(group["rows"]), 1, dtype=torch.bool), group["active"]), 1)
    failed = {r["row"] for r in report["results"][0]["rows"] if not r["active_prefix_passes"]}
    rows = [dict(row=i, previously_failed=i in failed,
        max_320_640_xy=float(xy[i, mask[i]].max()), max_320_640_angle=float(angle[i, mask[i]].max()),
        passes_320_640=bool((xy[i, mask[i]] <= .001).all() and (angle[i, mask[i]] <= .003).all()),
        ordered_320=bool(a["ordered_goal_success"][i]), ordered_640=bool(b["ordered_goal_success"][i]))
        for i in range(len(group["rows"]))]
    raw = Path(output).with_suffix(".pt")
    if raw.exists():
        raise FileExistsError("Preserve existing refinement diagnostics")
    torch.save(new, raw)
    result = dict(primary_report=str(report_path), primary_report_sha256=sha(report_path),
        original_160_320_failed_rows=sorted(failed), cases=len(rows), rows=rows,
        raw_path=str(raw), raw_sha256=sha(raw), preserves_primary_failures=True,
        planner_queries=0, original_recorded_actions_and_stopping_times=True)
    Path(output).with_suffix(".json").write_text(json.dumps(_plain(result), indent=2)+"\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    primary = json.loads((args.output/"verification.json").read_text())
    manifest = json.loads((args.output/"manifest.json").read_text())
    if len(primary["numerical_reports"]) != len(manifest["groups"]):
        raise ValueError("Complete the primary numerical audit before this diagnostic")
    paths = []
    for path in sorted((args.output/"numerics").glob("*.json")):
        data = json.loads(path.read_text())
        if any(not row["active_prefix_passes"] for result in data["results"] for row in result["rows"]):
            paths.append(path)
    destination = args.output/"finer_diagnostics"
    destination.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(3, max(1, len(paths)))) as pool:
        futures = [pool.submit(diagnose, path, destination/path.stem) for path in paths]
        reports = [future.result() for future in futures]
    result = dict(post_hoc=True, selection="Every source group with at least one failed registered 160/320 trajectory",
        primary_thresholds_unchanged=True, source_sha256=sha(__file__), reports=reports)
    (args.output/"finer_diagnostics.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(groups=len(paths), cases=sum(r["cases"] for r in reports),
        all_320_640_pass=all(row["passes_320_640"] for r in reports for row in r["rows"]),
        changed_ordered_labels=sum(row["ordered_320"] != row["ordered_640"] for r in reports for row in r["rows"]))))


if __name__ == "__main__":
    main()
