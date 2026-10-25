"""Run the archived package directly, without importing the active UAV package.

The fresh mission mode needs only the compact archived model files. Replay mode
also needs the separately retained raw mission. Outputs belong outside this
read-only archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np


ARCHIVE = Path(__file__).resolve().parent
REPO = ARCHIVE/"repo"
sys.path.insert(0, str(REPO))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mission", "replay"), default="mission")
    parser.add_argument("--raw-root", type=Path,
        help="Original or relocated repository root containing the ignored historical raw data")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.is_relative_to(ARCHIVE):
        raise ValueError("Write verification outputs outside the immutable archive")
    if output.exists(): raise FileExistsError("Preserve existing verification output")
    capture = json.loads((ARCHIVE/"capture.json").read_text())
    for record in capture["files"]:
        if sha(ARCHIVE/record["archive_path"]) != record["sha256"]:
            raise AssertionError("Captured source/evidence changed: "+record["archive_path"])
    import mujoco
    import uav_transport
    from uav_transport.local_model import load_model
    from uav_transport.mission import run_mission
    from uav_transport.mission_replay import replay_mission
    assert Path(uav_transport.__file__).resolve().parent == REPO/"uav_transport"
    evidence = REPO/"experiments/uav_transport/v2"
    manifest = json.loads((evidence/"mission_v2.json").read_text())
    entry = next(item for item in manifest["results"]
        if item["summary"]["planner"] == "wm_small" and item["summary"]["model_seed"] == 0
        and item["summary"]["case_id"] == "mass026_donor0_route0")
    source = None
    if args.raw_root is not None:
        raw = args.raw_root.resolve()/entry["raw_path"]
        if sha(raw) != entry["raw_sha256"]: raise AssertionError("Historical raw mission hash differs")
        with raw.open("rb") as stream: source = pickle.load(stream)
    if args.mode == "replay" and source is None:
        raise ValueError("Replay requires --raw-root pointing to the original raw data")
    started = time.perf_counter()
    if args.mode == "replay":
        result = replay_mission(source)
        checks = dict(physical_states_bitwise=result["source_physical_bitwise"],
            integration_states_bitwise=result["source_integration_bitwise"],
            scores_reproduced=result["source_scores_reproduced"], score=result["score"],
            control_steps=result["control_steps"], physics_steps=result["physics_steps"])
        assert all(checks[key] for key in ("physical_states_bitwise", "integration_states_bitwise", "scores_reproduced"))
    else:
        models = json.loads((evidence/"models/models_v1_manifest.json").read_text())
        model_entry = models["models"]["small/seed0"]
        prefix = evidence/"models"/Path(model_entry["prefix"]).name
        model = load_model(prefix)
        case = next(case for case in manifest["cases"] if case["case_id"] == entry["summary"]["case_id"])
        generated = run_mission(case, "wm_small", model, model_seed=0, margins=model_entry["margins"])
        score_fields = ("delivery_completion", "constrained_completion", "waypoint_successes",
            "option_completions", "actual_slack_successes", "total_seconds", "control_steps", "physics_steps")
        checks = dict(summary=generated["summary"], published_scores_match=all(
            generated["summary"][key] == entry["summary"][key] for key in score_fields))
        assert checks["published_scores_match"]
        if source is not None:
            for name in ("physical_states", "integration_states", "actions", "external_forces"):
                checks[name+"_bitwise"] = bool(np.array_equal(generated["trace"][name], source["trace"][name]))
                assert checks[name+"_bitwise"], name
    report = dict(mode=args.mode, package_path="repo/uav_transport", package_import_is_archived=True,
        model_family="self-built MuJoCo quadrotor baseline v2", case_id=entry["summary"]["case_id"],
        original_raw_reference=entry["raw_path"], original_raw_sha256=entry["raw_sha256"],
        versions=dict(python=sys.version.split()[0], mujoco=mujoco.__version__, numpy=np.__version__),
        wall_seconds=time.perf_counter()-started, checks=checks)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(output), mode=args.mode, case_id=report["case_id"],
        archived_package=True, wall_seconds=report["wall_seconds"], passed=True)), flush=True)


if __name__ == "__main__": main()
