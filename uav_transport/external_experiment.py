"""Full Menagerie missions with the pinned upstream Lee flight controller.

Only the two rule-based geometry arms are exposed. The study's recovery and
payload allocator are shared; flight position/attitude feedback is upstream C.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import pickle
import time

import numpy as np

from .collect import plain
from .environment import MENAGERIE, make_config, source_paths
from .external_controller import ExternalLeeController, VENDOR, verify_upstream, _sha
from .mission import MissionConfig, mission_cases, run_mission


ARMS = ("rule_fixed", "rule_reconfigure")


def schedule(arms=ARMS):
    if not arms or len(arms)!=len(set(arms)) or set(arms)-set(ARMS):
        raise ValueError("The external controller experiment supports unique rule arms only")
    return [(case, arm) for arm in arms for case in mission_cases()]


def worker(case, arm, output, hashes):
    if arm not in ARMS: raise ValueError("External learned/oracle arms are not supported")
    for path,digest in hashes.items():
        if _sha(path)!=digest: raise AssertionError("External experiment input changed: "+path)
    path=Path(f"{output}_{arm}_seed0_{case['case_id']}.pkl")
    if path.exists(): raise FileExistsError("Preserve completed external-controller runs")
    started=time.perf_counter()
    result=run_mission(case,arm,plant_family=MENAGERIE,controller_factory=ExternalLeeController)
    with path.open("wb") as stream: pickle.dump(result,stream,protocol=5)
    return dict(summary=result["summary"],raw_path=str(path),raw_sha256=_sha(path),
        controller_configuration=result["controller_config"],wall_seconds=time.perf_counter()-started)


def summary(results):
    output=[]
    for arm in ARMS:
        rows=[x["summary"] for x in results if x["summary"]["planner"]==arm]
        if not rows: continue
        output.append(dict(planner=arm,controller_family=ExternalLeeController.family,missions=len(rows),
            delivery_completion=sum(x["delivery_completion"] for x in rows),
            constrained_completion=sum(x["constrained_completion"] for x in rows),
            mean_mission_seconds=float(np.mean([x["total_seconds"] for x in rows])),
            mean_actuator_work=float(np.mean([x["absolute_actuator_work"] for x in rows])),
            maximum_operation_excursion=max(max(x["option_peak_excursions"]) for x in rows),
            maximum_donor_tension=max(max(x["option_peak_donor_tensions"]) for x in rows),
            maximum_rotor_force=max(x["maximum_rotor_force"] for x in rows),
            minimum_uav_separation=min(x["minimum_uav_separation"] for x in rows),
            all_finite=all(x["finite"] for x in rows),collisions=sum(x["collision"] for x in rows),
            warnings=sum(x["warning_count"] for x in rows),
            predictor_evaluations=sum(x["predictor_evaluations"] for x in rows)))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("data/uav_transport/menagerie_external_lee_v1"))
    parser.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=4: raise ValueError("Use one through four external-baseline workers")
    jobs=schedule(tuple(args.arms));args.out.parent.mkdir(parents=True,exist_ok=True)
    report_path=Path(str(args.out)+".json");registration=Path(str(args.out)+"_registration.json")
    if report_path.exists() or registration.exists(): raise FileExistsError("Use a fresh external-baseline prefix")
    # Compile once before workers start; each UAV still owns an independent
    # mutable upstream controller state and retains the published gains.
    calibrated=ExternalLeeController(make_config(MENAGERIE))
    files=[Path(__file__),Path(__file__).with_name("external_controller.py")]
    files += [Path("uav_transport")/(name+".py") for name in
        ("mission","collect","local_model","control","evaluation","snapshot")]
    files += source_paths(MENAGERIE)
    files += [p for p in VENDOR.iterdir() if p.is_file()]
    hashes={str(path):_sha(path) for path in files}
    declaration=dict(plant_family=MENAGERIE,controller_family=ExternalLeeController.family,
        planners=args.arms,model_seeds=[0],mission_count=len(jobs),physical_cases=len(mission_cases()),
        cases=mission_cases(),mission_config=asdict(MissionConfig()),source_sha256=hashes,model_sha256={},
        upstream=verify_upstream(),controller_configuration=calibrated.configuration(),
        role_separation="Shared study payload/tension allocation and recovery/formation references; pinned unchanged upstream Lee C position/attitude feedback; shared bounded rotor mixer",
        clock_adaptation="The wrapper configures the source scheduler for the 50Hz study control period; upstream default is500Hz. Published feedback gains remain unchanged.",
        calibration="Mass and diagonal inertia are supplied from the official Menagerie Crazyflie2 body model",
        allowed_planners="rule_fixed and rule_reconfigure only; no learned or exact-option predictions with an external controller",
        score="Original three-waypoint/two-operation delivery and registered excursion/donor-tension limits",
        no_training_or_gain_search=True)
    registration.write_text(json.dumps(plain(declaration),indent=2,allow_nan=False)+"\n")
    started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(worker,case,arm,args.out,hashes) for case,arm in jobs]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print(json.dumps(dict(completed=len(results),total=len(jobs),**result["summary"])),flush=True)
    if len({(x["summary"]["planner"],x["summary"]["case_id"]) for x in results})!=len(jobs):
        raise AssertionError("External controller matrix is incomplete or duplicated")
    for path,digest in hashes.items():
        if _sha(path)!=digest: raise AssertionError("Registered external input changed during execution: "+path)
    results.sort(key=lambda x:(x["summary"]["planner"],x["summary"]["case_id"]))
    report=dict(declaration,registration_path=str(registration),registration_sha256=_sha(registration),
        results=results,summary=summary(results),verified_frozen_inputs=True,wall_seconds=time.perf_counter()-started)
    report_path.write_text(json.dumps(plain(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(report_path),summary=report["summary"])),flush=True)


if __name__=="__main__": main()
