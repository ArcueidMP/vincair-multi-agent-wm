"""Registered feedback-only transport recovery and fixed-action refinement.

Eighteen physical conditions compare the original support geometry with a
physically reconfigured support geometry from identical settled snapshots.
Every maneuver is replayed at its original stop with three integration steps.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np

from .control import RecoveryConfig, TransportController
from .evaluation import fork, replay_option, run_option
from .plant import TransportConfig, TransportEnv
from .snapshot import capture, integration_state, physical_state


TOLERANCES = dict(position_max_m=.002, donor_peak_relative=.20,
                  retension_impulse_absolute_ns=.002, retension_impulse_relative=.10)
ARMS = ("fixed_geometry", "reconfigured")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def case_matrix():
    return [dict(case_id=i,payload_mass=mass,donor=donor,formation_yaw=yaw)
        for i,(mass,donor,yaw) in enumerate(itertools.product((.024,.03,.036),(0,1,2),(0.,np.pi/4)))]


def compare_replay(nominal, replay, tolerances=TOLERANCES):
    source=np.asarray(nominal["trace"]["physical_states"])
    fine=np.asarray(replay["physical_states"])
    a,b=nominal["summary"],replay["summary"]
    if source.shape!=fine.shape or abs(a["elapsed_seconds"]-b["elapsed_seconds"])>1e-12:
        raise ValueError("Refinement must retain every original control endpoint and stopping time")
    payload=np.linalg.norm(source[:,:3]-fine[:,:3],axis=1)
    drones=np.stack([np.linalg.norm(source[:,j:j+3]-fine[:,j:j+3],axis=1) for j in (7,14,21)],axis=1)
    position=max(float(payload.max()),float(drones.max()))
    peak_difference=abs(a["peak_donor_tension"]-b["peak_donor_tension"])
    peak_relative=peak_difference/max(abs(b["peak_donor_tension"]),1e-12)
    impulse_difference=abs(a["retension_tension_impulse"]-b["retension_tension_impulse"])
    impulse_relative=impulse_difference/max(abs(b["retension_tension_impulse"]),1e-12)
    passes=dict(position=position<=tolerances["position_max_m"],
        donor_peak=peak_relative<=tolerances["donor_peak_relative"],
        retension_impulse=(impulse_difference<=tolerances["retension_impulse_absolute_ns"]
            or impulse_relative<=tolerances["retension_impulse_relative"]))
    return dict(source_dt=replay["source_dt"],replay_dt=replay["replay_dt"],
        maximum_payload_position_difference_m=float(payload.max()),
        maximum_uav_position_difference_m=float(drones.max()),
        maximum_all_body_position_difference_m=position,
        donor_peak_difference_n=peak_difference,donor_peak_relative_difference=peak_relative,
        retension_impulse_difference_ns=impulse_difference,
        retension_impulse_relative_difference=impulse_relative,passes=passes,
        all_continuous_checks_pass=all(passes.values()),
        source_completion=bool(a["completion"]),replay_completion=bool(b["completion"]),
        completion_changed=bool(a["completion"]!=b["completion"]),
        source_safety=dict(finite=a["finite"],collision=a["collision"],warning_count=a["warning_count"]),
        replay_safety=dict(finite=b["finite"],collision=b["collision"],warning_count=b["warning_count"]))


def case_key(row):
    return (row["payload_mass"],row["donor"],row["formation_yaw"])


def paired_summary(rows):
    indexed={}
    for row in rows:
        key=(case_key(row),row["arm"])
        if key in indexed:raise ValueError("Duplicated physical condition and arm")
        indexed[key]=row
    conditions=sorted({key[0] for key in indexed})
    if any((condition,arm) not in indexed for condition in conditions for arm in ARMS):
        raise ValueError("Every physical condition must retain both actual feedback operators")
    metrics=("peak_payload_excursion","peak_donor_tension","retension_tension_impulse",
             "elapsed_seconds","absolute_actuator_work","total_squared_thrust_integral",
             "final_payload_error","final_payload_speed","donor_slack_seconds")
    pairs=[]
    for condition in conditions:
        fixed,indexed_reconfigured=indexed[(condition,ARMS[0])],indexed[(condition,ARMS[1])]
        a,b=fixed["summary"],indexed_reconfigured["summary"]
        pairs.append(dict(payload_mass=condition[0],donor=condition[1],formation_yaw=condition[2],
            fixed_completion=bool(a["completion"]),reconfigured_completion=bool(b["completion"]),
            fixed={m:float(a[m]) for m in metrics},reconfigured={m:float(b[m]) for m in metrics},
            reconfigured_minus_fixed={m:float(b[m]-a[m]) for m in metrics}))
    summary={}
    for metric in metrics:
        a=np.array([p["fixed"][metric] for p in pairs]);b=np.array([p["reconfigured"][metric] for p in pairs])
        summary[metric]=dict(fixed_mean=float(a.mean()),reconfigured_mean=float(b.mean()),
            mean_difference=float((b-a).mean()),median_difference=float(np.median(b-a)),
            lower_count=int((b<a-1e-12).sum()),higher_count=int((b>a+1e-12).sum()),
            equal_count=int((np.abs(b-a)<=1e-12).sum()))
    return dict(cases=len(pairs),fixed_completed=sum(x["fixed_completion"] for x in pairs),
        reconfigured_completed=sum(x["reconfigured_completion"] for x in pairs),
        metrics=summary,pairs=pairs)


def write_pickle(path,value):
    if path.exists():raise FileExistsError("Preserve the existing physics artifact: "+str(path))
    with path.open("wb") as stream:pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
    return dict(path=str(path),sha256=sha(path))


def worker(case,output,hashes):
    for path,digest in hashes.items():
        if sha(path)!=digest:raise AssertionError("Registered source changed before simulation: "+path)
    started=time.perf_counter()
    prefix=Path(f"{output}_case{case['case_id']:02d}")
    report_path=prefix.with_suffix(".json")
    if report_path.exists():raise FileExistsError("Preserve existing physical cases")
    env=TransportEnv(TransportConfig(payload_mass=case["payload_mass"],formation_yaw=case["formation_yaw"]))
    controller=TransportController(env.config)
    goal=env.observe()["payload_pos"].copy()
    prelude=dict(start_snapshot=capture(env),start_controller_memory=controller.capture_memory(),
        physical_states=[physical_state(env)],integration_states=[integration_state(env)],actions=[])
    for _ in range(round(2./env.config.control_dt)):
        command=controller.command(env.observe(),goal)
        env.step(command)
        prelude["actions"].append(command.copy());prelude["physical_states"].append(physical_state(env))
        prelude["integration_states"].append(integration_state(env))
    for key in ("physical_states","integration_states","actions"):prelude[key]=np.asarray(prelude[key])
    prelude["end_snapshot"]=capture(env);prelude["end_controller_memory"]=controller.capture_memory()
    common_state=integration_state(env).copy()
    prelude_file=write_pickle(prefix.with_name(prefix.name+"_prelude").with_suffix(".pkl"),prelude)
    rows=[]
    for arm in ARMS:
        clone,feedback=fork(env,controller)
        result=run_option(clone,feedback,RecoveryConfig(),case["donor"],arm=="reconfigured",max_seconds=18.,record=True)
        if not np.array_equal(result["start_snapshot"].state,common_state):
            raise AssertionError("Paired physical operator did not begin at the identical complete state")
        nominal_file=write_pickle(prefix.with_name(prefix.name+"_"+arm).with_suffix(".pkl"),
            dict(case=case,arm=arm,result=result,source_sha256=hashes,prelude_file=prelude_file))
        replays=[]
        for dt in (.002,.001,.0005):
            replay=replay_option(result,dt=dt)
            comparison=compare_replay(result,replay)
            exact=np.array_equal(replay["integration_states"],result["trace"]["integration_states"])
            if dt==.002 and not exact:raise AssertionError("Nominal full integration state did not replay bitwise")
            replay_file=write_pickle(prefix.with_name(prefix.name+"_"+arm+f"_dt{dt:.4f}.pkl"),replay)
            replays.append(dict(dt=dt,summary=replay["summary"],comparison=comparison,
                full_integration_bitwise=bool(exact),file=replay_file))
        force=result["trace"]["actions"]
        rows.append(dict(**case,arm=arm,summary=result["summary"],nominal_file=nominal_file,
            replays=replays,identical_complete_start=True,
            command_min_n=float(force.min()),command_max_n=float(force.max()),
            rotor_upper_bound_n=env.config.rotor_max,
            actual_motor_peak_n=float(result["end_snapshot"].metrics["maximum_rotor_force"].max()),
            peak_tilt_deg=float(np.degrees(result["summary"]["peak_uav_tilt"]))))
    if not np.array_equal(integration_state(env),common_state):
        raise AssertionError("The common source environment was modified by a counterfactual branch")
    report=dict(case=case,rows=rows,prelude_file=prelude_file,wall_seconds=time.perf_counter()-started,
        source_sha256=hashes,common_start_immutable=True)
    report_path.write_text(json.dumps(plain(report),indent=2,allow_nan=False)+"\n")
    return str(report_path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("data/uav_transport/physics_v1"))
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--register-only",action="store_true")
    args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError("Use one through four independent workers")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__)]+[Path("uav_transport")/(x+".py") for x in ("plant","control","snapshot","evaluation")]
    hashes={str(path):sha(path) for path in paths}
    declaration=dict(source_sha256=hashes,cases=case_matrix(),arms=ARMS,
        plant_config=asdict(TransportConfig()),recovery_config=asdict(RecoveryConfig()),
        physical_model="Three free rigid quadrotors with a common central payload attachment in 3D",
        prelude_seconds=2.,maximum_option_seconds=18.,nominal_dt=.002,replay_dt=[.002,.001,.0005],
        control_dt=.02,tolerances=TOLERANCES,comparisons="Both original support geometry and physically reconfigured geometry use the full frozen feedback controller.",
        refinement="Saved rotor commands, complete settled start snapshots, original physical stopping times and independently rescored final dwell.",
        reporting="All physical cases retained; continuous error, tension and effort metrics are primary; completion changes are retained separately.")
    registration=args.output.with_name(args.output.name+"_registration").with_suffix(".json")
    if registration.exists():
        if json.loads(registration.read_text())!=json.loads(json.dumps(declaration)):
            raise ValueError("Prespecified registration differs; use a fresh output prefix")
    else:registration.write_text(json.dumps(declaration,indent=2)+"\n")
    if args.register_only:print(str(registration));return
    if args.output.with_suffix(".json").exists():raise FileExistsError("Preserve completed physical audit")
    started=time.perf_counter();completed=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(worker,case,args.output,hashes) for case in case_matrix()]
        for future in as_completed(futures):
            path=Path(future.result());part=json.loads(path.read_text())
            completed.append(dict(path=str(path),sha256=sha(path),**part))
            print(json.dumps(dict(case=part["case"],finished_cases=len(completed),total_cases=18,
                successes=[r["summary"]["completion"] for r in part["rows"]],
                peak_position=[r["summary"]["peak_payload_excursion"] for r in part["rows"]],
                peak_tension=[r["summary"]["peak_donor_tension"] for r in part["rows"]])),flush=True)
    completed.sort(key=lambda x:x["case"]["case_id"])
    rows=[row for part in completed for row in part["rows"]]
    if len(rows)!=36:raise AssertionError("Missing physical audit arm")
    for path,digest in hashes.items():
        if sha(path)!=digest:raise AssertionError("Frozen source changed during physical audit: "+path)
    refinement=[]
    for dt in (.002,.001,.0005):
        selected=[x for row in rows for x in row["replays"] if x["dt"]==dt]
        refinement.append(dict(dt=dt,cases=len(selected),completion=sum(x["summary"]["completion"] for x in selected),
            continuous_passes=sum(x["comparison"]["all_continuous_checks_pass"] for x in selected),
            completion_changed=sum(x["comparison"]["completion_changed"] for x in selected),
            maximum_position_discrepancy=max(x["comparison"]["maximum_all_body_position_difference_m"] for x in selected),
            maximum_donor_peak_relative_discrepancy=max(x["comparison"]["donor_peak_relative_difference"] for x in selected),
            maximum_impulse_absolute_discrepancy=max(x["comparison"]["retension_impulse_difference_ns"] for x in selected)))
    report=dict(declaration,rows=rows,paired=paired_summary(rows),refinement=refinement,
        case_reports=[{k:part[k] for k in ("path","sha256")} for part in completed],
        registration_path=str(registration),registration_sha256=sha(registration),
        versions={p:importlib.metadata.version(p) for p in ("mujoco","numpy","scipy")},
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        verified_frozen_sources=True,wall_seconds=time.perf_counter()-started)
    args.output.with_suffix(".json").write_text(json.dumps(plain(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output),paired=report["paired"]["metrics"],refinement=refinement)),flush=True)


if __name__=="__main__":main()
