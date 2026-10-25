"""Uninterrupted saved-action replay and original-stop mission rescoring.

The replay begins at the mission's complete initial snapshot and advances every
recorded joint rotor command. Option snapshots, predictors and controllers are
never used to replace an intermediate physical state. Stage goals and physical
stop times remain fixed while integration resolution changes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import pickle
import time

import numpy as np

from .collect import plain
from .evaluation import _SubstepObserver, _stack_records
from .plant import TransportConfig, TransportEnv
from .environment import config_from_dict, make_env, family, source_paths
from .snapshot import integration_state, physical_state, restore


TOLERANCES=dict(maximum_body_position_m=.002,maximum_body_attitude_rad=.02)
SEQUENCE=("settle","leg_1","operation_1","leg_2","operation_2","leg_3")
SCORE_FIELDS=("delivery_completion","constrained_completion","waypoint_successes",
              "option_completions","actual_slack_successes","finite","collision","warning_count")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_clock(result):
    base=int(result["initial_snapshot"].control_steps)
    final=int(result["summary"]["control_steps"])
    count=final-base
    if count<1:raise ValueError("A mission replay requires actual physical control steps")
    stages=result["stages"]
    if tuple(x["name"] for x in stages)!=SEQUENCE[:len(stages)]:
        raise ValueError("Recorded stages must retain the original ordered mission prefix")
    previous=base
    for stage in stages:
        if int(stage["begin_step"])!=previous or int(stage["end_step"])<=previous:
            raise ValueError("Stage clocks must form one uninterrupted physical execution")
        previous=int(stage["end_step"])
    if previous!=final:raise ValueError("The final stage stop must match the recorded mission stop")
    trace=result["trace"]
    if trace is None:raise ValueError("Whole-mission replay requires recorded joint actions")
    for field in ("actions","external_forces","phases","stages","references"):
        if len(trace[field])<count:raise ValueError("Recorded actions/references omit actual physical steps")
    if len(trace["physical_states"])<count+1 or len(trace["integration_states"])<count+1:
        raise ValueError("The trace must include the initial and every post-control state")
    for stage in stages:
        b,e=int(stage["begin_step"])-base,int(stage["end_step"])-base
        if any(x!=stage["name"] for x in trace["stages"][b:e]):
            raise ValueError("Per-action stage labels disagree with their original boundaries")
    return base,count


def _compact_observation(obs):
    return {key:obs[key] for key in ("payload_pos","payload_vel","cable_tensions",
        "cable_gaps","finite","warnings","collision_pairs")}


def _window(observations, begin, end, required):
    if required<1 or end-begin<required:return []
    return observations[end-required+1:end+1]


def score_mission(result,replay):
    """Measure every original stage independently, keeping failed option prefixes."""
    base,count=validate_clock(result)
    cfg=result["mission_config"]
    period=float(result["plant_config"]["control_dt"])
    dt=float(replay["dt"])
    obs=replay["observations"][:count+1]
    if len(obs)!=count+1:raise ValueError("Replay observations omit original control endpoints")
    raw=replay["substeps"]
    indices=np.asarray(raw["control_index"],dtype=int)
    active=(indices>=0)&(indices<count)
    if set(indices[active].tolist())!=set(range(count)):
        raise ValueError("Substep measurements must cover every actual control interval")
    finite=bool(all(x["finite"] for x in obs) and np.asarray(raw["finite"])[active].all())
    collision=bool(any(x["collision_pairs"] for x in obs) or np.asarray(raw["collision"])[active].any())
    warning_count=int(np.asarray(obs[-1]["warnings"]).sum())
    dwell=max(1,int(math.ceil(cfg["goal_dwell_seconds"]/period-1e-10)))
    waypoints=[];legs=[]
    stages={s["name"]:s for s in result["stages"]}
    for number in (1,2,3):
        name=f"leg_{number}"
        if name not in stages:
            waypoints.append(False);legs.append(dict(name=name,present=False,success=False));continue
        stage=stages[name];b=int(stage["begin_step"])-base;e=int(stage["end_step"])-base
        window=_window(obs,b,e,dwell)
        eligible=bool(window and (e-dwell+1-b)*period+1e-10>=cfg["move_seconds"])
        goal=np.asarray(stage["target"])
        within=[bool(np.linalg.norm(x["payload_pos"]-goal)<=cfg["position_tolerance"]
            and np.linalg.norm(x["payload_vel"])<=cfg["speed_tolerance"]
            and x["finite"] and not np.asarray(x["warnings"]).any()) for x in window]
        success=bool(eligible and all(within))
        waypoints.append(success)
        legs.append(dict(name=name,present=True,success=success,begin_step=b,end_step=e,
            required_dwell_steps=dwell,qualifying_states=sum(within),reference_finished=eligible,
            final_position_error=float(np.linalg.norm(obs[e]["payload_pos"]-goal)),
            final_speed=float(np.linalg.norm(obs[e]["payload_vel"]))))
    options=[];option_success=[];slack_success=[]
    for index in range(2):
        name=f"operation_{index+1}"
        if index>=len(result["operations"]) or name not in stages:
            option_success.append(False);slack_success.append(False)
            options.append(dict(name=name,present=False,completion=False,slack_success=False));continue
        operation=result["operations"][index];stage=stages[name]
        b=int(stage["begin_step"])-base;e=int(stage["end_step"])-base
        if int(operation["mission_begin_step"])-base!=b or int(operation["mission_end_step"])-base!=e:
            raise ValueError("An option's original endpoints differ from the whole-mission clock")
        selected=active&(indices>=b)&(indices<e)
        goal=np.asarray(operation["goal"]);rc=operation["recovery_config"]
        required=max(1,int(math.ceil(rc["verify_seconds"]/period-1e-10)))
        window=_window(obs,b,e,required)
        verified=bool(window and all(np.all(x["cable_tensions"]>max(.02,rc["support_threshold"]))
            and np.all(x["cable_gaps"]<=rc["gap_tolerance"])
            and np.linalg.norm(x["payload_pos"]-goal)<=rc["payload_error_limit"]
            and np.linalg.norm(x["payload_vel"])<=rc["payload_speed_limit"] for x in window))
        status=operation["end_protocol_memory"]
        protocol_finished=bool(status["done"] and not status["failed"])
        local_finite=bool(all(x["finite"] for x in obs[b:e+1]) and np.asarray(raw["finite"])[selected].all())
        local_collision=bool(any(x["collision_pairs"] for x in obs[b:e+1]) or np.asarray(raw["collision"])[selected].any())
        completion=bool(protocol_finished and verified and local_finite and not local_collision
            and not np.asarray(obs[e]["warnings"]).any())
        donor=int(operation["donor"])
        tensions=np.asarray(raw["cable_tensions"])[selected,donor]
        gaps=np.asarray(raw["cable_gaps"])[selected,donor]
        slack=float(((tensions<.005)&(gaps>.01)).sum()*dt)
        slack_pass=bool(slack+1e-10>=cfg["required_slack_seconds"])
        positions=np.asarray(raw["payload_position"])[selected]
        excursion=float(max(np.linalg.norm(positions-goal,axis=1).max(),
            np.linalg.norm(obs[b]["payload_pos"]-goal),np.linalg.norm(obs[e]["payload_pos"]-goal)))
        peak=float(max(tensions.max(),obs[b]["cable_tensions"][donor],obs[e]["cable_tensions"][donor]))
        options.append(dict(name=name,present=True,completion=completion,slack_success=slack_pass,
            protocol_finished=protocol_finished,original_protocol_failed=bool(status["failed"]),
            original_failure_reason=status["failure_reason"],verified_final_dwell=verified,
            begin_step=b,end_step=e,donor=donor,required_dwell_steps=required,
            peak_payload_excursion=excursion,peak_donor_tension=peak,donor_slack_seconds=slack,
            finite=local_finite,collision=local_collision,
            excursion_within_limit=excursion<=cfg["payload_excursion_limit"],
            tension_within_limit=peak<=cfg["donor_tension_limit"]))
        option_success.append(completion);slack_success.append(slack_pass)
    delivery=bool(all(waypoints) and all(option_success) and all(slack_success)
        and finite and not collision and warning_count==0)
    constrained=bool(delivery and all(x.get("excursion_within_limit",False)
        and x.get("tension_within_limit",False) for x in options))
    return dict(delivery_completion=delivery,constrained_completion=constrained,
        waypoint_successes=waypoints,option_completions=option_success,actual_slack_successes=slack_success,
        finite=finite,collision=collision,warning_count=warning_count,
        legs=legs,operations=options,control_steps=count,physics_steps=int(active.sum()),
        total_seconds=count*period,maximum_uav_tilt=float(np.asarray(raw["uav_tilt"])[active].max()),
        minimum_body_height=float(np.asarray(raw["minimum_body_height"])[active].min()),
        minimum_uav_separation=float(np.asarray(raw["minimum_uav_separation"])[active].min()))


def trajectory_errors(source,replayed,tolerances=TOLERANCES):
    source,replayed=np.asarray(source),np.asarray(replayed)
    if source.shape!=replayed.shape or source.ndim!=2 or source.shape[1]!=64:
        raise ValueError("Compare the same full64D physical endpoints at the same original clock")
    offsets=(0,7,14,21)
    positions=np.stack([np.linalg.norm(source[:,j:j+3]-replayed[:,j:j+3],axis=1) for j in offsets],axis=1)
    angles=[]
    for j in offsets:
        a=source[:,j+3:j+7];b=replayed[:,j+3:j+7]
        a=a/np.linalg.norm(a,axis=1,keepdims=True);b=b/np.linalg.norm(b,axis=1,keepdims=True)
        # The relative quaternion vector gives stable zero/near-zero angles.
        vector=a[:,:1]*b[:,1:]-b[:,:1]*a[:,1:]-np.cross(a[:,1:],b[:,1:])
        scalar=np.sum(a*b,axis=1)
        angles.append(2*np.arctan2(np.linalg.norm(vector,axis=1),np.abs(scalar)))
    angles=np.stack(angles,axis=1)
    p=float(positions.max());a=float(angles.max())
    return dict(maximum_body_position_m=p,maximum_payload_position_m=float(positions[:,0].max()),
        maximum_uav_position_m=float(positions[:,1:].max()),maximum_body_attitude_rad=a,
        per_body_position_max=positions.max(0),per_body_attitude_max=angles.max(0),
        position_pass=p<=tolerances["maximum_body_position_m"],
        attitude_pass=a<=tolerances["maximum_body_attitude_rad"],
        passes=bool(p<=tolerances["maximum_body_position_m"] and a<=tolerances["maximum_body_attitude_rad"]))


def replay_mission(result,dt=None):
    base,count=validate_clock(result)
    config=config_from_dict(result["plant_config"])
    if dt is not None:config=replace(config,dt=float(dt))
    env=make_env(config)
    restore(env,result["initial_snapshot"],allow_timestep_change=dt is not None)
    trace=result["trace"]
    states=[physical_state(env)];observations=[_compact_observation(env.observe())]
    exact_integration=bool(np.array_equal(integration_state(env),trace["integration_states"][0]))
    substep_labels=[]
    with _SubstepObserver(env,np.asarray(trace["references"][0])) as observer:
        for index in range(count):
            observer.goal=np.asarray(trace["references"][index]).copy()
            obs=observer.step(trace["phases"][index],trace["actions"][index],trace["external_forces"][index])
            observations.append(_compact_observation(obs));states.append(physical_state(env))
            exact_integration &= np.array_equal(integration_state(env),trace["integration_states"][index+1])
            substep_labels.extend([index]*env.substeps)
    raw=_stack_records(observer.records);raw["control_index"]=np.asarray(substep_labels)
    data=dict(dt=config.dt,observations=observations,substeps=raw)
    scores=score_mission(result,data)
    source=np.asarray(trace["physical_states"][:count+1]);states=np.asarray(states)
    changes={key:dict(source=result["summary"][key],replay=scores[key])
        for key in SCORE_FIELDS if result["summary"][key]!=scores[key]}
    return dict(dt=config.dt,source_dt=result["plant_config"]["dt"],physical_states=states,
        score=scores,trajectory=trajectory_errors(source,states),score_changes=changes,
        source_physical_bitwise=bool(np.array_equal(source,states)),
        source_integration_bitwise=bool(exact_integration),source_scores_reproduced=not changes,
        control_steps=count,physics_steps=len(observer.records),
        source_initial_control_step=base,omitted_padding_actions=len(trace["actions"])-count,
        interpretation="One uninterrupted physical replay; recorded rotor actions, external forces, references, phase bounds and original stops; final dwell and actual slack independently measured.")


def audit_file(entry,fine_dt,output,hashes):
    for path,digest in hashes.items():
        if sha(path)!=digest:raise AssertionError("Registered replay source changed: "+path)
    path=Path(entry["raw_path"])
    if sha(path)!=entry["raw_sha256"]:raise AssertionError("Mission artifact digest changed")
    with path.open("rb") as stream:result=pickle.load(stream)
    started=time.perf_counter()
    nominal=replay_mission(result,float(result["plant_config"]["dt"]))
    fine=replay_mission(result,fine_dt)
    output=Path(output);raw_path=output.with_suffix(".pkl");report_path=output.with_suffix(".json")
    if raw_path.exists() or report_path.exists():raise FileExistsError("Preserve completed mission replay artifacts")
    raw=dict(source_path=str(path),source_sha256=entry["raw_sha256"],nominal=nominal,fine=fine)
    with raw_path.open("wb") as stream:pickle.dump(raw,stream,protocol=5)
    report=dict(source_path=str(path),source_sha256=entry["raw_sha256"],identity={k:result["summary"][k]
        for k in ("case_id","planner","model_seed","payload_mass","donor","route_template")},
        nominal={k:v for k,v in nominal.items() if k!="physical_states"},
        fine={k:v for k,v in fine.items() if k!="physical_states"},
        raw_path=str(raw_path),raw_sha256=sha(raw_path),wall_seconds=time.perf_counter()-started)
    report_path.write_text(json.dumps(plain(report),indent=2,allow_nan=False)+"\n")
    if sha(path)!=entry["raw_sha256"]:raise AssertionError("Mission source changed while replaying")
    return str(report_path)


def aggregate(parts):
    groups={};changes=[]
    for part in parts:
        identity=part["identity"];key=(identity["planner"],identity["model_seed"])
        row=groups.setdefault(key,dict(planner=key[0],model_seed=key[1],missions=0,
            nominal_delivery=0,fine_delivery=0,nominal_constrained=0,fine_constrained=0,
            continuous_passes=0,delivery_lost=0,delivery_gained=0,constrained_lost=0,constrained_gained=0))
        row["missions"]+=1
        a,b=part["nominal"]["score"],part["fine"]["score"]
        for name,metric in (("delivery","delivery_completion"),("constrained","constrained_completion")):
            row["nominal_"+name]+=int(a[metric]);row["fine_"+name]+=int(b[metric])
            row[name+"_lost"]+=int(a[metric] and not b[metric]);row[name+"_gained"]+=int(b[metric] and not a[metric])
        row["continuous_passes"]+=int(part["fine"]["trajectory"]["passes"])
        for field,difference in part["fine"]["score_changes"].items():
            changes.append(dict(**identity,field=field,**difference))
        for index,(x,y) in enumerate(zip(a["operations"],b["operations"])):
            for field in ("verified_final_dwell","slack_success","excursion_within_limit","tension_within_limit"):
                if x.get(field)!=y.get(field):changes.append(dict(**identity,operation=index,
                    field=field,source=x.get(field),replay=y.get(field)))
    return [groups[key] for key in sorted(groups)],changes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--fine-dt",type=float,default=.001)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--out",type=Path,default=Path("data/uav_transport/mission_v1_replay"))
    args=parser.parse_args()
    if not 1<=args.workers<=8:raise ValueError("Use one through eight independent workers")
    manifest=json.loads(args.manifest.read_text());entries=manifest["results"]
    if len({x["raw_path"] for x in entries})!=len(entries):raise ValueError("Duplicate source missions")
    if args.out.with_suffix(".json").exists():raise FileExistsError("Preserve completed mission refinement")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    sources={str(p):sha(p) for p in [Path(__file__)]+[Path("uav_transport")/(n+".py")
        for n in ("plant","snapshot","evaluation","control","collect")]}
    sources.update({str(p):sha(p) for p in source_paths(family(manifest))})
    inputs={entry["raw_path"]:entry["raw_sha256"] for entry in entries}
    inputs[str(args.manifest)]=sha(args.manifest)
    for path,digest in inputs.items():
        if sha(path)!=digest:raise AssertionError("Source manifest hash mismatch: "+path)
    if not 0<args.fine_dt<.002:raise ValueError("Fine timestep must refine the nominal .002s resolution")
    declaration=dict(source_sha256=sources,input_sha256=inputs,manifest_path=str(args.manifest),
        fine_dt=args.fine_dt,tolerances=TOLERANCES,missions=len(entries),control_dt=.02,
        scoring="Original leg and option stopping times; complete source protocol status, actual slack, physical final dwell and substep excursion/tension limits",
        replay="Uninterrupted from initial full snapshot, with saved rotor actions and externalforces; no controller or predictor calls")
    registration=args.out.with_name(args.out.name+"_registration.json")
    if registration.exists():
        if json.loads(registration.read_text())!=plain(declaration):raise ValueError("Replay registration changed")
    else:registration.write_text(json.dumps(plain(declaration),indent=2)+"\n")
    started=time.perf_counter();parts={}
    report=dict(declaration,registration_path=str(registration),registration_sha256=sha(registration),
        parts=[],completed=0,complete=False)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures={pool.submit(audit_file,entry,args.fine_dt,args.out.with_name(args.out.name+f"_part{i:03d}"),sources):i
            for i,entry in enumerate(entries)}
        for future in as_completed(futures):
            index=futures[future];path=Path(future.result());part=json.loads(path.read_text())
            part.update(report_path=str(path),report_sha256=sha(path));parts[index]=part
            report["parts"]=[parts[i] for i in sorted(parts)]
            report["summary"],report["changed_scores"]=aggregate(report["parts"])
            report["completed"]=len(parts);report["wall_seconds"]=time.perf_counter()-started
            args.out.with_suffix(".json").write_text(json.dumps(plain(report),indent=2,allow_nan=False)+"\n")
            print(json.dumps(dict(completed=len(parts),total=len(entries),**part["identity"],
                nominal_bitwise=part["nominal"]["source_integration_bitwise"],
                source_scores_reproduced=part["nominal"]["source_scores_reproduced"],
                fine_pass=part["fine"]["trajectory"]["passes"],
                source_constrained=part["nominal"]["score"]["constrained_completion"],
                fine_constrained=part["fine"]["score"]["constrained_completion"])),flush=True)
    for path,digest in {**sources,**inputs}.items():
        if sha(path)!=digest:raise AssertionError("Frozen replay input changed: "+path)
    report.update(complete=True,verified_frozen_inputs=True,wall_seconds=time.perf_counter()-started,
        nominal_valid=all(p["nominal"]["source_physical_bitwise"] and p["nominal"]["source_integration_bitwise"]
            and p["nominal"]["source_scores_reproduced"] for p in parts.values()))
    args.out.with_suffix(".json").write_text(json.dumps(plain(report),indent=2,allow_nan=False)+"\n")
    if not report["nominal_valid"]:raise AssertionError("Nominal full-state replay or independent scoring differs; every diagnostic is retained")


if __name__=="__main__":main()
