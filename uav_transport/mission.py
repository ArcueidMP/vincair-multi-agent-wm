"""Frozen-model planning through complete cooperative aerial transport missions.

Mission targets enter ordinary flight feedback. Local outcome predictors see
only current physical state, controller memory and an operation's declared
parameters. Every recovery option is executed by bounded rotor commands.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import pickle
import time

import numpy as np

from .collect import candidate_bank, plain
from .control import RecoveryConfig, TransportController, smooth_transition
from .evaluation import fork, run_option, _SubstepObserver, _stack_records
from .local_model import features, load_model, TARGET_NAMES
from .plant import TransportConfig, TransportEnv
from .environment import FAMILIES, LEGACY, make_config, make_env, require_model_family, source_paths
from .snapshot import capture, integration_state, physical_state


PLANNERS = ("rule_fixed", "rule_reconfigure", "wm_small_direct", "wm_small",
            "wm_large_direct", "wm_large", "true_option_direct", "true_option")
LEARNED = ("wm_small_direct", "wm_small", "wm_large_direct", "wm_large")


@dataclass(frozen=True)
class MissionConfig:
    settle_seconds: float = 2.
    move_seconds: float = 3.
    leg_budget_seconds: float = 6.
    goal_dwell_seconds: float = .3
    position_tolerance: float = .025
    speed_tolerance: float = .05
    termination_position_tolerance: float = .020
    termination_speed_tolerance: float = .040
    option_budget_seconds: float = 18.
    required_slack_seconds: float = .1
    payload_excursion_limit: float = .05
    donor_tension_limit: float = .6

    def __post_init__(self):
        if not (0 < self.termination_position_tolerance <= self.position_tolerance
                and 0 < self.termination_speed_tolerance <= self.speed_tolerance):
            raise ValueError("Internal termination bounds must be positive and no looser than external goal bounds")


def mission_cases():
    routes = (
        ((.20,.10,.05),(.35,-.10,0.),(.50,0.,-.05)),
        ((-.10,.20,.05),(.10,.35,0.),(0.,.50,-.05)),
    )
    return [dict(case_id=f"mass{int(round(mass*1000)):03d}_donor{donor}_route{route}",
        payload_mass=mass, donor=donor, route_template=route,
        formation_yaw=(math.pi/6 if route==0 else -math.pi/3),
        slack_distance=(.07 if route==0 else .09), waypoint_offsets=routes[route])
        for mass in (.026,.034) for donor in range(3) for route in range(2)]


def model_kind(planner):
    if planner not in LEARNED: return None
    return "small" if planner.startswith("wm_small") else "large"


def _memory_equal(left, right):
    if isinstance(left,dict) and isinstance(right,dict):
        return set(left)==set(right) and all(_memory_equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)):
        return len(left)==len(right) and all(_memory_equal(a,b) for a,b in zip(left,right))
    if isinstance(left,np.ndarray) or isinstance(right,np.ndarray):
        return np.array_equal(left,right)
    return left==right


def outcome_scores(predictions, config=None, margins=(0.,0.)):
    """Identical constrained endpoint criterion for learned and exact outcomes."""
    cfg=MissionConfig() if config is None else config
    predictions=np.asarray(predictions,dtype=float)
    margins=np.asarray(margins,dtype=float)
    if predictions.ndim!=2 or predictions.shape[1]!=8 or margins.shape!=(2,):
        raise ValueError("Expected eight operation outcomes and two validation margins")
    if not np.isfinite(margins).all() or (margins<0).any():
        raise ValueError("Outcome margins must be finite and nonnegative")
    finite=np.isfinite(predictions).all(1)
    safe=np.nan_to_num(predictions,nan=1e12,posinf=1e12,neginf=-1e12)
    excursion=safe[:,0]+margins[0];tension=safe[:,1]+margins[1]
    duration=safe[:,2];completion=np.clip(safe[:,7],0.,1.)
    feasible=(finite & (excursion<=cfg.payload_excursion_limit) &
        (tension<=cfg.donor_tension_limit) & (duration<=cfg.option_budget_seconds) &
        (completion>=.5))
    base=duration+.02*safe[:,3]+.01*safe[:,4]
    violations=np.stack((np.maximum(excursion/cfg.payload_excursion_limit-1.,0.),
        np.maximum(tension/cfg.donor_tension_limit-1.,0.),
        np.maximum(duration/cfg.option_budget_seconds-1.,0.)),axis=1)
    fallback=1000*np.sum(violations**2,axis=1)+100*(1-completion)+duration
    scores=np.where(feasible,base,1e30) if feasible.any() else np.where(finite,fallback,1e30)
    if not finite.any():raise FloatingPointError("Every operation prediction was nonfinite")
    return scores,feasible,not bool(feasible.any())


def select_option(env, controller, donor, depth, planner, model=None, *,
                  config=None, margins=(0.,0.)):
    """Predict and select one complete local operation without moving the parent."""
    cfg=MissionConfig() if config is None else config
    if planner not in PLANNERS:raise ValueError("Unknown mission planner")
    if planner in LEARNED and model is None:raise ValueError("Learned planner requires a frozen model")
    if planner in LEARNED:require_model_family(model,env.config)
    started=time.perf_counter()
    before=capture(env);memory=controller.capture_memory()
    obs=env.observe()
    if planner.startswith("rule_"):
        mode=planner=="rule_reconfigure"
        bank=[(mode,RecoveryConfig(rearrange_seconds=2.,retension_seconds=2.,slack_distance=depth))]
        prediction=None;selected=0;feasible=None;scores=None;fallback=False
        source="fixed_feedback_rule";rollout_steps=0
    else:
        bank=candidate_bank(depth,expanded=not planner.endswith("_direct"))
        if planner in LEARNED:
            inputs=np.stack([features(obs,env.config,donor,rc,mode,memory) for mode,rc in bank])
            prediction=np.asarray(model.predict(inputs),dtype=float)
            source="frozen_local_model";rollout_steps=0
        else:
            outcomes=[];rollout_steps=0
            for mode,rc in bank:
                other,control=fork(env,controller)
                outcome=run_option(other,control,rc,donor,mode,
                    max_seconds=cfg.option_budget_seconds,record=False)
                outcomes.append(outcome["label"])
                rollout_steps+=outcome["summary"]["control_steps"]
            prediction=np.asarray(outcomes)
            source="true_option_simulator"
        scores,feasible,fallback=outcome_scores(prediction,cfg,margins)
        selected=int(np.argmin(scores))
    if not np.array_equal(before.state,capture(env).state) or not _memory_equal(memory,controller.capture_memory()):
        raise AssertionError("Option selection changed the parent simulator or controller")
    mode,rc=bank[selected]
    record=dict(planner=planner,prediction_source=source,donor=int(donor),slack_distance=float(depth),
        options=[dict(reconfigure=m,recovery_config=asdict(c)) for m,c in bank],
        selected_index=selected,selected_reconfigure=mode,selected_recovery_config=asdict(rc),
        predictions=prediction,selected_prediction=None if prediction is None else prediction[selected].copy(),
        ranking_scores=scores,predicted_feasible=feasible,feasibility_fallback=fallback,
        validation_margins=np.asarray(margins),candidate_count=len(bank),
        feasible_count=0 if feasible is None else int(feasible.sum()),
        minimum_risk_fallback=bool(fallback),
        predictor_evaluations=0 if prediction is None else len(bank),
        simulated_candidate_control_steps=rollout_steps,
        simulated_candidate_physics_steps=rollout_steps*env.substeps,
        decision_seconds=time.perf_counter()-started,start_snapshot=before,
        controller_memory=memory,observation=obs)
    return mode,rc,record


def _append_block(trace, block, *, stage, references):
    trace["physical_states"].extend(block["physical_states"][1:])
    trace["integration_states"].extend(block["integration_states"][1:])
    trace["actions"].extend(block["actions"])
    trace["external_forces"].extend(block["external_forces"])
    trace["stages"].extend([stage]*len(block["actions"]))
    trace["phases"].extend(block["phases"])
    trace["references"].extend(references)
    trace["observations"].extend(block["observations"])
    for i in range(len(block["substeps"]["time"])):
        item={key:value[i] for key,value in block["substeps"].items()}
        item["stage"]=stage;trace["substeps"].append(item)


def _flight_stage(env,controller,target,*,name,config,settle=False):
    start_step=env.control_steps;obs=env.observe();start=obs["payload_pos"].copy()
    maximum=config.settle_seconds if settle else config.leg_budget_seconds
    limit=round(maximum/env.config.control_dt)
    required=max(1,int(math.ceil(config.goal_dwell_seconds/env.config.control_dt-1e-10)))
    external_dwell=0;termination_dwell=0
    block=dict(physical_states=[physical_state(env)],integration_states=[integration_state(env)],
        actions=[],external_forces=[],phases=[],observations=[])
    references=[]
    with _SubstepObserver(env,target) as observer:
        for step in range(limit):
            if settle:
                reference,velocity=target,np.zeros(3)
            else:
                reference,velocity=smooth_transition(start,target,step*env.config.control_dt,config.move_seconds)
            observer.goal=np.asarray(reference).copy()
            command=controller.command(obs,reference,payload_velocity=velocity)
            obs=observer.step(name,command)
            block["physical_states"].append(physical_state(env));block["integration_states"].append(integration_state(env))
            block["actions"].append(command.copy());block["external_forces"].append(env.last_external_forces.copy())
            block["phases"].append(name);block["observations"].append(obs);references.append(np.asarray(reference).copy())
            eligible=(step+1)*env.config.control_dt+1e-10 >= config.move_seconds
            position_error=np.linalg.norm(obs["payload_pos"]-target)
            speed=np.linalg.norm(obs["payload_vel"])
            external_ok=(position_error<=config.position_tolerance and speed<=config.speed_tolerance)
            termination_ok=(position_error<=config.termination_position_tolerance and
                            speed<=config.termination_speed_tolerance)
            external_dwell=external_dwell+1 if external_ok and (settle or eligible) else 0
            termination_dwell=termination_dwell+1 if termination_ok and (settle or eligible) else 0
            if not obs["finite"] or obs["warnings"].any():break
            if not settle and termination_dwell>=required:break
        block["substeps"]=_stack_records(observer.records)
    # The tighter predicate controls paid stopping time only. External task
    # success remains the original dwell truth, including at the time budget.
    success=bool(external_dwell>=required)
    stage=dict(name=name,begin_step=start_step,end_step=env.control_steps,target=np.asarray(target).copy(),
        start_position=start,maximum_seconds=maximum,elapsed_seconds=(env.control_steps-start_step)*env.config.control_dt,
        success=bool(success),goal_dwell_steps=external_dwell,required_dwell_steps=required,
        termination_dwell_steps=termination_dwell,guard_confirmed=bool(termination_dwell>=required),
        termination_guard_applied=not settle,
        final_position_error=float(np.linalg.norm(obs["payload_pos"]-target)),
        final_speed=float(np.linalg.norm(obs["payload_vel"])),finite=bool(obs["finite"]))
    return stage,block,references


def run_mission(case, planner, model=None, *, model_seed=0, config=None,
                margins=(0.,0.), dt=.002, record=True, plant_family=LEGACY,
                controller_factory=None):
    """Execute both recovery operations and all waypoints, retaining failed prefixes."""
    cfg=MissionConfig() if config is None else config
    if planner not in PLANNERS:raise ValueError("Unknown mission planner")
    if planner in LEARNED and model is None:raise ValueError("Learned mission requires frozen local weights")
    if min(asdict(cfg).values())<=0 or cfg.move_seconds>cfg.leg_budget_seconds:
        raise ValueError("Mission durations and limits must be positive and the path fit its leg budget")
    plant_cfg=make_config(plant_family,payload_mass=float(case["payload_mass"]),formation_yaw=float(case["formation_yaw"]),dt=float(dt))
    if planner in LEARNED:require_model_family(model,plant_cfg)
    for duration in (cfg.settle_seconds,cfg.leg_budget_seconds,cfg.option_budget_seconds):
        ratio=duration/plant_cfg.control_dt
        if ratio<1 or not math.isclose(ratio,round(ratio),abs_tol=1e-10):
            raise ValueError("Physical stage budgets must be positive integer control periods")
    if controller_factory is not None and planner not in ("rule_fixed","rule_reconfigure"):
        raise ValueError("An external feedback baseline uses the prescribed recovery protocol; learned models retain their training controller")
    env=make_env(plant_cfg)
    controller=(TransportController if controller_factory is None else controller_factory)(plant_cfg)
    initial=env.observe();origin=initial["payload_pos"].copy();initial_snapshot=capture(env)
    initial_controller_memory=controller.capture_memory()
    waypoints=origin+np.asarray(case["waypoint_offsets"],dtype=float)
    if waypoints.shape!=(3,3) or not np.isfinite(waypoints).all():raise ValueError("Expected three finite mission waypoints")
    trace=dict(physical_states=[physical_state(env)],integration_states=[integration_state(env)],
        actions=[],external_forces=[],stages=[],phases=[],references=[],observations=[],substeps=[])
    stages=[];operations=[];decisions=[];stop_reason="mission_sequence_end"
    stage,block,references=_flight_stage(env,controller,origin,name="settle",config=cfg,settle=True)
    stages.append(stage);_append_block(trace,block,stage="settle",references=references)
    for leg,target in enumerate(waypoints):
        if not env.observe()["finite"] or env.observe()["warnings"].any():
            stop_reason="physical_state_failure";break
        name=f"leg_{leg+1}"
        stage,block,references=_flight_stage(env,controller,target,name=name,config=cfg)
        stages.append(stage);_append_block(trace,block,stage=name,references=references)
        if leg==2:continue
        if not env.observe()["finite"] or env.observe()["warnings"].any():
            stop_reason="physical_state_failure";break
        donor=(int(case["donor"])+leg)%3
        mode,rc,decision=select_option(env,controller,donor,float(case["slack_distance"]),planner,model,
            config=cfg,margins=margins)
        decision.update(operation_index=leg,mission_step=env.control_steps)
        decisions.append(decision)
        begin=env.control_steps
        result=run_option(env,controller,rc,donor,mode,max_seconds=cfg.option_budget_seconds,record=True)
        result["mission_begin_step"]=begin;result["mission_end_step"]=env.control_steps
        operations.append(result)
        name=f"operation_{leg+1}"
        stages.append(dict(name=name,begin_step=begin,end_step=env.control_steps,
            elapsed_seconds=result["summary"]["elapsed_seconds"],maximum_seconds=cfg.option_budget_seconds,
            success=bool(result["summary"]["completion"]),donor=donor,reconfigure=mode,
            target=result["goal"].copy()))
        _append_block(trace,result["trace"],stage=name,
            references=[result["goal"].copy()]*len(result["trace"]["actions"]))
    final=env.observe()
    waypoints_pass=[next((s["success"] for s in stages if s["name"]==f"leg_{i+1}"),False) for i in range(3)]
    options_pass=[bool(operations[i]["summary"]["completion"]) if i<len(operations) else False for i in range(2)]
    slack_pass=[operations[i]["summary"]["donor_slack_seconds"]+1e-10>=cfg.required_slack_seconds if i<len(operations) else False for i in range(2)]
    records=trace["substeps"]
    finite=initial["finite"] and final["finite"] and all(x["finite"] for x in records)
    collision=bool(initial["collision_pairs"]) or any(x["collision"] for x in records)
    warning_count=int(final["warnings"].sum())
    delivery=bool(all(waypoints_pass) and all(options_pass) and all(slack_pass) and finite and not collision and warning_count==0)
    constrained=delivery and all(x["summary"]["peak_payload_excursion"]<=cfg.payload_excursion_limit
        and x["summary"]["peak_donor_tension"]<=cfg.donor_tension_limit for x in operations)
    flight_records=[x for x in records if x["stage"].startswith("leg_")]
    summary=dict(case_id=case["case_id"],planner=planner,model_seed=int(model_seed),plant_family=plant_family,
        controller_family=getattr(controller,"family","study_geometric"),
        payload_mass=plant_cfg.payload_mass,donor=int(case["donor"]),route_template=int(case["route_template"]),
        slack_distance=float(case["slack_distance"]),dt=plant_cfg.dt,control_dt=plant_cfg.control_dt,
        delivery_completion=delivery,constrained_completion=bool(constrained),
        waypoint_successes=waypoints_pass,option_completions=options_pass,actual_slack_successes=slack_pass,
        option_peak_excursions=[x["summary"]["peak_payload_excursion"] for x in operations],
        option_peak_donor_tensions=[x["summary"]["peak_donor_tension"] for x in operations],
        option_slack_seconds=[x["summary"]["donor_slack_seconds"] for x in operations],
        selected_reconfigurations=[x["reconfigure"] for x in operations],
        selected_durations=[[x["recovery_config"]["rearrange_seconds"],x["recovery_config"]["retension_seconds"]] for x in operations],
        total_seconds=env.control_steps*plant_cfg.control_dt,control_steps=env.control_steps,physics_steps=env.physics_steps,
        maximum_seconds=cfg.settle_seconds+3*cfg.leg_budget_seconds+2*cfg.option_budget_seconds,
        absolute_actuator_work=float(env.metrics["absolute_actuator_work"].sum()),
        total_squared_thrust_integral=float(env.metrics["squared_rotor_force_integral"].sum()),
        rotor_force_integral=float(env.metrics["rotor_force_integral"].sum()),
        uav_path_length=float(env.metrics["uav_path_length"].sum()),
        maximum_rotor_force=float(env.metrics["maximum_rotor_force"].max()),
        motor_bound_seconds=float(env.metrics["saturation_time"].sum()),
        peak_uav_tilt=float(env.metrics["maximum_uav_tilt"].max()),
        minimum_body_height=float(env.metrics["minimum_body_height"].min()),
        minimum_uav_separation=float(min(x["minimum_uav_separation"] for x in records)),
        flight_tracking_rmse=float(np.sqrt(np.mean([x["payload_error"]**2 for x in flight_records]))) if flight_records else None,
        finite=bool(finite),collision=collision,warning_count=warning_count,
        predictor_evaluations=sum(x["predictor_evaluations"] for x in decisions),
        simulated_candidate_control_steps=sum(x["simulated_candidate_control_steps"] for x in decisions),
        simulated_candidate_physics_steps=sum(x["simulated_candidate_physics_steps"] for x in decisions),
        total_decision_seconds=sum(x["decision_seconds"] for x in decisions),
        minimum_risk_fallbacks=sum(x["minimum_risk_fallback"] for x in decisions),
        feasible_candidate_counts=[x["feasible_count"] for x in decisions],
        stop_reason=stop_reason,model_kind=model_kind(planner))
    result=dict(summary=summary,case=deepcopy(case),mission_config=asdict(cfg),plant_config=asdict(plant_cfg),
        controller_config=controller.configuration(),validation_margins=np.asarray(margins),
        waypoints=waypoints,initial_snapshot=initial_snapshot,initial_controller_memory=initial_controller_memory,
        final_snapshot=capture(env),final_controller_memory=controller.capture_memory(),
        stages=stages,operations=operations,decisions=decisions,resources=deepcopy(env.metrics))
    if record:
        result["trace"]={key:(value if key in ("stages","phases","observations") else np.asarray(value))
            for key,value in trace.items() if key!="substeps"}
        result["trace"]["substeps"]=_stack_records(records)
        if len(result["trace"]["actions"])!=env.control_steps:
            raise AssertionError("Whole-mission recording missed executed physical control steps")
    else:result["trace"]=None
    return result


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(case,planner,model_seed,model_entry,output,source_hashes,model_hashes,plant_family=LEGACY):
    for path,digest in {**source_hashes,**model_hashes}.items():
        if _sha(path)!=digest:raise AssertionError("Registered mission input changed: "+path)
    path=Path(f"{output}_{planner}_seed{model_seed}_{case['case_id']}.pkl")
    if path.exists():raise FileExistsError(path)
    kind=model_kind(planner)
    prefix=model_entry["prefix"] if kind else None
    model=load_model(prefix) if prefix else None
    margins=tuple(model_entry["margins"]) if kind else (0.,0.)
    started=time.perf_counter()
    result=run_mission(case,planner,model,model_seed=model_seed,margins=margins,plant_family=plant_family)
    with path.open("wb") as stream:pickle.dump(result,stream,protocol=5)
    return dict(summary=result["summary"],raw_path=str(path),raw_sha256=_sha(path),
        wall_seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("data/uav_transport/mission_v1"))
    parser.add_argument("--models",type=Path,default=Path("data/uav_transport/models_v1_manifest.json"))
    parser.add_argument("--planners",nargs="+",choices=PLANNERS,default=list(PLANNERS))
    parser.add_argument("--model-seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--register-only",action="store_true")
    parser.add_argument("--plant",choices=FAMILIES,default=LEGACY)
    args=parser.parse_args()
    if not 1<=args.workers<=8 or len(set(args.planners))!=len(args.planners) or len(set(args.model_seeds))!=len(args.model_seeds):
        raise ValueError("Use1..8workers and unique planners/seeds")
    cases=mission_cases()
    schedule=[(case,planner,seed) for planner in sorted(args.planners,key=lambda x:not x.startswith("true_option"))
        for seed in (args.model_seeds if planner in LEARNED else [0]) for case in cases]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.with_suffix(".json").exists():raise FileExistsError(args.out)
    paths=[Path(__file__)]+[Path("uav_transport")/(name+".py") for name in
        ("plant","snapshot","control","evaluation","collect","local_model")]
    paths += source_paths(args.plant)
    source_hashes={str(p):_sha(p) for p in paths}
    model_hashes={}
    model_manifest=json.loads(args.models.read_text()) if any(p in LEARNED for p in args.planners) else dict(models={})
    if model_manifest["models"] and model_manifest.get("plant_family",LEGACY)!=args.plant:
        raise ValueError("Use models trained on the requested plant family")
    if model_manifest["models"]:model_hashes[str(args.models)]=_sha(args.models)
    for planner in args.planners:
        kind=model_kind(planner)
        if kind:
            for seed in args.model_seeds:
                entry=model_manifest["models"][f"{kind}/seed{seed}"]
                for suffix,key in (("npz","weights_sha256"),("json","metadata_sha256")):
                    path=entry["prefix"]+"."+suffix
                    if _sha(path)!=entry[key]:raise AssertionError("Frozen model manifest mismatch: "+path)
                    model_hashes[path]=entry[key]
    registration=dict(plant_family=args.plant,cases=cases,planners=args.planners,model_seeds=args.model_seeds,
        mission_config=asdict(MissionConfig()),source_sha256=source_hashes,model_sha256=model_hashes,
        mission_count=len(schedule),physical_cases=len(cases),
        model_manifest_path=str(args.models) if model_manifest["models"] else None,
        validation_margins={key:value["margins"] for key,value in model_manifest["models"].items()},
        margin_policy="Frozen validation-only positive underprediction q95 for learned predictors; zero for exact dynamics",
        calibrated_parameters="Known payload mass supplied equally to all controllers and predictors",
        task="Three waypoint legs with two complete controlled unloading/recovery operations",
        candidate_bank="direct9; expanded18; geometrymode x rearrange1/2/3s x retension1.5/2/3s",
        source_target="Only prior goal-free local operations fit the frozen predictors",
        constrained_score="All delivery requirements plus each operation excursion<=.05m and donor tension<=.6N",
        goal_dwell="External scoring remains.3s after the3squinticreference reaches its endpoint; position<=.025m and speed<=.05m/s",
        termination_guard="Early flight-leg stopping requires.3s atposition<=.020m andspeed<=.040m/s; every extra physical step counts within the same6s budget; budget endpoints are scored using the original external bounds",
        historical_control="MissionConfig(termination_position_tolerance=.025,termination_speed_tolerance=.05) reproduces the v1 stopping predicate",
        selector="Predicted-feasible minimumelapsed+.02work+.01squaredthrust; ifnone,normalizedconstraint-excess penalty",
        numerical_note="Physicstimestep.002s, controlperiod.02s; actualmotorlag.03s; finitecompliancetension-onlycables")
    reg=args.out.with_name(args.out.name+"_registration.json")
    if reg.exists():
        if json.loads(reg.read_text())!=plain(registration):raise ValueError("Mission registration changed")
    else:reg.write_text(json.dumps(plain(registration),indent=2)+"\n")
    if args.register_only:print(reg);return
    started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(worker,case,p,seed,
            model_manifest["models"].get(f"{model_kind(p)}/seed{seed}"),args.out,source_hashes,model_hashes,args.plant)
            for case,p,seed in schedule]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print(json.dumps(dict(completed=len(results),total=len(schedule),**result["summary"])),flush=True)
    for path,digest in {**source_hashes,**model_hashes}.items():
        if _sha(path)!=digest:raise AssertionError("Frozen mission input changed: "+path)
    results.sort(key=lambda x:(x["summary"]["planner"],x["summary"]["model_seed"],x["summary"]["case_id"]))
    report=dict(registration,registration_path=str(reg),registration_sha256=_sha(reg),results=results,
        wall_seconds=time.perf_counter()-started,verified_frozen_inputs=True)
    args.out.with_suffix(".json").write_text(json.dumps(plain(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.out),missions=len(results),seconds=report["wall_seconds"])),flush=True)


if __name__=="__main__":main()
