"""Physical execution, complete-state forks and replay of local transport options.

Local options hold the payload at their starting pose by default. The mission
goal is external to this operator and to its outcome labels. Every reference
change is executed through the frozen controller and ordinary plant steps.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import math

import numpy as np

from .control import RecoveryConfig, RecoveryProtocol, TransportController
from .plant import TransportConfig, TransportEnv
from .snapshot import capture, restore, integration_state, physical_state


TARGET_NAMES = ("peak_payload_excursion", "peak_donor_tension", "elapsed_seconds",
    "absolute_actuator_work", "total_squared_thrust_integral", "final_payload_error",
    "final_payload_speed", "completion")


def fork(env, controller):
    """Fork simulator, actuator state, resources and feedback-controller memory."""
    child_env = env.clone()
    child_controller = TransportController(child_env.config, controller.config)
    child_controller.restore_memory(controller.capture_memory())
    return child_env, child_controller


def _state_measurement(env, tensions, goal):
    positions = env._positions()
    velocities = np.stack([env.data.qvel[a:a+3] for a in env.dof_addresses])
    quats = np.stack([env.data.qpos[a+3:a+7] for a in env.qpos_addresses[1:]])
    tilt = np.arccos(np.clip(1-2*(quats[:,1]**2+quats[:,2]**2),-1.,1.))
    distances = [np.linalg.norm(positions[a]-positions[b]) for a,b in ((1,2),(1,3),(2,3))]
    return dict(time=float(env.data.time), payload_position=positions[0].copy(),
        payload_error=float(np.linalg.norm(positions[0]-goal)),
        payload_speed=float(np.linalg.norm(velocities[0])), cable_tensions=tensions.copy(),
        cable_gaps=env.config.cable_length-env.data.ten_length.copy(),
        minimum_uav_separation=float(min(distances)), uav_tilt=tilt,
        minimum_body_height=float(positions[:,2].min()), collision=bool(env.data.ncon),
        floor_contact=any(0 in pair for pair in env.collision_pairs()),
        finite=bool(np.isfinite(physical_state(env)).all()), warnings=env.data.warning.number.copy())


class _SubstepObserver:
    """Read the already-computed cable forces once per actual physics step.

    The plant calls cable_tensions while accumulating substep resources and
    once more for the final observation. The latter is excluded by the physical
    clock, and exact sample counts are asserted so changes to that contract
    cannot silently alter force/impulse measurements.
    """
    def __init__(self, env, goal):
        self.env, self.goal = env, np.asarray(goal,dtype=float)
        self.records = []
        self.phase = None
        self.expected_end = None
        self.original = env.cable_tensions
        self.instance_override = "cable_tensions" in env.__dict__
        self.original_override = env.__dict__.get("cable_tensions")

    def __enter__(self):
        def observe():
            tensions = self.original()
            if self.expected_end is not None and self.env.physics_steps < self.expected_end:
                item = _state_measurement(self.env,tensions,self.goal)
                item.update(phase=self.phase, physics_step=self.env.physics_steps)
                self.records.append(item)
            return tensions
        self.env.cable_tensions = observe
        return self

    def step(self, phase, command, external_forces=None):
        before = len(self.records)
        start = self.env.physics_steps
        self.expected_end = start+self.env.substeps
        self.phase = phase
        observation = self.env.step(command,external_forces=external_forces)
        self.expected_end = None
        new = self.records[before:]
        if len(new) != self.env.substeps or [x["physics_step"] for x in new] != list(range(start,start+self.env.substeps)):
            raise AssertionError("Physical force observer did not record exactly one sample per substep")
        return observation

    def __exit__(self, *_):
        if self.instance_override:
            self.env.cable_tensions = self.original_override
        else:
            del self.env.__dict__["cable_tensions"]


def _stack_records(records):
    if not records:
        return {}
    return {key: (list(item[key] for item in records) if key=="phase" else
                  np.asarray([item[key] for item in records])) for key in records[0]}


def _resource_delta(start, end):
    additive = ("rotor_force_integral", "squared_rotor_force_integral",
        "signed_actuator_work", "absolute_actuator_work", "uav_path_length",
        "saturation_time", "clamp_intervention_time", "command_saturation_time",
        "command_force_integral", "rotor_upper_bound_time", "rotor_lower_bound_time",
        "contact_steps", "floor_contact_steps", "uav_collision_steps", "payload_uav_collision_steps")
    return {key: deepcopy(end[key])-deepcopy(start[key]) for key in additive}


def _summary(env, initial, final, substeps, resources, donor, goal, start_position,
             elapsed, protocol, verification, stop_reason):
    records = substeps
    tensions = np.asarray([x["cable_tensions"] for x in records])
    gaps = np.asarray([x["cable_gaps"] for x in records])
    errors = np.asarray([x["payload_error"] for x in records])
    phases = np.asarray([x["phase"] for x in records])
    positions = np.asarray([x["payload_position"] for x in records])
    retension = (phases=="retension")|(phases=="confirm")
    slack = (tensions[:,donor]<.005)&(gaps[:,donor]>.01)
    final_error = float(np.linalg.norm(final["payload_pos"]-goal))
    finite = all(x["finite"] for x in records) and initial["finite"] and final["finite"]
    collision = any(x["collision"] for x in records) or bool(initial["collision_pairs"]) or bool(final["collision_pairs"])
    warnings = final["warnings"]-initial["warnings"]
    initial_separation = min(np.linalg.norm(initial["uav_pos"][i]-initial["uav_pos"][j])
                            for i,j in ((0,1),(0,2),(1,2)))
    completion = bool(protocol.done and not protocol.failed and verification
        and finite and not collision and not final["warnings"].any())
    summary = dict(peak_payload_excursion=float(max(errors.max(),np.linalg.norm(initial["payload_pos"]-goal),final_error)),
        peak_payload_displacement=float(max(np.linalg.norm(positions-start_position,axis=1).max(),0.)),
        payload_tracking_rmse=float(np.sqrt(np.mean(errors**2))),
        peak_payload_speed=float(max(max(x["payload_speed"] for x in records),np.linalg.norm(initial["payload_vel"]))),
        peak_donor_tension=float(max(tensions[:,donor].max(),initial["cable_tensions"][donor],final["cable_tensions"][donor])),
        retension_peak_donor_tension=float(tensions[retension,donor].max()) if retension.any() else 0.,
        retension_tension_impulse=float(tensions[retension,donor].sum()*env.config.dt),
        donor_slack_seconds=float(slack.sum()*env.config.dt), donor_slack_observed=bool(slack.any()),
        elapsed_seconds=float(elapsed), absolute_actuator_work=float(resources["absolute_actuator_work"].sum()),
        total_squared_thrust_integral=float(resources["squared_rotor_force_integral"].sum()),
        final_payload_error=final_error, final_payload_speed=float(np.linalg.norm(final["payload_vel"])),
        completion=completion, protocol_done=bool(protocol.done),protocol_failed=bool(protocol.failed),
        verification_dwell_passed=bool(verification),stop_reason=stop_reason,
        phase=str(protocol.phase), gate_dwell_seconds=float(protocol.gate_dwell),
        minimum_uav_separation=float(min(min(x["minimum_uav_separation"] for x in records),initial_separation)),
        peak_uav_tilt=float(max(max(np.max(x["uav_tilt"]) for x in records),np.max(initial["uav_tilt"]))),
        minimum_body_height=float(min(min(x["minimum_body_height"] for x in records),
            initial["payload_pos"][2],np.min(initial["uav_pos"][:,2]))),
        collision=collision, floor_contact=bool(any(x["floor_contact"] for x in records) or final["floor_contact"] or initial["floor_contact"]),
        finite=finite, warning_count=int(warnings.sum()), cumulative_warning_count=int(final["warnings"].sum()),
        maximum_cable_tensions=np.maximum.reduce([tensions.max(0),initial["cable_tensions"],final["cable_tensions"]]),
        motor_bound_seconds=resources["saturation_time"].copy(),
        resources=resources, physics_steps=len(records), control_steps=round(elapsed/env.config.control_dt))
    return summary


def _verified(observations, goal, config, dt):
    count = int(math.ceil(config.verify_seconds/dt-1e-10))
    if len(observations)<count:
        return False
    return all(np.all(obs["cable_tensions"]>max(.02,config.support_threshold))
        and np.all(obs["cable_gaps"]<=config.gap_tolerance)
        and np.linalg.norm(obs["payload_pos"]-goal)<=config.payload_error_limit
        and np.linalg.norm(obs["payload_vel"])<=config.payload_speed_limit
        for obs in observations[-count:])


def run_option(env, controller, recovery_config, donor, reconfigure, goal=None,
               max_seconds=18., record=True):
    """Execute a complete local maneuver or retain its physical timeout/failure.

    The supplied environment and controller advance normally. To score another
    option from the same state use ``fork`` before calling this function.
    """
    cfg = RecoveryConfig() if recovery_config is None else recovery_config
    if not math.isfinite(max_seconds) or max_seconds < env.config.control_dt:
        raise ValueError("The option budget must permit at least one physical control step")
    initial = env.observe()
    goal = initial["payload_pos"].copy() if goal is None else np.asarray(goal,dtype=float).copy()
    if goal.shape!=(3,) or not np.isfinite(goal).all():
        raise ValueError("The local holding reference must be a finite 3D point")
    protocol = RecoveryProtocol(env.config,cfg,recovering_uav=donor,reconfigure=reconfigure)
    start_snapshot, start_memory = capture(env),controller.capture_memory()
    start_protocol = protocol.capture_memory()
    resources_before = deepcopy(env.metrics)
    start_control = env.control_steps
    states=[physical_state(env)] if record else []
    integrations=[integration_state(env)] if record else []
    commands, forces, phases, observations, protocol_states = [],[],[],[],[]
    budget_steps=int(math.floor(max_seconds/env.config.control_dt+1e-10))
    stop_reason="physical_time_budget"
    final=initial
    with _SubstepObserver(env,goal) as observer:
        for _ in range(budget_steps):
            reference=protocol.reference()
            command=controller.command(final,goal,formation_offsets=reference.offsets,
                formation_velocity=reference.offset_velocity,support_weights=reference.support_weights)
            final=observer.step(reference.phase,command)
            observations.append(final)
            protocol.update(final,goal)
            if record:
                states.append(physical_state(env));integrations.append(integration_state(env))
                commands.append(command.copy());forces.append(env.last_external_forces.copy())
                phases.append(reference.phase);protocol_states.append(protocol.capture_memory())
            if not final["finite"]:
                stop_reason="nonfinite_physical_state";break
            if final["warnings"].any():
                stop_reason="simulator_warning";break
            if protocol.done:
                stop_reason="verified_protocol_end";break
            if protocol.failed:
                stop_reason=str(protocol.failure_reason);break
    elapsed=(env.control_steps-start_control)*env.config.control_dt
    resources=_resource_delta(resources_before,env.metrics)
    verification=_verified(observations,goal,cfg,env.config.control_dt)
    summary=_summary(env,initial,final,observer.records,resources,int(donor),goal,
        initial["payload_pos"],elapsed,protocol,verification,stop_reason)
    trace = None
    if record:
        trace=dict(physical_states=np.asarray(states),integration_states=np.asarray(integrations),
            actions=np.asarray(commands),external_forces=np.asarray(forces),phases=phases,
            observations=observations,substeps=_stack_records(observer.records),
            protocol_memory=protocol_states,initial_observation=initial)
    return dict(summary=summary,label=np.asarray([float(summary[x]) for x in TARGET_NAMES]),
        label_names=TARGET_NAMES,trace=trace,start_snapshot=start_snapshot,end_snapshot=capture(env),
        start_controller_memory=start_memory,end_controller_memory=controller.capture_memory(),
        start_protocol_memory=start_protocol,end_protocol_memory=protocol.capture_memory(),
        plant_config=asdict(env.config),controller_config=controller.configuration(),
        recovery_config=asdict(cfg),donor=int(donor),reconfigure=bool(reconfigure),goal=goal,
        maximum_seconds=float(max_seconds),physics_specification=env.specification())


def replay_option(result, dt=None):
    """Replay saved rotor commands at their original physical stopping time."""
    trace=result["trace"]
    if trace is None:
        raise ValueError("Fixed-action replay requires a recorded option")
    cfg=TransportConfig(**result["plant_config"])
    if dt is not None:cfg=replace(cfg,dt=float(dt))
    env=TransportEnv(cfg)
    restore(env,result["start_snapshot"],allow_timestep_change=dt is not None)
    resources_before=deepcopy(env.metrics)
    initial=env.observe();states=[physical_state(env)];integrations=[integration_state(env)]
    observations=[]
    with _SubstepObserver(env,result["goal"]) as observer:
        for phase,command,forces in zip(trace["phases"],trace["actions"],trace["external_forces"]):
            observations.append(observer.step(phase,command,forces))
            states.append(physical_state(env));integrations.append(integration_state(env))
    if not observations:
        raise ValueError("A recorded option must contain executed physical actions")
    recovery=RecoveryConfig(**result["recovery_config"])
    protocol=RecoveryProtocol(cfg,recovery,recovering_uav=result["donor"],reconfigure=result["reconfigure"])
    protocol.restore_memory(result["end_protocol_memory"])
    verification=_verified(observations,result["goal"],recovery,cfg.control_dt)
    summary=_summary(env,initial,observations[-1],observer.records,
        _resource_delta(resources_before,env.metrics),result["donor"],result["goal"],initial["payload_pos"],
        len(trace["actions"])*cfg.control_dt,protocol,verification,result["summary"]["stop_reason"])
    return dict(summary=summary,physical_states=np.asarray(states),integration_states=np.asarray(integrations),
        substeps=_stack_records(observer.records),end_snapshot=capture(env),
        source_dt=result["plant_config"]["dt"],replay_dt=cfg.dt,
        source_replay_bitwise=bool(np.array_equal(np.asarray(states),trace["physical_states"])),
        interpretation="Recorded joint actions, original duration and source protocol transitions; measured final dwell is rescored.")
