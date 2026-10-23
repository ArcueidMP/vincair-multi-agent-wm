"""Matched actuator ablations of physical rebracing; no model training.

On/off denotes bounded drive command versus zero drive. All bodies retain their
finite inertia and contacts. A read-only observer records the engine's actual
sphere/box contact forces at integration substeps without recomputing physics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import torch

from vmas_pusht.abstraction_convergence import restore_across_substeps
from vmas_pusht.boundary import _finite, _at_wall, _plain
from vmas_pusht.physical_abstraction import physical_state
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.rebrace import RebraceController
from vmas_pusht.scenario import wrap
from vmas_pusht.snapshot import EntitySnapshot, SimulationSnapshot, capture, restore


BRANCHES = (("A", False, False), ("B", True, False), ("C", False, True), ("D", True, True))


class PhysicsObserver:
    """Non-invasive substep instrumentation of *actual* executed dynamics.

    Usage::

        with PhysicsObserver(env) as observer:
            observer.begin_step()
            env.step([primary_force, helper_force])
            measured = observer.end_step()

    Contact forces/torques are the exact arguments supplied by VMAS to
    update_env_forces, not a proximity proxy or independently recalculated force.
    Forces act on the agent; torque acts on its contacted T part about that part's
    center. Work is the actual actuator force dotted with the substep's physical
    displacement. It includes signed, absolute, positive and negative work.
    Peak times are zero-based substep indices within this environment step.
    Instrument only executed trajectories, not large planning candidate batches.
    """
    def __init__(self, env):
        self.env, self.world = env, env.world
        self.agents = {entity: i for i, entity in enumerate(env.world.agents)}
        self.parts = {env.scenario.bar: 0, env.scenario.stem: 1}
        if len(self.agents) != 2:
            raise ValueError("The recovery observer requires two agents")
        self.active = False
        self.installed = False
        self.current_substep = 0

    def __enter__(self):
        if self.installed:
            raise RuntimeError("Observer already installed")
        self.original_forces = self.world.update_env_forces
        self.original_integrate = self.world._integrate_state
        self.original_environment_force = self.world._apply_vectorized_enviornment_force

        def environment_force():
            self.original_environment_force()
            if self.active:
                self.current_substep += 1

        def update_forces(entity_a, force_a, torque_a, entity_b, force_b, torque_b):
            self.original_forces(entity_a, force_a, torque_a, entity_b, force_b, torque_b)
            if not self.active:
                return
            if entity_a in self.agents and entity_b in self.parts:
                agent, part, force, torque = self.agents[entity_a], self.parts[entity_b], force_a, torque_b
            elif entity_b in self.agents and entity_a in self.parts:
                agent, part, force, torque = self.agents[entity_b], self.parts[entity_a], force_b, torque_a
            else:
                return
            dt = self.world._sub_dt
            norm = force.norm(dim=-1)
            self.values["contact_force_impulse"][:, agent, part] += force * dt
            self.values["contact_force_norm_integral"][:, agent, part] += norm * dt
            self.values["contact_torque_impulse_about_part"][:, agent, part] += torch.as_tensor(torque).reshape(-1) * dt
            peak = self.values["contact_force_peak_norm"][:, agent, part]
            bigger = norm > peak
            peak[bigger] = norm[bigger]
            self.values["contact_force_peak_substep"][:, agent, part][bigger] = self.current_substep

        def integrate(entity, substep):
            if not self.active:
                return self.original_integrate(entity, substep)
            index = self.agents.get(entity)
            old_position = entity.state.pos.clone() if index is not None else None
            result = self.original_integrate(entity, substep)
            if index is not None:
                displacement = entity.state.pos - old_position
                force = entity.state.force
                work = (force * displacement).sum(-1)
                norm = force.norm(dim=-1)
                dt = self.world._sub_dt
                self.values["actuator_signed_work"][:, index] += work
                self.values["actuator_absolute_work"][:, index] += work.abs()
                self.values["actuator_positive_work"][:, index] += work.clamp_min(0)
                self.values["actuator_negative_work"][:, index] += work.clamp_max(0)
                self.values["force_norm_integral"][:, index] += norm * dt
                self.values["squared_force_integral"][:, index] += norm.square() * dt
                self.values["agent_path_length"][:, index] += displacement.norm(dim=-1)
                power = work.abs() / dt
                peak = self.values["actuator_peak_abs_power"][:, index]
                bigger = power > peak
                peak[bigger] = power[bigger]
                self.values["actuator_peak_abs_power_substep"][:, index][bigger] = substep
            if entity is self.env.scenario.bar:
                omega = entity.state.ang_vel[:, 0]
                bigger = omega.abs() > self.values["bar_peak_abs_omega"]
                self.values["bar_peak_abs_omega"][bigger] = omega.abs()[bigger]
                self.values["bar_peak_abs_omega_signed"][bigger] = omega[bigger]
                self.values["bar_peak_abs_omega_substep"][bigger] = substep
            return result

        self.world.update_env_forces = update_forces
        self.world._integrate_state = integrate
        self.world._apply_vectorized_enviornment_force = environment_force
        self.installed = True
        return self

    def begin_step(self):
        if not self.installed or self.active:
            raise RuntimeError("Observer must be installed and previous step finished")
        n, dev = self.env.num_envs, self.env.device
        z = lambda *shape: torch.zeros(n, *shape, device=dev)
        indices = lambda *shape: torch.full((n, *shape), -1, dtype=torch.long, device=dev)
        self.values = dict(contact_force_impulse=z(2, 2, 2), contact_force_norm_integral=z(2, 2),
            contact_force_peak_norm=z(2, 2), contact_force_peak_substep=indices(2, 2),
            contact_torque_impulse_about_part=z(2, 2), actuator_signed_work=z(2),
            actuator_absolute_work=z(2), actuator_positive_work=z(2), actuator_negative_work=z(2),
            force_norm_integral=z(2), squared_force_integral=z(2), agent_path_length=z(2),
            actuator_peak_abs_power=z(2), actuator_peak_abs_power_substep=indices(2),
            bar_peak_abs_omega=z(), bar_peak_abs_omega_signed=z(), bar_peak_abs_omega_substep=indices())
        self.current_substep = 0
        self.active = True

    def end_step(self):
        if not self.active:
            raise RuntimeError("No measured step is active")
        self.active = False
        if self.current_substep != self.world._substeps:
            raise AssertionError("Observer did not capture exactly one full environment step")
        return {key: value.clone() for key, value in self.values.items()}

    def __exit__(self, exc_type, exc, tb):
        self.world.update_env_forces = self.original_forces
        self.world._integrate_state = self.original_integrate
        self.world._apply_vectorized_enviornment_force = self.original_environment_force
        self.active = self.installed = False


def snapshot_dict(snapshot, indices=None):
    saved = asdict(snapshot)
    if indices is None:
        return saved
    idx = torch.as_tensor(indices, dtype=torch.long)
    def select(value):
        return None if value is None else value.index_select(0, idx.to(value.device)).clone()
    for entity in saved["entities"]:
        entity["state"] = {key: select(value) for key, value in entity["state"].items()}
        if entity["action"] is not None:
            entity["action"] = {key: select(value) for key, value in entity["action"].items()}
    saved["scenario"] = {key: select(value) for key, value in saved["scenario"].items()}
    saved["steps"] = select(saved["steps"])
    saved["batch_size"] = len(idx)
    return saved


def load_snapshot(saved):
    saved = dict(saved)
    saved["entities"] = tuple(EntitySnapshot(**item) for item in saved["entities"])
    return SimulationSnapshot(**saved)


def merge_snapshots(snapshots):
    if not snapshots:
        raise ValueError("No snapshots to merge")
    first = snapshots[0]
    if any(s["physics"] != first["physics"] for s in snapshots):
        raise ValueError("Only identical physical configuration can be merged")
    def cat(values):
        if all(value is None for value in values): return None
        if any(value is None for value in values): raise ValueError("Snapshot field presence differs")
        return torch.cat(values, 0)
    entities = []
    for i, entity in enumerate(first["entities"]):
        entries = [s["entities"][i] for s in snapshots]
        entities.append(dict(name=entity["name"],
            state={key: cat([item["state"][key] for item in entries]) for key in entity["state"]},
            action=None if entity["action"] is None else {
                key: cat([item["action"][key] for item in entries]) for key in entity["action"]}))
    return dict(batch_size=sum(s["batch_size"] for s in snapshots), physics=first["physics"],
        entities=tuple(entities), scenario={key: cat([s["scenario"][key] for s in snapshots]) for key in first["scenario"]},
        steps=cat([s["steps"] for s in snapshots]))


@torch.no_grad()
def reconstruct_retreats(source, *, source_substeps=160):
    """Replay original complete batches and preserve exact retreat-end states.

    Only the saved post-prelude snapshot is restored. Every intervening action
    comes from the source. All 24 physical coordinates are checked bitwise at
    every available source time, before any row is extracted or batch expanded.
    """
    snapshots, cases, checks = [], [], []
    groups = [group for group in source["groups"] if group["rows"][0]["substeps"] == source_substeps]
    if not groups:
        raise ValueError("No source groups at the requested resolution")
    profile = groups[0]["physics_profile"]
    for group_index, group in enumerate(groups):
        if group["physics_profile"] != profile:
            raise ValueError("Source groups must share one physics profile")
        rows = group["rows"]; n = len(rows)
        starts = [record for record in group["counterfactual_snapshots"]
                  if all(row["snapshot_stage"] == "post_prelude" for row in record["rows"])]
        if len(starts) != 1 or starts[0]["snapshot"]["batch_size"] != n:
            raise ValueError("Expected the intact, original-order post-prelude snapshot")
        if [row["composition_row"] for row in starts[0]["rows"]] != list(range(n)):
            raise ValueError("Source snapshot order differs from the original batch")
        env = make_profile(profile, num_envs=n, substeps=source_substeps, seed=101)
        restore(env, load_snapshot(starts[0]["snapshot"]))
        if not torch.equal(physical_state(env), group["trajectory"][:, 0]):
            raise AssertionError("Initial post-prelude snapshot differs from recorded trajectory")
        ends = torch.tensor([row["phase_end_steps"]["retreat"] for row in rows])
        captured = {}
        maximum_error = 0.
        for step in range(int(ends.max())):
            commands = group["actions"][:, step]
            env.step([commands[:, 0], commands[:, 1]])
            actual = physical_state(env)
            expected = group["trajectory"][:, step + 1]
            maximum_error = max(maximum_error, float((actual - expected).abs().max()))
            if not torch.equal(actual, expected):
                raise AssertionError(f"Original batch replay is not bitwise exact at group {group_index}, step {step+1}: {maximum_error}")
            selected = (ends == step + 1).nonzero().flatten().tolist()
            if selected:
                saved = capture(env)
                for i in selected:
                    captured[i] = snapshot_dict(saved, [i])
        for i, row in enumerate(rows):
            if i not in captured:
                raise AssertionError("A retreat endpoint was not retained")
            start, end = row["phase_end_steps"]["retreat"], row["phase_end_steps"]["recovery"]
            snapshots.append(captured[i])
            cases.append(dict(case_id=len(cases), source_group=group_index, source_row=i,
                support=row["support"], planner=row["planner"], first_goal_passed=row["first_goal_success_at_phase_boundary"],
                first_goal_rad=row["first_goal_rad"], final_goal_rad=row["relative_goal_rad"],
                initial_angle_deg=row["initial_angle_deg"], nuisance_id=row["nuisance_id"],
                initial_velocity=row["initial_velocity"], initial_angular_velocity=row["initial_angular_velocity"],
                source_retreat_end_step=start, historical_handoff_steps=end-start,
                historical_recovery_gate=row["contact_reacquired"],
                historical_handoff_state=group["trajectory"][i, end].clone(),
                historical_recovery_actions=group["actions"][i, start:end].clone(),
                source_initial_pose=group["initial_pose"][i].clone()))
        checks.append(dict(source_group=group_index, original_batch_size=n, row_order_preserved=True,
            replay_steps=int(ends.max()), physical_coordinates=24, bitwise_all_times=True,
            maximum_absolute_state_error=maximum_error, captured_cases=n))
    return dict(profile=profile, snapshot=merge_snapshots(snapshots), cases=cases, replay_checks=checks,
                source_substeps=source_substeps)


@torch.no_grad()
def run_factorial(prepared, *, substeps=160, horizon=80, instrument=True):
    """All four branches start from the same true state; only drive masks vary."""
    cases = prepared["cases"]
    if not cases or horizon < max(case["historical_handoff_steps"] for case in cases):
        raise ValueError("Recovery horizon must include every historical handoff")
    base = load_snapshot(prepared["snapshot"])
    index = torch.arange(len(cases)).repeat_interleave(4)
    expanded = load_snapshot(snapshot_dict(base, index))
    rows = [dict(case_id=case["case_id"], support=case["support"], branch=name,
                 main_on=main, helper_on=helper)
            for case in cases for name, main, helper in BRANCHES]
    n = len(rows)
    env = make_profile(prepared["profile"], num_envs=n, substeps=substeps, seed=101)
    restore_across_substeps(env, expanded)
    initial = physical_state(env).clone()
    grouped_initial = initial.reshape(len(cases), 4, 24)
    if not torch.equal(grouped_initial, grouped_initial[:, :1].expand_as(grouped_initial)):
        raise AssertionError("Factorial branches do not have exactly identical complete initial states")
    support_names = [row["support"] for row in rows]
    controller = RebraceController(env, support_names, max_steps=horizon)
    main_mask = torch.tensor([row["main_on"] for row in rows])
    helper_mask = torch.tensor([row["helper_on"] for row in rows])
    trajectory, actions, actual_forces, phases, anchors, readiness, measurements = [initial], [], [], [], [], [], []
    first_ready = torch.full((n,), -1, dtype=torch.long)
    finite, wall = _finite(env), _at_wall(env)
    observer = PhysicsObserver(env)
    observer.__enter__()
    try:
        for step in range(horizon):
            command = controller.step()
            command[~main_mask, 0] = 0.
            command[~helper_mask, 1] = 0.
            if not torch.isfinite(command).all() or command.norm(dim=-1).max() > 1.000001:
                raise AssertionError("Recovery drive exceeds the finite unit force bound")
            phases.append(controller.phase.clone())
            if instrument: observer.begin_step()
            env.step([command[:, 0], command[:, 1]])
            if instrument: measurements.append(observer.end_step())
            state = controller.update()
            new = state["ready"] & (first_ready < 0)
            first_ready[new] = step + 1
            trajectory.append(physical_state(env).clone())
            actions.append(command.clone())
            actual_forces.append(torch.stack([agent.state.force.clone() for agent in env.world.agents], 1))
            anchors.append(controller.anchor.clone())
            readiness.append(state["ready"].clone())
            finite &= _finite(env); wall |= _at_wall(env)
    finally:
        observer.__exit__(None, None, None)
    result = dict(rows=rows, trajectory=torch.stack(trajectory, 1), actions=torch.stack(actions, 1),
        actual_actuator_forces=torch.stack(actual_forces, 1), controller_phase=torch.stack(phases, 1),
        helper_anchor=torch.stack(anchors, 1), ready=torch.stack(readiness, 1), first_ready_step=first_ready,
        finite=finite, wall=wall, substeps=substeps, dt=float(env.world.dt), horizon=horizon,
        controller_configuration=controller.configuration(),
        measurements={key: torch.stack([step[key] for step in measurements], 1) for key in measurements[0]} if measurements else {})
    if not finite.all():
        raise AssertionError("A recovery branch became nonfinite")
    if instrument:
        m=result["measurements"]
        if not m["actuator_absolute_work"][~main_mask, :, 0].eq(0).all():
            raise AssertionError("Disabled primary performed actuator work")
        if not m["actuator_absolute_work"][~helper_mask, :, 1].eq(0).all():
            raise AssertionError("Disabled helper performed actuator work")
    return result


def _case_metadata(case):
    return {key: value for key, value in case.items() if key not in
        ("historical_handoff_state", "historical_recovery_actions", "source_initial_pose")}


def endpoint_metrics(run, prepared):
    """Common fixed-time and historical-D-time comparisons, never own stopping times."""
    output=[]; contrasts=[]; historical_checks=[]; stage_peaks=[]
    traj=run["trajectory"]; metrics=run["measurements"]; dt=run["dt"]; ss=run["substeps"]
    for case_index,case in enumerate(prepared["cases"]):
        base=case_index*4
        historical=int(case["historical_handoff_steps"])
        d_state=traj[base+3,historical]
        historical_checks.append(dict(**_case_metadata(case), substeps=ss,
            d_historical_handoff_max_state_error=float((d_state-case["historical_handoff_state"]).abs().max()),
            d_historical_handoff_bar_angle_error_rad=float(wrap(d_state[16]-case["historical_handoff_state"][16]).abs()),
            d_historical_handoff_bar_xy_error_units=float((d_state[12:14]-case["historical_handoff_state"][12:14]).norm()),
            d_historical_action_max_error=float((run["actions"][base+3,:historical]-case["historical_recovery_actions"]).abs().max()),
            new_d_first_ready_step=int(run["first_ready_step"][base+3])))
        for endpoint,t in ((f"fixed_{run['horizon']}_steps",run["horizon"]),("historical_D_handoff",historical)):
            branch_values=[]
            for branch,(label,_,_) in enumerate(BRANCHES):
                i=base+branch
                initial,state=traj[i,0],traj[i,t]
                theta=float(state[16]-initial[16])
                peak_steps=metrics["bar_peak_abs_omega"][i,:t]
                peak_step=int(peak_steps.argmax())
                entry=dict(**_case_metadata(case), endpoint=endpoint, endpoint_steps=t, endpoint_time=t*dt,
                    substeps=ss,branch=label,delta_angle_rad=theta,wrapped_delta_angle_rad=float(wrap(state[16]-initial[16])),
                    final_bar_angular_velocity=float(state[17]), delta_angular_velocity=float(state[17]-initial[17]),
                    bar_displacement_xy_units=(state[12:14]-initial[12:14]).tolist(),
                    bar_displacement_units=float((state[12:14]-initial[12:14]).norm()),
                    agent_displacement_units=[float((state[j*6:j*6+2]-initial[j*6:j*6+2]).norm()) for j in range(2)],
                    signed_actuator_work_by_agent=metrics["actuator_signed_work"][i,:t].sum(0).tolist(),
                    absolute_actuator_work_by_agent=metrics["actuator_absolute_work"][i,:t].sum(0).tolist(),
                    positive_actuator_work_by_agent=metrics["actuator_positive_work"][i,:t].sum(0).tolist(),
                    negative_actuator_work_by_agent=metrics["actuator_negative_work"][i,:t].sum(0).tolist(),
                    force_norm_integral_by_agent=metrics["force_norm_integral"][i,:t].sum(0).tolist(),
                    squared_force_integral_by_agent=metrics["squared_force_integral"][i,:t].sum(0).tolist(),
                    path_length_by_agent=metrics["agent_path_length"][i,:t].sum(0).tolist(),
                    contact_normal_impulse_by_agent_part=metrics["contact_force_norm_integral"][i,:t].sum(0).tolist(),
                    contact_peak_force_by_agent_part=metrics["contact_force_peak_norm"][i,:t].amax(0).tolist(),
                    contact_torque_impulse_by_agent_part=metrics["contact_torque_impulse_about_part"][i,:t].sum(0).tolist(),
                    peak_absolute_bar_omega=float(peak_steps[peak_step]),
                    peak_bar_omega_signed=float(metrics["bar_peak_abs_omega_signed"][i,peak_step]),
                    peak_bar_omega_time=float((peak_step+(int(metrics["bar_peak_abs_omega_substep"][i,peak_step])+1)/ss)*dt),
                    first_ready_step=int(run["first_ready_step"][i]), ready_at_endpoint=bool(run["ready"][i,t-1]),
                    finite=bool(run["finite"][i]),wall=bool(run["wall"][i]))
                output.append(entry);branch_values.append(entry)
            a,b,c,d=branch_values
            contrast=dict(**_case_metadata(case),endpoint=endpoint,endpoint_steps=t,substeps=ss)
            for field in ("delta_angle_rad","final_bar_angular_velocity","bar_displacement_units"):
                contrast[field+"_B_minus_A"]=b[field]-a[field]
                contrast[field+"_C_minus_A"]=c[field]-a[field]
                contrast[field+"_D_minus_B_minus_C_plus_A"]=d[field]-b[field]-c[field]+a[field]
            contrast.update(A_angle=a["delta_angle_rad"],B_angle=b["delta_angle_rad"],C_angle=c["delta_angle_rad"],D_angle=d["delta_angle_rad"])
            contrasts.append(contrast)
        # Phase-local peaks under both the actual branch phase and common D
        # phase clock. The latter avoids silently comparing different time spans.
        d_phases=run["controller_phase"][base+3]
        for branch,(label,_,_) in enumerate(BRANCHES):
            i=base+branch
            for clock in ("own_controller_phase","matched_D_phase_clock"):
                phases=run["controller_phase"][i] if clock=="own_controller_phase" else d_phases
                for phase in (0,1,2):
                    selected=(phases==phase).nonzero().flatten()
                    if not len(selected):continue
                    omega=metrics["bar_peak_abs_omega"][i,selected]
                    maximum=int(selected[int(omega.argmax())])
                    force=metrics["contact_force_peak_norm"][i,selected]
                    peak_values,peak_positions=force.max(0)
                    absolute_steps=selected[peak_positions]
                    sub=metrics["contact_force_peak_substep"][i,absolute_steps,
                         torch.arange(2)[:,None].expand(2,2),torch.arange(2)[None,:].expand(2,2)]
                    contact_times=(absolute_steps+(sub.clamp_min(0).float()/ss))*dt
                    contact_times[sub<0]=-1
                    step_angle=traj[i,1:,16]-traj[i,:-1,16]
                    stage_peaks.append(dict(**_case_metadata(case),substeps=ss,branch=label,phase_clock=clock,
                        controller_phase=phase,observed_steps=len(selected),
                        first_step=int(selected.min()),last_step=int(selected.max()),
                        angle_increment_during_selected_steps=float(step_angle[selected].sum()),
                        peak_absolute_bar_omega=float(metrics["bar_peak_abs_omega"][i,maximum]),
                        peak_bar_omega_signed=float(metrics["bar_peak_abs_omega_signed"][i,maximum]),
                        peak_bar_omega_time=float((maximum+(int(metrics["bar_peak_abs_omega_substep"][i,maximum])+1)/ss)*dt),
                        contact_peak_force_by_agent_part=peak_values.tolist(),
                        contact_peak_time_by_agent_part=contact_times.tolist(),
                        signed_actuator_work_by_agent=metrics["actuator_signed_work"][i,selected].sum(0).tolist(),
                        absolute_actuator_work_by_agent=metrics["actuator_absolute_work"][i,selected].sum(0).tolist()))
    return dict(endpoints=output,contrasts=contrasts,historical_D_checks=historical_checks,phase_peaks=stage_peaks)


def summarize_analysis(analysis):
    from collections import defaultdict
    endpoint_groups=defaultdict(list);contrast_groups=defaultdict(list)
    for row in analysis["endpoints"]:
        endpoint_groups[(row["substeps"],row["endpoint"],row["support"],row["first_goal_passed"],row["branch"])].append(row)
    for row in analysis["contrasts"]:
        contrast_groups[(row["substeps"],row["endpoint"],row["support"],row["first_goal_passed"])].append(row)
    endpoints=[];contrasts=[]
    for key,rows in sorted(endpoint_groups.items()):
        n=len(rows);entry=dict(zip(("substeps","endpoint","support","first_goal_passed","branch"),key),cases=n)
        for field in ("delta_angle_rad","final_bar_angular_velocity","peak_absolute_bar_omega","bar_displacement_units"):
            values=torch.tensor([row[field] for row in rows]);entry[field+"_mean"]=float(values.mean());entry[field+"_min"]=float(values.min());entry[field+"_max"]=float(values.max())
        for field in ("signed_actuator_work_by_agent","absolute_actuator_work_by_agent","force_norm_integral_by_agent","path_length_by_agent","contact_normal_impulse_by_agent_part","contact_peak_force_by_agent_part"):
            entry[field+"_mean"]=torch.tensor([row[field] for row in rows]).mean(0).tolist()
        entry["ready_at_endpoint_count"]=sum(row["ready_at_endpoint"] for row in rows)
        endpoints.append(entry)
    for key,rows in sorted(contrast_groups.items()):
        entry=dict(zip(("substeps","endpoint","support","first_goal_passed"),key),cases=len(rows))
        for field in rows[0]:
            if "_minus_" in field or field in ("A_angle","B_angle","C_angle","D_angle"):
                values=torch.tensor([row[field] for row in rows]);entry[field+"_mean"]=float(values.mean());entry[field+"_min"]=float(values.min());entry[field+"_max"]=float(values.max())
        contrasts.append(entry)
    return dict(endpoint_summary=endpoints,contrast_summary=contrasts)


def compare_resolutions(runs,prepared):
    if len(runs)<2:return []
    output=[];base=runs[0]
    for candidate in runs[1:]:
        if not torch.equal(base["trajectory"][:,0],candidate["trajectory"][:,0]):
            raise AssertionError("Resolution check changed full initial physical states")
        state=base["trajectory"]-candidate["trajectory"]
        for case_index,case in enumerate(prepared["cases"]):
            for branch,(name,_,_) in enumerate(BRANCHES):
                i=case_index*4+branch;h=case["historical_handoff_steps"]
                output.append(dict(**_case_metadata(case),branch=name,
                    source_substeps=base["substeps"],target_substeps=candidate["substeps"],
                    identical_complete_initial_state=True,
                    max_bar_xy_error_units=float(state[i,:,12:14].norm(dim=-1).max()),
                    max_bar_angle_error_rad=float(wrap(state[i,:,16]).abs().max()),
                    final_bar_angle_error_rad=float(wrap(state[i,-1,16]).abs()),
                    historical_handoff_angle_error_rad=float(wrap(state[i,h,16]).abs()),
                    max_action_difference=float((base["actions"][i]-candidate["actions"][i]).abs().max())))
    return output


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("data/novelty/stable_composition_selected_matched.pt"))
    parser.add_argument("--output",type=Path,default=Path("data/novelty/recovery_attribution"))
    parser.add_argument("--substeps",type=int,nargs="+",default=[160,320])
    parser.add_argument("--horizon",type=int,default=80)
    args=parser.parse_args();torch.set_num_threads(1);started=time.perf_counter()
    sources=[Path(__file__),Path("vmas_pusht/rebrace.py"),Path("vmas_pusht/snapshot.py"),
        Path("vmas_pusht/scenario.py"),Path("vmas_pusht/physics_profiles.py"),Path("vmas_pusht/abstraction_convergence.py")]
    import vmas.simulator.core
    sources.append(Path(vmas.simulator.core.__file__))
    model_paths=sorted(Path("data/novelty").glob("stable_local_*_models.pt"))
    hashes={str(path.resolve()):file_sha(path) for path in [args.source,*sources,*model_paths]}
    source=torch.load(args.source,map_location="cpu")
    prepared=reconstruct_retreats(source,source_substeps=160)
    print(json.dumps(dict(stage="exact_replay_complete",cases=len(prepared["cases"]),checks=prepared["replay_checks"],elapsed_seconds=time.perf_counter()-started)),flush=True)
    runs=[];analyses=[]
    for substeps in args.substeps:
        run=run_factorial(prepared,substeps=substeps,horizon=args.horizon)
        analysis=endpoint_metrics(run,prepared);runs.append(run);analyses.append(analysis)
        print(json.dumps(dict(stage="factorial_complete",substeps=substeps,trajectories=len(run["rows"]),
            summary=summarize_analysis(analysis),elapsed_seconds=time.perf_counter()-started)),flush=True)
    combined={key:[row for analysis in analyses for row in analysis[key]] for key in analyses[0]}
    refinement=compare_resolutions(runs,prepared)
    for path,sha in hashes.items():
        if file_sha(path)!=sha:raise AssertionError("An immutable source or model changed: "+path)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    raw_path=args.output.with_suffix(".pt")
    torch.save(dict(prepared=prepared,runs=runs,source_sha256=hashes),raw_path)
    torch.load(raw_path,map_location="cpu")
    report=dict(schema_version=1,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        profile=prepared["profile"],replay_checks=prepared["replay_checks"],case_count=len(prepared["cases"]),
        trajectory_count=sum(len(run["rows"]) for run in runs),source_sha256=hashes,raw_sha256=file_sha(raw_path),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        versions={"torch":str(torch.__version__),"vmas":importlib.metadata.version("vmas")},
        elapsed_seconds=time.perf_counter()-started,**combined,**summarize_analysis(combined),refinement=refinement,
        measurement_definition={"contact":"Actual VMAS agent-T-part contact forces sampled at every integration substep; normal impulses and peaks; agent-agent collisions remain active but are not included in per-part force summaries.",
            "work":"Sum of actual actuator force dot physical substep displacement; signed, absolute, positive and negative separately. Scene units, not SI-calibrated energy.",
            "phase":"Original RebraceController staging=0, acquisition=1, frozen-anchor confirmation=2. Own-phase and common-D-phase clocks are both reported.",
            "angle":"Unwrapped bar angle displacement is used for factorial contrasts; wrapped displacement is additionally reported.",
            "off":"Zero final drive command only. Bodies retain mass, velocity, dynamics and collisions; passive contact forces can remain nonzero.",
            "resolution":"Identical full 160-substep retreat-end snapshots and expanded batch/order; only substeps/sub_dt change at initialization."},
        limitations=[
            "B-A and C-A are conditional controller effects against coasting/colliding zero-drive bodies, not effects of removing either agent.",
            "D-B-C+A includes genuine mechanical interaction and feedback/contact-phase changes caused by the intervention. It is not a pure open-loop force superposition term.",
            "The historical-D handoff endpoint uses each source case's original recovery duration. The fixed80 endpoint continues recovery/holding for80steps and must not be described as the historical handoff.",
            "Original passive support semantics are preserved, including zero helper drive after confirmation. Active world_anchor/preloaded_anchor are the primary attribution strata; passive is reported separately.",
            "Every first-goal failure is retained and stratified. Source conditions are deterministic cases, not independent draws supporting population confidence intervals.",
            "The original batch replay is checked bitwise before extraction. Expanded factorial batches can introduce floating-point differences, explicitly measured by D-versus-source comparisons.",
            "This diagnostic does not train, query or alter a world model; all saved model artifacts remain hash-identical.",
        ])
    args.output.with_suffix(".json").write_text(json.dumps(_plain(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output),cases=report["case_count"],trajectories=report["trajectory_count"],elapsed_seconds=report["elapsed_seconds"])),flush=True)


if __name__=="__main__":main()
