"""True-simulator receding-horizon recovery, with no WM fitting or queries.

Each decision evaluates complete remaining-time shooting plans, executes at
most five real steps, then replans from the new full state. Plans contain a
bank13 first push followed by coast, repeated pushes, or quantized feedback.
The expanded library can first execute either finite-force exterior contact
route. A committed route retains its actual progress across MPC decisions.

This is a finite search over declared rollout templates, not an exhaustive
oracle or a certificate that failed states are globally unreachable.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from vmas_pusht.abstraction_convergence import restore_across_substeps, _weld_residual
from vmas_pusht.abstraction_transfer import task_metrics, ANGLE_TOLERANCE, TRANSLATION_TOLERANCE
from vmas_pusht.boundary import _at_wall, _finite, _plain
from vmas_pusht.physical_abstraction import helper_action, physical_state
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.recovery_primitives import (PrimitiveState, start_primitive,
    route_control, advance_primitive, orient_push, configuration)
from vmas_pusht.scenario import wrap
from vmas_pusht.snapshot import capture, restore
from vmas_pusht.stable_composition import (candidates_for_bank, merge_snapshot_records,
    snapshot_from_record)


MACRO = 5
HORIZON = 60
TAILS = ("coast", "repeat", "quantized_feedback")
ROUTES = ("continue", "upper_right", "lower_left")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    return dict(no_retraining=True, physics="dry_light", source_substeps=160,
        remaining_real_steps=HORIZON, replan_every_steps=MACRO,
        lookahead="all remaining real steps, including navigation and reacquisition",
        force_norm_limit=1., helper="unchanged original support law and frozen handoff anchor",
        candidates="bank13 first five contact-ready push steps; then one of three fixed tails",
        tails=list(TAILS), expanded_routes=list(ROUTES),
        candidate_counts=dict(bank13=39, expanded=117),
        feedback_tail="nearest bank13 direction0 strength to clamp(8*signed_angle_error-.8*signed_omega,0,.9)",
        route_commitment="at most one exterior route; retain its progress, acquired face and unfinished five-step force macro across replanning",
        scoring="five consecutive in-tolerance post-step states, terminal pose, finite and no wall; original first-goal gate retained",
        angle_tolerance=ANGLE_TOLERANCE, translation_tolerance=TRANSLATION_TOLERANCE,
        search_order="predicted goal dwell success first, then terminal normalized pose error, then 1e-4*steps",
        computation="candidate count differs by library; report candidate environment steps, not equal-compute claim",
        primitives=configuration(),
        limitations=["Finite template MPC; a negative result is not exhaustive reachability.",
            "Full privileged physical state and exact simulator are used, not the frozen WM.",
            "Source states are held fixed; grouped replay may differ in float roundoff from original execution.",
            "Expanded choices include finite navigation feedback and its failure braking; successful routes and failure braking are reported separately."])


def trajectory_at_wall(trajectory, env):
    """The original geometric wall predicate, evaluated on recorded full states."""
    touched = torch.zeros(trajectory.shape[:-1], dtype=torch.bool)
    limit = trajectory.new_tensor([env.world.x_semidim, env.world.y_semidim])
    for i, entity in enumerate((*env.world.agents, env.scenario.bar, env.scenario.stem)):
        position = trajectory[..., 6*i:6*i+2]
        if i < 2:
            extent = entity.shape.radius
        else:
            angle = trajectory[..., 6*i+4]
            c, s = angle.cos().abs(), angle.sin().abs()
            l, w = entity.shape.length/2, entity.shape.width/2
            extent = torch.stack((c*l+s*w, s*l+c*w), -1)
        touched |= ((position.abs()+extent) >= limit-1e-6).any(-1)
    return touched


def handoff_record(artifact, source_substeps=160):
    records = []
    geometry = make_profile("dry_light", num_envs=1, substeps=source_substeps, seed=101)
    for group_index, group in enumerate(artifact["groups"]):
        if group["rows"][0]["substeps"] != source_substeps:
            continue
        for record in group["counterfactual_snapshots"]:
            if record["rows"][0]["snapshot_stage"] != "post_recovery":
                continue
            rows = []
            for source in record["rows"]:
                i = source["composition_row"]
                original = group["rows"][i]
                endpoint = original["phase_end_steps"]["recovery"]
                commands = torch.cat((group["prelude_actions"][i], group["actions"][i, :endpoint]), 0)
                prefix = torch.cat((group["prelude_trajectory"][i], group["trajectory"][i, :endpoint+1]), 0)
                rows.append(dict(source, source_group=group_index, source_row=i,
                    initial_pose=group["initial_pose"][i].tolist(),
                    first_goal_pass=original["first_goal_success_at_phase_boundary"],
                    original_recovery_pass=original["contact_reacquired"],
                    original_retreat_pass=original["retreated_clear"],
                    original_ordered_success=original["ordered_goal_success"],
                    prefix_steps=endpoint+original["prelude_steps"],
                    prefix_force_integral=(commands.norm(dim=-1).sum(0)*.1).tolist(),
                    original_final_pose=group["terminal_pose"][i].tolist(),
                    original_prefix_finite=bool(torch.isfinite(prefix).all()),
                    original_prefix_wall=bool(trajectory_at_wall(prefix, geometry).any())))
            records.append(dict(record, rows=rows))
    merged = merge_snapshot_records(records)
    keys = [(r["source_group"], r["source_row"]) for r in merged["rows"]]
    if len(keys) != len(set(keys)):
        raise AssertionError("Repeated source handoff")
    return merged


def select_record(record, indices):
    """Select a debugging/review subset without synthesizing physical states."""
    idx = torch.as_tensor(indices, dtype=torch.long)
    snapshot = snapshot_from_record(record)
    def select(t):
        return None if t is None else t[idx].clone()
    saved = asdict(snapshot)
    for ent in saved["entities"]:
        ent["state"] = {k: select(v) for k, v in ent["state"].items()}
        if ent["action"] is not None:
            ent["action"] = {k: select(v) for k, v in ent["action"].items()}
    saved["scenario"] = {k: select(v) for k, v in saved["scenario"].items()}
    saved["steps"] = select(saved["steps"])
    saved["batch_size"] = len(idx)
    return dict(snapshot=saved, rows=[record["rows"][i] for i in idx.tolist()],
        anchor=record["anchor"][idx].clone(), history=record["history"][idx].clone())


def template_table(library):
    if library not in ("bank13", "expanded"):
        raise ValueError("Unknown library")
    return torch.tensor([(route, action, tail) for route in range(1 if library == "bank13" else 3)
                         for action in range(13) for tail in range(3)], dtype=torch.long)


def _assign_state(target, source, mask):
    for item in fields(PrimitiveState):
        value = getattr(target, item.name)
        if torch.is_tensor(value):
            value[mask] = getattr(source, item.name)[mask]
    return target


def begin_plan(env, parent_state, templates, carry=None):
    state = parent_state.clone()
    switch = templates[:, 0] > 0
    # New navigation starts only at the original face. A second geometric
    # route would need a separate physically validated transition contract.
    valid = ~switch | (parent_state.mode == 3)
    if switch.any():
        names = [ROUTES[int(x)] if int(x) else "stay" for x in templates[:, 0]]
        fresh = start_primitive(env, names)
        valid &= ~switch | fresh.admissible_start
        _assign_state(state, fresh, switch)
    n = env.num_envs
    return dict(state=state, templates=templates.clone(), valid=valid,
        pushed_steps=torch.zeros(n, dtype=torch.long),
        locked_steps=torch.zeros(n, dtype=torch.long) if carry is None else carry["locked_steps"].clone(),
        held=torch.zeros(n, 2) if carry is None else carry["held"].clone())


def unfinished_macro(plan):
    """A route may acquire contact between global MPC query times."""
    remaining = (-plan["pushed_steps"]) % MACRO
    remaining = torch.where(plan["locked_steps"] > 0, plan["locked_steps"], remaining)
    return dict(locked_steps=remaining.clone(), held=plan["held"].clone())


def feedback_index(env, state, initial, goals):
    sign = torch.where(state.mode < 2, -1., 1.)
    error = sign * wrap(initial[:, 2] + goals - env.scenario.bar.state.rot[:, 0])
    omega = sign * env.scenario.bar.state.ang_vel[:, 0]
    desired = (8.*error-.8*omega).clamp(0., .9)
    levels = desired.new_tensor([0., .15, .3, .6, .9])
    indices = (desired[:, None]-levels).abs().argmin(1)
    # bank13: zero, direction-20 four strengths, direction0 four, +20 four.
    return torch.where(indices == 0, 0, indices+4)


def plan_command(env, plan, initial, goals, anchor, actuated, active):
    navigation, ready = route_control(env, plan["state"])
    renew = ready & (plan["locked_steps"] == 0) & (plan["pushed_steps"] % MACRO == 0)
    indices = plan["templates"][:, 1].clone()
    later = plan["pushed_steps"] >= MACRO
    indices[later & (plan["templates"][:, 2] == 0)] = 0
    use_feedback = later & (plan["templates"][:, 2] == 2)
    suggested = feedback_index(env, plan["state"], initial, goals)
    indices[use_feedback] = suggested[use_feedback]
    bank = candidates_for_bank(env.scenario.bar.state.rot[:, 0], "bank13")
    push = orient_push(env, plan["state"], bank[torch.arange(env.num_envs), indices])
    plan["held"][renew] = push[renew]
    primary = torch.where(ready[:, None], plan["held"], navigation)
    helper = helper_action(env, "world_anchor", anchor) * actuated[:, None]
    command = torch.stack((primary, helper), 1) * active[:, None, None]
    return command, ready


def after_step(env, plan, ready, active):
    locked = plan["locked_steps"] > 0
    plan["locked_steps"] -= (ready & active & locked).long()
    plan["pushed_steps"] += (ready & active & ~locked).long()
    advance_primitive(env, plan["state"], active_mask=active)


class ShootingMPC:
    def __init__(self, n, library, substeps):
        self.table = template_table(library)
        self.n, self.k = n, len(self.table)
        self.indices = torch.arange(n).repeat_interleave(self.k)
        self.templates = self.table.repeat(n, 1)
        self.env = make_profile("dry_light", num_envs=n*self.k, substeps=substeps, seed=101)

    @torch.no_grad()
    def choose(self, env, state, initial, goals, anchor, actuated, remaining, actual_dwell, carry=None):
        n, k, idx, sim = self.n, self.k, self.indices, self.env
        restore(sim, capture(env), idx)
        branched_carry = None if carry is None else {key: value[idx].clone() for key, value in carry.items()}
        plan = begin_plan(sim, state.clone(idx), self.templates, branched_carry)
        reference, target = initial[idx], goals[idx]
        anchors, mask = anchor[idx], actuated[idx]
        active = torch.ones(n*k, dtype=torch.bool)
        dwell = actual_dwell[idx].clone()
        success = torch.zeros(n*k, dtype=torch.bool)
        finite, wall = _finite(sim), _at_wall(sim)
        terminal = sim.scenario.t_pose().clone()
        stop = torch.full((n*k,), remaining, dtype=torch.long)
        first_macro_state = None
        for t in range(remaining):
            command, ready = plan_command(sim, plan, reference, target, anchors, mask, active)
            sim.step([command[:, 0], command[:, 1]])
            after_step(sim, plan, ready, active)
            finite &= ~active | _finite(sim)
            wall |= active & _at_wall(sim)
            ok, _, _ = task_metrics(sim.scenario.t_pose(), reference, target)
            dwell = torch.where(active & ok, dwell+1, torch.zeros_like(dwell))
            newly = active & (dwell >= MACRO) & finite & ~wall
            terminal[active] = sim.scenario.t_pose()[active]
            stop[newly] = t+1
            success |= newly
            active &= ~newly
            if t == min(MACRO, remaining)-1:
                first_macro_state = physical_state(sim).clone()
        _, error, shift = task_metrics(terminal, reference, target)
        cost = (error/ANGLE_TOLERANCE).square() + (shift/TRANSLATION_TOLERANCE).square()
        score = (~success).float()*1e6 + cost + 1e-4*stop
        score[~plan["valid"] | ~finite | wall] = torch.inf
        score = score.reshape(n, k)
        chosen = score.argmin(1)
        flat = torch.arange(n)*k+chosen
        return self.table[chosen].clone(), dict(chosen=chosen, scores=score,
            predicted_success=success.reshape(n, k), predicted_terminal_cost=cost.reshape(n, k),
            selected_first_macro_state=first_macro_state[flat],
            candidate_environment_steps=n*k*remaining, horizon=remaining)


@torch.no_grad()
def run_mpc(record, library, *, substeps=160, horizon=HORIZON, observe=True):
    if horizon < MACRO or horizon % MACRO:
        raise ValueError("Execution budget must be a positive multiple of five")
    from vmas_pusht.recovery_attribution import PhysicsObserver
    from contextlib import nullcontext
    n = len(record["rows"])
    env = make_profile("dry_light", num_envs=n, substeps=substeps, seed=101)
    restore_across_substeps(env, snapshot_from_record(record))
    initial = torch.tensor([r["initial_pose"] for r in record["rows"]])
    goals = torch.tensor([r["relative_goal_rad"] for r in record["rows"]])
    anchor = record["anchor"].clone()
    actuated = torch.tensor([r["support"] != "passive" for r in record["rows"]])
    state = start_primitive(env, "stay")
    optimizer = ShootingMPC(n, library, substeps)
    active = torch.ones(n, dtype=torch.bool)
    finite, wall = _finite(env), _at_wall(env)
    dwell = torch.zeros(n, dtype=torch.long)
    success = torch.zeros(n, dtype=torch.bool)
    actual_steps = torch.zeros(n, dtype=torch.long)
    terminal = env.scenario.t_pose().clone()
    actions, states, active_trace, modes, phases, observations, queries = [], [physical_state(env).clone()], [], [], [], [], []
    navigation_steps = torch.zeros(n, dtype=torch.long)
    started = time.perf_counter()
    with PhysicsObserver(env) if observe else nullcontext() as observer:
        for t in range(horizon):
            if not active.any():
                break
            if t % MACRO == 0:
                carry = None if t == 0 else unfinished_macro(plan)
                templates, query = optimizer.choose(env, state, initial, goals, anchor,
                    actuated, horizon-t, dwell, carry)
                plan = begin_plan(env, state, templates, carry)
                # Finished rows still accompany the same simulator batch but
                # must not acquire a route that was never physically executed.
                _assign_state(plan["state"], state, ~active)
                queries.append(dict(step=t, templates=templates, active=active.clone(), **query))
                print(json.dumps(dict(library=library, substeps=substeps, step=t, rows=n,
                    active=int(active.sum()), successes=int(success.sum()), elapsed_seconds=time.perf_counter()-started)), flush=True)
            command, ready = plan_command(env, plan, initial, goals, anchor, actuated, active)
            if observer is not None:
                observer.begin_step()
            env.step([command[:, 0], command[:, 1]])
            if observer is not None:
                observations.append(observer.end_step())
            after_step(env, plan, ready, active)
            state = plan["state"]
            navigation_steps += (active & ~ready & (state.mode < 2)).long()
            finite &= ~active | _finite(env)
            wall |= active & _at_wall(env)
            ok, _, _ = task_metrics(env.scenario.t_pose(), initial, goals)
            dwell = torch.where(active & ok, dwell+1, torch.zeros_like(dwell))
            newly = active & (dwell >= MACRO) & finite & ~wall
            terminal[active] = env.scenario.t_pose()[active]
            actual_steps += active
            actions.append(command.clone()); states.append(physical_state(env).clone())
            active_trace.append(active.clone()); modes.append(state.mode.clone()); phases.append(state.phase.clone())
            success |= newly
            active &= ~newly
    trajectory = torch.stack(states, 1)
    action_trace = torch.stack(actions, 1)
    execution_mask = torch.stack(active_trace, 1)
    measurements = {key: torch.stack([o[key] for o in observations], 1) for key in observations[0]} if observations else {}
    if action_trace.norm(dim=-1).max() > 1.000001:
        raise AssertionError("Exceeded actuator force limit")
    terminal_ok, angle_error, shift = task_metrics(terminal, initial, goals)
    first_pass = torch.tensor([r["first_goal_pass"] for r in record["rows"]])
    prefix_ok = torch.tensor([r["original_prefix_finite"] and not r["original_prefix_wall"] for r in record["rows"]])
    continuation_success = success & terminal_ok & finite & ~wall
    full_success = first_pass & prefix_ok & continuation_success
    rows = []
    for i, row in enumerate(record["rows"]):
        item = dict(row, library=library, execution_substeps=substeps,
            continuation_goal_success=bool(continuation_success[i]), ordered_goal_success=bool(full_success[i]),
            final_goal_dwell=bool(success[i]), terminal_goal_success=bool(terminal_ok[i]),
            continuation_steps=int(actual_steps[i]), total_steps=int(actual_steps[i])+row["prefix_steps"],
            continuation_budget=horizon, maximum_original_task_steps=230,
            final_angle_error_rad=float(angle_error[i]), final_translation_units=float(shift[i]),
            navigation_steps=int(navigation_steps[i]), final_route_mode=int(state.mode[i]),
            final_route_phase=int(state.phase[i]), route_failed=bool(state.failed[i]),
            route_first_push_step=int(state.first_push_step[i]),
            finite=bool(finite[i]), wall=bool(wall[i]),
            maximum_action_norm_by_agent=action_trace[i].norm(dim=-1).amax(0).tolist(),
            candidate_environment_steps=sum(q["candidate_environment_steps"]//n for q in queries),
            actual_mpc_queries=int(sum(bool(q["active"][i]) for q in queries)))
        for key in ("actuator_signed_work", "actuator_absolute_work", "actuator_positive_work",
                    "actuator_negative_work", "force_norm_integral", "squared_force_integral", "agent_path_length"):
            if key in measurements:
                item[key+"_by_agent"] = (measurements[key][i]*execution_mask[i, :, None]).sum(0).tolist()
        rows.append(item)
    return dict(rows=rows, start=record, library=library, substeps=substeps,
        trajectory=trajectory, actions=action_trace, active=execution_mask,
        route_mode=torch.stack(modes, 1), route_phase=torch.stack(phases, 1),
        measurements=measurements, queries=queries, terminal_pose=terminal,
        elapsed_seconds=time.perf_counter()-started)


def summary(rows):
    output = []
    for library in sorted({r["library"] for r in rows}):
        for support in ("all", "passive", "world_anchor", "preloaded_anchor"):
            for eligible in (False, True):
                picked = [r for r in rows if r["library"] == library and (support == "all" or r["support"] == support)
                          and (not eligible or r["first_goal_pass"])]
                if not picked:
                    continue
                output.append(dict(library=library, support=support, first_goal_pass_only=eligible,
                    rows=len(picked), continuation_success=sum(r["continuation_goal_success"] for r in picked),
                    ordered_success=sum(r["ordered_goal_success"] for r in picked),
                    route_attempted=sum(r["final_route_mode"] < 2 for r in picked),
                    route_acquired=sum(r["final_route_mode"] < 2 and r["final_route_phase"] == 3 for r in picked),
                    route_failed=sum(r["route_failed"] for r in picked),
                    mean_final_angle_error_rad=sum(r["final_angle_error_rad"] for r in picked)/len(picked),
                    finite_all=all(r["finite"] for r in picked), wall_count=sum(r["wall"] for r in picked)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/novelty/stable_composition_selected_matched.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/novelty/recovery_mpc"))
    parser.add_argument("--libraries", nargs="+", choices=("bank13", "expanded"), default=["bank13", "expanded"])
    parser.add_argument("--substeps", type=int, default=160)
    parser.add_argument("--indices", nargs="+", type=int, help="Explicit diagnostic subset; always recorded")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    registration = args.output.with_name(args.output.name+"_registration.json")
    declaration = dict(protocol=protocol(), arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    if not registration.exists():
        registration.write_text(json.dumps(declaration, indent=2)+"\n")
    elif json.dumps(json.loads(registration.read_text())["protocol"], sort_keys=True) != json.dumps(declaration["protocol"], sort_keys=True):
        raise ValueError("Protocol changed: preserve the earlier registration and choose a new output prefix")
    if args.register_only:
        print(str(registration)); return
    source_paths = [Path(__file__), Path("vmas_pusht/recovery_primitives.py"), Path("vmas_pusht/recovery_attribution.py"),
        Path("vmas_pusht/physics_profiles.py"), Path("vmas_pusht/snapshot.py"),
        Path("vmas_pusht/stable_composition.py"), Path("vmas_pusht/abstraction_convergence.py"),
        Path("vmas_pusht/physical_abstraction.py"), Path("vmas_pusht/abstraction_transfer.py"),
        Path("vmas_pusht/scenario.py"), Path("vmas_pusht/boundary.py")]
    model_paths = [Path("data/novelty/stable_local_original_models.pt"), Path("data/novelty/stable_local_selected_models.pt")]
    source_hashes = {str(p): sha(p) for p in source_paths}
    model_hashes = {str(p): sha(p) for p in model_paths}
    source_digest = sha(args.source)
    artifact = torch.load(args.source, map_location="cpu")
    record = handoff_record(artifact)
    if args.indices:
        record = select_record(record, args.indices)
    results = []
    for library in args.libraries:
        result = run_mpc(record, library, substeps=args.substeps)
        if sha(args.source) != source_digest:
            raise AssertionError("Source trajectory artifact changed during experiment")
        results.append(result)
        raw = args.output.with_name(args.output.name+"_"+library+".pt")
        torch.save(result, raw)
        report = dict(protocol=protocol(), arguments=declaration["arguments"],
            git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            source_path=str(args.source), source_sha256=source_digest, source_hashes=source_hashes,
            registration_path=str(registration), registration_sha256=sha(registration),
            frozen_models_sha256=model_hashes, raw_path=str(raw), raw_sha256=sha(raw),
            rows=result["rows"], summary=summary(result["rows"]), elapsed_seconds=result["elapsed_seconds"])
        raw.with_suffix(".json").write_text(json.dumps(_plain(report), indent=2, allow_nan=False)+"\n")
    if any(sha(p) != h for p, h in model_hashes.items()):
        raise AssertionError("Frozen models changed")
    if any(sha(p) != h for p, h in source_hashes.items()):
        raise AssertionError("Source code changed during experiment")
    print(json.dumps(dict(output=str(args.output), summary=summary([r for result in results for r in result["rows"]]))), flush=True)


if __name__ == "__main__":
    main()
