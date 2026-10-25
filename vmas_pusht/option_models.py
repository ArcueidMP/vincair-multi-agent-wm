"""Goal-free local outcome models for finite-force contact options.

Training uses independently generated local motion histories and atomic options,
never composed-task trajectories, task goals, or MPC-generated task labels.
Models predict endpoint bar displacement, real duration, and a completion score.
The latter is clipped regression, not a calibrated success probability.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from vmas_pusht.abstraction_models import (ResponseMLP, body_frame, matched_width,
    parameter_count, world_frame_response)
from vmas_pusht.abstraction_transfer import reposition_action
from vmas_pusht.action_interface import ActionInterface, model_inputs
from vmas_pusht.boundary import _at_wall, _finite, _plain, _rotate
from vmas_pusht.physical_abstraction import (PRIMARY_LOCAL, coarse_state, helper_action,
    norm_bound, physical_state, setup_scene)
from vmas_pusht.physics_profiles import get_profile, make_profile
from vmas_pusht.scenario import wrap
from vmas_pusht.snapshot import capture, restore
from vmas_pusht.stable_composition import (history_inputs, merge_snapshot_records,
    snapshot_from_record, snapshot_record)
from vmas_pusht.stable_local_wm import (full_body_features, history_body_features,
    history_record, push_history)


SUPPORTS = ("passive", "world_anchor", "preloaded_anchor")
MODELS = ("coarse_body_small", "full_matched", "history_matched", "full_large")
CONTEXT_COUNTS = dict(train=48, val=12, test=24)
MODE_NAMES = {3: "original", 0: "upper", 1: "lower"}
PREHISTORY_STEPS = 20
OUTPUT_NAMES = ("dx", "dy", "dtheta", "elapsed_seconds", "completion_score")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registered_protocol():
    return dict(schema_version=1, profile="dry_light", reference_substeps=160,
        context_counts=CONTEXT_COUNTS, context_seed=48271,
        prehistory_steps=PREHISTORY_STEPS,
        prehistory_kinds=["zero", "push_then_stop", "push_then_hold"],
        continuation_queries="One extra query per acquired face/context:5-step nonzero bank push, followed by immediate query,5zero-primary steps,or5bounded face-hold steps, cycling bycontextindex independentlyofgoals.",
        train=dict(directions=[-20., 0., 20.], strengths=[.15, .3, .6, .9], durations=[5, 10, 20], zero_controls=True),
        val=dict(directions=[-10., 10.], strengths=[.45, .75], durations=[5, 10, 20], zero_controls=False),
        test=dict(directions=[-35., 35.], strengths=[.45, .75], durations=[15, 25], zero_controls=False),
        initial_orientation_train_val_rad=[-.2, .2], heldout_orientation_deg=30.,
        primary_lateral_error=[-.02, .02], primary_outward_error=[.001, .008],
        initial_velocity=[-.03, .03], initial_angular_velocity=[-.2, .2],
        anchor_insets=dict(passive=[0.], world_anchor=[0., .005, .01], preloaded_anchor=[.005, .01, .01]),
        current_faces=[3, 0, 1], face_preparation="Real bounded zero-push options after actual prehistory; failed face preparations remain recorded and are not relabeled as acquired.",
        outputs=list(OUTPUT_NAMES), completion_score="Clipped joint regression output; not a calibrated probability.",
        data_scope="Goal-free atomic options only. No full task, original handoff, recovery-MPC trajectory, or task success label is loaded.",
        failure_policy="Attempted option failures are retained. Failed preparation coverage is reported separately; no query is asserted to start on an unacquired face.",
        features=dict(coarse=24, full=38, history=58, extra_history_storage=34,
                      last_six="Current face onehot(left,upper,lower), requested route onehot(continue,upper,lower)"),
        training=dict(updates=1000, seeds=[0, 1, 2], batch_size=256, lr=.003,
            architectures=list(MODELS), parameter_budget=parameter_count(24, 32, 5)),
        normalization="Training rows only; pooled across supports within an architecture. Validation only calibrates elapsed-time residuals.",
        collection_budget="Same rows and optimizer updates for all architectures of a support condition; acquired-face availability can differ across supports and is reported.",
        frozen_before_task_evaluation=True)


def local_contexts(counts=None, seed=48271):
    counts = CONTEXT_COUNTS if counts is None else dict(counts)
    generator = torch.Generator().manual_seed(seed)
    result = []
    for split in ("train", "val", "test"):
        for i in range(int(counts.get(split, 0))):
            values = torch.rand(7, generator=generator).tolist()
            label = "test_context" if split == "test" and i % 2 else "test_action" if split == "test" else split
            angle = math.pi/6 if label == "test_context" else -.2+.4*values[0]
            result.append(dict(context_id=f"{split}_{i:03d}", split=label, program_split=split,
                angle_rad=angle, lateral_error=-.02+.04*values[1], outward_error=.001+.007*values[2],
                velocity_omega=[-.03+.06*values[3], -.03+.06*values[4], -.2+.4*values[5]],
                helper_outward_error=.003*values[6], history_kind=("zero", "push_then_stop", "push_then_hold")[i % 3],
                anchor_index=i % 3))
    return result


def local_programs(split):
    key = "test" if split.startswith("test") else split
    spec = registered_protocol()[key]
    programs = []
    for duration in spec["durations"]:
        if spec["zero_controls"]:
            programs.append(dict(direction_deg=0., strength=0., push_steps=duration))
        for direction, strength in itertools.product(spec["directions"], spec["strengths"]):
            programs.append(dict(direction_deg=direction, strength=strength, push_steps=duration))
    for i, program in enumerate(programs):
        program["program_id"] = f"{key}_{i:03d}"
    return programs


def _support_command(env, anchor, support):
    return torch.zeros(env.num_envs, 2) if support == "passive" else helper_action(env, "world_anchor", anchor)


@torch.no_grad()
def prepare_local_faces(support, contexts, *, substeps=160):
    """Generate complete current-face snapshots and real histories locally."""
    n = len(contexts)
    if not n or support not in SUPPORTS:
        raise ValueError("Nonempty contexts and a declared support are required")
    env = make_profile("dry_light", num_envs=n, substeps=substeps, seed=48271)
    angle = torch.tensor([c["angle_rad"] for c in contexts])
    primary = torch.tensor([[PRIMARY_LOCAL[0]+c["lateral_error"], PRIMARY_LOCAL[1]+c["outward_error"]] for c in contexts])
    outward = torch.tensor([c["helper_outward_error"] for c in contexts])
    anchor = setup_scene(env, angle=angle, hidden_v=[c["velocity_omega"][:2] for c in contexts],
        hidden_omega=[c["velocity_omega"][2] for c in contexts], primary_local=primary,
        helper_error=torch.stack((outward, -outward), 1))
    insets = registered_protocol()["anchor_insets"][support]
    inset = torch.tensor([insets[c["anchor_index"] % len(insets)] for c in contexts])
    anchor += _rotate(torch.stack((-inset, inset), 1), angle)
    geometry, actions = [coarse_state(env).clone()], []
    max_norm = torch.zeros(n)
    finite, wall = _finite(env), _at_wall(env)
    kinds = [c["history_kind"] for c in contexts]
    initial_force = torch.stack((angle.sin(), -angle.cos()), 1)*.6
    for step in range(PREHISTORY_STEPS):
        primary_force = torch.zeros(n, 2)
        pushing = torch.tensor([k == "push_then_stop" and step < 15 or k == "push_then_hold" and step < 5 for k in kinds])
        primary_force[pushing] = initial_force[pushing]
        holding = torch.tensor([k == "push_then_hold" and step >= 5 for k in kinds])
        if holding.any():
            primary_force[holding] = reposition_action(env, 0, PRIMARY_LOCAL)[holding]
        command = torch.stack((primary_force, _support_command(env, anchor, support)), 1)
        env.step([command[:, 0], command[:, 1]])
        geometry.append(coarse_state(env).clone()); actions.append(command.clone())
        max_norm = torch.maximum(max_norm, command.norm(dim=-1).amax(1))
        finite &= _finite(env); wall |= _at_wall(env)
    if not finite.all() or max_norm.max() > 1.000001:
        raise AssertionError("Local prehistory must remain finite and bounded")
    base_snapshot = capture(env)
    idx = torch.arange(n).repeat_interleave(3)
    requested_modes = torch.tensor([3, 0, 1]).repeat(n)
    prep = make_profile("dry_light", num_envs=3*n, substeps=substeps, seed=48271)
    restore(prep, base_snapshot, idx)
    prep_anchor = anchor[idx].clone()
    trace_geometry = [g[idx].clone() for g in geometry]
    trace_actions = [a[idx].clone() for a in actions]
    rows = [dict(contexts[i], support=support, requested_mode=int(requested_modes[j]),
                 preparation_index=j, anchor_inset=float(inset[i]),
                 prehistory_finite=bool(finite[i]), prehistory_wall=bool(wall[i]),
                 prehistory_max_action_norm=float(max_norm[i])) for j, i in enumerate(idx.tolist())]
    interface = ActionInterface(prep, support, prep_anchor)
    options = torch.zeros(3*n, 4)
    options[:, 0] = torch.where(requested_modes == 0, 1, torch.where(requested_modes == 1, 2, 0))
    options[:, 3] = 5
    need_route = requested_modes != 3
    interface.start(options, mask=need_route)
    records = []
    statuses = []

    def record_mask(mask, preparation_steps, completed):
        indices = mask.nonzero().flatten().tolist()
        if not indices:
            return
        for i in indices:
            row = rows[i]
            row["preparation_completed"] = bool(completed[i])
            row["preparation_steps"] = int(preparation_steps[i])
            row["environment_start_step"] = float(prep.steps[i])
            row["current_mode"] = int(requested_modes[i]) if completed[i] else None
            row["entry_origin"] = "post_prelude" if requested_modes[i] == 3 else "post_route"
            row["entry_phase"] = int(interface.state.phase[i])
            statuses.append(dict(row))
        records.append(snapshot_record(prep, indices, prep_anchor,
            history_inputs(trace_geometry, trace_actions), rows, "local_option_query"))

    record_mask(~need_route, torch.zeros(3*n, dtype=torch.long), ~need_route)
    for _ in range(interface.state.config.acquisition_limit+5):
        active = interface.busy.clone()
        if not active.any():
            break
        command = interface.command(active)
        prep.step([command[:, 0], command[:, 1]])
        status = interface.observe(active)
        trace_geometry.append(coarse_state(prep).clone()); trace_actions.append(command.clone())
        record_mask(status["newly_done"], interface.elapsed, interface.completed)
    if interface.busy.any() or len(statuses) != 3*n:
        raise AssertionError("Every local face preparation must terminate and be recorded")
    merged = merge_snapshot_records(records)
    if not merged["history"][:, :, 16].eq(1).all():
        raise AssertionError("Every local query requires genuine past observations")
    prepared = dict(record=merged, statuses=statuses,
        original_contexts=contexts, support=support,
        generated_from_task_trajectories=False,
        acquisition_success_count=sum(r["preparation_completed"] for r in statuses),
        preparation_case_count=len(statuses))
    return _add_continuation_queries(prepared, support, substeps)


def _face_holding_command(env, modes):
    local = torch.tensor([PRIMARY_LOCAL if int(m) == 3 else (.13, .101) if int(m) == 0
                          else (-.13, -.101) for m in modes])
    bar, main = env.scenario.bar, env.world.agents[0]
    offset = _rotate(local, bar.state.rot[:, 0])
    target_velocity = bar.state.vel+bar.state.ang_vel*torch.stack((-offset[:, 1], offset[:, 0]), 1)
    return norm_bound(30.*(bar.state.pos+offset-main.state.pos)+4.*(target_velocity-main.state.vel))


@torch.no_grad()
def _add_continuation_queries(prepared, support, substeps):
    """Add real post-push/stop/hold states without reading any task artifact."""
    record = prepared["record"]
    idx = torch.tensor([i for i, r in enumerate(record["rows"]) if r["preparation_completed"]])
    if not len(idx):
        prepared["continuation_queries"] = []
        return prepared
    n = len(idx)
    env = make_profile("dry_light", num_envs=n, substeps=substeps, seed=48271)
    restore(env, snapshot_from_record(record), idx)
    anchor, past = record["anchor"][idx].clone(), record["history"][idx].clone()
    rows = [dict(record["rows"][int(i)], parent_prepared_record_index=int(i)) for i in idx]
    context_numbers = [int(r["context_id"].split("_")[-1]) for r in rows]
    origins = [("after_push", "after_stop", "after_hold")[(i//3) % 3] for i in context_numbers]
    modes = torch.tensor([r["current_mode"] for r in rows])
    options = torch.tensor([[0., (-20., 0., 20.)[(i//4) % 3], (.15, .3, .6, .9)[i % 4], 5.]
                            for i in context_numbers])
    interface = ActionInterface(env, support, anchor)
    interface.reset_contact(torch.ones(n, dtype=torch.bool), mode=modes)
    interface.start(options)
    start_geometry = coarse_state(env).clone()
    actions = []
    for _ in range(5):
        command = interface.command()
        env.step([command[:, 0], command[:, 1]])
        interface.observe()
        actions.append(command.clone())
    if not interface.completed.all():
        raise AssertionError("A prepared face must execute its declared continuation push")
    past = push_history(past, history_record(start_geometry, torch.stack(actions, 1)))
    records, continuation_rows = [record], []

    def retain(mask, extra_steps):
        indices = mask.nonzero().flatten().tolist()
        if not indices:
            return
        for i in indices:
            rows[i].update(entry_origin=origins[i], continuation_push=options[i].tolist(),
                continuation_extra_steps=extra_steps, environment_start_step=float(env.steps[i]),
                preparation_steps=rows[i]["preparation_steps"]+extra_steps,
                continuation_finite=bool(_finite(env)[i]), continuation_wall=bool(_at_wall(env)[i]))
            continuation_rows.append(dict(rows[i]))
        records.append(snapshot_record(env, indices, anchor, past, rows, "local_option_continuation"))

    immediate = torch.tensor([o == "after_push" for o in origins])
    retain(immediate, 5)
    active = ~immediate
    if active.any():
        start_geometry = coarse_state(env).clone(); commands = []
        holding = torch.tensor([o == "after_hold" for o in origins])
        for _ in range(5):
            primary = torch.zeros(n, 2)
            if holding.any():
                primary[holding] = _face_holding_command(env, modes)[holding]
            command = torch.stack((primary, _support_command(env, anchor, support)), 1)*active[:, None, None]
            env.step([command[:, 0], command[:, 1]])
            commands.append(command.clone())
        past = push_history(past, history_record(start_geometry, torch.stack(commands, 1)))
        retain(active, 10)
    if len(continuation_rows) != n or not all(r["continuation_finite"] for r in continuation_rows):
        raise AssertionError("Every acquired face must retain its finite local continuation query")
    prepared["record"] = merge_snapshot_records(records)
    prepared["continuation_queries"] = continuation_rows
    return prepared


@torch.no_grad()
def collect_support(support, *, substeps=160, contexts=None, chunk_size=768):
    contexts = local_contexts() if contexts is None else contexts
    prepared = prepare_local_faces(support, contexts, substeps=substeps)
    record = prepared["record"]
    sample_rows, source_indices, options = [], [], []
    for source_index, entry in enumerate(record["rows"]):
        if not entry["preparation_completed"]:
            continue
        current_mode = entry["current_mode"]
        for route in ((0, 1, 2) if current_mode == 3 else (0,)):
            for program in local_programs(entry["program_split"]):
                row = dict(entry, **program, route_id=route,
                    query_id=f"{support}/{entry['context_id']}/{current_mode}/{entry['entry_origin']}/{route}/{program['program_id']}",
                    family_id=f"{support}/{entry['context_id']}/{current_mode}/{entry['entry_origin']}/{route}/{program['direction_deg']}/{program['strength']}",
                    prepared_source_index=source_index)
                sample_rows.append(row); source_indices.append(source_index)
                options.append([route, program["direction_deg"], program["strength"], program["push_steps"]])
    if not sample_rows:
        raise ValueError("No acquired local contexts remain")
    all_tensors = defaultdict(list)
    started = time.perf_counter()
    for begin in range(0, len(sample_rows), chunk_size):
        stop = min(begin+chunk_size, len(sample_rows))
        rows = sample_rows[begin:stop]
        index = torch.tensor(source_indices[begin:stop])
        command_options = torch.tensor(options[begin:stop], dtype=torch.float32)
        n = len(rows)
        env = make_profile("dry_light", num_envs=n, substeps=substeps, seed=48271)
        restore(env, snapshot_from_record(record), index)
        anchor = record["anchor"][index].clone()
        modes = torch.tensor([r["current_mode"] for r in rows])
        interface = ActionInterface(env, support, anchor)
        interface.reset_contact(torch.ones(n, dtype=torch.bool), mode=modes)
        coarse, full = model_inputs(env, anchor, modes, command_options, reference_substeps=substeps)
        past = record["history"][index].clone()
        start_pose = env.scenario.t_pose().clone()
        start_physical = physical_state(env).clone()
        start_clock = env.steps.clone()
        interface.start(command_options)
        end_pose = start_pose.clone(); end_physical = start_physical.clone()
        end_clock = start_clock.clone()
        finite, wall = _finite(env), _at_wall(env)
        max_norm = torch.zeros(n)
        force_integral = torch.zeros(n, 2)
        for _ in range(interface.state.config.acquisition_limit+int(command_options[:, 3].max())):
            active = interface.busy.clone()
            if not active.any():
                break
            command = interface.command(active)
            env.step([command[:, 0], command[:, 1]])
            status = interface.observe(active)
            finite &= ~active | _finite(env)
            wall |= active & _at_wall(env)
            norms = command.norm(dim=-1)
            max_norm = torch.maximum(max_norm, norms.amax(1))
            force_integral += norms*env.world.dt
            done = status["newly_done"]
            end_pose[done] = env.scenario.t_pose()[done]
            end_physical[done] = physical_state(env)[done]
            end_clock[done] = env.steps[done]
        if interface.busy.any() or not finite.all() or max_norm.max() > 1.000001:
            raise AssertionError("Atomic options must terminate with finite bounded execution")
        if not torch.equal(end_clock-start_clock, interface.elapsed.to(end_clock)):
            raise AssertionError("Option duration must equal actual environment clock increments")
        response = end_pose-start_pose
        response[:, 2] = wrap(response[:, 2])
        targets = torch.cat((response, (interface.elapsed*env.world.dt)[:, None], interface.completed.float()[:, None]), 1)
        for i, row in enumerate(rows):
            row.update(option_completed=bool(interface.completed[i]), option_failed=bool(interface.failed[i]),
                actual_steps=int(interface.elapsed[i]), actual_elapsed_seconds=float(targets[i, 3]),
                pushed_steps=int(interface.pushed_steps[i]), environment_end_step=float(end_clock[i]),
                finite=bool(finite[i]), wall=bool(wall[i]), max_action_norm=float(max_norm[i]),
                force_integral_by_agent=force_integral[i].tolist(), final_mode=int(interface.mode[i]))
        for key, value in dict(X_coarse=coarse, X_full=full, X_history=past, Y=targets,
                               physical_start=start_physical, physical_end=end_physical,
                               options=command_options).items():
            all_tensors[key].append(value.clone())
        if begin == 0 or stop == len(sample_rows) or stop % (4*chunk_size) == 0:
            print(json.dumps(dict(support=support, collected=stop, total=len(sample_rows),
                                  elapsed_seconds=time.perf_counter()-started)), flush=True)
    result = {key: torch.cat(values, 0) for key, values in all_tensors.items()}
    result.update(rows=sample_rows, preparation=prepared, profile=get_profile("dry_light"),
                  reference_substeps=substeps, protocol=registered_protocol())
    validate_dataset(result)
    return result


def collect_local(*, substeps=160, supports=SUPPORTS, contexts=None, chunk_size=768):
    parts = [collect_support(s, substeps=substeps, contexts=contexts, chunk_size=chunk_size) for s in supports]
    return combine_datasets(parts)


def combine_datasets(parts):
    if not parts or any(p["profile"] != parts[0]["profile"] or p["reference_substeps"] != parts[0]["reference_substeps"] for p in parts):
        raise ValueError("Datasets must have identical physical configuration")
    result = {key: torch.cat([p[key] for p in parts], 0) for key in
        ("X_coarse", "X_full", "X_history", "Y", "physical_start", "physical_end", "options")}
    result.update(rows=[r for p in parts for r in p["rows"]],
        preparations=[p["preparation"] for p in parts], profile=parts[0]["profile"],
        reference_substeps=parts[0]["reference_substeps"], protocol=parts[0]["protocol"])
    validate_dataset(result)
    return result


def validate_dataset(data):
    n = len(data["rows"])
    for key, shape in (("X_coarse", (n, 24)), ("X_full", (n, 38)), ("X_history", (n, 2, 17)), ("Y", (n, 5))):
        if data[key].shape != shape or not torch.isfinite(data[key]).all():
            raise ValueError(f"Invalid finite {key} shape")
    if not torch.equal(data["X_coarse"][:, :18], data["X_full"][:, :18]) or not torch.equal(data["X_coarse"][:, 18:], data["X_full"][:, 32:]):
        raise ValueError("Coarse/full geometry and mode/route indicators must match")
    if not data["X_history"][:, :, 16].eq(1).all():
        raise ValueError("Training/query histories must contain two real past intervals")
    if not torch.isin(data["Y"][:, 4], torch.tensor([0., 1.])).all():
        raise ValueError("Completion labels must be observed binary outcomes")
    families = defaultdict(set); identifiers = set(); by_support = defaultdict(list)
    for i, row in enumerate(data["rows"]):
        if row["query_id"] in identifiers:
            raise ValueError("Duplicate local query identifier")
        identifiers.add(row["query_id"]); families[row["family_id"]].add(row["split"])
        by_support[row["support"]].append(i)
        if not row["preparation_completed"] or row["current_mode"] not in MODE_NAMES:
            raise ValueError("Query cannot claim an unacquired current contact face")
        if row["environment_start_step"] < PREHISTORY_STEPS:
            raise ValueError("Physical prehistory is missing")
        if row["environment_end_step"]-row["environment_start_step"] != row["actual_steps"]:
            raise ValueError("Physical option clock differs from elapsed time")
        if abs(float(data["Y"][i, 3])-.1*row["actual_steps"]) > 1e-5:
            raise ValueError("Elapsed-time target differs from executed steps")
        if bool(data["Y"][i, 4]) != row["option_completed"]:
            raise ValueError("Completion target differs from executed option result")
    if any(len(splits) != 1 for splits in families.values()):
        raise ValueError("An option family crosses data splits")
    # Exact repeated input across splits is leakage even when labels differ.
    for support, indices in by_support.items():
        seen = {}
        for i in indices:
            key = data["X_coarse"][i].contiguous().numpy().tobytes()
            if key in seen and seen[key] != data["rows"][i]["split"]:
                raise ValueError("Exact coarse option input crosses data splits")
            seen[key] = data["rows"][i]["split"]
    return by_support


def entry_features(entry, coarse, full=None, history=None):
    if coarse.ndim != 2 or coarse.shape[1] != 24:
        raise ValueError("Expected Bx24 coarse option inputs")
    c = coarse[:, :18].clone()
    c[:, 15] = float(entry["reference_substeps"])
    extras = coarse[:, 18:]
    representation = entry["representation"]
    if representation == "option_body_coarse24":
        return torch.cat((body_frame(c), extras), 1)
    if representation == "option_body_full38":
        if full is None or full.shape != (len(c), 38):
            raise ValueError("Full option model requires Bx38 inputs")
        f = full[:, :32].clone(); f[:, 15] = c[:, 15]
        if not torch.equal(full[:, 32:], extras):
            raise ValueError("Full and coarse mode indicators differ")
        return torch.cat((full_body_features(c, f), extras), 1)
    if representation == "option_body_history58":
        return torch.cat((history_body_features(c, history), extras), 1)
    raise ValueError("Unknown option-model representation")


def normalization(x, y):
    x, y = x.double(), y.double()
    spread = x.std(0, correction=0)
    return dict(x_mean=x.mean(0).float(), x_scale=torch.where(spread < 1e-6, torch.ones_like(spread), spread).float(),
        y_mean=y.mean(0).float(), y_scale=y.std(0, correction=0).clamp_min(torch.tensor([1e-4, 1e-4, 1e-3, .1, .1], dtype=torch.float64)).float())


def fit_model(x, y, norm, width, *, steps=1000, seed=0, batch_size=256, lr=.003):
    torch.manual_seed(seed)
    model = ResponseMLP(x.shape[1], width, outputs=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    xs = (x-norm["x_mean"])/norm["x_scale"]
    ys = (y-norm["y_mean"])/norm["y_scale"]
    generator = torch.Generator().manual_seed(seed+94007)
    for _ in range(steps):
        index = torch.randint(len(xs), (min(batch_size, len(xs)),), generator=generator)
        loss = (model(xs[index])-ys[index]).square().mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model.eval()


class FrozenOptionPredictor:
    def __init__(self, entry):
        self.entry = entry
        self.model = ResponseMLP(entry["input_dim"], entry["width"], outputs=5)
        self.model.load_state_dict(entry["state_dict"])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, coarse, full=None, history=None):
        features = entry_features(self.entry, coarse, full, history)
        norm = self.entry["normalization"]
        value = self.model((features-norm["x_mean"])/norm["x_scale"])*norm["y_scale"]+norm["y_mean"]
        value = world_frame_response(coarse[:, :18], value)
        value[:, 2] = wrap(value[:, 2])
        value[:, 3] = value[:, 3].clamp_min(0.)
        value[:, 4] = value[:, 4].clamp(0., 1.)
        return value


def predict_entry(entry, coarse, full=None, history=None):
    """Convenience API; use load_model_pack predictors for repeated queries."""
    return FrozenOptionPredictor(entry)(coarse, full, history)


def load_model_pack(path):
    raw = torch.load(path, map_location="cpu")
    return dict(raw, model_path=str(path), model_sha256=sha(path),
                predictors={key: FrozenOptionPredictor(entry) for key, entry in raw["checkpoints"].items()})


def metrics(y, prediction):
    error = prediction-y
    return dict(rows=len(y), rmse_by_output=error.square().mean(0).sqrt().tolist(),
        translation_rmse_units=float(error[:, :2].square().sum(1).mean().sqrt()),
        angular_rmse_rad=float(error[:, 2].square().mean().sqrt()),
        elapsed_mae_seconds=float(error[:, 3].abs().mean()),
        completion_accuracy=float(((prediction[:, 4] >= .5) == y[:, 4].bool()).float().mean()),
        observed_completion_fraction=float(y[:, 4].mean()))


def _summaries(seed_metrics):
    grouped = defaultdict(list)
    for row in seed_metrics:
        grouped[(row["support"], row["model"], row["split"], row["stratum"], str(row["stratum_value"]))].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        record = dict(zip(("support", "model", "split", "stratum", "stratum_value"), key))
        record.update(seeds=len(values), rows_per_seed=values[0]["rows"])
        for field in ("translation_rmse_units", "angular_rmse_rad", "elapsed_mae_seconds", "completion_accuracy", "observed_completion_fraction"):
            record[field+"_mean"] = float(np.mean([r[field] for r in values]))
            record[field+"_std_over_seeds"] = float(np.std([r[field] for r in values]))
        result.append(record)
    return result


def train_models(data, *, steps=1000, seeds=(0, 1, 2), batch_size=256, lr=.003, models=MODELS):
    by_support = validate_dataset(data)
    coarse, full, history, truth = [data[k] for k in ("X_coarse", "X_full", "X_history", "Y")]
    budget = parameter_count(24, 32, 5)
    definitions = dict(coarse_body_small=("option_body_coarse24", 24, 32),
        full_matched=("option_body_full38", 38, matched_width(38, budget, 5)),
        history_matched=("option_body_history58", 58, matched_width(58, budget, 5)),
        full_large=("option_body_full38", 38, 128))
    if set(models)-set(definitions):
        raise ValueError("Unknown option model")
    train_all = torch.tensor([i for i,r in enumerate(data["rows"]) if r["split"] == "train"])
    if not len(train_all):
        raise ValueError("Missing local training split")
    target = truth.clone(); target[:, :3] = body_frame(coarse[:, :18], truth[:, :3])[1]
    caches = {}
    for name in models:
        representation, dim, width = definitions[name]
        features = entry_features(dict(representation=representation, reference_substeps=data["reference_substeps"]), coarse, full, history)
        caches[name] = (features, normalization(features[train_all], target[train_all]), representation, dim, width)
    checkpoints, predictions, seed_metrics, budgets = {}, {}, [], {}
    for support, indices in sorted(by_support.items()):
        train = [i for i in indices if data["rows"][i]["split"] == "train"]
        val = [i for i in indices if data["rows"][i]["split"] == "val"]
        if not train or not val:
            raise ValueError("Every support requires train and validation rows")
        budgets[support] = dict(Counter(data["rows"][i]["split"] for i in indices))
        budgets[support]["train_families"] = len({data["rows"][i]["family_id"] for i in train})
        for name in models:
            features, norm, representation, dim, width = caches[name]
            for seed in seeds:
                model = fit_model(features[train], target[train], norm, width,
                    steps=steps, seed=seed, batch_size=batch_size, lr=lr)
                entry = dict(state_dict=model.state_dict(), input_dim=dim, width=width, output_dim=5,
                    representation=representation, normalization=norm, reference_substeps=data["reference_substeps"],
                    profile=data["profile"], support=support, model=name, seed=int(seed),
                    parameter_count=parameter_count(dim, width, 5), optimizer_steps=steps,
                    train_row_indices=torch.tensor(train), output_names=list(OUTPUT_NAMES),
                    completion_semantics="clipped regression score, not calibrated probability",
                    helper_learned_parameters=0, extra_history_storage=34 if name == "history_matched" else 0)
                predictor = FrozenOptionPredictor(entry)
                predicted = predictor(coarse[indices], full[indices], history[indices])
                lookup = {i:j for j,i in enumerate(indices)}
                val_predictions = predicted[[lookup[i] for i in val]]
                entry["elapsed_time_q95_absolute_residual_seconds"] = float(torch.quantile((val_predictions[:, 3]-truth[val, 3]).abs(), .95))
                entry["pose_q95_absolute_residual"] = torch.quantile((val_predictions[:, :3]-truth[val, :3]).abs(), .95, dim=0)
                key = f"{support}/{name}/seed{seed}"
                checkpoints[key] = entry
                predictions[key] = dict(row_indices=torch.tensor(indices), prediction=predicted)
                for split in ("train", "val", "test_action", "test_context", "test"):
                    base = [i for i in indices if data["rows"][i]["split"] == split or split == "test" and data["rows"][i]["split"].startswith("test")]
                    groups = [("all", "all", base)]
                    if split == "test":
                        for field in ("current_mode", "route_id", "entry_origin", "history_kind"):
                            groups += [(field, value, [i for i in base if data["rows"][i][field] == value])
                                       for value in sorted({data["rows"][i][field] for i in base})]
                    for stratum, value, picked in groups:
                        if not picked:
                            continue
                        estimate = predicted[[lookup[i] for i in picked]]
                        seed_metrics.append(dict(support=support, model=name, seed=int(seed), split=split,
                            stratum=stratum, stratum_value=value, parameters=entry["parameter_count"],
                            **metrics(truth[picked], estimate)))
                print(json.dumps(dict(trained=key, train_rows=len(train), parameters=entry["parameter_count"])), flush=True)
    report = dict(protocol=registered_protocol(), profile=data["profile"], reference_substeps=data["reference_substeps"],
        rows_per_support=budgets, learned_parameters={name:parameter_count(definitions[name][1],definitions[name][2],5) for name in models},
        optimizer=dict(updates=steps,seeds=list(seeds),batch_size=batch_size,lr=lr),
        seed_metrics=seed_metrics, summary=_summaries(seed_metrics),
        preparation_coverage=[dict(support=p["support"], cases=p["preparation_case_count"], acquired=p["acquisition_success_count"]) for p in data.get("preparations", [data.get("preparation")]) if p],
        limitations=["Atomic endpoint predictions are not a full learned simulator rollout.",
            "Every mode change consumes observed physical duration; elapsed calibration is validation-based, not an OOD guarantee.",
            "Full-state and controller geometry/velocity sensing remain privileged resources. History adds34 stored floats.",
            "Goal-free physical preprocessing differs across supports and acquired-face coverage is reported; this is not a randomized support-effect estimate.",
            "Three model seeds measure optimizer variation, not independent physical success-rate samples.",
            "No complete task trajectory, goal, or task success label enters collection or fitting."])
    return report, dict(checkpoints=checkpoints, predictions=predictions)


def _collect_worker(support, substeps, chunk_size, path, counts):
    torch.set_num_threads(1)
    data = collect_support(support, substeps=substeps, contexts=local_contexts(counts), chunk_size=chunk_size)
    torch.save(data, path)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/novelty/option_interface_local"))
    parser.add_argument("--substeps", type=int, default=160)
    parser.add_argument("--supports", nargs="+", choices=SUPPORTS, default=list(SUPPORTS))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=768)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    registration_path = args.output.with_name(args.output.name+"_registration.json")
    declaration = dict(protocol=registered_protocol(), arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    if not registration_path.exists():
        registration_path.write_text(json.dumps(declaration,indent=2)+"\n")
    elif json.loads(registration_path.read_text())["protocol"] != declaration["protocol"]:
        raise ValueError("Protocol changed; preserve the previous run and choose another prefix")
    if args.register_only:
        print(str(registration_path)); return
    data_path = args.data or args.output.with_name(args.output.name+"_data.pt")
    model_path = args.output.with_name(args.output.name+"_models.pt")
    if model_path.exists() or (args.data is None and data_path.exists()):
        raise FileExistsError("Use a fresh output prefix to preserve existing artifacts")
    paths = [Path(__file__)] + [Path("vmas_pusht")/(name+".py") for name in
        ("action_interface","recovery_primitives","stable_local_wm","stable_composition","abstraction_models",
         "abstraction_transfer","physical_abstraction","physics_profiles","snapshot","scenario")]
    source_hashes = {str(p):sha(p) for p in paths}
    old_models = [Path("data/novelty")/(name+".pt") for name in ("stable_local_original_models","stable_local_selected_models")]
    old_hashes = {str(p):sha(p) for p in old_models if p.exists()}
    started = time.perf_counter()
    if args.data:
        data = torch.load(args.data,map_location="cpu")
        if data["reference_substeps"] != args.substeps or data["profile"] != get_profile("dry_light"):
            raise ValueError("Reused data do not match declared physics/resolution")
    else:
        worker_paths = [args.output.with_name(args.output.name+"_"+support+"_data.pt") for support in args.supports]
        if any(p.exists() for p in worker_paths):
            raise FileExistsError("Local support artifacts already exist")
        with ProcessPoolExecutor(max_workers=min(args.workers,len(args.supports)), mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_collect_worker,s,args.substeps,args.chunk_size,p,CONTEXT_COUNTS) for s,p in zip(args.supports,worker_paths)]
            for future in as_completed(futures):
                print(json.dumps(dict(collected_support_artifact=future.result())),flush=True)
        data = combine_datasets([torch.load(p,map_location="cpu") for p in worker_paths])
        torch.save(data,data_path)
    report, pack = train_models(data,steps=args.steps,seeds=args.seeds)
    provenance = dict(source_hashes=source_hashes, old_model_sha256=old_hashes,
        data_path=str(data_path),data_sha256=sha(data_path),registration_path=str(registration_path),registration_sha256=sha(registration_path),
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),torch_version=str(torch.__version__),
        elapsed_seconds=time.perf_counter()-started)
    for path,digest in {**source_hashes,**old_hashes}.items():
        if sha(path) != digest:
            raise AssertionError("Frozen source or old model changed during training: "+path)
    torch.save(dict(pack,provenance=provenance,profile=data["profile"],reference_substeps=data["reference_substeps"]),model_path)
    torch.load(model_path,map_location="cpu")
    report.update(provenance=provenance,model_path=str(model_path),model_sha256=sha(model_path))
    args.output.with_name(args.output.name+"_models.json").write_text(json.dumps(_plain(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(model_path=str(model_path),rows=len(data["rows"]),elapsed_seconds=provenance["elapsed_seconds"])),flush=True)


if __name__ == "__main__":
    main()
