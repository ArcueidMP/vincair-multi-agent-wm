"""Local-only response models for a separately validated physics profile.

Training contains no goal, planner, retreat, reacquisition, or composed task.
Every label is a five-step response to one constant primary force. Longer local
primitive episodes provide genuine continuation geometry and motion history.
Whole primitive families stay in one split, including shared prefixes of the
5/10/20-step training programs. Those duplicate prefixes are counted explicitly.

This deliberately pose-only interface is not the released 15D world model.
The helper is fixed, goal-independent finite-force PD, not a learned policy.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from vmas_pusht.abstraction_models import (
    ResponseMLP, alias_statistics, body_frame, error_metrics, fit_mlp,
    matched_width, normalization, parameter_count, predict, save_artifact,
    summarize_seed_metrics, world_frame_response,
)
from vmas_pusht.boundary import _at_wall, _finite, _rotate
from vmas_pusht.physical_abstraction import (
    PRIMARY_LOCAL, coarse_state, helper_action, model_features, physical_state,
    setup_scene,
)
from vmas_pusht.physics_profiles import get_profile, make_profile
from vmas_pusht.scenario import wrap


SUPPORTS = ("passive", "world_anchor", "preloaded_anchor")
MODELS = ("coarse_small", "coarse_body_small", "full_matched", "history_matched")
MACRO_STEPS = 5
PRELUDE_STEPS = 10
HISTORY_RECORD_WIDTH = 17


def registered_protocol():
    """The complete local sampling/training protocol; independent of task results."""
    return {
        "schema_version": 1,
        "macro_steps": MACRO_STEPS,
        "zero_primary_prelude_steps": PRELUDE_STEPS,
        "train": {"directions_deg": [-20, 0, 20], "strengths": [.15, .3, .6, .9],
                  "durations_steps": [5, 10, 20]},
        "validation": {"directions_deg": [-10, 10], "strengths": [.45, .75],
                       "durations_steps": [10, 20]},
        "test_action": {"directions_deg": [-35, 35], "strengths": [.45, .75],
                        "durations_steps": [15, 30]},
        "primary_x_offsets": [-.015, 0., .015],
        "training_initial_velocity_omega": [[0., 0., 0.], [.03, 0., 0.], [-.03, 0., 0.],
            [0., .03, 0.], [0., -.03, 0.], [0., 0., .2], [0., 0., -.2]],
        "heldout_velocity_omega": [[.045, 0., .3], [-.045, 0., -.3]],
        "training_angles_deg": [0.], "heldout_angle_deg": 30.,
        "helper": {"kp": 30., "kd": 4., "preload": .01, "learned_parameters": 0,
                   "goal_input": False, "primary_action_input": False},
        "optimizer": {"updates": 1000, "seeds": [0, 1, 2], "batch_size": 256, "lr": .003},
        "models": list(MODELS), "parameter_budget": parameter_count(18, 32),
        "history": "Last two genuine contiguous five-step records: geometry12 at interval start, mean joint force command4, valid1; current geometry is in coarse18. All models share a ten-step zero-primary prelude, included in resource accounting.",
        "normalization": "Training rows only, pooled across supports; reference_substeps fixed at training resolution during all inference. Duplicate nested-duration prefixes are removed before fitting.",
        "selection": "No validation or task-result architecture, update, helper or seed selection. Validation residuals calibrate intervals only.",
    }


def local_programs():
    registration = registered_protocol()
    programs = []
    for split, label in (("train", "train"), ("validation", "val"), ("test_action", "test_action")):
        spec = registration[split]
        for direction, strength, duration in itertools.product(
                spec["directions_deg"], spec["strengths"], spec["durations_steps"]):
            programs.append(dict(direction_deg=float(direction), strength=float(strength),
                                 duration_steps=int(duration), action_split=label))
    # Zero controls are training primitives, not spurious heldout action tests.
    for duration in registration["train"]["durations_steps"]:
        programs.append(dict(direction_deg=0., strength=0., duration_steps=duration, action_split="train"))
    for i, program in enumerate(programs):
        program["program_id"] = i
    return programs


def local_contexts():
    registration = registered_protocol()
    contexts = []
    for offset, velocity in itertools.product(registration["primary_x_offsets"],
                                              registration["training_initial_velocity_omega"]):
        contexts.append(dict(angle_deg=0., primary_x_offset=offset,
                             velocity_omega=velocity, context_split="train"))
    for velocity in registration["heldout_velocity_omega"]:
        contexts.append(dict(angle_deg=0., primary_x_offset=0.,
                             velocity_omega=velocity, context_split="test_nuisance"))
    for velocity in [[0., 0., 0.], *registration["heldout_velocity_omega"]]:
        contexts.append(dict(angle_deg=registration["heldout_angle_deg"], primary_x_offset=0.,
                             velocity_omega=velocity, context_split="test_context"))
    for i, context in enumerate(contexts):
        context["context_id"] = i
    return contexts


def empty_history(coarse):
    """Two unavailable records, padded with current geometry and explicit flags."""
    result = coarse.new_zeros((len(coarse), 2, HISTORY_RECORD_WIDTH))
    result[:, :, :12] = coarse[:, None, :12]
    return result


def history_record(geometry, joint_actions):
    """Create [geometry12, mean joint action4, valid1] from one observed interval.

    joint_actions is B x steps x 2 x 2, B x steps x 4, or an already averaged
    B x 4 command. No hidden velocity or later geometry enters this record.
    """
    if geometry.ndim != 2 or geometry.shape[1] != 12:
        raise ValueError("history geometry must be B x 12")
    if joint_actions.ndim == 4 and joint_actions.shape[-2:] == (2, 2):
        action = joint_actions.flatten(-2).mean(1)
    elif joint_actions.ndim == 3 and joint_actions.shape[-1] == 4:
        action = joint_actions.mean(1)
    elif joint_actions.ndim == 2 and joint_actions.shape[-1] == 4:
        action = joint_actions
    else:
        raise ValueError("history actions must contain two agents' 2D forces")
    if len(action) != len(geometry):
        raise ValueError("history batch sizes differ")
    return torch.cat((geometry, action, geometry.new_ones((len(geometry), 1))), -1)


def push_history(history, record):
    if history.shape[1:] != (2, HISTORY_RECORD_WIDTH) or record.shape != (len(history), HISTORY_RECORD_WIDTH):
        raise ValueError("history shape must be B x 2 x 17")
    return torch.stack((history[:, 1], record), 1)


def _rotate_to_current(coarse, values):
    angle = torch.atan2(coarse[:, 6], coarse[:, 7])
    return _rotate(values, -angle)


def full_body_features(coarse, full):
    if full is None or full.shape != (len(coarse), 32):
        raise ValueError("full_matched requires B x 32 complete input")
    if not torch.equal(full[:, :15], coarse[:, :15]) or not torch.equal(full[:, 16:18], coarse[:, 16:18]):
        raise ValueError("full input must have the same coarse prefix")
    output = full.clone()
    output[:, :18] = body_frame(coarse)
    for start in (18, 21, 24, 27):
        output[:, start:start + 2] = _rotate_to_current(coarse, full[:, start:start + 2])
    return output


def history_body_features(coarse, history):
    if history is None or history.shape != (len(coarse), 2, HISTORY_RECORD_WIDTH):
        raise ValueError("history_matched requires B x 2 x 17 observations")
    angle = torch.atan2(coarse[:, 6], coarse[:, 7])
    transformed = history.clone()
    for h in range(2):
        for start in (0, 2, 4, 8):
            transformed[:, h, start:start + 2] = _rotate_to_current(
                coarse, history[:, h, start:start + 2] - coarse[:, 4:6])
        for start in (6, 10):
            local_angle = torch.atan2(history[:, h, start], history[:, h, start + 1]) - angle
            transformed[:, h, start] = local_angle.sin()
            transformed[:, h, start + 1] = local_angle.cos()
        for start in (12, 14):
            transformed[:, h, start:start + 2] = _rotate_to_current(coarse, history[:, h, start:start + 2])
        # Missing observations contain no geometry/action information. Their
        # zeros and explicit validity flag are invariant to the current pose.
        transformed[:, h, :16] *= history[:, h, 16:17]
    return torch.cat((body_frame(coarse), transformed.flatten(1)), 1)


def entry_features(entry, coarse, full=None, history=None):
    if coarse.ndim != 2 or coarse.shape[1] != 18:
        raise ValueError("coarse response input must be B x 18")
    x = coarse.clone()
    x[:, 15] = float(entry["reference_substeps"])
    representation = entry["representation"]
    if representation == "world_coarse18":
        return x
    if representation == "body_coarse18":
        return body_frame(x)
    if representation == "body_full32":
        return full_body_features(x, full)
    if representation == "body_history52":
        return history_body_features(x, history)
    raise ValueError(f"Unknown representation {representation}")


@torch.no_grad()
def predict_entry(entry, coarse, full=None, history=None):
    """Frozen world-frame [dx,dy,dtheta] prediction; never changes entry weights."""
    model = ResponseMLP(entry["input_dim"], entry["width"])
    model.load_state_dict(entry["state_dict"])
    model.eval()
    features = entry_features(entry, coarse, full, history)
    prediction = predict(model, features, entry["normalization"])
    if entry["representation"] != "world_coarse18":
        prediction = world_frame_response(coarse, prediction)
    return prediction


def _row_split(context, program):
    return context["context_split"] if context["context_split"] != "train" else program["action_split"]


def deduplicate_prefixes(data):
    """Remove repeated local prefixes, verifying that they are truly identical.

    Duration is not part of the future five-step control or past trajectory.
    Programs that end later therefore share exact prefix observations/labels.
    Their canonical logical key includes the complete context and split.
    """
    seen, kept = {}, []
    tensor_names = ("X_coarse", "X_full", "X_history", "Y", "physical_start")
    for i, row in enumerate(data["rows"]):
        key = (row["mode"], row["context_id"], row["direction_deg"], row["strength"], row["window_start"])
        if key in seen:
            previous = seen[key]
            if row["split"] != data["rows"][previous]["split"]:
                raise ValueError("Identical primitive prefix crosses data splits")
            for name in tensor_names:
                if name in data and not torch.equal(data[name][i], data[name][previous]):
                    raise ValueError(f"Logical duplicate prefix has different {name}")
        else:
            seen[key] = i
            kept.append(i)
    result = dict(data)
    result["nominal_window_count"] = len(data["rows"])
    result["duplicate_prefixes_removed"] = len(data["rows"]) - len(kept)
    result["rows"] = [data["rows"][i] for i in kept]
    for name in tensor_names:
        if name in data:
            result[name] = data[name][kept]
    return result


@torch.no_grad()
def constructed_alias_audit(profile, *, substeps, supports=SUPPORTS):
    """Independent exact-alias mechanism audit, never added to WM training.

    This branch intentionally has no prelude: rigid-body-consistent initial
    velocity variants share exactly the same geometry. Each tested primary
    force is held for five steps. Finite-support ambiguity and action authority
    are kept separate from task success and from the postprelude local dataset.
    """
    profile = get_profile(profile)
    unique = {}
    for program in local_programs():
        unique.setdefault((program["direction_deg"], program["strength"]), program)
    programs = list(unique.values())
    velocities = (registered_protocol()["training_initial_velocity_omega"]
                  + registered_protocol()["heldout_velocity_omega"])
    rows = [dict(mode=mode, angle_deg=angle, hidden_id=vid, velocity_omega=velocity,
                 direction_deg=program["direction_deg"], strength=program["strength"],
                 action_split=program["action_split"], duration_steps=MACRO_STEPS)
            for angle, (vid, velocity), program, mode in itertools.product(
                (0., 30.), enumerate(velocities), programs, supports)]
    n = len(rows)
    env = make_profile(profile, num_envs=n, substeps=substeps, seed=12019, device="cpu")
    angles = torch.tensor([math.radians(row["angle_deg"]) for row in rows])
    anchor = setup_scene(env, angle=angles, hidden_v=[r["velocity_omega"][:2] for r in rows],
                         hidden_omega=[r["velocity_omega"][2] for r in rows])
    physical = physical_state(env).clone()
    matched = physical.reshape(-1, len(supports), 24)
    if not torch.equal(matched, matched[:, :1].expand_as(matched)):
        raise AssertionError("Alias supports start at different physical states")
    preload = torch.tensor([r["mode"] == "preloaded_anchor" for r in rows])
    active = torch.tensor([r["mode"] != "passive" for r in rows])
    anchor += preload[:, None] * _rotate(torch.tensor([-.01, .01]).expand(n, 2), angles)
    direction = angles + torch.tensor([math.radians(r["direction_deg"]) for r in rows])
    primary = torch.stack((direction.sin(), -direction.cos()), 1) * torch.tensor([r["strength"] for r in rows])[:, None]
    coarse, full = model_features(env, primary, MACRO_STEPS, anchor)
    pose = env.scenario.t_pose().clone()
    finite, wall = _finite(env), _at_wall(env)
    force_integral, effort = torch.zeros(n, 2), torch.zeros(n, 2)
    for _ in range(MACRO_STEPS):
        helper = helper_action(env, "world_anchor", anchor, kp=30., kd=4.) * active[:, None]
        command = torch.stack((primary, helper), 1)
        env.step([command[:, 0], command[:, 1]])
        norms = command.norm(dim=-1)
        force_integral += norms * env.scenario.agent_force * env.world.dt
        effort += norms.square() * env.world.dt
        finite &= _finite(env); wall |= _at_wall(env)
    response = env.scenario.t_pose() - pose
    response[:, 2] = wrap(response[:, 2])
    summary = []
    for mode in supports:
        for angle in (0., 30.):
            picked = [i for i, r in enumerate(rows) if r["mode"] == mode and r["angle_deg"] == angle]
            controls = {r["hidden_id"]: response[i] for i, r in enumerate(rows)
                        if r["mode"] == mode and r["angle_deg"] == angle and r["strength"] == 0}
            driven = [i for i in picked if rows[i]["strength"] > 0]
            effects = torch.stack([response[i] - controls[rows[i]["hidden_id"]] for i in driven])
            ambiguity = alias_statistics(coarse[picked], response[picked])
            summary.append(dict(mode=mode, angle_deg=angle, **ambiguity,
                matched_zero_primary_angular_effect_rms_rad=float(effects[:, 2].square().mean().sqrt()),
                matched_zero_primary_max_positive_angular_effect_rad=float(effects[:, 2].max()),
                mean_helper_force_integral=float(force_integral[picked, 1].mean()),
                mean_helper_squared_effort=float(effort[picked, 1].mean()),
                finite_all=bool(finite[picked].all()), wall_count=int(wall[picked].sum())))
    return dict(profile=profile, substeps=int(substeps), rows=rows, X_coarse=coarse,
                X_full=full, Y=response, physical_initial=physical, summary=summary,
                used_in_training=False, constructed_initial_conditions=True, zero_primary_prelude_steps=0,
                limitation="Fixed-profile initial-velocity aliases, not friction aliases or reached-history states; no claim that postprelude model inputs remain aliased.")


@torch.no_grad()
def collect_local(profile, *, substeps, supports=SUPPORTS, contexts=None, programs=None):
    """Collect all local programs vectorized, with disjoint five-step labels."""
    profile = get_profile(profile)
    contexts = local_contexts() if contexts is None else contexts
    programs = local_programs() if programs is None else programs
    if not supports or set(supports) - set(SUPPORTS):
        raise ValueError("Unknown or empty support set")
    # Simulate each causal action prefix once, rather than simulating duplicate
    # duration branches and then asking large-batch floating point arithmetic
    # to be bitwise identical. The shorter requested durations are observed
    # prefixes of this same local constant-action trajectory, not new samples.
    nominal_windows = len(contexts) * len(supports) * sum(p["duration_steps"] // MACRO_STEPS for p in programs)
    program_families = {}
    for program in programs:
        key = (program["direction_deg"], program["strength"])
        if key in program_families:
            family = program_families[key]
            if family["action_split"] != program["action_split"]:
                raise ValueError("Shared action prefix crosses data splits")
            family["duration_steps"] = max(family["duration_steps"], program["duration_steps"])
            family["requested_durations_steps"].append(program["duration_steps"])
        else:
            program_families[key] = {**program, "requested_durations_steps": [program["duration_steps"]]}
    programs = list(program_families.values())
    episodes = []
    for context, program, support in itertools.product(contexts, programs, supports):
        if program["duration_steps"] < MACRO_STEPS or program["duration_steps"] % MACRO_STEPS:
            raise ValueError("Local durations must be positive multiples of five")
        family = f"c{context['context_id']}:d{program['direction_deg']}:s{program['strength']}"
        episodes.append({**context, **program, "mode": support, "split": _row_split(context, program),
                         "episode_id": f"c{context['context_id']}:p{program['program_id']}",
                         "family_id": family})
    n = len(episodes)
    if n == 0:
        raise ValueError("No local episodes")
    env = make_profile(profile, num_envs=n, substeps=substeps, seed=9121, device="cpu")
    angles = torch.tensor([math.radians(e["angle_deg"]) for e in episodes])
    primary_local = torch.tensor([[PRIMARY_LOCAL[0] + e["primary_x_offset"], PRIMARY_LOCAL[1]] for e in episodes])
    initial_anchor = setup_scene(env, angle=angles,
        hidden_v=[e["velocity_omega"][:2] for e in episodes],
        hidden_omega=[e["velocity_omega"][2] for e in episodes], primary_local=primary_local)
    initial = physical_state(env).clone()
    matched = initial.reshape(-1, len(supports), 24)
    if not torch.equal(matched, matched[:, :1].expand_as(matched)):
        raise AssertionError("Supports must start at exactly matching physical states")
    anchor = initial_anchor.clone()
    preload = torch.tensor([e["mode"] == "preloaded_anchor" for e in episodes])
    anchor += preload[:, None] * _rotate(torch.tensor([-.01, .01]).expand(n, 2), angles)
    active = torch.tensor([e["mode"] != "passive" for e in episodes])
    directions = angles + torch.tensor([math.radians(e["direction_deg"]) for e in episodes])
    strength = torch.tensor([e["strength"] for e in episodes])
    primary = torch.stack((directions.sin(), -directions.cos()), 1) * strength[:, None]
    durations = torch.tensor([e["duration_steps"] for e in episodes])
    coarse0, _ = model_features(env, primary, MACRO_STEPS, anchor)
    history = empty_history(coarse0)
    columns = defaultdict(list)
    rows, episode_records = [], []
    finite, wall = _finite(env), _at_wall(env)
    max_action = torch.zeros(n)
    force_integral = torch.zeros(n, 2)
    effort = torch.zeros(n, 2)
    prelude_start = env.scenario.t_pose().clone()
    # Genuine observed history, not a padded copy of a single initial pose.
    # This same prelude is part of every downstream planner's task protocol.
    for _ in range(PRELUDE_STEPS // MACRO_STEPS):
        geometry = coarse_state(env).clone()
        commands = []
        for _ in range(MACRO_STEPS):
            helper = helper_action(env, "world_anchor", anchor, kp=30., kd=4.) * active[:, None]
            command = torch.stack((torch.zeros_like(primary), helper), 1)
            env.step([command[:, 0], command[:, 1]])
            commands.append(command.clone())
            command_norm = command.norm(dim=-1)
            max_action = torch.maximum(max_action, command_norm.amax(-1))
            force_integral += command_norm * env.scenario.agent_force * env.world.dt
            effort += command_norm.square() * env.world.dt
            finite &= _finite(env)
            wall |= _at_wall(env)
        history = push_history(history, history_record(geometry, torch.stack(commands, 1)))
    if not torch.all(history[:, :, 16] == 1):
        raise AssertionError("Training queries require two observed history intervals")
    prelude_response = env.scenario.t_pose() - prelude_start
    prelude_response[:, 2] = wrap(prelude_response[:, 2])
    start_pose = env.scenario.t_pose().clone()
    final_response = torch.zeros(n, 3)
    for window_start in range(0, int(durations.max()), MACRO_STEPS):
        selected = torch.nonzero(durations > window_start, as_tuple=False).flatten()
        xc, xf = model_features(env, primary, MACRO_STEPS, anchor)
        geometry = coarse_state(env).clone()
        pose_before = env.scenario.t_pose().clone()
        physical_before = physical_state(env).clone()
        commands = []
        for step in range(MACRO_STEPS):
            helper = helper_action(env, "world_anchor", anchor, kp=30., kd=4.) * active[:, None]
            command = torch.stack((primary, helper), 1)
            command *= (window_start + step < durations)[:, None, None]
            env.step([command[:, 0], command[:, 1]])
            commands.append(command.clone())
            command_norm = command.norm(dim=-1)
            max_action = torch.maximum(max_action, command_norm.amax(-1))
            force_integral += command_norm * env.scenario.agent_force * env.world.dt
            effort += command_norm.square() * env.world.dt
            finite &= _finite(env)
            wall |= _at_wall(env)
        response = env.scenario.t_pose() - pose_before
        response[:, 2] = wrap(response[:, 2])
        for name, values in (("X_coarse", xc), ("X_full", xf), ("X_history", history),
                              ("Y", response), ("physical_start", physical_before)):
            columns[name].append(values[selected].clone())
        for i in selected.tolist():
            rows.append({**episodes[i], "window_start": window_start, "window_end": window_start + MACRO_STEPS,
                         "window_id": f"{episodes[i]['episode_id']}:w{window_start}",
                         "substeps": int(substeps), "reference_substeps": int(substeps),
                         "environment_start_step": PRELUDE_STEPS + window_start,
                         "is_reset_alias": False, "is_local_program_start": window_start == 0})
        ended = durations == window_start + MACRO_STEPS
        delta = env.scenario.t_pose() - start_pose
        delta[:, 2] = wrap(delta[:, 2])
        final_response[ended] = delta[ended]
        for i in torch.nonzero(ended, as_tuple=False).flatten().tolist():
            episode_records.append({**episodes[i], "response": final_response[i].tolist(),
                "zero_primary_prelude_steps": PRELUDE_STEPS,
                "total_simulated_steps": PRELUDE_STEPS + episodes[i]["duration_steps"],
                "prelude_response": prelude_response[i].tolist(),
                "finite": bool(finite[i]), "wall": bool(wall[i]),
                "max_action_norm": float(max_action[i]), "force_integral": force_integral[i].tolist(),
                "squared_control_effort": effort[i].tolist()})
        history = push_history(history, history_record(geometry, torch.stack(commands, 1)))
    result = {name: torch.cat(values) for name, values in columns.items()}
    result.update(rows=rows, episodes=episode_records, protocol=registered_protocol(),
                  profile=profile, reference_substeps=int(substeps),
                  exact_matched_initial_physical_states=True,
                  helper_anchor_note="Effective preload anchor is shifted .01; physical starts and coarse geometry match across supports, anchor coordinates need not.")
    result = deduplicate_prefixes(result)
    result["nominal_window_count"] = nominal_windows
    result["duplicate_prefixes_removed"] = nominal_windows - len(result["rows"])
    result["prefix_deduplication"] = "Structural before simulation: each context/direction/strength family is simulated once up to its longest registered duration. Shorter duration endpoints are prefixes, never independent samples."
    validate_local_dataset(result)
    return result


def validate_local_dataset(data):
    rows = data["rows"]
    n = len(rows)
    for name, shape in (("X_coarse", (n, 18)), ("X_full", (n, 32)),
                        ("X_history", (n, 2, 17)), ("Y", (n, 3))):
        value = data[name]
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid {name} shape or finite values")
    if not torch.equal(data["X_full"][:, :18], data["X_coarse"]):
        raise ValueError("Full state must have the same coarse prefix")
    by_mode, families = defaultdict(list), defaultdict(set)
    for i, row in enumerate(rows):
        if row["window_end"] - row["window_start"] != MACRO_STEPS:
            raise ValueError("Every response must span exactly five steps")
        if row["window_end"] > row["duration_steps"]:
            raise ValueError("A response crossed an action-program endpoint")
        if row["split"] == "train" and (row["context_split"] != "train" or row["action_split"] != "train"):
            raise ValueError("Heldout primitive leaked into training")
        by_mode[row["mode"]].append(i)
        families[(row["mode"], row["family_id"])].add(row["split"])
    if any(len(splits) != 1 for splits in families.values()):
        raise ValueError("Shared primitive family crosses train/test split")
    reference = None
    for mode, ids in by_mode.items():
        keys = [(rows[i]["window_id"], rows[i]["split"]) for i in ids]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate local window ID in a support")
        if reference is not None and set(keys) != reference:
            raise ValueError("Supports have unequal primitive/window budgets")
        reference = set(keys)
    if not any(r["split"] == "train" for r in rows):
        raise ValueError("No training windows")
    return by_mode


def _calibration(y, prediction):
    if len(y) == 0:
        return None
    return torch.quantile((prediction - y).abs(), .95, dim=0)


def train_models(data, *, steps=1000, seeds=(0, 1, 2), models=MODELS,
                 batch_size=256, lr=.003):
    by_mode = validate_local_dataset(data)
    coarse, full, history, y = [data[k].float() for k in ("X_coarse", "X_full", "X_history", "Y")]
    rows = data["rows"]
    train = torch.tensor([r["split"] == "train" for r in rows])
    pooled_scale = normalization(coarse[train], y[train])["y_scale"]
    reference_substeps = int(data["reference_substeps"])
    budget = parameter_count(18, 32)
    definitions = {
        "coarse_small": ("world_coarse18", 18, 32),
        "coarse_body_small": ("body_coarse18", 18, 32),
        "full_matched": ("body_full32", 32, matched_width(32, budget)),
        "history_matched": ("body_history52", 52, matched_width(52, budget)),
    }
    if set(models) - set(definitions):
        raise ValueError("Unknown model specification")
    caches = {}
    for name in models:
        representation, dim, width = definitions[name]
        skeleton = dict(representation=representation, reference_substeps=reference_substeps)
        inputs = entry_features(skeleton, coarse, full, history)
        target = y if representation == "world_coarse18" else body_frame(coarse, y)[1]
        caches[name] = (inputs, target, normalization(inputs[train], target[train]), representation, dim, width)
    metrics, checkpoints, predictions = [], {}, {}
    splits = sorted(set(r["split"] for r in rows))
    alias, budgets = {}, {}
    for mode, indices in sorted(by_mode.items()):
        indices = sorted(indices, key=lambda i: rows[i]["window_id"])
        train_ids = [i for i in indices if rows[i]["split"] == "train"]
        val_ids = [i for i in indices if rows[i]["split"] == "val"]
        budgets[mode] = {split: sum(rows[i]["split"] == split for i in indices) for split in splits}
        budgets[mode]["unique_training_coarse_inputs"] = len(torch.unique(coarse[train_ids], dim=0))
        budgets[mode]["unique_training_full_inputs"] = len(torch.unique(full[train_ids], dim=0))
        budgets[mode]["training_primitive_families"] = len(set(rows[i]["family_id"] for i in train_ids))
        # Strip duplicate program-duration prefixes before alias statistics so
        # repeated identical trajectories are not mistaken for nuisance draws.
        reset = {}
        for i in indices:
            if rows[i]["window_start"] == 0:
                reset.setdefault((rows[i]["context_id"], rows[i]["direction_deg"], rows[i]["strength"]), i)
        reset_ids = list(reset.values())
        alias[mode] = alias_statistics(coarse[reset_ids], y[reset_ids])
        alias[mode]["note"] += " These queries follow the ten-step prelude and are not asserted to remain exact nuisance aliases. Separate constructed initial-state audit is needed."
        for name in models:
            inputs, target, norm, representation, dim, width = caches[name]
            for seed in seeds:
                model = fit_mlp(inputs[train_ids], target[train_ids], norm, width,
                                steps, seed, batch_size, lr)
                entry = dict(state_dict=model.state_dict(), input_dim=dim, width=width,
                    normalization=norm, representation=representation,
                    reference_substeps=reference_substeps, profile=data["profile"], macro_steps=MACRO_STEPS,
                    train_row_indices=torch.tensor(train_ids), optimizer_steps=steps,
                    seed=int(seed), parameter_count=parameter_count(dim, width), model=name,
                    support=mode, helper=registered_protocol()["helper"])
                pred = predict_entry(entry, coarse[indices], full[indices], history[indices])
                lookup = {row_id: j for j, row_id in enumerate(indices)}
                calibration = _calibration(y[val_ids], pred[[lookup[i] for i in val_ids]])
                entry["calibration_validation_absolute_q95"] = calibration
                key = f"{mode}/{name}/seed{seed}"
                checkpoints[key] = entry
                predictions[key] = {"row_indices": torch.tensor(indices), "prediction": pred}
                for split in splits + ["all_heldout"]:
                    for section in ("all", "reset", "continuation"):
                        picked = [j for j, i in enumerate(indices)
                            if (rows[i]["split"] != "train" if split == "all_heldout" else rows[i]["split"] == split)
                            and (section == "all" or (rows[i]["window_start"] == 0) == (section == "reset"))]
                        if not picked:
                            continue
                        truth = y[[indices[j] for j in picked]]
                        estimate = pred[picked]
                        result = error_metrics(truth, estimate, pooled_scale)
                        result.update(mode=mode, model=name, seed=int(seed),
                            split=split if section == "all" else f"{split}/{section}", parameters=entry["parameter_count"])
                        if calibration is not None:
                            covered = (estimate - truth).abs() <= calibration
                            result["validation_q95_coverage_by_axis"] = covered.float().mean(0).tolist()
                            result["validation_q95_joint_coverage"] = float(covered.all(-1).float().mean())
                        metrics.append(result)
    report = dict(schema_version=1, profile=data["profile"], reference_substeps=reference_substeps,
        protocol=data["protocol"], training=dict(updates=steps, seeds=list(seeds), batch_size=batch_size, lr=lr,
        learned_parameters_by_model={name: parameter_count(definitions[name][1], definitions[name][2]) for name in models},
        helper_learned_parameters=0, helper_memory_floats=2,
        model_extra_history_floats=34, history_observation_frames=3,
        all_models_use_the_same_primitive_windows=True),
        rows_per_mode=budgets, seed_metrics=metrics, summary=summarize_seed_metrics(metrics),
        nominal_window_count=data.get("nominal_window_count", len(rows)),
        duplicate_prefixes_removed=data.get("duplicate_prefixes_removed", 0),
        postprelude_exact_input_statistics=alias,
        limitations=[
            "The deliberately coarse pose representation omits velocity; this does not show information loss in the released 15D WM.",
            "The helper is hand-written bounded PD with extra velocity sensing and two-float anchor memory, not a learned abstraction.",
            "Longer local primitives share exact prefixes, which are deduplicated before fitting. No shared family crosses splits.",
            "Continuation geometry differs legitimately across supports. Initial physical states match, but the preloaded effective anchor differs.",
            "The history baseline adds34 stored floats and observed joint-force history; parameter matching alone is not total-system complexity matching.",
            "Three seeds measure optimizer variation. They are not independent physics replications or confidence intervals.",
            "Validation-calibrated intervals are marginal residual quantiles, not guaranteed simultaneous or OOD coverage.",
            "Coarse/full/history body models use declared geometric invariance. The fixed resolution feature prevents numerical refinement becoming an OOD input.",
        ])
    return report, dict(checkpoints=checkpoints, predictions=predictions)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Previously validated profile JSON/name")
    parser.add_argument("--substeps", type=int, required=True, help="Frozen training/reference resolution")
    parser.add_argument("--output", type=Path, default=Path("data/novelty/stable_local"))
    parser.add_argument("--data", type=Path, help="Reuse an already collected local dataset")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    parser.add_argument("--supports", choices=SUPPORTS, nargs="+", default=list(SUPPORTS))
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    registration = dict(protocol=registered_protocol(), profile=get_profile(args.profile),
                        reference_substeps=args.substeps, updates=args.steps, seeds=args.seeds,
                        models=args.models, supports=args.supports)
    registration_path = args.output.with_name(args.output.name + "_registration.json")
    registration_path.write_text(json.dumps(registration, indent=2, allow_nan=False) + "\n")
    if args.register_only:
        print(str(registration_path))
        return
    data_path = args.data or args.output.with_name(args.output.name + "_data.pt")
    if args.data:
        data = torch.load(args.data, map_location="cpu")
        if data["profile"] != registration["profile"] or data["reference_substeps"] != args.substeps:
            raise ValueError("Reused dataset does not match frozen profile/resolution")
    else:
        data = collect_local(args.profile, substeps=args.substeps, supports=args.supports)
        torch.save(data, data_path)
        torch.load(data_path, map_location="cpu")
    print(json.dumps({"collected_windows": len(data["rows"]), "seconds": time.perf_counter()-started}), flush=True)
    report, artifacts = train_models(data, steps=args.steps, seeds=args.seeds, models=args.models)
    source_files = [Path(__file__), Path(__file__).with_name("physics_profiles.py"),
                    Path(__file__).with_name("physical_abstraction.py"),
                    Path(__file__).with_name("abstraction_models.py"), Path(__file__).with_name("scenario.py")]
    provenance = dict(source_sha256={str(p): _sha256(p) for p in source_files},
        data_path=str(data_path), data_sha256=_sha256(data_path),
        registration_path=str(registration_path), registration_sha256=_sha256(registration_path),
        torch_version=str(torch.__version__), profile=registration["profile"], reference_substeps=args.substeps,
        local_only_training=True, elapsed_seconds=time.perf_counter()-started)
    model_path = args.output.with_name(args.output.name + "_models.pt")
    save_artifact(model_path, provenance, artifacts)
    report.update(provenance=provenance, model_path=str(model_path), model_sha256=_sha256(model_path))
    args.output.with_name(args.output.name + "_models.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"model_path": str(model_path), "seconds": provenance["elapsed_seconds"],
                      "rows_per_mode": report["rows_per_mode"], "training": report["training"]}, indent=2))


if __name__ == "__main__":
    main()
