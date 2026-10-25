"""Plan over the finite contact-option interface without hidden simulator use.

Learned planners query local endpoint models only. The separate true_option
baseline evaluates the identical options using complete simulator snapshots.
Both use one complete, variable-duration option of lookahead, not the previous
full-remaining-horizon template MPC. Actual task scoring remains stepwise.
"""
from __future__ import annotations

import math

import torch

from vmas_pusht.action_interface import ActionInterface, candidate_table, model_inputs, valid_options
from vmas_pusht.abstraction_transfer import ANGLE_TOLERANCE, TRANSLATION_TOLERANCE
from vmas_pusht.boundary import _at_wall, _finite
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.scenario import wrap
from vmas_pusht.snapshot import capture, restore


MODEL_NAMES = dict(wm_coarse="coarse_body_small", wm_full="full_matched",
    wm_history="history_matched", wm_large="full_large", wm_full_direct="full_matched")
PLANNERS = tuple(MODEL_NAMES) + ("feedback_upper", "feedback_lower", "true_option")


def validate_models(rows, model_pack):
    for row in rows:
        name = row["planner"]
        if name not in PLANNERS:
            raise ValueError(f"Unknown option planner {name}")
        if name in MODEL_NAMES:
            key = model_key(row)
            if model_pack is None or key not in model_pack.get("checkpoints", {}) or key not in model_pack.get("predictors", {}):
                raise ValueError(f"Missing frozen option model {key}")


def model_key(row):
    return f"{row['support']}/{MODEL_NAMES[row['planner']]}/seed{row.get('model_seed', 0)}"


def protocol():
    return dict(planners=list(PLANNERS), model_mapping=MODEL_NAMES,
        options="continue or one physically executed reverse-contact route, then a constant push",
        directions_degrees=[-20, 0, 20], strengths=[0, .15, .3, .6, .9], push_steps=[5, 10, 20],
        first_rotation="direct options only", final_rotation="expanded except wm_full_direct",
        learned_lookahead="one complete local option; no true simulator or full-task training labels",
        score="squared normalized endpoint angle and translation errors + .001*predicted_seconds + 100*(1-completion_score)",
        completion_threshold=.5, elapsed_margin="validation-only absolute residual q95",
        time_budget="direct pushing uses its exact declared duration; routes use predicted elapsed plus validation margin; actual phase budget is always enforced",
        fallback="a direct zero-force five-step option, interrupted by the real phase deadline if needed",
        feedback="fixed upper/lower route after overshoot; quantize clamp(8*signed_error-.8*signed_omega,0,.9) to bank13",
        resource_note="Atomic-route controller memory/sensing and actual acquisition time belong to every planner.")


def endpoint_score(pose, prediction, initial_pose, goals):
    endpoint = pose[:, None] + prediction[:, :, :3]
    angle = wrap(endpoint[:, :, 2] - initial_pose[:, None, 2] - goals[:, None])
    shift = (endpoint[:, :, :2] - initial_pose[:, None, :2]).norm(dim=-1)
    seconds = prediction[:, :, 3].clamp_min(0)
    completed = prediction[:, :, 4].clamp(0, 1)
    return (angle/ANGLE_TOLERANCE).square() + (shift/TRANSLATION_TOLERANCE).square() + .001*seconds + 100*(1-completed)


@torch.no_grad()
def true_option_predictions(env, interface, table, indices, legal):
    """Privileged diagnostic only; the live environment is never stepped."""
    nq, k = len(indices), len(table)
    repeat = indices.repeat_interleave(k)
    sim = make_profile(getattr(env, "physics_profile", "dry_light"), num_envs=nq*k,
                       substeps=env.scenario.substeps, seed=101)
    restore(sim, capture(env), repeat)
    options = interface.clone_to(sim, repeat)
    enabled = legal[indices].reshape(-1)
    commands = table.repeat(nq, 1)
    start = sim.scenario.t_pose().clone()
    terminal = start.clone()
    options.start(commands, enabled)
    finite, wall = _finite(sim), _at_wall(sim)
    # The controller's acquisition limit is finite; pushing is also bounded.
    maximum = options.state.config.acquisition_limit + int(table[:, 3].max()) + 1
    for _ in range(maximum):
        active = options.busy.clone() & enabled
        if not active.any():
            break
        action = options.command(active)
        sim.step([action[:, 0], action[:, 1]])
        options.observe(active)
        finite &= ~active | _finite(sim)
        wall |= active & _at_wall(sim)
        finished = active & ~options.busy
        terminal[finished] = sim.scenario.t_pose()[finished]
    if (options.busy & enabled).any():
        raise AssertionError("A bounded option did not terminate")
    response = terminal-start
    response[:, 2] = wrap(response[:, 2])
    completed = options.done & ~options.failed & finite & ~wall & enabled
    result = torch.cat((response, options.elapsed[:, None]*sim.world.dt, completed[:, None].float()), 1)
    return result.reshape(nq, k, 5)


def _feedback_choice(env, interface, rows, goals, initial_pose, allow_routes, table, legal, query):
    n = env.num_envs
    chosen = torch.full((n,), -1, dtype=torch.long)
    error = wrap(initial_pose[:, 2]+goals-env.scenario.bar.state.rot[:, 0])
    for i, row in enumerate(rows):
        if not query[i] or not row["planner"].startswith("feedback_"):
            continue
        route = 0
        mode = int(interface.mode[i])
        if mode == 3 and allow_routes[i] and error[i] < -ANGLE_TOLERANCE:
            route = 1 if row["planner"] == "feedback_upper" else 2
        sign = -1. if mode < 2 or route else 1.
        desired = (8.*sign*error[i] - .8*sign*env.scenario.bar.state.ang_vel[i, 0]).clamp(0, .9)
        levels = desired.new_tensor([0., .15, .3, .6, .9])
        strength = levels[(levels-desired).abs().argmin()]
        match = (table[:, 0] == route) & (table[:, 1] == 0) & (table[:, 2] == strength) & (table[:, 3] == 5)
        match &= legal[i]
        found = match.nonzero().flatten()
        if len(found):
            chosen[i] = found[0]
    return chosen


@torch.no_grad()
def select_options(env, interface, rows, initial_pose, goals, remaining, history,
                   query_mask, model_pack=None, allow_routes=None):
    """Select an atomic option for each queried row; no live simulator steps.

    Inference reads only current observations, controller mode/anchor memory,
    declared actions and frozen weights. Targets enter the cost, never the WM.
    """
    n = env.num_envs
    if len(rows) != n or query_mask.shape != (n,) or remaining.shape != (n,):
        raise ValueError("Rows, query mask and remaining budgets must match the environment")
    if (query_mask & interface.busy).any():
        raise ValueError("A committed option must finish or be cancelled before replanning")
    validate_models(rows, model_pack)
    allow_routes = torch.zeros(n, dtype=torch.bool) if allow_routes is None else allow_routes.bool().clone()
    allow_routes &= query_mask
    allow_routes &= torch.tensor([r["planner"] != "wm_full_direct" for r in rows])
    table = candidate_table(expanded=bool(allow_routes.any()))
    k = len(table)
    legal = valid_options(env, interface.mode, table, state=interface.state)
    legal &= query_mask[:, None]
    legal &= (table[None, :, 0] == 0) | allow_routes[:, None]
    legal &= table[None, :, 3] <= remaining[:, None]
    predictions = torch.full((n, k, 5), float("nan"))
    elapsed_margins = torch.zeros(n)
    learned = query_mask & torch.tensor([r["planner"] in MODEL_NAMES for r in rows])
    if learned.any():
        coarse, full = model_inputs(env, interface.anchor, interface.mode, table[None].expand(n, -1, -1),
                                    reference_substeps=160)
        for key in sorted({model_key(rows[i]) for i in learned.nonzero().flatten().tolist()}):
            indices = torch.tensor([i for i in range(n) if learned[i] and model_key(rows[i]) == key])
            entry = model_pack["checkpoints"][key]
            if entry.get("reference_substeps", 160) != 160:
                raise ValueError("This evaluation requires the frozen 160-substep training reference")
            prediction = model_pack["predictors"][key](coarse[indices].reshape(-1, 24),
                full=full[indices].reshape(-1, 38),
                history=history[indices, None].expand(-1, k, -1, -1).reshape(-1, 2, 17))
            predictions[indices] = prediction.reshape(len(indices), k, 5)
            elapsed_margins[indices] = float(entry.get("elapsed_time_q95_absolute_residual_seconds", 0.))
    privileged = query_mask & torch.tensor([r["planner"] == "true_option" for r in rows])
    if privileged.any():
        indices = privileged.nonzero().flatten()
        predictions[indices] = true_option_predictions(env, interface, table, indices, legal)
    queried = learned | privileged
    # A direct option has an exact duration by construction. Only navigation
    # requires learned timing. Keep raw predictions for unbiased error audits.
    raw_predictions = predictions.clone()
    direct = table[None, :, 0] == 0
    known_seconds = table[None, :, 3]*env.world.dt
    predicted_seconds = torch.where(direct, known_seconds,
        torch.maximum(predictions[:, :, 3], known_seconds))
    margins = torch.where(direct, torch.zeros_like(predicted_seconds), elapsed_margins[:, None])
    predictions[:, :, 3] = predicted_seconds
    feasible = legal & torch.isfinite(predictions).all(-1)
    feasible &= predictions[:, :, 4] >= .5
    feasible &= predicted_seconds + margins <= remaining[:, None]*env.world.dt + 1e-6
    score = endpoint_score(env.scenario.t_pose(), predictions, initial_pose, goals)
    score[~feasible] = torch.inf
    chosen = score.argmin(1)
    fallback = query_mask & (~feasible.any(1)) & queried
    chosen[~queried] = -1
    chosen[fallback] = -1
    feedback = _feedback_choice(env, interface, rows, goals, initial_pose, allow_routes, table, legal, query_mask)
    feedback_rows = query_mask & torch.tensor([r["planner"].startswith("feedback_") for r in rows])
    chosen[feedback_rows] = feedback[feedback_rows]
    fallback |= feedback_rows & (feedback < 0)
    selected = torch.zeros(n, 4)
    selected[:, 3] = 5
    valid_choice = query_mask & (chosen >= 0)
    selected[valid_choice] = table[chosen[valid_choice]]
    selected_prediction = torch.full((n, 5), float("nan"))
    model_selected = valid_choice & queried
    ids = model_selected.nonzero().flatten()
    selected_prediction[ids] = raw_predictions[ids, chosen[ids]]
    selected_steps = torch.full((n,), float("nan"))
    selected_steps[ids] = predicted_seconds[ids, chosen[ids]]/env.world.dt
    selected_margin = torch.zeros(n)
    selected_margin[ids] = margins[ids, chosen[ids]]
    selected_score = torch.full((n,), float("nan"))
    selected_score[ids] = score[ids, chosen[ids]]
    return selected, dict(chosen_index=chosen, selected_prediction=selected_prediction,
        selected_score=selected_score, candidate_count=legal.sum(1), queried_mask=queried,
        budget_fallback=fallback, predicted_steps=selected_steps,
        elapsed_validation_margin_seconds=elapsed_margins,
        selected_elapsed_margin_seconds=selected_margin,
        source=["world_model" if r["planner"] in MODEL_NAMES else
                ("true_simulator" if r["planner"] == "true_option" else "feedback") for r in rows])
