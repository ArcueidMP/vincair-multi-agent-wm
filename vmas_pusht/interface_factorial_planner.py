"""Frozen predictor by contact-interface controls with explicit query budgets.

The endpoint objective and execution options are unchanged. The budget39 arm
redistributes the original force/duration programs across contact routes; it
does not introduce new forces, durations, model inputs, or training examples.
"""
from __future__ import annotations

import time

import torch

from vmas_pusht.action_interface import candidate_table, model_inputs, valid_options
from vmas_pusht.option_planner import (MODEL_NAMES, endpoint_score, model_key,
    true_option_predictions, validate_models)


POLICIES = ("direct", "expanded", "budget39_s0", "budget39_s1", "budget39_s2")
BASE_PREDICTORS = ("wm_full", "wm_large", "true_option")


def allocated_table(shift):
    """Exactly the direct39 force/duration multiset, assigned to three routes.

    All zero programs stay direct. Each of 12 nonzero force programs visits
    every route across the three durations and across the three fixed shifts.
    Allocation is fixed before task evaluation and never reads state or goals.
    """
    if shift not in (0, 1, 2):
        raise ValueError("Use one of the three prespecified allocation shifts")
    table = candidate_table(expanded=False)
    for index in range(len(table)):
        duration_index, pulse_index = divmod(index, 13)
        if pulse_index:
            table[index, 0] = (duration_index + pulse_index - 1 + shift) % 3
    return table


def option_tables(env, interface, allow_routes, policy):
    """Return B,K,4 tables and a compact shared table when one is available."""
    if policy not in POLICIES:
        raise ValueError("Unknown candidate policy: "+str(policy))
    if policy in ("direct", "expanded"):
        shared = candidate_table(expanded=policy == "expanded" and bool(allow_routes.any()))
        return shared[None].expand(env.num_envs, -1, -1), shared
    direct = candidate_table(expanded=False)
    allocated = allocated_table(int(policy[-1]))
    # When routes cannot be executed, spend the same budget on the full bank
    # on the current face. This uses immediate state/memory legality only.
    routed = allocated[:, 0] > 0
    route_legal = valid_options(env, interface.mode, allocated, state=interface.state)
    can_route = allow_routes & route_legal[:, routed].all(1)
    tables = direct[None].expand(env.num_envs, -1, -1).clone()
    tables[can_route] = allocated
    return tables, None


@torch.no_grad()
def _oracle_predictions(env, interface, tables, shared, indices, legal):
    def padded_steps(result, enabled):
        elapsed = (result[:, :, 3]/env.world.dt).round().long()
        maximum = int(torch.where(enabled, elapsed, 0).max())
        return torch.full((len(result),), result.shape[1]*maximum, dtype=torch.long)
    if shared is not None:
        # Preserve the historical batch layout for the unrestricted controls.
        result = true_option_predictions(env, interface, shared, indices, legal)
        return result, padded_steps(result, legal[indices]), 1
    result = torch.full((len(indices), tables.shape[1], 5), float("nan"))
    padded = torch.zeros(len(indices), dtype=torch.long)
    # There are at most two tables: original direct39 and fixed allocated39.
    # Each simulator call still uses complete snapshots and controller memory.
    distinct, inverse = torch.unique(tables[indices].flatten(1), dim=0, return_inverse=True)
    for group, flat in enumerate(distinct):
        local = (inverse == group).nonzero().flatten()
        result[local] = true_option_predictions(env, interface,
            flat.reshape(-1, 4), indices[local], legal)
        padded[local] = padded_steps(result[local], legal[indices[local]])
    return result, padded, len(distinct)


@torch.no_grad()
def select_options(env, interface, rows, initial_pose, goals, remaining, history,
                   query_mask, model_pack=None, allow_routes=None, *, policy="expanded"):
    """Select one complete option with a frozen WM or exact simulator outcomes.

    Candidate evaluations, legal options and active oracle rollout steps are
    separate counters. Equal candidate budgets are per decision, not per task
    or per unit of simulator computation. Padded simulator integration work is
    also counted: every batch slot advances until its longest option ends.
    Goals affect only the endpoint cost.
    """
    started = time.perf_counter()
    n = env.num_envs
    if len(rows) != n or query_mask.shape != (n,) or remaining.shape != (n,):
        raise ValueError("Rows, query mask and remaining budgets must align")
    if any(row["planner"] not in BASE_PREDICTORS for row in rows):
        raise ValueError("This factorial uses only matched, large, and true-option predictors")
    if (query_mask & interface.busy).any():
        raise ValueError("A committed option must finish before replanning")
    validate_models(rows, model_pack)
    allowed = torch.zeros(n, dtype=torch.bool) if allow_routes is None else allow_routes.bool().clone()
    allowed &= query_mask
    if policy == "direct":
        allowed[:] = False
    tables, shared = option_tables(env, interface, allowed, policy)
    k = tables.shape[1]
    legal = valid_options(env, interface.mode, tables, state=interface.state)
    legal &= query_mask[:, None]
    legal &= (tables[:, :, 0] == 0) | allowed[:, None]
    legal &= tables[:, :, 3] <= remaining[:, None]
    predictions = torch.full((n, k, 5), float("nan"))
    elapsed_margins = torch.zeros(n)
    evaluated = torch.zeros(n, dtype=torch.long)
    simulator_steps = torch.zeros(n, dtype=torch.long)
    padded_steps = torch.zeros(n, dtype=torch.long)
    oracle_calls = 0
    learned = query_mask & torch.tensor([r["planner"] in MODEL_NAMES for r in rows])
    if learned.any():
        coarse, full = model_inputs(env, interface.anchor, interface.mode, tables,
                                    reference_substeps=160)
        for key in sorted({model_key(rows[i]) for i in learned.nonzero().flatten().tolist()}):
            indices = torch.tensor([i for i in range(n) if learned[i] and model_key(rows[i]) == key])
            entry = model_pack["checkpoints"][key]
            if entry.get("reference_substeps", 160) != 160:
                raise ValueError("Expected frozen models with the 160-substep reference")
            prediction = model_pack["predictors"][key](coarse[indices].reshape(-1, 24),
                full=full[indices].reshape(-1, 38),
                history=history[indices, None].expand(-1, k, -1, -1).reshape(-1, 2, 17))
            predictions[indices] = prediction.reshape(len(indices), k, 5)
            elapsed_margins[indices] = float(entry.get("elapsed_time_q95_absolute_residual_seconds", 0.))
            evaluated[indices] = k
    privileged = query_mask & torch.tensor([r["planner"] == "true_option" for r in rows])
    if privileged.any():
        indices = privileged.nonzero().flatten()
        predicted, padded, oracle_calls = _oracle_predictions(env, interface, tables, shared, indices, legal)
        predictions[indices] = predicted
        padded_steps[indices] = padded
        evaluated[indices] = legal[indices].sum(1)
        simulator_steps[indices] = torch.where(legal[indices],
            (predictions[indices, :, 3]/env.world.dt).round().long(), 0).sum(1)
    raw_predictions = predictions.clone()
    direct = tables[:, :, 0] == 0
    known_seconds = tables[:, :, 3]*env.world.dt
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
    fallback = query_mask & ~feasible.any(1)
    chosen[~query_mask | fallback] = -1
    selected = torch.zeros(n, 4)
    selected[:, 3] = 5
    ids = (query_mask & (chosen >= 0)).nonzero().flatten()
    selected[ids] = tables[ids, chosen[ids]]
    selected_prediction = torch.full((n, 5), float("nan"))
    selected_prediction[ids] = raw_predictions[ids, chosen[ids]]
    selected_steps = torch.full((n,), float("nan"))
    selected_steps[ids] = predicted_seconds[ids, chosen[ids]]/env.world.dt
    selected_margin = torch.zeros(n)
    selected_margin[ids] = margins[ids, chosen[ids]]
    selected_score = torch.full((n,), float("nan"))
    selected_score[ids] = score[ids, chosen[ids]]
    return selected, dict(chosen_index=chosen, selected_prediction=selected_prediction,
        selected_score=selected_score, candidate_count=legal.sum(1), queried_mask=query_mask.clone(),
        budget_fallback=fallback, predicted_steps=selected_steps,
        elapsed_validation_margin_seconds=elapsed_margins,
        selected_elapsed_margin_seconds=selected_margin,
        source=["world_model" if r["planner"] in MODEL_NAMES else "true_simulator" for r in rows],
        candidate_policy=policy, candidate_table=tables.clone(),
        candidate_budget=torch.full((n,), k, dtype=torch.long), evaluated_candidates=evaluated,
        simulated_candidate_steps=simulator_steps, active_simulated_candidate_steps=simulator_steps.clone(),
        padded_simulator_steps=padded_steps, padded_simulator_substeps=padded_steps*env.scenario.substeps,
        oracle_calls=oracle_calls, query_wall_seconds=time.perf_counter()-started,
        candidate_completion_scores=raw_predictions[:, :, 4].clone())
