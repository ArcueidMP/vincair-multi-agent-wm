"""Centralized selection over both agents' options, learned or simulator based."""
import time

import torch

from vmas_pusht.boundary import _rotate
from vmas_pusht.joint_interface import features, joint_table
from vmas_pusht.joint_models import execute
from vmas_pusht.option_planner import endpoint_score
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.snapshot import capture, restore


def tables_for(env, interface, allow_routes, *, planned_helper, expanded, allocation):
    direct = joint_table(planned_helper=planned_helper)
    routed = joint_table(planned_helper=planned_helper, expanded=True, allocation=allocation)
    tables = direct[None].expand(env.num_envs, -1, -1).clone()
    if expanded:
        can_route = interface.valid_options(routed)
        route_columns = routed[:, 0] != 0
        enabled = allow_routes & can_route[:, route_columns].all(1)
        tables[enabled] = routed
    return tables


@torch.no_grad()
def oracle(env, interface, tables, query, legal):
    """Evaluate identical options from full state and cloned controller memory."""
    n, k = tables.shape[:2]
    predictions = torch.full((n, k, 5), float("nan"))
    active_steps = torch.zeros(n, dtype=torch.long)
    padded_steps = 0
    indices = query.nonzero().flatten()
    if not len(indices):
        return predictions, active_steps, padded_steps
    repeat = indices.repeat_interleave(k)
    sim = make_profile("dry_light", num_envs=len(repeat), substeps=env.scenario.substeps, seed=173)
    restore(sim, capture(env), repeat)
    memory = interface.clone_to(sim, repeat)
    result = execute(sim, memory, tables[indices].reshape(-1, 5), legal[indices].reshape(-1))
    predictions[indices] = result["Y"].reshape(-1, k, 5)
    active_steps[indices] = result["active_steps"].reshape(-1, k).sum(1)
    return predictions, active_steps, result["padded_steps"]


@torch.no_grad()
def select_options(env, interface, rows, initial_pose, goals, remaining, history,
                   query_mask, model_pack=None, allow_routes=None, *, predictor=None,
                   planned_helper=True, expanded=False, allocation=0):
    """Goals enter endpoint cost only; learned inference never calls a simulator."""
    if (query_mask & interface.busy).any():
        raise ValueError("Complete the current joint option before replanning")
    started = time.perf_counter()
    n = env.num_envs
    allowed = torch.zeros(n, dtype=torch.bool) if allow_routes is None else allow_routes
    tables = tables_for(env, interface, allowed, planned_helper=planned_helper,
                        expanded=expanded, allocation=allocation)
    k = tables.shape[1]
    legal = interface.valid_options(tables) & query_mask[:, None]
    legal &= (tables[:, :, 0] == 0) | allowed[:, None]
    legal &= tables[:, :, 3] <= remaining[:, None]
    active_steps = torch.zeros(n, dtype=torch.long)
    padded_steps = 0
    evaluated = torch.zeros(n, dtype=torch.long)
    if predictor is None:
        predictions, active_steps, padded_steps = oracle(env, interface, tables, query_mask, legal)
        margin = 0.
        evaluated = legal.sum(1)
    else:
        predictions = torch.full((n, k, 5), float("nan"))
        x = features(env, interface, tables)
        indices = query_mask.nonzero().flatten()
        if len(indices):
            body = predictor(x[indices].reshape(-1, x.shape[-1])).reshape(len(indices), k, 5)
            angles = env.scenario.bar.state.rot[indices, 0, None].expand(-1, k).reshape(-1)
            body[..., :2] = _rotate(body[..., :2].reshape(-1, 2), angles).reshape(len(indices), k, 2)
            predictions[indices] = body
        margin = predictor.entry["duration_margin"]
        evaluated[query_mask] = k
    raw = predictions.clone()
    direct = tables[:, :, 0] == 0
    duration = torch.where(direct, tables[:, :, 3]*env.world.dt,
                           torch.maximum(predictions[:, :, 3], tables[:, :, 3]*env.world.dt))
    feasible = legal & torch.isfinite(predictions).all(-1) & (predictions[:, :, 4] >= .5)
    feasible &= duration + (~direct)*margin <= remaining[:, None]*env.world.dt + 1e-6
    predictions[:, :, 3] = duration
    scores = endpoint_score(env.scenario.t_pose(), predictions, initial_pose, goals)
    scores[~feasible] = torch.inf
    chosen = scores.argmin(1)
    fallback = query_mask & ~feasible.any(1)
    chosen[~query_mask | fallback] = -1
    options = torch.zeros(n, 5)
    options[:, 3] = 5
    selected_prediction = torch.full((n, 5), float("nan"))
    selected_score = torch.full((n,), float("nan"))
    ids = (chosen >= 0).nonzero().flatten()
    options[ids] = tables[ids, chosen[ids]]
    selected_prediction[ids] = raw[ids, chosen[ids]]
    selected_score[ids] = scores[ids, chosen[ids]]
    return options, dict(chosen_index=chosen, selected_prediction=selected_prediction,
        selected_score=selected_score, candidate_count=legal.sum(1),
        queried_mask=query_mask.clone(), budget_fallback=fallback,
        candidate_table=tables, candidate_budget=k, evaluated_candidates=evaluated,
        active_simulator_steps=active_steps, padded_simulator_steps=padded_steps,
        helper_choice=options[:, 4].long(), elapsed_margin_seconds=margin,
        selected_options_are_joint=True, query_wall_seconds=time.perf_counter()-started)
