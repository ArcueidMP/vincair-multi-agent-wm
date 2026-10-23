"""Fixed-marginal action-sequence pairing on exactly restored VMAS contexts.

Each agent has N frozen full sequences and uses each sequence once in a balanced
N-trial replicate. A method can only choose the permutation pairing those banks.
The independent expectation is the mean of the entire N x N outcome table. The
true-simulator optimum is a privileged diagnostic, not an online planning method.

    python -m vmas_pusht.pairing --checkpoint data/vmas_pusht/wm_full_h_agg2.pt

The frozen model and a pulse-overlap heuristic select their permutations before
the simulator outcome table is evaluated. Every method keeps identical uniform
marginals over complete sequences, including pulse timing. No candidate is
resampled and no single best pair is selected for execution.
"""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _square_matrix(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or not len(scores):
        raise ValueError("scores must be a nonempty square matrix")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return scores


def coupling(permutation):
    """A balanced deterministic matching with row and column mass 1/N."""
    p = np.asarray(permutation)
    if p.ndim != 1 or not len(p) or not np.issubdtype(p.dtype, np.integer):
        raise ValueError("permutation must be a nonempty integer vector")
    n = len(p)
    if not np.array_equal(np.sort(p), np.arange(n)):
        raise ValueError("each candidate must occur exactly once")
    gamma = np.zeros((n, n), dtype=np.float64)
    gamma[np.arange(n), p] = 1 / n
    return gamma


def optimal_permutation(scores, maximize=True):
    """Max/min expected score subject to exactly uniform empirical marginals."""
    scores = _square_matrix(scores)
    rows, cols = linear_sum_assignment(scores, maximize=maximize)
    p = np.empty(len(scores), dtype=np.int64)
    p[rows] = cols
    return p


def permutation_value(scores, permutation):
    scores = _square_matrix(scores)
    gamma = coupling(permutation)
    if gamma.shape != scores.shape:
        raise ValueError("permutation and score dimensions differ")
    return float((scores * gamma).sum())


def additive_projection(scores):
    """Row + column effects: every feasible coupling has exactly the same mean."""
    scores = _square_matrix(scores)
    return scores.mean(1, keepdims=True) + scores.mean(0, keepdims=True) - scores.mean()


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def timing_banks(direction, n=8, horizon=60, pulse_steps=20, seed=0,
                 family="timing", max_steering_degrees=30):
    """(2,N,H,2) banks with identical force magnitude and pulse duration.

    For ``timing``, agent 1's candidate order is shuffled once before scores are
    available and every vector impulse is identical. ``timing-steering`` pairs
    independently permuted onsets and steering angles per agent: vector impulse
    directions differ, while pulse duration and impulse magnitude stay equal.
    """
    if n < 2 or pulse_steps < 1 or horizon < pulse_steps or n > horizon - pulse_steps + 1:
        raise ValueError("need distinct pulse starts and at least two candidates")
    direction = torch.as_tensor(direction, dtype=torch.float32).reshape(2)
    if not torch.isclose(direction.norm(), torch.tensor(1.0), atol=1e-6):
        raise ValueError("direction must have unit Euclidean norm")
    starts = np.rint(np.linspace(0, horizon - pulse_steps, n)).astype(np.int64)
    order = np.random.default_rng(seed).permutation(n)
    offsets = np.stack([starts, starts[order]])
    steering = np.zeros((2, n))
    if family == "timing-steering":
        if not 0 < max_steering_degrees <= 90:
            raise ValueError("steering magnitude must be in (0,90] degrees")
        # N8 default gives [-30,-20,-10,0,0,10,20,30] degrees.
        half = np.linspace(0, max_steering_degrees, (n + 1) // 2)
        grid = np.concatenate([-half[:0:-1], half]) if n % 2 else np.concatenate([-half[::-1], half])
        for agent in range(2):
            rng = np.random.default_rng(seed + 101 + 7919 * agent)
            offsets[agent] = starts[rng.permutation(n)]
            steering[agent] = np.deg2rad(grid[rng.permutation(n)])
    elif family != "timing":
        raise ValueError("unknown candidate bank family")
    banks = torch.zeros(2, n, horizon, 2)
    for agent in range(2):
        for i, start in enumerate(offsets[agent]):
            c, s = math.cos(steering[agent, i]), math.sin(steering[agent, i])
            steered = torch.stack([c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]])
            banks[agent, i, start:start + pulse_steps] = steered
    return banks, offsets


def all_pairs(banks):
    """(2,N,H,2) -> (N*N,H,2,2), row-major (agent 0, agent 1)."""
    if banks.ndim != 4 or banks.shape[0] != 2 or banks.shape[-1] != 2:
        raise ValueError("banks must have shape (2,N,H,2)")
    n, horizon = banks.shape[1:3]
    a = banks[0, :, None].expand(n, n, horizon, 2)
    b = banks[1, None, :].expand(n, n, horizon, 2)
    return torch.stack([a, b], dim=3).reshape(n * n, horizon, 2, 2)


def overlap_scores(banks):
    """Transparent fixed timing heuristic; no simulator or learned outcomes."""
    magnitudes = banks.norm(dim=-1)
    return (magnitudes[0] @ magnitudes[1].T).cpu().numpy()


def _plain(value):
    """Store snapshot data as tensors and primitives, without pickle-only classes."""
    if dataclasses.is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    return value


def _snapshot_error(first, second):
    """Check all captured tensors and metadata, including hidden rigid-body state."""
    first, second = _plain(first), _plain(second)
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, atol=1e-7, rtol=0)
        return float((first.double() - second.double()).abs().max()) if first.numel() else 0.0
    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise AssertionError("snapshot keys differ")
        return max((_snapshot_error(first[k], second[k]) for k in first), default=0.0)
    if isinstance(first, list):
        if len(first) != len(second):
            raise AssertionError("snapshot list lengths differ")
        return max((_snapshot_error(a, b) for a, b in zip(first, second)), default=0.0)
    if first != second:
        raise AssertionError(f"snapshot metadata differ: {first!r}, {second!r}")
    return 0.0


def _com(env):
    sc = env.scenario
    return (sc.bar.state.pos * sc.bar.mass + sc.stem.state.pos * sc.stem.mass) / (sc.bar.mass + sc.stem.mass)


@torch.no_grad()
def evaluate_simulator(env, initial, actions, direction):
    from vmas_pusht.scenario import state
    from vmas_pusht.snapshot import capture, restore

    restore(env, initial)
    start_com = _com(env).clone()
    start_err = env.scenario.pose_err().clone()
    ever_success = env.scenario.success().clone()
    for t in range(actions.shape[1]):
        env.step([actions[:, t, agent] for agent in range(2)])
        ever_success |= env.scenario.success()
    displacement = _com(env) - start_com
    metrics = {
        "progress": start_err - env.scenario.pose_err(),
        "terminal_success": env.scenario.success(),
        "ever_success": ever_success,
        "signed_com_displacement": displacement @ direction.to(actions.device),
        "com_displacement_norm": displacement.norm(dim=-1),
        "terminal_pose_error": env.scenario.pose_err(),
    }
    return {k: v.cpu().numpy() for k, v in metrics.items()}, state(env).cpu(), capture(env)


@torch.no_grad()
def evaluate_model(model_data, hist_s, hist_a, actions, initial_error, goal):
    """Score precisely the same sequences by frozen terminal T-pose progress."""
    model, mean, std, reader, frameskip = model_data
    if actions.shape[1] % frameskip:
        raise ValueError("horizon must be divisible by checkpoint frameskip")
    n, horizon = actions.shape[:2]
    blocks = actions.reshape(n, horizon // frameskip, frameskip, 2, 2).permute(0, 1, 3, 2, 4).reshape(n, -1, 2, 2 * frameskip)
    z = model.encode((hist_s - mean) / std).expand(n, -1, -1, -1)
    pred = model.rollout(z, hist_a.expand(n, -1, -1, -1), blocks)[:, -1]
    planar = reader(pred)
    # Current release fixes angle mean=0/std=1; retain the inverse transform here.
    angle = torch.atan2(planar[:, 6], planar[:, 7]) * std[10] + mean[10]
    from vmas_pusht.scenario import wrap
    weight = 4 * 0.05 / math.radians(20)
    error = (planar[:, 4:6] - goal[:, :2]).norm(dim=1) + weight * wrap(angle - goal[:, 2]).abs()
    return (initial_error - error).cpu().numpy(), planar.cpu()


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_budget(candidates, horizon, has_model, method_count, replay_check=True):
    """Separate selection costs from extra offline verification; no compute parity claim."""
    pairs = candidates ** 2
    return {
        "fixed_full_sequences_per_agent": candidates,
        "unique_joint_sequences": pairs,
        "frozen_model_rollouts_for_selection": pairs if has_model else 0,
        "heuristic_pair_scores": pairs,
        "true_rollouts_for_shared_oracle_best_worst_table": pairs,
        "extra_true_rollouts_for_same_batch_replay": pairs if replay_check else 0,
        "extra_true_rollouts_for_balanced_cyclic_reference": pairs,
        "extra_true_rollouts_for_selected_balanced_execution": candidates * method_count,
        "environment_steps_per_rollout": horizon,
        "equal_online_compute_claimed": False,
        "note": "All selections share the same candidate banks. Oracle uses privileged simulator outcomes. Extra verification outcomes never change non-oracle selections; no CEM, resampling, or replanning.",
    }


def _method_result(metrics, permutation):
    gamma = coupling(permutation)
    n = len(permutation)
    if not (np.allclose(gamma.sum(0), 1 / n, atol=1e-15) and np.allclose(gamma.sum(1), 1 / n, atol=1e-15)):
        raise AssertionError("marginal constraint violated")
    out = {key: permutation_value(matrix, permutation) for key, matrix in metrics.items()}
    out.update(permutation=permutation.tolist(), progress_gain_vs_independent=out["progress"] - float(metrics["progress"].mean()))
    return out


def run(args):
    from vmas_pusht.boundary import place_contact
    from vmas_pusht.scenario import make, state
    from vmas_pusht.snapshot import capture

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_data, frameskip, history = None, 5, 3
    if args.checkpoint:
        from vmas_pusht.plan import head_reader, load
        config = torch.load(args.checkpoint, map_location="cpu")["config"]
        frameskip, history = config["frameskip"], config["history"]
        model, mean, std = load(args.checkpoint, args.device)
        if model.state_head is None:
            raise ValueError("the frozen baseline needs a trained state head")
        model_data = (model, mean, std, head_reader(model, mean, std), frameskip)
    if args.horizon % frameskip:
        raise ValueError("horizon must be a multiple of frameskip")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    physics = {"friction_ratio": args.friction_ratio, "substeps": args.substeps}
    initial_env = make(num_envs=1, device=args.device, seed=args.seed, **physics)
    sim = make(num_envs=args.candidates ** 2, device=args.device, seed=args.seed + 1, **physics)
    sim.reset()
    results, raw_contexts = [], []
    t0 = time.time()
    for layout in args.layouts:
        for degrees in args.angles:
            idx = len(results)
            initial_env.reset()
            direction = place_contact(initial_env, angles=math.radians(degrees), layout=layout,
                                      goal_distance=args.goal_distance, gap=args.gap)[0]
            # Real observed history, not repeated/padded encoder frames.
            states = [state(initial_env).clone()]
            zero = torch.zeros(1, 2, device=args.device)
            for t in range(frameskip * (history - 1)):
                initial_env.step([zero, zero])
                if (t + 1) % frameskip == 0:
                    states.append(state(initial_env).clone())
            hist_s = torch.stack(states, dim=1)
            hist_a = torch.zeros(1, history - 1, 2, 2 * frameskip, device=args.device)
            initial = capture(initial_env)
            goal = initial_env.scenario.goal_pose.clone()
            initial_error = initial_env.scenario.pose_err().clone()
            banks, offsets = timing_banks(direction.cpu(), args.candidates, args.horizon, args.pulse_steps, args.seed + idx,
                                         family=args.bank_family, max_steering_degrees=args.max_steering_degrees)
            impulses = banks.sum(dim=2)
            unit = direction.cpu()
            steering_degrees = torch.rad2deg(torch.atan2(unit[0] * impulses[..., 1] - unit[1] * impulses[..., 0],
                                                        (impulses * unit).sum(dim=-1)))
            bank_hash = tensor_sha256(banks)
            actions = all_pairs(banks).to(args.device)
            heuristic = overlap_scores(banks)
            selections = {"timing_overlap": optimal_permutation(heuristic)}
            predicted = predicted_planar = None
            # Freeze both non-oracle selections before observing any future simulator outcome.
            if model_data is not None:
                predicted, predicted_planar = evaluate_model(model_data, hist_s, hist_a, actions, initial_error, goal)
                predicted = predicted.reshape(args.candidates, args.candidates)
                selections["frozen_world_model"] = optimal_permutation(predicted)
            if tensor_sha256(banks) != bank_hash:
                raise AssertionError("candidate bank mutated during scoring")
            flat_metrics, final_state, final_snapshot = evaluate_simulator(sim, initial, actions, direction)
            metrics = {key: value.reshape(args.candidates, args.candidates) for key, value in flat_metrics.items()}
            replay_error = None
            if not args.skip_replay_check:
                repeated, repeated_state, repeated_snapshot = evaluate_simulator(sim, initial, actions, direction)
                replay_error = _snapshot_error(final_snapshot, repeated_snapshot)
                torch.testing.assert_close(final_state, repeated_state, atol=1e-7, rtol=0)
                for key in repeated:
                    np.testing.assert_allclose(flat_metrics[key], repeated[key], atol=1e-7, rtol=0)
            truth = metrics["progress"]
            selections["oracle_best"] = optimal_permutation(truth)
            selections["oracle_worst"] = optimal_permutation(truth, maximize=False)
            additive = additive_projection(truth)
            additive_gain = permutation_value(additive, optimal_permutation(additive)) - float(additive.mean())
            if abs(additive_gain) > 1e-12:
                raise AssertionError("additive negative control has nonzero coupling gain")
            methods = {name: _method_result(metrics, p) for name, p in selections.items()}
            balanced_audit = {}
            balanced_raw = {}
            # Execute N-trial balanced replicas separately. Float32 VMAS can vary
            # with batch size: expose discrepancies rather than silently relaxing
            # the strict same-batch snapshot replay tolerance. N cyclic balanced
            # replicas cover every pair once and supply an independent reference
            # with the same batch size as each selected N-trial execution.
            balanced_env = make(num_envs=args.candidates, device=args.device, seed=args.seed + 2, **physics)
            balanced_env.reset()
            balanced_tables = {key: np.empty_like(value) for key, value in metrics.items()}
            cyclic_perms = []
            rows = np.arange(args.candidates)
            for shift in range(args.candidates):
                permutation = (rows + shift) % args.candidates
                cyclic_perms.append(permutation)
                pair_indices = rows * args.candidates + permutation
                selected_metrics, _, _ = evaluate_simulator(balanced_env, initial, actions[pair_indices], direction)
                for key, value in selected_metrics.items():
                    balanced_tables[key][rows, permutation] = value
            balanced_independent = {key: float(value.mean()) for key, value in balanced_tables.items()}
            cross_batch_error = float(np.max(np.abs(balanced_tables["progress"] - truth)))
            for name, permutation in selections.items():
                pair_indices = rows * args.candidates + permutation
                selected_metrics, selected_state, selected_snapshot = evaluate_simulator(
                    balanced_env, initial, actions[pair_indices], direction)
                metric_errors = {key: float(np.max(np.abs(value.astype(float) - flat_metrics[key][pair_indices].astype(float))))
                                 for key, value in selected_metrics.items()}
                executed_means = {key: float(value.mean()) for key, value in selected_metrics.items()}
                balanced_audit[name] = {"trials": args.candidates, "max_metric_errors_vs_full_table": metric_errors,
                    "max_state_error_vs_full_table": float((selected_state - final_state[pair_indices]).abs().max()),
                    "executed_means": executed_means,
                    "executed_progress_gain_vs_same_batch_independent": executed_means["progress"] - balanced_independent["progress"],
                    "agent_0_counts": [1] * args.candidates,
                    "agent_1_counts": np.bincount(permutation, minlength=args.candidates).tolist()}
                balanced_raw[name] = {"final_snapshot": _plain(selected_snapshot), "final_state": selected_state,
                    "metrics": {k: torch.from_numpy(v) for k, v in selected_metrics.items()}}
            rng = np.random.default_rng(args.seed + 10000 + idx)
            random_perms = np.stack([rng.permutation(args.candidates) for _ in range(args.random_permutations)])
            random_values = np.array([permutation_value(truth, p) for p in random_perms])
            random_gains = random_values - truth.mean()
            independent = {key: float(value.mean()) for key, value in metrics.items()}
            result = {
                "context": {"layout": layout, "angle_degrees": degrees, "goal": goal.cpu().tolist()[0],
                            "direction": direction.cpu().tolist(), "initial_pose_error": float(initial_error),
                            "physics": physics},
                "bank_sha256": bank_hash,
                "bank_family": args.bank_family,
                "candidate_sha256": [[tensor_sha256(banks[a, i]) for i in range(args.candidates)] for a in range(2)],
                "pulse_onsets": offsets.tolist(),
                "steering_degrees": steering_degrees.tolist(),
                "action_impulses": impulses.tolist(),
                "action_impulse_magnitudes": impulses.norm(dim=-1).tolist(),
                "independent": independent,
                "methods": methods,
                "random_permutation_progress": {"mean": float(random_values.mean()), "std": float(random_values.std()),
                    "gain_mean": float(random_gains.mean()), "gain_quantiles_025_50_975": np.quantile(random_gains, [.025, .5, .975]).tolist()},
                "additive_control_gain": additive_gain,
                "interaction_rms": float(np.sqrt(np.mean((truth - additive) ** 2))),
                "full_snapshot_replay_max_abs_error": replay_error,
                "fresh_balanced_execution_audit": balanced_audit,
                "balanced_batch_independent": balanced_independent,
                "balanced_batch_progress_matrix_max_error": cross_batch_error,
                "true_progress_matrix": truth.tolist(),
                "predicted_progress_matrix": predicted.tolist() if predicted is not None else None,
                "timing_overlap_matrix": heuristic.tolist(),
                "uniform_row_mass": 1 / args.candidates,
                "uniform_column_mass": 1 / args.candidates,
                "pair_count_per_matrix": args.candidates ** 2,
            }
            results.append(result)
            raw_contexts.append({"context": result["context"], "bank_sha256": bank_hash, "banks": banks,
                "pulse_onsets": torch.from_numpy(offsets), "initial_snapshot": _plain(initial),
                "history_states": hist_s.cpu(), "history_actions": hist_a.cpu(), "goal": goal.cpu(),
                "metrics": {k: torch.from_numpy(v) for k, v in metrics.items()},
                "predicted_progress": torch.from_numpy(predicted) if predicted is not None else None,
                "predicted_planar_state": predicted_planar, "final_state": final_state,
                "final_snapshot": _plain(final_snapshot), "permutations": {k: torch.from_numpy(v) for k, v in selections.items()},
                "fresh_balanced_execution": balanced_raw,
                "balanced_batch_tables": {k: torch.from_numpy(v) for k, v in balanced_tables.items()},
                "balanced_batch_cover_permutations": torch.from_numpy(np.stack(cyclic_perms)),
                "random_permutations": torch.from_numpy(random_perms), "random_progress": torch.from_numpy(random_values)})
            brief = {"layout": layout, "angle": degrees, "independent_progress": independent["progress"],
                     "gain": {name: v["progress_gain_vs_independent"] for name, v in methods.items()},
                     "replay_error": replay_error, "cross_batch_progress_max_error": cross_batch_error,
                     "actual_balanced_gain": {name: value["executed_progress_gain_vs_same_batch_independent"]
                                              for name, value in balanced_audit.items()}}
            print(json.dumps(brief), flush=True)
    methods = list(results[0]["methods"])
    aggregate = {name: {key: float(np.mean([r["methods"][name][key] for r in results]))
                       for key in ("progress", "progress_gain_vs_independent", "terminal_success", "ever_success", "signed_com_displacement")}
                 for name in methods}
    for name in methods:
        aggregate[name]["actual_balanced_progress_gain_vs_same_batch_independent"] = float(np.mean([
            r["fresh_balanced_execution_audit"][name]["executed_progress_gain_vs_same_batch_independent"] for r in results]))
    source_paths = [Path(__file__), Path(__file__).with_name("scenario.py"), Path(__file__).with_name("snapshot.py"),
                    Path(__file__).with_name("boundary.py"), Path(__file__).with_name("model.py"), Path(__file__).with_name("plan.py")]
    manifest = {
        "schema_version": 1, "arguments": vars(args), "duration_seconds": time.time() - t0,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p.resolve()): _file_sha256(p) for p in source_paths},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "vmas", "scipy", "numpy")},
        "checkpoint_sha256": _file_sha256(args.checkpoint) if args.checkpoint else None,
        "budget_per_context": execution_budget(args.candidates, args.horizon, bool(args.checkpoint), len(methods),
                                               replay_check=not args.skip_replay_check),
        "objective": "initial T pose error minus terminal T pose error; angle weight = 4*pos_tol/rot_tol",
        "protocol": "N full sequences per agent; coupling matrix is a permutation/N; every sequence occurs once per balanced replicate",
        "history": {"frames": history, "frameskip": frameskip, "real_zero_action_env_steps": frameskip * (history - 1), "padding": False},
        "limitations": ["Finite candidate/context diagnostic, not a proof of singleton impossibility or general cooperation.",
            "Oracle best/worst consume the complete true outcome matrix and are privileged diagnostic bounds.",
            "Timing baseline uses known pulse overlap; model and heuristic select before any true outcomes.",
            "Fixed uniform marginals hold over complete action sequences per context, not a single best execution.",
            "VMAS float32 trajectories can differ across batch size/composition; separate balanced-batch executions and a same-size cyclic independent reference quantify this effect.",
            "No confidence interval over environmental variation: listed contexts are deterministic designed cases.",
            "Random-permutation quantiles describe pairing variability, not uncertainty over new tasks."],
        "aggregate": aggregate, "contexts": results,
        "cross_batch_numerical_audit": {
            "max_progress_table_entry_difference": max(r["balanced_batch_progress_matrix_max_error"] for r in results),
            "reference": "N-sample cyclic balanced cover versus N^2-sample complete matrix; selected N-sample executions are measured separately",
        },
    }
    raw_path = out.with_suffix(".pt")
    torch.save({"schema_version": 1, "arguments": vars(args), "contexts": raw_contexts}, raw_path)
    manifest["raw_file"] = str(raw_path.resolve())
    manifest["raw_sha256"] = _file_sha256(raw_path)
    out.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"aggregate": aggregate, "json": str(out), "raw": str(raw_path), "seconds": time.time() - t0}), flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="Optional frozen state-head checkpoint; no fitting uses outcomes")
    p.add_argument("--out", default="data/novelty/pairing_pilot.json")
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--pulse-steps", type=int, default=20)
    p.add_argument("--bank-family", choices=["timing", "timing-steering"], default="timing")
    p.add_argument("--max-steering-degrees", type=float, default=30,
                   help="Declared steering range for timing-steering; never fitted to simulator outcomes")
    p.add_argument("--layouts", nargs="+", choices=["wide", "narrow", "side"], default=["wide", "narrow", "side"])
    p.add_argument("--angles", type=float, nargs="+", default=[0, 45])
    p.add_argument("--goal-distance", type=float, default=.2)
    p.add_argument("--gap", type=float, default=.001)
    p.add_argument("--friction-ratio", type=float, default=1.5)
    p.add_argument("--substeps", type=int, default=5)
    p.add_argument("--random-permutations", type=int, default=256)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-replay-check", action="store_true", help="Disable the default exact full-matrix replay audit")
    args = p.parse_args()
    if args.random_permutations < 1:
        p.error("--random-permutations must be positive")
    run(args)


if __name__ == "__main__":
    main()
