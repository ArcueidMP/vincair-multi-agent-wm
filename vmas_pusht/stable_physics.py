"""Prespecified profile selection using local physics, never task success.

Every comparison starts from identical complete physical states at identical
environment timestamps. Closed-loop feedback and exact finest joint-action
replay are audited separately. Selection requires both useful matched-zero
rotation authority and finite-grid convergence; motionless systems cannot pass.
"""
from dataclasses import asdict, dataclass
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from vmas_pusht.abstraction_convergence import (Tolerances, compare_trajectories,
    make_rows, rollout, rollout_summary)
from vmas_pusht.physical_abstraction import setup_scene
from vmas_pusht.physics_profiles import PROFILES, get_profile, make_profile
from vmas_pusht.snapshot import capture


@dataclass(frozen=True)
class SelectionCriteria:
    max_xy_units: float = .001
    max_angle_rad: float = .003
    minimum_max_authority_rad: float = .12
    minimum_mean_authority_rad: float = .04
    maximum_weld_xy_units: float = .01
    maximum_weld_angle_rad: float = .08
    maximum_action_norm: float = 1.000001


def local_rows(split="development", screen=False):
    if split not in ("development", "validation"):
        raise ValueError("Unknown local split")
    if split == "development":
        angles, hidden = (0.,), ((0., 0., 0.),)
        directions, strengths = ((0.,), (.6, .9)) if screen else ((-20., 0., 20.), (.3, .6, .9))
        pulses = (20,) if screen else (5, 20)
    else:
        angles = (30.,)
        hidden = ((0., 0., 0.), (.03, 0., .2), (-.03, 0., -.2), (0., .03, 0.), (0., -.03, 0.))
        directions, strengths, pulses = (-35., 35.), (.45, .75), (15, 30)
    programs = [dict(direction_deg=d, strength=s, pulse_steps=p)
                for d, s, p in itertools.product(directions, strengths, pulses)]
    # Matched zero controls are indexed by angle, hidden state and helper mode.
    programs.append(dict(direction_deg=0., strength=0., pulse_steps=20))
    return make_rows(angles=angles, nuisances=hidden, programs=programs)


def preregistration():
    return dict(selection_uses="local development physics only; no task goals, WM fits or composition outcomes",
        profile_order=["dry_light", "dry_medium", "dry_lowrotation", "damped_heavy6", "damped_heavy12"],
        original_control="original", horizon_steps=60, physical_dt=.1,
        refinement_substeps=[40, 80, 160, 320], criteria=asdict(SelectionCriteria()),
        proposed_nominal_substeps=80,
        selection_rule="first dry profile passing full development convergence and authority, then independent validation; fallback profiles only if dry family fails",
        control_bounds="Both ordinary dynamic agents retain force multiplier1 and Euclidean action norm<=1. All joints have finite stiffness.",
        development_rows=local_rows("development"), validation_rows=local_rows("validation"),
        validation_rule="Independent heldout local angle/actions/initial velocities. A failed validation is reported, not silently used for tuning.",
        five_step_check="Report authority by step5 separately to contextualize old local-controller horizons; not optimized against task goals.")


def _authority(run, rows, horizon=None):
    from vmas_pusht.scenario import wrap
    trajectory = run["trajectory"]
    if horizon is not None:
        trajectory = trajectory[:, :horizon+1]
    result = []
    for mode in dict.fromkeys(row["mode"] for row in rows):
        zero = {(r["angle_deg"], r["hidden_id"]): i for i, r in enumerate(rows)
                if r["mode"] == mode and r["strength"] == 0.}
        moving = [i for i, r in enumerate(rows) if r["mode"] == mode and r["strength"] > 0.]
        if not moving or not zero:
            raise ValueError("Authority requires matched zero controls in every mode/context")
        controls = [zero[(rows[i]["angle_deg"], rows[i]["hidden_id"])] for i in moving]
        delta = wrap(trajectory[moving, :, 16] - trajectory[controls, :, 16])
        peak = delta.abs().amax(1)
        result.append(dict(mode=mode, moving_cases=len(moving), max_authority_rad=float(peak.max()),
                           mean_peak_authority_rad=float(peak.mean()),
                           median_peak_authority_rad=float(peak.median()),
                           minimum_peak_authority_rad=float(peak.min())))
    return result


def qualification(summary, *, criteria=SelectionCriteria(), nominal=None):
    steps = summary["substeps"]
    nominal = steps[-2] if nominal is None else nominal
    comparisons = [c for c in summary["comparisons"] if c["candidate_substeps"] == nominal
                   and c["reference_substeps"] > nominal and c["pair"] == "vs_finest"]
    if len(comparisons) != 2:
        raise ValueError("Qualification requires both feedback and identical-action finest comparisons")
    converges = all(c["max_xy_units"] <= criteria.max_xy_units
                    and c["max_angle_rad"] <= criteria.max_angle_rad for c in comparisons)
    authority = summary["authority_full_horizon"]
    active = [a for a in authority if a["mode"] != "passive"]
    useful = any(a["max_authority_rad"] >= criteria.minimum_max_authority_rad
                 and a["mean_peak_authority_rad"] >= criteria.minimum_mean_authority_rad for a in active)
    finest = summary["feedback_runs"][-1]["helper_modes"]
    finite_fidelity = all(a["wall_cases"] == 0 and a["max_action_norm"] <= criteria.maximum_action_norm
                         and a["max_weld_xy_units"] <= criteria.maximum_weld_xy_units
                         and a["max_weld_angle_rad"] <= criteria.maximum_weld_angle_rad for a in finest)
    return dict(nominal_substeps=nominal, finest_substeps=steps[-1], converges=converges,
                useful_primary_authority=useful, finite_force_and_weld_fidelity=finite_fidelity,
                qualified=converges and useful and finite_fidelity,
                criteria=asdict(criteria))


def run_profile(profile, *, rows=None, substeps=(80, 160), horizon=60, seed=42):
    cfg = get_profile(profile)
    rows = local_rows() if rows is None else rows
    steps = sorted(set(substeps))
    if len(steps) < 2 or steps[0] < 1:
        raise ValueError("At least two positive refinement resolutions are required")
    n, finest = len(rows), steps[-1]
    source = make_profile(cfg, num_envs=n, substeps=finest, seed=seed)
    angles = torch.tensor([math.radians(row["angle_deg"]) for row in rows])
    anchor = setup_scene(source, angle=angles,
                         hidden_v=[[row["vx"], row["vy"]] for row in rows],
                         hidden_omega=[row["omega"] for row in rows])
    snapshot = capture(source)
    started = time.time()
    feedback = {finest: rollout(source, snapshot, rows, anchor, horizon=horizon)}
    replay = {finest: feedback[finest]}
    for ss in steps[:-1]:
        env = make_profile(cfg, num_envs=n, substeps=ss, seed=seed)
        feedback[ss] = rollout(env, snapshot, rows, anchor, horizon=horizon)
        replay[ss] = rollout(env, snapshot, rows, anchor, horizon=horizon,
                             exact_actions=feedback[finest]["actions"])
    comparisons = []
    for kind, runs in (("fixed_feedback_law", feedback), ("identical_joint_action_replay", replay)):
        for ss in steps[:-1]:
            item = compare_trajectories(runs[ss], runs[finest], tolerance=Tolerances())
            comparisons.append(dict(kind=kind, pair="vs_finest", **item))
        for left, right in zip(steps[:-1], steps[1:]):
            item = compare_trajectories(runs[left], runs[right], tolerance=Tolerances())
            comparisons.append(dict(kind=kind, pair="adjacent", **item))
    summary = dict(profile=cfg, profile_sha256=profile_hash(cfg), cases=n, substeps=steps,
        horizon=horizon, physical_seconds=horizon * source.world.dt,
        initial_full_state_identical=True, actual_normal_forces_recorded=True,
        comparisons=comparisons, feedback_runs=[rollout_summary(feedback[ss], rows) for ss in steps],
        authority_full_horizon=_authority(feedback[finest], rows),
        authority_first5=_authority(feedback[finest], rows, min(5,horizon)),
        wall_seconds=time.time()-started)
    summary["qualification"] = qualification(summary)
    raw = dict(profile=cfg, rows=rows, snapshot=snapshot, anchor=anchor, feedback=feedback, replay=replay)
    return summary, raw


def profile_hash(profile):
    return hashlib.sha256(json.dumps(get_profile(profile), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def finalize_selection(development_path, validation_path, candidate_path, out):
    """Freeze coefficients and the cheapest validated numerical resolution.

    This consumes local physical diagnostics only. It cannot access a response
    model or composition result. The candidate was chosen on development before
    validation; material parameters cannot be changed at this stage.
    """
    paths = [Path(p) for p in (development_path, validation_path, candidate_path)]
    development, validation, candidate = [json.loads(p.read_text()) for p in paths]
    if development["split"] != "development" or validation["split"] != "validation":
        raise ValueError("Selection needs development and independent local validation")
    cfg = get_profile(candidate)
    dev = next(p for p in development["profiles"] if p["profile"] == cfg)
    val = next(p for p in validation["profiles"] if p["profile"] == cfg)
    if not dev["qualification"]["qualified"]:
        raise ValueError("Candidate did not pass development")
    resolutions = [qualification(val, nominal=n) for n in val["substeps"][:-1]]
    valid = [x for x in resolutions if x["qualified"]]
    if not valid:
        raise ValueError("Independent validation did not establish usable convergent physics")
    selected = min(valid, key=lambda x: x["nominal_substeps"])
    result = dict(profile=cfg, profile_sha256=profile_hash(cfg),
        selection_status="frozen_after_development_and_independent_validation",
        selected_before_wm_training=True, selected_before_composition_evaluation=True,
        nominal_substeps=selected["nominal_substeps"], reference_substeps=selected["finest_substeps"],
        selection_rule="Keep development-selected physics; select lowest numerical resolution meeting every local validation accuracy/authority/fidelity gate.",
        development_qualification=dev["qualification"], validation_qualifications=resolutions,
        qualification=selected, source_paths=[str(p) for p in paths],
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        unix_timestamp=time.time(), interpretation="Finite-grid local numerical evidence, not proof of all-state convergence or physical realism.")
    Path(out).write_text(json.dumps(result, indent=2)+"\n")
    return result


def provenance():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "vmas_pusht/physics_profiles.py", root / "vmas_pusht/scenario.py",
             root / "vmas_pusht/physical_abstraction.py", root / "vmas_pusht/abstraction_convergence.py",
             root / "vmas_pusht/rebrace.py", root / "vmas_pusht/snapshot.py"]
    return dict(git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        git_branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
        versions={key: importlib.metadata.version(key) for key in ("vmas", "torch", "numpy")},
        torch_threads=torch.get_num_threads(),
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=["dry_light", "dry_medium", "dry_lowrotation", "original"])
    parser.add_argument("--substeps", nargs="+", type=int, default=[80, 160])
    parser.add_argument("--split", choices=["development", "validation"], default="development")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("data/novelty/stable_physics"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prereg = args.out.with_name("stable_physics_preregistration.json")
    if not prereg.exists():
        prereg.write_text(json.dumps(preregistration(), indent=2)+"\n")
    report = dict(provenance=provenance(), split=args.split, screen=args.screen,
                  preregistration_path=str(prereg), profiles=[])
    rows = local_rows(args.split, screen=args.screen)
    for name in args.profiles:
        summary, raw = run_profile(name, rows=rows, substeps=args.substeps, horizon=args.horizon)
        raw_path = args.out.with_name(args.out.name+"_"+summary["profile"]["name"]+".pt")
        torch.save(raw, raw_path)
        summary["raw_path"] = str(raw_path)
        summary["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        report["profiles"].append(summary)
        args.out.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(dict(profile=name, qualification=summary["qualification"],
            authority=summary["authority_full_horizon"],
            errors=[dict(kind=c["kind"], xy=c["max_xy_units"], angle=c["max_angle_rad"])
                    for c in summary["comparisons"] if c["pair"] == "vs_finest"],
            wall_seconds=summary["wall_seconds"])), flush=True)


if __name__ == "__main__":
    main()
