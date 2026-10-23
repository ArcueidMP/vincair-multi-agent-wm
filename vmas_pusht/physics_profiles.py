"""Explicit finite-force physics profiles for the numerical stability audit.

The ordinary scenario's defaults are untouched. ``original`` reproduces them
exactly. Other profiles only alter declared material/inertial parameters before
any rollout. No body is clamped, teleported, made static, or given extra force.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from vmas_pusht.scenario import make


PROFILES = {
    "original": dict(name="original", scenario=dict(t_mass=2., friction_ratio=1.5,
        rot_friction_ratio=1.2, agent_force=1.), world={}),
    "dry_light": dict(name="dry_light", scenario=dict(t_mass=2., friction_ratio=.25,
        rot_friction_ratio=.10, agent_force=1.), world={}),
    "dry_medium": dict(name="dry_medium", scenario=dict(t_mass=2., friction_ratio=.5,
        rot_friction_ratio=.15, agent_force=1.), world={}),
    "dry_lowrotation": dict(name="dry_lowrotation", scenario=dict(t_mass=2., friction_ratio=.5,
        rot_friction_ratio=.05, agent_force=1.), world={}),
    # Prespecified fallback: finite inertia and the unchanged VMAS per-env-step
    # drag remove Coulomb sign switching without any kinematic fixture.
    "damped_heavy6": dict(name="damped_heavy6", scenario=dict(t_mass=6., friction_ratio=0.,
        rot_friction_ratio=0., agent_force=1.), world=dict(torque_constraint_force=5.)),
    "damped_heavy12": dict(name="damped_heavy12", scenario=dict(t_mass=12., friction_ratio=0.,
        rot_friction_ratio=0., agent_force=1.), world=dict(torque_constraint_force=5.)),
}


def get_profile(profile="original"):
    """Load a named profile, profile dictionary, or saved JSON selection.

    A selection JSON may wrap the profile under ``profile``. A fresh owned
    dictionary is returned, never a mutable reference into the profile registry.
    """
    if isinstance(profile, (str, Path)):
        key = str(profile)
        if key in PROFILES:
            profile = PROFILES[key]
        else:
            profile = json.loads(Path(key).read_text())
    if not isinstance(profile, dict):
        raise ValueError("Profile must be a name, JSON path, or dictionary")
    if "profile" in profile:
        profile = profile["profile"]
    result = deepcopy(profile)
    if set(result) != {"name", "scenario", "world"}:
        raise ValueError("Profile requires exactly name, scenario and world")
    if not isinstance(result["name"], str) or not result["name"]:
        raise ValueError("Profile requires a nonempty name")
    if set(result["scenario"]) != {"t_mass", "friction_ratio", "rot_friction_ratio", "agent_force"}:
        raise ValueError("Only declared material, mass and force parameters are allowed")
    if set(result["world"]) - {"torque_constraint_force"}:
        raise ValueError("Only finite joint torque stiffness may be overridden")
    for key, value in result["scenario"].items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Profile parameters must be finite and nonnegative")
    if result["scenario"]["t_mass"] <= 0 or result["scenario"]["agent_force"] != 1.:
        raise ValueError("Finite positive mass and unchanged unit agent force are required")
    for value in result["world"].values():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Joint stiffness must be finite and positive")
    return result


def make_profile(profile="original", num_envs=32, substeps=5, seed=0, device="cpu", max_steps=None):
    """Build a profile before rollout; dt=.1 and finite dynamic bodies remain."""
    cfg = get_profile(profile)
    env = make(num_envs=num_envs, substeps=substeps, seed=seed, device=device,
               max_steps=max_steps, **cfg["scenario"])
    for key, value in cfg["world"].items():
        setattr(env.world, "_" + key, value)
    env.physics_profile = deepcopy(cfg)
    return env
