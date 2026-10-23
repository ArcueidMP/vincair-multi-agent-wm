"""Exact, detached snapshots for this deterministic VMAS PushT scenario.

The 15D learning state omits the stem's independent physical state, agent
rotation, and reward/time bookkeeping. It is not a simulator checkpoint. This
module saves those fields and permits batched replay with repeated source rows.

Snapshots require the same fixed physics, topology and task configuration. They
support noiseless, unscripted Holonomic agents in PushT, not arbitrary VMAS
scenarios or stochastic replay: RNG state is deliberately not saved. World force
accumulators are recomputed on every physics substep; joint anchor caches contain
only fixed geometry. Neither is a hidden integration state. Exact numerical
replay is tested with matching device/backend and rollout batch configuration.
Reordered/expanded snapshots preserve every stored state value, but changing
batch size or batch companions can change VMAS's batched collision calculations
and floating-point roundoff. Cross-batch or cross-device bit equality is not
promised; audit these separately for sensitive contact trajectories.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from vmas.simulator.core import Agent
from vmas.simulator.dynamics.holonomic import Holonomic

from vmas_pusht.scenario import PushT


@dataclass(frozen=True)
class EntitySnapshot:
    name: str
    state: dict[str, Optional[torch.Tensor]]
    action: Optional[dict[str, Optional[torch.Tensor]]]


@dataclass(frozen=True)
class SimulationSnapshot:
    """A torch.save-compatible snapshot whose tensors own their storage.

    Treat saved tensors as read-only. Restoring always copies them and cannot
    mutate either this snapshot or the original environment through aliasing.
    """

    batch_size: int
    physics: tuple
    entities: tuple[EntitySnapshot, ...]
    scenario: dict[str, Optional[torch.Tensor]]
    steps: torch.Tensor


def _freeze(value):
    """Convert small configuration values to comparable, serializable tuples."""
    if torch.is_tensor(value):
        return (str(value.dtype), tuple(value.shape), _freeze(value.detach().cpu().tolist()))
    if isinstance(value, dict):
        return tuple((key, _freeze(val)) for key, val in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(val) for val in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported physics configuration value: {type(value).__name__}")


def _values(obj, names):
    return tuple((name, _freeze(getattr(obj, name))) for name in names)


def _configuration(env):
    if type(env.scenario) is not PushT:
        raise ValueError("Snapshots support the deterministic PushT scenario only")
    for agent in env.world.agents:
        if (type(agent.dynamics) is not Holonomic or agent.action_script is not None
                or torch.as_tensor(agent.action.u_noise).ne(0).any()
                or agent.c_noise != 0 or agent.obs_noise != 0):
            raise ValueError("Snapshots require noiseless, unscripted Holonomic agents")
    if env.world.dim_c != 0:
        raise ValueError("Snapshots require PushT without communication state")
    world = env.world
    entity_config = []
    for entity in world.entities:
        config = (
            type(entity).__name__, entity.name,
            type(entity.shape).__name__, _freeze(vars(entity.shape)),
            _values(entity, ("movable", "rotatable", "collide", "mass", "max_speed", "v_range",
                             "drag", "linear_friction", "angular_friction", "gravity")),
        )
        if isinstance(entity, Agent):
            config += (
                _values(entity, ("max_f", "f_range", "max_t", "t_range", "action_size", "silent")),
                _values(entity.action, ("u_range", "u_multiplier", "u_noise")),
            )
        entity_config.append(config)
    joint_config = tuple(
        (joint.entity_a.name, joint.entity_b.name,
         _values(joint, ("anchor_a", "anchor_b", "dist", "rotate", "fixed_rotation")))
        for joint in world.joints
    )
    collision_filters = tuple(
        bool(a.collision_filter(b)) for a in world.entities for b in world.entities
    )
    return (
        _values(env, ("max_steps", "continuous_actions", "clamp_action", "terminated_truncated")),
        _values(env.scenario, ("n_agents", "t_mass", "friction_ratio", "rot_friction_ratio",
                               "agent_force", "pos_tol", "rot_tol", "substeps", "semidim")),
        _values(world, ("_dt", "_substeps", "_sub_dt", "_dim_p", "_dim_c", "_x_semidim", "_y_semidim",
                        "_drag", "_gravity", "_linear_friction", "_angular_friction", "_collision_force",
                        "_joint_force", "_contact_margin", "_torque_constraint_force")),
        tuple(entity_config), joint_config, collision_filters,
    )


def _copy(value):
    return None if value is None else value.detach().clone()


def capture(env) -> SimulationSnapshot:
    """Capture all dynamic PushT state without sharing tensors with ``env``."""
    physics = _configuration(env)
    entities = []
    for entity in env.world.entities:
        fields = ["pos", "vel", "rot", "ang_vel"]
        action = None
        if isinstance(entity, Agent):
            fields += ["force", "torque", "c"]
            action = {key: _copy(getattr(entity.action, key)) for key in ("u", "c")}
        entities.append(EntitySnapshot(entity.name, {key: _copy(getattr(entity.state, key)) for key in fields}, action))
    task = {key: _copy(getattr(env.scenario, key, None)) for key in ("goal_pose", "prev_err", "rew")}
    return SimulationSnapshot(env.world.batch_dim, physics, tuple(entities), task, _copy(env.steps))


def restore(env, snapshot: SimulationSnapshot, indices=None):
    """Restore selected source rows into the entire target environment batch.

    ``indices[j]`` selects the snapshot row for target row ``j``; repeats and
    reordering are supported. Without indices, a singleton snapshot broadcasts,
    or equal-size batches map one to one. Other batch sizes require an explicit
    map. Configuration/indices are validated before any environment mutation.
    """
    if _configuration(env) != snapshot.physics:
        raise ValueError("Snapshot and target must have identical physics, topology and task configuration")
    batch_size = env.world.batch_dim
    if indices is None:
        if snapshot.batch_size == 1:
            indices = torch.zeros(batch_size, dtype=torch.long)
        elif snapshot.batch_size == batch_size:
            indices = torch.arange(batch_size)
        else:
            raise ValueError("Different batch sizes require explicit source indices")
    else:
        indices = torch.as_tensor(indices)
        if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
            raise ValueError("Snapshot indices must be integers")
        indices = indices.to(dtype=torch.long)
    if indices.ndim != 1 or indices.numel() != batch_size:
        raise ValueError("Snapshot indices must have one entry for every target environment")
    if bool(((indices < 0) | (indices >= snapshot.batch_size)).any()):
        raise ValueError("Snapshot source index is out of range")

    def selected(value):
        if value is None:
            return None
        return value.index_select(0, indices.to(value.device)).to(env.device).clone()

    for entity, saved in zip(env.world.entities, snapshot.entities):
        for key, value in saved.state.items():
            copied = selected(value)
            if copied is None:
                # VMAS 1.5.2 public tensor setters do not accept absent buffers.
                setattr(entity.state, "_" + key, None)
            else:
                setattr(entity.state, key, copied)
        if saved.action is not None:
            for key, value in saved.action.items():
                copied = selected(value)
                if copied is None:
                    setattr(entity.action, "_" + key, None)
                else:
                    setattr(entity.action, key, copied)
    for key, value in snapshot.scenario.items():
        copied = selected(value)
        if copied is None:
            if hasattr(env.scenario, key):
                delattr(env.scenario, key)
        else:
            setattr(env.scenario, key, copied)
    env.steps = selected(snapshot.steps)
