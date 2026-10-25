"""Complete deterministic MuJoCo and plant-bookkeeping snapshots.

The integration state includes motor activations, controls, warm starts, applied
forces and simulation time as well as configuration/velocity coordinates.
Controller memory is owned and snapshotted by the caller, outside this plant.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass
class TransportSnapshot:
    version: int
    model_sha256: str
    config: dict
    state_spec: int
    state: np.ndarray
    control_steps: int
    physics_steps: int
    last_command: np.ndarray
    last_requested_command: np.ndarray
    last_external_forces: np.ndarray
    metrics: dict
    warning_number: np.ndarray
    warning_lastinfo: np.ndarray


def integration_state(env):
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    values = np.empty(mujoco.mj_stateSize(env.model, spec))
    mujoco.mj_getState(env.model, env.data, values, spec)
    return values


def physical_state(env):
    """Physical coordinates and rotor activation, excluding derived caches."""
    return np.concatenate((env.data.qpos, env.data.qvel, env.data.act)).copy()


def capture(env):
    return TransportSnapshot(version=1, model_sha256=env.model_hash,
        config=asdict(env.config),state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),
        state=integration_state(env),control_steps=env.control_steps,physics_steps=env.physics_steps,
        last_command=env.last_command.copy(),last_requested_command=env.last_requested_command.copy(),
        last_external_forces=env.last_external_forces.copy(),metrics=deepcopy(env.metrics),
        warning_number=env.data.warning.number.copy(),warning_lastinfo=env.data.warning.lastinfo.copy())


def restore(env, snapshot, *, allow_timestep_change=False):
    if snapshot.version != 1 or snapshot.state_spec != int(mujoco.mjtState.mjSTATE_INTEGRATION):
        raise ValueError("Unsupported integration-state snapshot version/specification")
    if env.model_hash != snapshot.model_sha256:
        original, current = deepcopy(snapshot.config), asdict(env.config)
        if allow_timestep_change:
            original.pop("dt");current.pop("dt")
        if not allow_timestep_change or original != current:
            raise ValueError("Snapshot model/configuration differs beyond an explicitly permitted timestep change")
    if snapshot.state.shape != (mujoco.mj_stateSize(env.model,snapshot.state_spec),):
        raise ValueError("Snapshot integration-state size does not match the model")
    mujoco.mj_setState(env.model,env.data,snapshot.state,snapshot.state_spec)
    mujoco.mj_forward(env.model,env.data)
    # Rebuild all derived caches without altering the recorded warmstart or
    # other integration inputs. The next step then sees exactly those inputs.
    mujoco.mj_setState(env.model,env.data,snapshot.state,snapshot.state_spec)
    env.control_steps = snapshot.control_steps
    env.physics_steps = (snapshot.control_steps*env.substeps if allow_timestep_change
                         else snapshot.physics_steps)
    env.last_command = snapshot.last_command.copy()
    env.last_requested_command = snapshot.last_requested_command.copy()
    env.last_external_forces = snapshot.last_external_forces.copy()
    env.metrics = deepcopy(snapshot.metrics)
    env.data.warning.number[:] = snapshot.warning_number
    env.data.warning.lastinfo[:] = snapshot.warning_lastinfo
    return env.observe()
