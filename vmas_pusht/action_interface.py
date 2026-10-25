"""Finite-force atomic options with explicit contact modes and real durations.

Options contain (route_id, direction_degrees, strength, pushing_steps). The
route is either continue=0, upper_right=1 or lower_left=2. Contact-mode IDs
match recovery_primitives: original left/top=3, upper/right=0, lower/left=1.
An exterior route may be started only from the original face. Every navigation,
confirmation, pushing, and failed-acquisition braking step is physically
executed and counted. This interface never reads a task goal or writes body
state; its caller owns the episode budget and task termination conditions.
"""
from __future__ import annotations

from dataclasses import fields
import itertools
import math

import torch

from vmas_pusht.boundary import _rotate
from vmas_pusht.physical_abstraction import helper_action, model_features
from vmas_pusht.recovery_primitives import (PrimitiveState, advance_primitive,
    route_control, start_primitive)


ROUTE_NAMES = ("continue", "upper_right", "lower_left")
SUPPORTS = ("passive", "world_anchor", "preloaded_anchor")
MODE_ORDER = (3, 0, 1)


def candidate_table(expanded=True, *, durations=(5, 10, 20),
                    directions=(-20., 0., 20.), strengths=(.15, .3, .6, .9)):
    """Return K,4 options, with one zero push per route/duration.

    The first 39 default expanded entries are exactly the direct-action table.
    Additional directions/durations support separately declared heldout probes.
    """
    durations, directions, strengths = tuple(durations), tuple(directions), tuple(strengths)
    if not durations or any(not math.isfinite(x) or x < 1 or x != int(x) for x in durations):
        raise ValueError("Pushing durations must be positive integers")
    if any(not math.isfinite(x) for x in directions):
        raise ValueError("Directions must be finite")
    if any(not math.isfinite(x) or not 0 < x <= 1 for x in strengths):
        raise ValueError("Nonzero candidate strengths must be in (0,1]")
    if any(len(x) != len(set(x)) for x in (durations, directions, strengths)):
        raise ValueError("Candidate factors must not contain duplicates")
    pushes = [(0., 0.)] + list(itertools.product(directions, strengths))
    return torch.tensor([(route, direction, strength, duration)
        for route in range(3 if expanded else 1) for duration in durations
        for direction, strength in pushes], dtype=torch.float32)


def _mode(env, mode):
    values = torch.as_tensor(mode, device=env.device)
    if values.ndim == 0:
        values = values.expand(env.num_envs)
    if values.shape != (env.num_envs,) or not ((values == 3) | (values == 0) | (values == 1)).all():
        raise ValueError("Expected one contact mode (3,0,1) per environment")
    return values.long()


def _structural_valid(options):
    return (torch.isfinite(options).all(-1)
        & ((options[..., 0] == 0) | (options[..., 0] == 1) | (options[..., 0] == 2))
        & (options[..., 2] >= 0) & (options[..., 2] <= 1)
        & (options[..., 3] >= 1) & (options[..., 3] == options[..., 3].floor()))


def _options(env, options, *, table):
    values = torch.as_tensor(options, dtype=env.scenario.bar.state.pos.dtype, device=env.device)
    if table and values.ndim == 2 and values.shape[-1] == 4:
        values = values[None].expand(env.num_envs, -1, -1)
    elif not table and values.ndim == 1 and values.shape == (4,):
        values = values[None].expand(env.num_envs, -1)
    expected = 3 if table else 2
    if values.ndim != expected or values.shape[0] != env.num_envs or values.shape[-1] != 4:
        raise ValueError("Expected K,4 or B,K,4 table" if table else "Expected B,4 options")
    return values


def valid_options(env, mode, options, *, state=None):
    """Return B,K immediate option legality, using current geometry only.

    This is not a predicted-success gate. It checks the declared face, route
    precondition and command bounds. It does not promise sustained contact.
    Online callers must supply the interface state: failed/incomplete routes
    retain their unknown contact status and permit only direct zero drive.
    """
    mode = _mode(env, mode)
    options = _options(env, options, table=True)
    local = _rotate(env.world.agents[0].state.pos-env.scenario.bar.state.pos,
                    -env.scenario.bar.state.rot[:, 0])
    route_start = (local[:, 1] >= .095) | (local[:, 0] <= -.27)
    switching = options[..., 0] != 0
    result = (_structural_valid(options)
        & (~switching | ((mode == 3) & route_start)[:, None]))
    if state is not None:
        if state.phase.shape != mode.shape or not torch.equal(state.mode, mode):
            raise ValueError("Route memory and current contact modes do not match")
        contact_ready = (state.phase >= 3) & ~state.failed
        safe_zero = (options[..., 0] == 0) & (options[..., 2] == 0)
        result &= contact_ready[:, None] | safe_zero
    return result


def _target_mode(mode, routes):
    return torch.where(routes == 1, 0, torch.where(routes == 2, 1, mode))


def model_inputs(env, anchor, mode, options, *, reference_substeps=160):
    """Current-state option features; no goal or future observation is read.

    B,4 options return B,24 and B,38; B,K,4 return B,K,24 and B,K,38.
    Their first 18/32 coordinates retain physical_abstraction.model_features'
    original layout. The last six values encode current face (left,upper,lower)
    and requested route (continue,upper,lower). Pushing duration excludes route
    time; the actual total option duration is a separate training target.
    """
    mode = _mode(env, mode)
    value = torch.as_tensor(options, dtype=env.scenario.bar.state.pos.dtype, device=env.device)
    is_single = value.ndim == 2
    if is_single:
        value = _options(env, value, table=False)[:, None]
    else:
        value = _options(env, value, table=True)
    if not _structural_valid(value).all():
        raise ValueError("Model inputs require well-formed bounded options")
    if not math.isfinite(reference_substeps) or reference_substeps <= 0:
        raise ValueError("Reference resolution must be finite and positive")
    anchor = torch.as_tensor(anchor, dtype=value.dtype, device=env.device)
    if anchor.shape != (env.num_envs, 2) or not torch.isfinite(anchor).all():
        raise ValueError("Expected finite B,2 helper anchors")
    n, k = value.shape[:2]
    c, f = model_features(env, torch.zeros(n, 2, device=env.device), 1, anchor)
    c, f = c[:, None].expand(n, k, -1).clone(), f[:, None].expand(n, k, -1).clone()
    routes = value[..., 0].long()
    target_mode = _target_mode(mode[:, None], routes)
    angle = env.scenario.bar.state.rot[:, 0, None] + value[..., 1] * math.pi / 180.
    force = torch.stack((angle.sin(), -angle.cos()), -1) * value[..., 2, None]
    force *= torch.where(target_mode == 1, -1., 1.)[..., None]
    for inputs in (c, f):
        inputs[..., 12:14] = force
        inputs[..., 14] = value[..., 3] * env.world.dt
        inputs[..., 15] = float(reference_substeps)
    face = torch.where(mode == 3, 0, torch.where(mode == 0, 1, 2))
    extra = torch.cat((torch.nn.functional.one_hot(face, 3)[:, None].expand(n, k, 3),
                       torch.nn.functional.one_hot(routes, 3)), -1).to(value.dtype)
    coarse, full = torch.cat((c, extra), -1), torch.cat((f, extra), -1)
    return (coarse[:, 0], full[:, 0]) if is_single else (coarse, full)


def _copy_state(target, source, mask):
    for field in fields(PrimitiveState):
        value = getattr(target, field.name)
        if torch.is_tensor(value):
            value[mask] = getattr(source, field.name)[mask]


class ActionInterface:
    """Own option memory while the caller performs every simulator step.

    Call start once per selected option, command before each real env.step, and
    observe immediately afterward. Idle/completed rows emit zero joint drive.
    Geometry-inadmissible route attempts brake for one real step and report
    failure. Unsupported mode switches and malformed commands are rejected.
    """
    _tensor_fields = ("anchor", "busy", "done", "failed", "cancelled", "elapsed",
        "pushed_steps", "options", "held_force", "push_started", "safe_idle", "_pending",
        "_pending_ready", "_pending_steps")

    def __init__(self, env, supports, anchor, *, config=None):
        self.env = env
        self.supports = [supports] * env.num_envs if isinstance(supports, str) else list(supports)
        if len(self.supports) != env.num_envs or any(s not in SUPPORTS for s in self.supports):
            raise ValueError("Expected one declared support mode per environment")
        self.anchor = torch.as_tensor(anchor, dtype=env.scenario.bar.state.pos.dtype,
                                      device=env.device).clone()
        if self.anchor.shape != (env.num_envs, 2) or not torch.isfinite(self.anchor).all():
            raise ValueError("Expected finite B,2 helper anchors")
        self.state = start_primitive(env, "stay", config=config)
        n, dev = env.num_envs, env.device
        for name in ("busy", "done", "failed", "cancelled", "push_started", "safe_idle", "_pending", "_pending_ready"):
            setattr(self, name, torch.zeros(n, dtype=torch.bool, device=dev))
        self.elapsed = torch.zeros(n, dtype=torch.long, device=dev)
        self.pushed_steps = torch.zeros_like(self.elapsed)
        self.options = torch.zeros(n, 4, device=dev)
        self.held_force = torch.zeros(n, 2, device=dev)
        self._pending_steps = env.steps.clone()

    @property
    def mode(self):
        return self.state.mode

    @property
    def completed(self):
        return self.done & ~self.failed & ~self.cancelled

    @property
    def contact_ready(self):
        return (self.state.phase >= 3) & ~self.state.failed

    def valid_options(self, options):
        return valid_options(self.env, self.mode, options, state=self.state)

    def _mask(self, mask):
        if mask is None:
            return torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.env.device)
        if mask.shape != (self.env.num_envs,):
            raise ValueError("Expected a B-element mask")
        return mask

    def start(self, options, mask=None):
        mask = self._mask(mask)
        value = _options(self.env, options, table=False)
        if not _structural_valid(value)[mask].all():
            raise ValueError("Malformed or unbounded option")
        if (self.busy & mask).any() or (self._pending & mask).any():
            raise ValueError("Cancel or complete an active option before replacing it")
        switching = value[:, 0] != 0
        if (mask & switching & (self.mode != 3)).any():
            raise ValueError("A second contact-side switch is outside this interface")
        unavailable = ~self.contact_ready
        safe_idle = ~switching & (value[:, 2] == 0) & unavailable
        if (mask & ~switching & unavailable & ~safe_idle).any():
            raise ValueError("Cannot continue a contact face before acquisition succeeds")
        if (mask & switching).any():
            names = ["upper_right" if bool(mask[i]) and int(value[i, 0]) == 1 else
                     "lower_left" if bool(mask[i]) and int(value[i, 0]) == 2 else "stay"
                     for i in range(self.env.num_envs)]
            fresh = start_primitive(self.env, names, config=self.state.config)
            _copy_state(self.state, fresh, mask & switching)
        self.options[mask] = value[mask]
        self.safe_idle[mask] = safe_idle[mask]
        self.state.strength[mask] = value[mask, 2]
        self.busy[mask] = True
        for field in ("done", "failed", "cancelled", "push_started", "_pending", "_pending_ready"):
            getattr(self, field)[mask] = False
        self.elapsed[mask] = 0
        self.pushed_steps[mask] = 0
        self.held_force[mask] = 0
        return self.status()

    def command(self, active_mask=None):
        active = self.busy & self._mask(active_mask)
        # Repeated reads before the same physics step are allowed; skipping an
        # observation after stepping is not, since it would erase elapsed time.
        if (active & self._pending & (self.env.steps != self._pending_steps)).any():
            raise RuntimeError("Observe the previous real step before issuing another command")
        navigation, ready = route_control(self.env, self.state)
        beginning = active & ready & ~self.safe_idle & ~self.push_started
        angle = self.env.scenario.bar.state.rot[:, 0] + self.options[:, 1] * math.pi / 180.
        force = torch.stack((angle.sin(), -angle.cos()), -1) * self.options[:, 2, None]
        force *= torch.where(self.mode == 1, -1., 1.)[:, None]
        self.held_force[beginning] = force[beginning]
        self.push_started[beginning] = True
        primary = torch.where(ready[:, None], self.held_force, navigation)
        primary[self.safe_idle] = 0.
        actuated = torch.tensor([s != "passive" for s in self.supports], device=self.env.device)
        helper = helper_action(self.env, "world_anchor", self.anchor) * actuated[:, None]
        self._pending[active] = True
        self._pending_ready[active] = (ready | self.safe_idle)[active]
        self._pending_steps[active] = self.env.steps[active]
        return torch.stack((primary, helper), 1) * active[:, None, None]

    def observe(self, active_mask=None):
        active = self._pending & self.busy if active_mask is None else self._mask(active_mask) & self.busy
        if (active & ~self._pending).any():
            raise RuntimeError("No command was issued for an observed active row")
        if not (self.env.steps[active] - self._pending_steps[active] == 1).all():
            raise RuntimeError("Each observation must follow exactly one real environment step")
        self.elapsed[active] += 1
        self.pushed_steps += (active & self._pending_ready).long()
        routed = active & ~self.safe_idle
        advance_primitive(self.env, self.state, active_mask=routed)
        self.failed[routed] |= self.state.failed[routed]
        newly_done = active & (self.failed | (self.pushed_steps >= self.options[:, 3]))
        self.done[newly_done] = True
        self.busy[newly_done] = False
        self._pending[active] = False
        self._pending_ready[active] = False
        return dict(self.status(), newly_done=newly_done.clone())

    def cancel(self, mask):
        active = self._mask(mask) & self.busy
        self.cancelled[active] = True
        self.done[active] = True
        self.busy[active] = False
        self._pending[active] = False
        self._pending_ready[active] = False
        return self.status()

    def reset_contact(self, mask, *, anchor=None, mode=3):
        """Reset memory after a caller-observed phase boundary, without placement.

        The declared face is bookkeeping, not a claim of sustained contact.
        A caller restoring an acquired-face snapshot may pass its known mode.
        """
        mask = self._mask(mask)
        mode = _mode(self.env, mode)
        new_anchor = self.anchor if anchor is None else torch.as_tensor(anchor,
            dtype=self.anchor.dtype, device=self.env.device)
        if new_anchor.shape == (2,):
            new_anchor = new_anchor[None].expand(self.env.num_envs, 2)
        if new_anchor.shape != self.anchor.shape or not torch.isfinite(new_anchor[mask]).all():
            raise ValueError("Expected finite selected helper anchors")
        fresh = start_primitive(self.env, "stay", config=self.state.config)
        fresh.mode[:] = mode
        _copy_state(self.state, fresh, mask)
        self.anchor[mask] = new_anchor[mask].clone()
        for name in ("busy", "done", "failed", "cancelled", "push_started", "safe_idle", "_pending", "_pending_ready"):
            getattr(self, name)[mask] = False
        self.elapsed[mask] = 0
        self.pushed_steps[mask] = 0
        self.options[mask] = 0
        self.held_force[mask] = 0
        return self.status()

    def status(self):
        return {name: getattr(self, name).clone() for name in
            ("busy", "done", "failed", "cancelled", "completed", "contact_ready", "elapsed", "pushed_steps", "mode")}

    def clone_to(self, env, indices=None):
        """Copy owned memory into an independently restored environment batch."""
        idx = torch.arange(self.env.num_envs, device=self.env.device) if indices is None else torch.as_tensor(
            indices, dtype=torch.long, device=self.env.device)
        if idx.ndim != 1 or len(idx) != env.num_envs or ((idx < 0) | (idx >= self.env.num_envs)).any():
            raise ValueError("Expected one valid source-memory index per target environment")
        result = ActionInterface(env, [self.supports[i] for i in idx.tolist()], self.anchor[idx].to(env.device),
                                 config=self.state.config)
        result.state = self.state.clone(idx)
        for field in fields(PrimitiveState):
            value = getattr(result.state, field.name)
            if torch.is_tensor(value):
                setattr(result.state, field.name, value.to(env.device).clone())
        for name in self._tensor_fields:
            setattr(result, name, getattr(self, name)[idx].to(env.device).clone())
        return result
