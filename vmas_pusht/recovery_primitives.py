"""Finite-force, body-relative reverse-contact options for simulator MPC.

The primary changes contact through an exterior route before pushing. Every
transition consumes ordinary environment steps; no function writes body state.
The helper command is supplied by the caller so its frozen support law and
anchor need not change. These are restricted contact options, not a complete
manipulation planner or a proof that any failed state is unrecoverable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from vmas.simulator.physics import _get_closest_point_box, _get_inner_point_box
from vmas.simulator.utils import LINE_MIN_DIST

from vmas_pusht.boundary import _rotate
from vmas_pusht.physical_abstraction import norm_bound


PRIMITIVES = ("reverse_upper", "reverse_lower", "brake", "stay")
ALIASES = {"upper_right": "reverse_upper", "lower_left": "reverse_lower"}


@dataclass(frozen=True)
class PrimitiveConfig:
    kp: float = 30.
    kd: float = 4.
    waypoint_tolerance: float = .025
    waypoint_relative_speed: float = .16
    contact_tolerance: float = .025
    contact_relative_speed: float = .12
    contact_force_minimum: float = .001
    contact_confirm_steps: int = 2
    acquisition_limit: int = 45
    brake_gain: float = 8.


@dataclass
class PrimitiveState:
    """Owned batched memory; clone(indices) branches/repeats source rows."""
    name: str
    mode: torch.Tensor
    strength: torch.Tensor
    phase: torch.Tensor
    elapsed: torch.Tensor
    dwell: torch.Tensor
    initial_local: torch.Tensor
    admissible_start: torch.Tensor
    failed: torch.Tensor
    first_push_step: torch.Tensor
    config: PrimitiveConfig

    def clone(self, indices=None):
        def copy(value):
            if indices is None:
                return value.clone()
            idx = torch.as_tensor(indices, dtype=torch.long, device=value.device)
            return value.index_select(0, idx).clone()
        return PrimitiveState(self.name, *[copy(getattr(self, field)) for field in
            ("mode", "strength", "phase", "elapsed", "dwell", "initial_local", "admissible_start",
             "failed", "first_push_step")], self.config)


def start_primitive(env, name, *, strength=.6, config=None):
    names = [name] * env.num_envs if isinstance(name, str) else list(name)
    names = [ALIASES.get(n, n) for n in names]
    if len(names) != env.num_envs or any(n not in PRIMITIVES for n in names):
        raise ValueError("Expected one declared primitive name per environment")
    mode = torch.tensor([PRIMITIVES.index(n) for n in names], dtype=torch.long, device=env.device)
    c = config or PrimitiveConfig()
    if min(c.kp, c.kd, c.waypoint_tolerance, c.waypoint_relative_speed,
           c.contact_tolerance, c.contact_relative_speed, c.brake_gain) <= 0:
        raise ValueError("Controller gains and tolerances must be positive")
    if c.contact_confirm_steps < 1 or c.acquisition_limit < c.contact_confirm_steps:
        raise ValueError("Invalid finite acquisition budget")
    strength = torch.as_tensor(strength, dtype=torch.float32, device=env.device).expand(env.num_envs).clone()
    if not torch.isfinite(strength).all() or (strength < 0).any() or (strength > 1).any():
        raise ValueError("Push strength must be finite in [0,1]")
    bar, main = env.scenario.bar, env.world.agents[0]
    local = _rotate(main.state.pos-bar.state.pos, -bar.state.rot[:, 0]).clone()
    # The upper half-space or left exterior admits the declared routes. An
    # unexpected lower/right start is retained as a failed option and brakes;
    # it is never silently projected into a convenient initial contact.
    admissible = (local[:, 1] >= .095) | (local[:, 0] <= -.27)
    admissible |= mode >= 2
    zeros = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    phase = zeros.clone()
    phase[mode == 3] = 3
    return PrimitiveState(names[0] if len(set(names)) == 1 else "mixed", mode, strength, phase, zeros.clone(), zeros.clone(),
        local, admissible, ~admissible, torch.full_like(zeros, -1), c)


def _target_local(state):
    n = len(state.phase)
    targets = state.initial_local.new_zeros(n, 2)
    # Withdraw vertically above the current contact; a start already left
    # of the entire body uses the left exterior instead of a diagonal cut.
    x = torch.where(state.initial_local[:, 1] >= .095,
                    state.initial_local[:, 0], state.initial_local.new_full((n,), -.29))
    upper_x = torch.where(state.phase == 0, x, .13)
    upper_y = torch.where(state.phase < 2, .20, .101)
    lower_x = torch.where(state.phase < 2, -.29, -.13)
    lower_y = torch.where(state.phase == 0, .16, torch.where(state.phase == 1, -.12, -.101))
    targets[:, 0] = torch.where(state.mode == 1, lower_x, upper_x)
    targets[:, 1] = torch.where(state.mode == 1, lower_y, upper_y)
    return targets


def target_state(env, state):
    local = _target_local(state)
    bar = env.scenario.bar
    offset = _rotate(local, bar.state.rot[:, 0])
    target = bar.state.pos + offset
    velocity = bar.state.vel + bar.state.ang_vel * torch.stack((-offset[:, 1], offset[:, 0]), 1)
    return target, velocity


def primary_contact_forces(env):
    """Actual VMAS sphere/box normal force on the primary, bar then stem."""
    main = env.world.agents[0]
    forces = []
    for part in (env.scenario.bar, env.scenario.stem):
        closest = _get_closest_point_box(part.state.pos, part.state.rot,
            part.shape.width, part.shape.length, main.state.pos)
        inner, distance = _get_inner_point_box(main.state.pos, closest, part.state.pos)
        force, _ = env.world._get_constraint_forces(main.state.pos, inner,
            dist_min=main.shape.radius+LINE_MIN_DIST+distance,
            force_multiplier=env.world._collision_force)
        if not main.collide or not part.collide:
            force = torch.zeros_like(force)
        forces.append(force)
    return torch.stack(forces, 1)


def route_control(env, state):
    """Return (navigation force, ready mask), without choosing a task push.

    At ready contacts (including ``stay``), navigation force is zero so MPC can
    replace it with the same candidate bank. Failed routes explicitly brake.
    This call reads simulator state; call advance_primitive after env.step.
    """
    if len(state.phase) != env.num_envs:
        raise ValueError("Primitive memory must match the full environment batch")
    main, bar, c = env.world.agents[0], env.scenario.bar, state.config
    brake = norm_bound(-c.brake_gain * main.state.vel)
    target, velocity = target_state(env, state)
    command = norm_bound(c.kp*(target-main.state.pos)+c.kd*(velocity-main.state.vel))
    ready = (state.phase >= 3) & ~state.failed & (state.mode != 2)
    command = torch.where(ready[:, None], torch.zeros_like(command), command)
    command = torch.where((state.failed | (state.mode == 2))[:, None], brake, command)
    return command, ready


def orient_push(env, state, world_down_actions):
    """Orient a body-down bank at the acquired face; lower contact pushes up.

    Input may be B,2 or B,K,2 and is normally the caller's existing bank13
    world-frame forces. Norms/strengths/direction offsets remain unchanged.
    """
    if world_down_actions.ndim not in (2, 3) or world_down_actions.shape[0] != env.num_envs or world_down_actions.shape[-1] != 2:
        raise ValueError("Expected world-frame downward bank of shape B,2 or B,K,2")
    sign = torch.where(state.mode == 1, -1., 1.).to(world_down_actions)
    return world_down_actions * sign.reshape((-1,) + (1,) * (world_down_actions.ndim-1))


def primary_action(env, state, *, strength=None, direction_deg=0.):
    """Convenience rollout law: navigate, then a fixed declared normal push."""
    command, ready = route_control(env, state)
    force = state.strength if strength is None else torch.as_tensor(strength, device=env.device).expand(env.num_envs)
    if not torch.isfinite(force).all() or (force < 0).any() or (force > 1).any():
        raise ValueError("Push strength must be finite in [0,1]")
    direction = env.scenario.bar.state.rot[:, 0] + torch.as_tensor(direction_deg, device=env.device) * torch.pi/180.
    down = torch.stack((direction.sin(), -direction.cos()), 1) * force[:, None]
    push = orient_push(env, state, down)
    return torch.where(ready[:, None], push, command)


def primitive_action(env, state, helper_command):
    """Joint command with the caller's unchanged helper support action."""
    if helper_command.shape != (env.num_envs, 2) or not torch.isfinite(helper_command).all():
        raise ValueError("Helper force must be a finite (batch,2) tensor")
    if (helper_command.norm(dim=-1) > 1.000001).any():
        raise ValueError("Helper force exceeds unit Euclidean norm")
    return torch.stack((primary_action(env, state), helper_command), 1)


def advance_primitive(env, state, active_mask=None):
    """Update only option memory after one genuine simulator step.

    A route transitions on physical proximity/speed; the contact transition also
    requires the actual bar normal force for consecutive steps. Acquisition
    timeout brakes and remains a recorded failure. Push phase persists until the
    caller chooses another option, including the explicit brake option.
    """
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device) if active_mask is None else active_mask.bool()
    if active.shape != state.phase.shape:
        raise ValueError("active_mask must match the full environment batch")
    state.elapsed[active] += 1
    main, c = env.world.agents[0], state.config
    target, velocity = target_state(env, state)
    distance = (main.state.pos-target).norm(dim=-1)
    speed = (main.state.vel-velocity).norm(dim=-1)
    moving_route = state.mode < 2
    move = active & ~state.failed & (state.phase < 2) & moving_route
    reached = move & (distance < c.waypoint_tolerance) & (speed < c.waypoint_relative_speed)
    # Capture the original phase so a stage advances at most once per step.
    contact_stage = active & ~state.failed & (state.phase == 2) & moving_route
    normal_force = primary_contact_forces(env)[:, 0].norm(dim=-1)
    contacting = contact_stage & (distance < c.contact_tolerance) & (speed < c.contact_relative_speed)
    contacting &= normal_force >= c.contact_force_minimum
    state.dwell[contact_stage] = torch.where(contacting[contact_stage], state.dwell[contact_stage]+1, 0)
    acquired = contact_stage & (state.dwell >= c.contact_confirm_steps)
    state.phase[reached] += 1
    state.phase[acquired] = 3
    state.first_push_step[acquired & (state.first_push_step < 0)] = state.elapsed[acquired & (state.first_push_step < 0)]
    state.failed |= active & moving_route & (state.elapsed >= c.acquisition_limit) & (state.phase < 3)
    return state


def configuration(config=None):
    return dict(controllers=PRIMITIVES, **asdict(config or PrimitiveConfig()),
        primary_observation="primary pose/velocity; bar pose/velocity; actual primary-bar contact force",
        helper="Caller supplies unchanged bounded support command and its existing anchor.",
        reverse_upper_waypoints="(initial x,.20), (.13,.20), (.13,.101), body-down force",
        reverse_lower_waypoints="(-.29,.16), (-.29,-.12), (-.13,-.101), body-up force",
        admissible_start="primary local y>=.095 or x<=-.27; other starts brake and fail, without dropping rows",
        no_simulator_state_writes=True, all_acquisition_steps_consume_real_budget=True)
