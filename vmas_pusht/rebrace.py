"""Finite-force, goal-independent recovery of the T's right concave contact.

Only geometric approach and contact confirmation are implemented here. No goal,
world-model prediction, or task progress is observed. The recovery coordinator
reads both agent poses/velocities and the T pose/velocities. The post-handoff
helper reads only its own position/velocity and its frozen anchor. An actuated helper's new fixed
PD reference contains a declared 0.01 inward preload: this is an interface
change for the old ``world_anchor`` reset model, not retraining that model.

The helper remains an ordinary finite-mass body. ``step`` returns bounded forces;
it never writes simulator state. A timeout is a failure. Contact at any previous
time is insufficient: readiness requires consecutive observations under the
exact frozen-anchor helper command subsequently used after handoff.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from vmas.simulator.physics import _get_closest_point_box, _get_inner_point_box
from vmas.simulator.utils import LINE_MIN_DIST

from vmas_pusht.boundary import _finite, _rotate, surface_gaps
from vmas_pusht.physical_abstraction import HELPER_LOCAL, PRIMARY_LOCAL, helper_action, norm_bound
from vmas_pusht.scenario import STEM_OFFSET, wrap


@dataclass(frozen=True)
class RebraceConfig:
    kp: float = 30.0
    kd: float = 4.0
    preload: float = .01
    # Geometry is a sanity check around the engine's collision shell. Force,
    # not a narrower arbitrary distance threshold, defines actual contact.
    gap_tolerance: float = LINE_MIN_DIST + 1e-6
    minimum_contact_force: float = .001
    penetration_limit: float = .006
    helper_region_tolerance: float = .025
    primary_position_tolerance: float = .02
    relative_speed_tolerance: float = .04
    raw_force_limit: float = .95
    weld_position_tolerance: float = .01
    weld_angle_tolerance: float = .08
    staging_tolerance: float = .025
    staging_speed_tolerance: float = .1
    confirm_steps: int = 5
    max_steps: int = 80


def contact_reference(env, local):
    bar = env.scenario.bar
    offset = _rotate(torch.as_tensor(local, dtype=bar.state.pos.dtype, device=env.device)
                     .expand(env.num_envs, 2), bar.state.rot[:, 0])
    target = bar.state.pos + offset
    velocity = bar.state.vel + bar.state.ang_vel * torch.stack((-offset[:, 1], offset[:, 0]), -1)
    return target, velocity


def main_holding_action(env, config=None):
    config = config or RebraceConfig()
    target, velocity = contact_reference(env, PRIMARY_LOCAL)
    agent = env.world.agents[0]
    return norm_bound(config.kp * (target - agent.state.pos) + config.kd * (velocity - agent.state.vel))


def support_action(env, anchor, supports, config=None):
    """Exact post-handoff helper law; passive always has zero actuator force."""
    config = config or RebraceConfig()
    act = helper_action(env, "world_anchor", anchor, kp=config.kp, kd=config.kd)
    mask = torch.tensor([s != "passive" for s in supports], device=env.device)
    return act * mask[:, None]


def readiness_from_metrics(metrics, config=None):
    """Pure instantaneous gate. Persistence is checked separately by the FSM."""
    c = config or RebraceConfig()
    gaps = metrics["helper_gaps"]
    return (metrics["finite"] & (gaps <= c.gap_tolerance).all(-1)
            & (metrics["helper_normal_forces"] >= c.minimum_contact_force).all(-1)
            & (gaps >= -c.penetration_limit).all(-1)
            & (metrics["helper_region_error"] <= c.helper_region_tolerance)
            & (metrics["primary_position_error"] <= c.primary_position_tolerance)
            & (metrics["primary_relative_speed"] <= c.relative_speed_tolerance)
            & (metrics["helper_relative_speed"] <= c.relative_speed_tolerance)
            & (metrics["raw_support_force_norm"] < c.raw_force_limit)
            & (metrics["weld_position_residual"] <= c.weld_position_tolerance)
            & (metrics["weld_angle_residual"] <= c.weld_angle_tolerance))


def helper_contact_forces(env):
    """Read-only VMAS 1.5.2 sphere/box normal forces at the current state.

    VMAS includes LINE_MIN_DIST=0.0066667 in its collision distance; a positive
    geometric surface gap can therefore carry real reaction force. Use the
    engine's closest/inner-point and constraint functions, rather than calling
    proximity alone 'contact'. This does not integrate or update force buffers.
    """
    helper = env.world.agents[1]
    forces = []
    for part in (env.scenario.bar, env.scenario.stem):
        closest = _get_closest_point_box(part.state.pos, part.state.rot,
                                         part.shape.width, part.shape.length, helper.state.pos)
        inner, distance = _get_inner_point_box(helper.state.pos, closest, part.state.pos)
        force, _ = env.world._get_constraint_forces(helper.state.pos, inner,
            dist_min=helper.shape.radius + LINE_MIN_DIST + distance,
            force_multiplier=env.world._collision_force)
        if not part.collide or not helper.collide:
            force = torch.zeros_like(force)
        forces.append(force)
    return torch.stack(forces, 1)


def readiness_metrics(env, anchor, supports, config=None):
    c = config or RebraceConfig()
    main, helper = env.world.agents
    bar, stem = env.scenario.bar, env.scenario.stem
    primary_target, primary_velocity = contact_reference(env, PRIMARY_LOCAL)
    helper_target, helper_velocity = contact_reference(env, HELPER_LOCAL)
    helper_local = _rotate(helper.state.pos - bar.state.pos, -bar.state.rot[:, 0])
    actuated = torch.tensor([s != "passive" for s in supports], device=env.device)
    raw_force = (c.kp * (anchor - helper.state.pos) - c.kd * helper.state.vel) * actuated[:, None]
    stem_offset = _rotate(bar.state.pos.new_tensor([0., STEM_OFFSET]).expand(env.num_envs, 2),
                          bar.state.rot[:, 0])
    metrics = {
        "finite": _finite(env),
        "helper_gaps": surface_gaps(env)[:, 1],
        "helper_normal_forces": helper_contact_forces(env).norm(dim=-1),
        "helper_local_position": helper_local,
        "helper_region_error": (helper_local - helper_local.new_tensor(HELPER_LOCAL)).norm(dim=-1),
        "primary_position_error": (main.state.pos - primary_target).norm(dim=-1),
        "primary_relative_speed": (main.state.vel - primary_velocity).norm(dim=-1),
        "helper_relative_speed": (helper.state.vel - helper_velocity).norm(dim=-1),
        "raw_support_force_norm": raw_force.norm(dim=-1),
        "weld_position_residual": (stem.state.pos - bar.state.pos - stem_offset).norm(dim=-1),
        "weld_angle_residual": wrap(stem.state.rot[:, 0] - bar.state.rot[:, 0]).abs(),
    }
    metrics["instantaneous_ready"] = readiness_from_metrics(metrics, c)
    return metrics


class RebraceController:
    """Batched controller: reset at recovery entry, step, simulate, then update.

    phase 0 stages clear of the concave corner; phase 1 acquires both contacts
    with a moving finite PD reference; phase 2 freezes that reference and checks
    actual post-handoff stability. ``ready`` is current, not a past-touch latch.
    ``accepted`` is the separate monotonic diagnostic for the first ready event.
    ``update(active_mask=...)`` leaves other rows' clocks and state untouched.
    """

    def __init__(self, env, supports, *, max_steps=80, confirm_steps=5, config=None):
        if len(supports) != env.num_envs or any(s not in ("passive", "world_anchor", "preloaded_anchor") for s in supports):
            raise ValueError("supports must name one supported helper law per environment")
        if config is not None and (max_steps != 80 or confirm_steps != 5):
            raise ValueError("Pass either config or explicit time budgets")
        self.config = config or RebraceConfig(max_steps=max_steps, confirm_steps=confirm_steps)
        if self.config.max_steps < 1 or self.config.confirm_steps < 1:
            raise ValueError("Recovery and persistence budgets must be positive")
        self.env, self.supports = env, tuple(supports)
        n, dev = env.num_envs, env.device
        self.anchor = torch.zeros(n, 2, device=dev)
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.elapsed = torch.zeros_like(self.phase)
        self.dwell = torch.zeros_like(self.phase)
        self.ready = torch.zeros(n, dtype=torch.bool, device=dev)
        self.accepted = self.ready.clone()
        self.failed = self.ready.clone()
        self.anchor_ood = torch.tensor([s == "world_anchor" for s in supports], device=dev)
        self.primary_staged = self.ready.clone()
        self.helper_staged = self.ready.clone()
        self.reset()

    def reset(self, mask=None):
        mask = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device) if mask is None else mask.bool()
        reference, _ = contact_reference(self.env, HELPER_LOCAL)
        self.anchor[mask] = reference[mask]
        self.phase[mask] = self.elapsed[mask] = self.dwell[mask] = 0
        self.ready[mask] = self.accepted[mask] = self.failed[mask] = False
        # Primary has a safe direct path if already above the bar. Otherwise
        # stage outside its left end before proceeding to the top contact.
        bar = self.env.scenario.bar
        main_local = _rotate(self.env.world.agents[0].state.pos - bar.state.pos, -bar.state.rot[:, 0])
        helper_local = _rotate(self.env.world.agents[1].state.pos - bar.state.pos, -bar.state.rot[:, 0])
        self.primary_staged[mask] = (main_local[:, 1] > .095)[mask]
        self.helper_staged[mask] = ((helper_local[:, 0] > .095) & (helper_local[:, 1] < -.095)
                                    & (helper_local.norm(dim=-1) < .24))[mask]

    def _pd(self, agent_index, local):
        target, velocity = contact_reference(self.env, local)
        agent = self.env.world.agents[agent_index]
        return norm_bound(self.config.kp * (target - agent.state.pos)
                          + self.config.kd * (velocity - agent.state.vel))

    def step(self):
        c = self.config
        main = torch.where(self.primary_staged[:, None], main_holding_action(self.env, c),
                           self._pd(0, (-.29, .17)))
        acquisition = (HELPER_LOCAL[0] - c.preload, HELPER_LOCAL[1] + c.preload)
        helper = torch.where(self.helper_staged[:, None], self._pd(1, acquisition),
                             self._pd(1, (.18, -.18)))
        helper = torch.where((self.phase == 2)[:, None], support_action(self.env, self.anchor, self.supports, c), helper)
        return torch.stack((main, helper), 1)

    def update(self, active_mask=None):
        env, c = self.env, self.config
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device) if active_mask is None else active_mask.bool()
        active = active & ~self.failed
        self.elapsed[active] += 1
        for agent_index, local, staged in ((0, (-.29, .17), self.primary_staged), (1, (.18, -.18), self.helper_staged)):
            target, velocity = contact_reference(env, local)
            agent = env.world.agents[agent_index]
            good = ((agent.state.pos - target).norm(dim=-1) < c.staging_tolerance) & ((agent.state.vel - velocity).norm(dim=-1) < c.staging_speed_tolerance)
            staged[active & good] = True
        self.phase[active & (self.phase == 0) & self.primary_staged & self.helper_staged] = 1
        metrics = readiness_metrics(env, self.anchor, self.supports, c)
        acquire = ((self.phase == 1) & (metrics["helper_gaps"] <= c.gap_tolerance).all(-1)
                   & (metrics["helper_gaps"] >= -c.penetration_limit).all(-1)
                   & (metrics["helper_region_error"] < c.helper_region_tolerance)
                   & (metrics["helper_relative_speed"] < c.relative_speed_tolerance)
                   & (metrics["primary_position_error"] < c.primary_position_tolerance))
        # Freeze exactly the acquisition reference. The next call switches to
        # the final fixed-anchor law, so this acquisition sample earns no dwell.
        was_confirm = self.phase == 2
        acquisition = (HELPER_LOCAL[0] - c.preload, HELPER_LOCAL[1] + c.preload)
        reference, _ = contact_reference(env, acquisition)
        newly = active & acquire
        self.anchor[newly] = reference[newly]
        self.phase[newly] = 2
        metrics = readiness_metrics(env, self.anchor, self.supports, c)
        valid = active & was_confirm & metrics["instantaneous_ready"]
        self.dwell[active] = torch.where(valid[active], self.dwell[active] + 1, 0)
        self.ready[active] = self.dwell[active] >= c.confirm_steps
        self.accepted |= self.ready
        self.failed |= active & (self.elapsed >= c.max_steps) & ~self.ready
        return {**metrics, "ready": self.ready.clone(), "current_ready": self.ready.clone(),
                "accepted": self.accepted.clone(), "failed": self.failed.clone(),
                "phase": self.phase.clone(), "dwell": self.dwell.clone(),
                "elapsed": self.elapsed.clone(), "anchor": self.anchor.clone(),
                "anchor_ood": self.anchor_ood.clone()}

    def configuration(self):
        return {**asdict(self.config), "dt": self.env.world.dt,
                "vmas_collision_line_min_dist": LINE_MIN_DIST,
                "recovery_observation": "both agent positions/velocities; bar/stem poses/velocities; contact geometry",
                "post_handoff_helper_observation": "own position/velocity and frozen 2D anchor memory",
                "handoff_law": "passive zero; actuated fixed-anchor finite PD with declared preload",
                "world_anchor_recovery_reference_is_ood": True}


@torch.no_grad()
def calibration_audit(source_path="data/novelty/abstraction_transfer.pt", output_path="data/novelty/rebrace_calibration.pt"):
    """Only the declared 0deg/friction1.5/ss20/zero-nuisance development slice.

    Replay actual original actions to step 70. No saved physical state is assigned
    in the middle of a trajectory. Duplicate each development row for zero and
    0.3 primary-force handoff checks; neither handoff command is a task planner.
    """
    import hashlib
    import json
    from pathlib import Path
    from vmas_pusht.boundary import _plain
    from vmas_pusht.physical_abstraction import physical_state, setup_scene
    from vmas_pusht.scenario import make

    source_path, output_path = Path(source_path), Path(output_path)
    source = torch.load(source_path, weights_only=True)
    matches = [g for g in source["groups"] if g["rows"][0]["friction_ratio"] == 1.5
               and g["rows"][0]["substeps"] == 20
               and g["rows"][0]["task"] == "rotate_retreat_rebrace_rotate"]
    if len(matches) != 1:
        raise ValueError("Expected one original development group")
    group = matches[0]
    indices = [i for i, row in enumerate(group["rows"])
               if row["initial_angle_deg"] == 0 and row["nuisance_id"] == 0]
    indices = torch.tensor(indices).repeat_interleave(2)
    rows = [{"source_row": int(i), "support": group["rows"][i]["support"],
             "source_planner": group["rows"][i]["planner"], "handoff_main_force": force}
            for i in indices[::2].tolist() for force in (0., .3)]
    n = len(rows)
    env = make(num_envs=n, device="cpu", friction_ratio=1.5, substeps=20, seed=101)
    setup_scene(env)
    for step in range(70):
        command = group["actions"][indices, step]
        env.step([command[:, 0], command[:, 1]])
    replay_error = (physical_state(env) - group["trajectory"][indices, 70]).abs().amax(1)
    supports = [row["support"] for row in rows]
    controller = RebraceController(env, supports)
    first_ready = torch.full((n,), -1, dtype=torch.long)
    post_count = torch.zeros(n, dtype=torch.long)
    post_force_contact_count = torch.zeros_like(post_count)
    post_geometric_contact_count = torch.zeros_like(post_count)
    primary_force = torch.tensor([row["handoff_main_force"] for row in rows])
    trajectory, actions, metrics = [physical_state(env).clone()], [], []
    initial_pose = env.scenario.t_pose().clone()
    for step in range(controller.config.max_steps + 20):
        recovering = (first_ready < 0) & ~controller.failed
        post = (first_ready >= 0) & (post_count < 20)
        command = controller.step()
        angle = env.scenario.bar.state.rot[:, 0]
        force = torch.stack((angle.sin(), -angle.cos()), -1) * primary_force[:, None]
        command[post, 0] = force[post]
        command[~recovering, 1] = support_action(env, controller.anchor, supports)[~recovering]
        command[~recovering & ~post, 0] = main_holding_action(env)[~recovering & ~post]
        env.step([command[:, 0], command[:, 1]])
        status = controller.update(active_mask=recovering)
        now = readiness_metrics(env, controller.anchor, supports)
        new = recovering & status["ready"]
        first_ready[new] = step + 1
        post_count[post] += 1
        force_contact = (now["helper_normal_forces"] >= controller.config.minimum_contact_force).all(-1)
        geometric = (now["helper_gaps"] <= controller.config.gap_tolerance).all(-1)
        post_force_contact_count[post] += force_contact[post]
        post_geometric_contact_count[post] += geometric[post]
        trajectory.append(physical_state(env).clone()); actions.append(command.clone())
        metrics.append({k: value.clone() for k, value in now.items()})
    for i, row in enumerate(rows):
        row.update(first_ready_step=int(first_ready[i]), ready=bool(first_ready[i] >= 0),
                   timed_out=bool(controller.failed[i]), post_handoff_steps=int(post_count[i]),
                   post_handoff_dual_force_steps=int(post_force_contact_count[i]),
                   post_handoff_near_contact_steps=int(post_geometric_contact_count[i]),
                   continuous_dual_force_20=bool(post_count[i] == 20 and post_force_contact_count[i] == 20),
                   replay_max_state_error=float(replay_error[i]))
    result = {"rows": rows, "configuration": controller.configuration(),
              "development_only": {"friction_ratio": 1.5, "substeps": 20, "initial_angle_deg": 0, "nuisance_id": 0},
              "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "trajectory": torch.stack(trajectory, 1), "actions": torch.stack(actions, 1),
              "initial_pose": initial_pose, "final_anchor": controller.anchor.clone(),
              "metrics": {k: torch.stack([m[k] for m in metrics], 1) for k in metrics[0]}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    summary = {k: value for k, value in result.items() if k not in ("trajectory", "actions", "metrics")}
    summary["artifact_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_path.with_suffix(".json").write_text(json.dumps(_plain(summary), indent=2) + "\n")
    return result


@torch.no_grad()
def reaudit_old_contact(source_path="data/novelty/abstraction_transfer.pt",
                        output_path="data/novelty/rebrace_calibration_old_contact.json"):
    """Recompute true collision forces on *recorded* old states, without rollout.

    This separates a measurement correction from controller improvement. Loading
    stored state into a temporary measurement environment is not a new physical
    trajectory; no transition or control-performance result is inferred from it.
    """
    import hashlib
    import json
    from pathlib import Path
    from vmas_pusht.boundary import _plain
    from vmas_pusht.scenario import make

    source_path = Path(source_path)
    source = torch.load(source_path, weights_only=True)
    reports = []
    for group in source["groups"]:
        if group["rows"][0]["task"] != "rotate_retreat_rebrace_rotate":
            continue
        rows, states = group["rows"], group["trajectory"]
        env = make(num_envs=len(rows), device="cpu", friction_ratio=rows[0]["friction_ratio"],
                   substeps=rows[0]["substeps"], seed=101)
        measured = []
        for index in range(86, 96):
            for j, entity in enumerate((*env.world.agents, env.scenario.bar, env.scenario.stem)):
                values = states[:, index, j * 6:(j + 1) * 6]
                entity.set_pos(values[:, :2].clone(), batch_index=None)
                entity.set_vel(values[:, 2:4].clone(), batch_index=None)
                entity.set_rot(values[:, 4:5].clone(), batch_index=None)
                entity.set_ang_vel(values[:, 5:6].clone(), batch_index=None)
            measured.append(readiness_metrics(env, group["helper_anchor"][:, index],
                                              [r["support"] for r in rows]))
        forces = torch.stack([m["helper_normal_forces"] for m in measured], 1)
        gaps = torch.stack([m["helper_gaps"] for m in measured], 1)
        force_contact = forces >= RebraceConfig().minimum_contact_force
        for i, row in enumerate(rows):
            other_old_gates = all(row[k] for k in ("terminal_success", "first_goal_success_at_phase_boundary",
                                                   "retreated_clear", "primary_repositioned"))
            boundary_any = bool(force_contact[i, 4].any())
            boundary_dual = bool(force_contact[i, 4].all())
            predual = bool(force_contact[i, :5].all())
            postdual = bool(force_contact[i, 5:].all())
            reports.append({"support": row["support"], "planner": row["planner"],
                "friction_ratio": row["friction_ratio"], "substeps": row["substeps"],
                "initial_angle_deg": row["initial_angle_deg"], "nuisance_id": row["nuisance_id"],
                "old_reported_contact": row["contact_reacquired"],
                "boundary_any_real_contact": boundary_any, "boundary_dual_real_contact": boundary_dual,
                "last5_recovery_dual_real_contact": predual, "first5_handoff_dual_real_contact": postdual,
                "boundary_normal_forces": forces[i, 4], "boundary_geometric_gaps": gaps[i, 4],
                "other_old_completion_gates": other_old_gates,
                "corrected_any_contact_old_completion": other_old_gates and boundary_any,
                "corrected_dual_contact_old_completion": other_old_gates and boundary_dual,
                "corrected_persistent_handoff_old_completion": other_old_gates and predual and postdual,
            })
    names = [key for key, value in reports[0].items() if isinstance(value, bool)]
    result = {"source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "measurement_only_no_new_rollout": True,
              "contact_force_threshold": RebraceConfig().minimum_contact_force,
              "vmas_collision_line_min_dist": LINE_MIN_DIST,
              "counts": {name: sum(row[name] for row in reports) for name in names},
              "n": len(reports), "rows": reports}
    Path(output_path).write_text(json.dumps(_plain(result), indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = calibration_audit()
    for row in result["rows"]:
        print(row)
    print("Old contact measurement correction:", reaudit_old_contact()["counts"])
