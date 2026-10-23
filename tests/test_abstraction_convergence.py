import copy
import math

import pytest
import torch

from vmas_pusht.abstraction_convergence import (
    Tolerances, compare_trajectories, default_programs, make_rows,
    restore_across_substeps, rollout, run_group,
)
from vmas_pusht.physical_abstraction import physical_state, setup_scene
from vmas_pusht.scenario import make
from vmas_pusht.snapshot import capture


def test_full_state_cross_resolution_copy_preserves_independent_stem_and_goals():
    src = make(num_envs=2, substeps=5)
    setup_scene(src, angle=[0., .3], hidden_v=(.03, 0.), hidden_omega=.2)
    src.step([torch.tensor([[0., -.6]]).expand(2, 2), torch.zeros(2, 2)])
    snap = capture(src)
    dst = make(num_envs=2, substeps=10)
    restore_across_substeps(dst, snap)
    assert torch.equal(physical_state(src), physical_state(dst))
    assert torch.equal(src.steps, dst.steps)
    assert torch.equal(src.scenario.goal_pose, dst.scenario.goal_pose)
    assert dst.scenario.substeps == 10
    assert capture(dst).physics != snap.physics
    # Copied tensors must not alias the source or saved snapshot.
    dst.scenario.stem.state.pos.add_(.001)
    assert torch.equal(physical_state(src)[:, 18:20], next(e for e in snap.entities if e.name == "T stem").state["pos"])


def test_cross_resolution_rejects_changed_physics_and_geometry_before_mutation():
    src = make(num_envs=1, substeps=5)
    setup_scene(src)
    snap = capture(src)
    for changed in (make(num_envs=1, substeps=10, friction_ratio=1.4),
                    make(num_envs=1, substeps=10, agent_force=.9)):
        before = physical_state(changed).clone()
        with pytest.raises(ValueError, match="only substeps"):
            restore_across_substeps(changed, snap)
        assert torch.equal(before, physical_state(changed))
    geometry = make(num_envs=1, substeps=10)
    geometry.scenario.bar.shape._length = .5
    with pytest.raises(ValueError, match="only substeps"):
        restore_across_substeps(geometry, snap)


def _fake_run():
    trajectory = torch.zeros(2, 4, 24)
    trajectory[..., 16] = math.pi - .01
    trajectory[:, :, 22] = math.pi - .01
    return dict(trajectory=trajectory, actions=torch.zeros(2, 3, 2, 2),
                contact=torch.zeros(2, 4, 2, 2, dtype=torch.bool),
                time=torch.arange(4) * .1, substeps=10)


def test_all_time_comparison_wraps_angles_and_catches_nonterminal_excursions():
    ref = _fake_run()
    candidate = copy.deepcopy(ref)
    candidate["trajectory"][:, 1, 16] = -math.pi + .01
    candidate["trajectory"][:, 1, 12] = .002
    metrics = compare_trajectories(candidate, ref)
    assert metrics["max_angle_rad"] == pytest.approx(.02, abs=1e-6)
    assert metrics["max_xy_units"] == pytest.approx(.002)
    assert metrics["terminal_rms_angle_rad"] == 0
    assert metrics["terminal_rms_xy_units"] == 0
    assert not metrics["passes_preregistered_tolerances"]


def test_comparison_rejects_misaligned_time_or_initial_state():
    ref = _fake_run()
    candidate = copy.deepcopy(ref)
    candidate["time"] = candidate["time"] + .01
    with pytest.raises(ValueError, match="timestamps"):
        compare_trajectories(candidate, ref)
    candidate = copy.deepcopy(ref)
    candidate["trajectory"][0, 0, 20] = .001  # Hidden stem velocity, not just bar pose.
    with pytest.raises(ValueError, match="Initial complete"):
        compare_trajectories(candidate, ref)


def test_exact_joint_action_replay_and_feedback_are_separate_records():
    rows = make_rows(angles=(0.,), nuisances=((0., 0., 0.),),
                     programs=[default_programs()[4], default_programs()[-1]],
                     modes=("world_anchor",))
    summary, raw = run_group(substeps=(5, 10), rows=rows, horizon=3, compare_horizons=(3,))
    assert summary["initial_state_identical"]
    torch.testing.assert_close(raw["replay"][5]["actions"], raw["feedback"][10]["actions"], rtol=0, atol=0)
    assert {c["kind"] for c in summary["comparisons"]} == {
        "fixed_feedback_law", "identical_joint_action_replay"}
    for run in raw["feedback"].values():
        assert run["trajectory"].shape == (2, 4, 24)
        assert run["finite"].all()
        assert run["actions"].norm(dim=-1).max() <= 1.000001
        assert torch.isfinite(run["weld_xy"]).all()


def test_representative_selection_keeps_matched_zero_controls_and_new_batch_reruns():
    from vmas_pusht.abstraction_convergence import select_representative_rows
    rows = make_rows(angles=(0.,), nuisances=((0., 0., 0.),),
                     programs=[default_programs()[4], default_programs()[-1]])
    low = torch.zeros(len(rows), 3, 24)
    high = low.clone()
    for i, row in enumerate(rows):
        if row["strength"]:
            low[i, 1:, 12] = .001 * (i + 1)
            high[i, 1:, 16] = .002 * (i + 1)
    selected, metadata = select_representative_rows(
        dict(rows=rows, feedback={80: dict(trajectory=low), 160: dict(trajectory=high)}))
    assert metadata["all_selected_rows_rerun_at_all_requested_substeps"]
    for row in selected:
        if row["strength"]:
            assert any(other["strength"] == 0. and other["mode"] == row["mode"]
                       and other["angle_deg"] == row["angle_deg"]
                       and other["hidden_id"] == row["hidden_id"] for other in selected)


def test_recovery_rollout_records_a_common_physical_clock_and_real_force_metrics():
    from vmas_pusht.abstraction_convergence import recovery_rollout
    src = make(num_envs=1, substeps=5)
    setup_scene(src)
    saved = capture(src)
    rows = [dict(angle_deg=0., hidden_id=0, vx=0., vy=0., omega=0.,
                 mode="preloaded_anchor", direction_deg=20., strength=.6,
                 pulse_steps=20, handoff_force=.3)]
    dst = make(num_envs=1, substeps=10)
    data = recovery_rollout(dst, saved, rows, max_recovery=3, post_steps=2)
    assert data["trajectory"].shape == (1, 6, 24)
    assert data["helper_normal_force_norms"].shape == (1, 6, 2)
    assert torch.equal(data["trajectory"][:, 0], physical_state(src))
    assert torch.equal(data["time"], torch.arange(6) * .1)
    # Five confirmations cannot be achieved inside a three-step recovery budget.
    assert data["first_ready_step"].item() == -1
    assert data["post_handoff_steps"].item() == 0
    assert not data["full_20_step_dual_force"].item()
    assert data["recovery_failed"].item()
    assert data["actions"].norm(dim=-1).max() <= 1.000001
