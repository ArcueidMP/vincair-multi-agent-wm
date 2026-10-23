import copy
import json

import pytest
import torch

from vmas_pusht.abstraction_convergence import restore_across_substeps
from vmas_pusht.physical_abstraction import physical_state, setup_scene
from vmas_pusht.physics_profiles import get_profile, make_profile
from vmas_pusht.scenario import make
from vmas_pusht.snapshot import capture
from vmas_pusht.stable_physics import (finalize_selection, local_rows, profile_hash,
                                      qualification, run_profile)


def test_original_profile_reproduces_complete_default_physics_and_initial_state():
    original = make(num_envs=2, substeps=10, seed=43)
    named = make_profile("original", num_envs=2, substeps=10, seed=43)
    assert capture(original).physics == capture(named).physics
    torch.testing.assert_close(physical_state(original), physical_state(named), rtol=0, atol=0)
    assert named.world.dt == .1
    assert all(e.movable for e in (*named.world.agents, named.scenario.bar, named.scenario.stem))


def test_profile_load_is_owned_validated_and_roundtrips_selection_json(tmp_path):
    cfg = get_profile("dry_light")
    cfg["scenario"]["friction_ratio"] = 99.
    assert get_profile("dry_light")["scenario"]["friction_ratio"] == .25
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(dict(profile=get_profile("dry_light"))))
    assert get_profile(path) == get_profile("dry_light")
    assert profile_hash(path) == profile_hash("dry_light")
    for key, value in (("agent_force", 2.), ("t_mass", float("inf")), ("t_mass", 0.),
                       ("rot_friction_ratio", -.1)):
        bad = get_profile("dry_light")
        bad["scenario"][key] = value
        with pytest.raises(ValueError):
            get_profile(bad)


def test_explicit_finite_stiffness_is_snapshot_checked_and_resolution_copy_exact():
    source = make_profile("damped_heavy6", num_envs=2, substeps=5)
    setup_scene(source, angle=[0., .5], hidden_v=(.03, 0.), hidden_omega=.2)
    source.step([torch.tensor([[0., -.3]]).expand(2, 2), torch.zeros(2, 2)])
    target = make_profile("damped_heavy6", num_envs=2, substeps=10)
    restore_across_substeps(target, capture(source))
    assert torch.equal(physical_state(source), physical_state(target))
    assert target.world._torque_constraint_force == 5.
    different = make_profile("dry_light", num_envs=2, substeps=10)
    with pytest.raises(ValueError, match="only substeps"):
        restore_across_substeps(different, capture(source))


def test_selection_grid_has_matched_no_primary_controls_and_distinct_validation():
    for split in ("development", "validation"):
        rows = local_rows(split)
        for row in rows:
            if row["strength"]:
                assert any(r["strength"] == 0 and r["angle_deg"] == row["angle_deg"]
                           and r["hidden_id"] == row["hidden_id"] and r["mode"] == row["mode"] for r in rows)
    assert {r["angle_deg"] for r in local_rows("development")} == {0.}
    assert {r["angle_deg"] for r in local_rows("validation")} == {30.}
    assert {r["direction_deg"] for r in local_rows("development") if r["strength"]}.isdisjoint(
        {r["direction_deg"] for r in local_rows("validation") if r["strength"]})


def _summary():
    return dict(substeps=[80, 160], comparisons=[dict(candidate_substeps=80, reference_substeps=160,
        pair="vs_finest", max_xy_units=.0001, max_angle_rad=.001, kind=k) for k in ("feedback", "replay")],
        authority_full_horizon=[dict(mode="world_anchor", max_authority_rad=.2, mean_peak_authority_rad=.08)],
        feedback_runs=[dict(helper_modes=[dict(wall_cases=0, max_action_norm=1.,
                                              max_weld_xy_units=.001, max_weld_angle_rad=.01)])])


def test_motionless_or_inaccurate_system_cannot_qualify_as_stable_controllable_physics():
    summary = _summary()
    assert qualification(summary)["qualified"]
    still = copy.deepcopy(summary)
    still["authority_full_horizon"][0].update(max_authority_rad=0., mean_peak_authority_rad=0.)
    assert not qualification(still)["qualified"]
    inaccurate = copy.deepcopy(summary)
    inaccurate["comparisons"][1]["max_angle_rad"] = .01
    assert not qualification(inaccurate)["qualified"]
    bent = copy.deepcopy(summary)
    bent["feedback_runs"][0]["helper_modes"][0]["max_weld_angle_rad"] = .1
    assert not qualification(bent)["qualified"]


def test_audit_runs_finite_actions_with_identical_joint_replay_and_full_initial_state():
    rows = local_rows(screen=True)
    summary, raw = run_profile("dry_light", rows=rows, substeps=(5, 10), horizon=3)
    assert summary["initial_full_state_identical"]
    assert summary["actual_normal_forces_recorded"]
    assert torch.equal(raw["feedback"][5]["trajectory"][:, 0], raw["feedback"][10]["trajectory"][:, 0])
    assert torch.equal(raw["replay"][5]["actions"], raw["feedback"][10]["actions"])
    assert raw["feedback"][10]["helper_normal_force_norms"].shape == (len(rows), 4, 2)
    for run in raw["feedback"].values():
        assert run["actions"].norm(dim=-1).max() <= 1.000001
        assert run["finite"].all()


def test_final_selection_increases_resolution_without_changing_selected_physics(tmp_path):
    cfg = get_profile("dry_light")
    dev = _summary()
    dev.update(profile=cfg, qualification=qualification(dev))
    val = _summary()
    val.update(profile=cfg, substeps=[80, 160, 320])
    val["comparisons"] = [dict(candidate_substeps=n, reference_substeps=320, pair="vs_finest",
        kind=k, max_xy_units=.0005, max_angle_rad=.004 if n == 80 else .001)
        for n in (80, 160) for k in ("feedback", "replay")]
    development, validation, candidate, out = [tmp_path / name for name in
        ("development.json", "validation.json", "candidate.json", "selected.json")]
    development.write_text(json.dumps(dict(split="development", profiles=[dev])))
    validation.write_text(json.dumps(dict(split="validation", profiles=[val])))
    candidate.write_text(json.dumps(dict(profile=cfg)))
    selected = finalize_selection(development, validation, candidate, out)
    assert selected["profile"] == cfg
    assert selected["nominal_substeps"] == 160
    assert selected["reference_substeps"] == 320
    assert not selected["validation_qualifications"][0]["qualified"]
    assert selected["validation_qualifications"][1]["qualified"]
    assert selected["source_sha256"]
    for comparison in val["comparisons"]:
        comparison["max_angle_rad"] = .01
    validation.write_text(json.dumps(dict(split="validation", profiles=[val])))
    with pytest.raises(ValueError, match="did not establish"):
        finalize_selection(development, validation, candidate, tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()
