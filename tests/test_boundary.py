"""Physical and intervention invariants for the empirical boundary audit."""

import math

import pytest
import torch

from vmas_pusht.boundary import (
    BRANCHES, audit_group, intervention_actions, place_contact, surface_gaps,
)
from vmas_pusht.scenario import make


@pytest.mark.parametrize("layout", ["wide", "narrow", "side", "centered"])
def test_contact_layout_has_equal_gap_and_unit_force_under_rotation(layout):
    torch.set_num_threads(1)
    env = make(num_envs=3)
    direction = place_contact(env, angles=torch.tensor([0.0, math.pi / 4, math.pi / 2]),
                              layout=layout, gap=0.001)
    assert torch.allclose(direction.norm(dim=-1), torch.ones(3))
    assert torch.allclose(surface_gaps(env).amin(-1), torch.full((3, 2), 0.001), atol=1e-7)
    assert ((env.world.agents[0].state.pos - env.world.agents[1].state.pos).norm(dim=-1) > 0.1).all()
    assert torch.allclose(env.scenario.goal_pose[:, :2] - env.scenario.bar.state.pos, 0.2 * direction)
    for entity in (*env.world.agents, env.scenario.bar, env.scenario.stem):
        assert torch.count_nonzero(entity.state.vel) == 0
        assert torch.count_nonzero(entity.state.ang_vel) == 0


def test_delayed_factorial_controls_match_and_preserve_norm():
    direction = torch.tensor([[1.0, 1.0]]).expand(4, -1) / math.sqrt(2)
    branch = torch.tensor(BRANCHES)
    for step in range(8):
        actions = intervention_actions(direction, torch.full((4,), 0.7),
                                       torch.full((4,), 3), branch, step)
        assert torch.equal(actions[1, 0], actions[3, 0])
        assert torch.equal(actions[2, 1], actions[3, 1])
        assert torch.count_nonzero(actions[0]) == 0
        assert torch.count_nonzero(actions[1, 1]) == 0
        assert torch.count_nonzero(actions[2, 0]) == 0
        assert (actions.norm(dim=-1) <= 0.700001).all()
        assert bool(actions[3, 1].any()) == (step >= 3)


def test_zero_intervention_branches_have_identical_physical_outcomes():
    torch.set_num_threads(1)
    group = audit_group(friction_ratio=1.5, substeps=5, angles_deg=(0, 45),
                        layouts=("wide",), strengths=(0,), delays=(0,),
                        horizons=(3,), prelude=10)
    assert group["snapshot_matched_exactly"]
    for row in group["rows"]:
        outcomes = list(row["interventions"].values())
        assert all(b == outcomes[0] for b in outcomes)
        assert row["directed_com_interaction"] == 0
        assert outcomes[0]["finite_all_steps"]


def test_swapping_solo_agent_reflects_symmetric_wide_push():
    torch.set_num_threads(1)
    group = audit_group(friction_ratio=1.5, substeps=5, angles_deg=(0,),
                        layouts=("wide",), strengths=(1,), delays=(0,),
                        horizons=(10,), prelude=10)
    a, b = (group["rows"][0]["interventions"][name] for name in ("10", "01"))
    assert a["finite_all_steps"] and b["finite_all_steps"]
    assert a["directed_com_displacement"] == pytest.approx(b["directed_com_displacement"], abs=2e-6)
    assert a["signed_angle_change_rad"] == pytest.approx(-b["signed_angle_change_rad"], abs=2e-6)
    # An inactive agent remains in the physics and is moved by passive contact.
    assert len(a["contact_fraction_by_agent"]) == 2
    assert a["maximum_action_norm_by_agent"] == [1.0, 0.0]
    assert b["maximum_action_norm_by_agent"] == [0.0, 1.0]


def test_early_horizon_does_not_inherit_later_success():
    torch.set_num_threads(1)
    group = audit_group(friction_ratio=1.5, substeps=5, angles_deg=(0,),
                        layouts=("wide",), strengths=(1,), delays=(0,),
                        horizons=(5, 100), prelude=10)
    early, late = [row["interventions"]["11"] for row in group["rows"]]
    assert not early["success_ever"]
    assert early["first_success_step"] == -1
    assert late["success_ever"]
    assert 5 < late["first_success_step"] <= 100
