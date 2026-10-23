import torch

from vmas_pusht.abstraction_release import BRANCHES, audit_group


def test_release_branches_share_real_history_and_primary_commands():
    torch.set_num_threads(1)
    summary, raw = audit_group(friction=1.5, substeps=5, horizon=20)
    original = raw["branches"]["keep"]
    for branch in BRANCHES:
        records = raw["branches"][branch]
        torch.testing.assert_close(records["trajectory"][:, 0], raw["initial_state"], atol=0, rtol=0)
        torch.testing.assert_close(records["actions"][:, :, 0], original["actions"][:, :, 0], atol=0, rtol=0)
        assert summary["branches"][branch]["finite_all"]
        assert summary["branches"][branch]["maximum_action_norm"] <= 1 + 1e-6
    for i, row in enumerate(raw["rows"]):
        torch.testing.assert_close(raw["initial_state"][i], raw["warmup_trajectory"][i, row["warm_steps"]], atol=0, rtol=0)
    assert raw["branches"]["zero_force"]["actions"][:, :, 1].count_nonzero() == 0
    assert summary["branches"]["contact_disabled"]["helper_any_contact_fraction"] == 0
    assert raw["branches"]["retreat_return"]["actions"][:, :, 1].abs().max() > 0
