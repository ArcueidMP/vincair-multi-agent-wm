import pytest
import torch

from vmas_pusht.friction_resolution_probe import probe


def test_below_cap_motion_scales_with_integration_step():
    torch.set_num_threads(1)
    coarse, fine = probe(5, .5, 5), probe(20, .5, 5)
    assert coarse["angle_rad"] == pytest.approx(coarse["angle_if_below_cap_creep_formula"], rel=1e-5)
    assert fine["angle_rad"] == pytest.approx(fine["angle_if_below_cap_creep_formula"], rel=1e-5)
    # Leading order is 1/substeps; per-env-step drag adds an O(sub_dt**2)
    # correction because friction is computed before drag in VMAS.
    assert 3.8 < coarse["angle_rad"] / fine["angle_rad"] < 4


def test_above_cap_torque_retains_motion_at_finer_resolution():
    torch.set_num_threads(1)
    low = probe(20, .5, 10)
    high = probe(20, 1.5, 10)
    high_fine = probe(40, 1.5, 10)
    assert high["angle_rad"] > 10 * low["angle_rad"]
    assert high_fine["angle_rad"] > .9 * high["angle_rad"]
