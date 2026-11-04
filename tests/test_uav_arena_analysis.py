import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")         # uav_arena.analysis takes the change layout from the harness
from scipy.stats import spearmanr

from uav_arena.analysis import interval, partial, partial_formula, position, spearman, verdict


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 30))
    y = np.round(x+rng.normal(size=(4, 30)), 1)
    z = rng.integers(0, 5, size=(4, 30)).astype(float)     # many ties, as J has per condition
    C = spearman(x, y, z)
    for b in range(4):
        assert np.allclose(C[b], spearmanr(np.column_stack([x[b], y[b], z[b]])).correlation)


def test_partial_correlation_forms_agree():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(3, 30))
    x = z+rng.normal(size=(3, 30))
    y = z-x+rng.normal(size=(3, 30))
    C = spearman(x, y, z, rng.normal(size=(3, 30)))
    assert np.allclose(partial_formula(C[:, :3, :3]), partial(C[:, :3, :3]))
    assert np.all(np.abs(partial(C)) <= 1)


def test_partial_correlation_removes_the_controlled_variable():
    """x and y both follow z: strongly correlated, but not given z."""
    rng = np.random.default_rng(3)
    z = rng.normal(size=(1, 2000))
    C = spearman(z+0.3*rng.normal(size=(1, 2000)), z+0.3*rng.normal(size=(1, 2000)), z)
    assert C[0, 0, 1] > 0.8 and abs(partial_formula(C)[0]) < 0.1


def test_interval_and_decision_rule():
    iv = interval(np.concatenate([[-0.4], np.linspace(-0.6, -0.2, 1001)]))
    assert iv["estimate"] == -0.4 and np.isclose(iv["low"], -0.59) and np.isclose(iv["high"], -0.21)
    assert position(iv) == "below 0" and verdict(iv, 30) == "supported" and verdict(iv, 4) == "uninformative"
    assert verdict(dict(estimate=0.1, low=-0.1, high=0.3), 30) == "not established"
    assert verdict(dict(estimate=0.3, low=0.1, high=0.5), 30) == "unexpected direction"
