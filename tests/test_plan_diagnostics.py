"""Rank diagnostics must not invent information when true candidate costs tie."""
import json

import pytest
import torch
from scipy.stats import spearmanr

from vmas_pusht.plan import json_safe, spearman_rows


@pytest.mark.parametrize("a,b", [
    ([1., 1., 2., 3., 3.], [2., 1., 2., 2., 4.]),
    ([2., 5., 1., 3., 4.], [5., 3., 1., 2., 4.]),
    ([1., 1., 2., 2., 3.], [3., 3., 2., 2., 1.]),
    ([0., 1.], [1., 0.]),
])
def test_average_tie_ranks_agree_with_scipy(a, b):
    actual = spearman_rows(torch.tensor([a]), torch.tensor([b]))[0].item()
    assert actual == pytest.approx(spearmanr(a, b).statistic, abs=1e-12)


def test_constant_rows_are_undefined_including_two_sample_elites():
    a = torch.tensor([[1., 2.], [1., 1.], [2., 2.], [1., 2.]])
    b = torch.tensor([[3., 3.], [2., 1.], [1., 1.], [2., 1.]])
    correlations = spearman_rows(a, b)
    assert correlations[:3].isnan().all()
    assert correlations[3].item() == pytest.approx(-1.)


def test_nonfinite_rows_are_undefined():
    a = torch.tensor([[1., float("nan")], [1., float("inf")]])
    assert spearman_rows(a, torch.tensor([[1., 2.], [1., 2.]])).isnan().all()


def test_diagnostic_json_uses_null_for_undefined_values():
    diagnostic = {"summary": {"spearman": float("nan")}, "replans": [{"ranks": [1., float("nan"), float("inf")]}]}
    encoded = json.dumps(json_safe(diagnostic), allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded == {"summary": {"spearman": None}, "replans": [{"ranks": [1., None, None]}]}
