import itertools

import numpy as np
import pytest
import torch

from vmas_pusht.pairing import (
    additive_projection,
    all_pairs,
    coupling,
    optimal_permutation,
    overlap_scores,
    permutation_value,
    tensor_sha256,
    timing_banks,
)


def test_uniform_complete_sequence_marginals():
    gamma = coupling(np.array([2, 0, 3, 1]))
    np.testing.assert_array_equal(gamma.sum(0), np.full(4, .25))
    np.testing.assert_array_equal(gamma.sum(1), np.full(4, .25))
    assert gamma.sum() == 1
    with pytest.raises(ValueError):
        coupling(np.array([0, 0, 2]))
    with pytest.raises(ValueError):
        coupling(np.array([0., 1.]))


def test_optimum_agrees_with_brute_force_and_not_best_single_pair():
    scores = np.array([[10., 9., 0.], [9., 0., 0.], [0., 0., 1.]])
    values = [permutation_value(scores, np.array(p)) for p in itertools.permutations(range(3))]
    assert permutation_value(scores, optimal_permutation(scores)) == pytest.approx(max(values))
    assert permutation_value(scores, optimal_permutation(scores, maximize=False)) == pytest.approx(min(values))
    # Greedily selecting the maximum (0,0) prevents the better (0,1),(1,0) coupling.
    np.testing.assert_array_equal(optimal_permutation(scores), [1, 0, 2])
    assert max(values) < scores.max()


def test_additive_control_has_zero_gain_for_every_pairing():
    scores = np.random.default_rng(11).normal(size=(5, 5))
    additive = additive_projection(scores)
    for p in itertools.permutations(range(5)):
        assert permutation_value(additive, np.array(p)) == pytest.approx(additive.mean(), abs=1e-12)
    best = permutation_value(scores, optimal_permutation(scores))
    assert best >= scores.mean() - 1e-12


def test_candidate_reordering_preserves_optimum_value_and_marginals():
    # Test value, not assignment identity: equivalent ties may choose other permutations.
    scores = np.random.default_rng(4).normal(size=(7, 7))
    row_order = np.array([3, 1, 4, 6, 2, 5, 0])
    col_order = np.array([4, 2, 6, 1, 0, 3, 5])
    reordered = scores[row_order][:, col_order]
    p = optimal_permutation(reordered)
    original_p = np.empty(7, dtype=int)
    original_p[row_order] = col_order[p]
    assert permutation_value(scores, original_p) == pytest.approx(permutation_value(scores, optimal_permutation(scores)))
    np.testing.assert_allclose(coupling(original_p).sum(0), np.full(7, 1 / 7))


def test_timing_bank_impulse_mapping_and_no_mutation():
    banks, offsets = timing_banks([.6, .8], n=8, horizon=40, pulse_steps=12, seed=2)
    digest = tensor_sha256(banks)
    expected = torch.tensor([.6, .8]) * 12
    torch.testing.assert_close(banks.sum(dim=2), expected.expand(2, 8, 2))
    pairs = all_pairs(banks)
    for i in range(8):
        for j in range(8):
            torch.testing.assert_close(pairs[8 * i + j, :, 0], banks[0, i])
            torch.testing.assert_close(pairs[8 * i + j, :, 1], banks[1, j])
    chosen = optimal_permutation(overlap_scores(banks))
    np.testing.assert_array_equal(offsets[0], offsets[1, chosen])
    assert tensor_sha256(banks) == digest


def test_input_validation():
    with pytest.raises(ValueError):
        optimal_permutation(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        optimal_permutation(np.array([[np.nan]]))
    with pytest.raises(ValueError):
        timing_banks([1., 1.])
    with pytest.raises(ValueError):
        timing_banks([0., -1.], n=8, horizon=10, pulse_steps=5)


def test_steering_banks_keep_duration_norm_impulse_magnitude_and_reproduce():
    banks, offsets = timing_banks([0., -1.], n=8, horizon=25, pulse_steps=10, seed=17, family="timing-steering")
    repeated, repeated_offsets = timing_banks([0., -1.], n=8, horizon=25, pulse_steps=10, seed=17, family="timing-steering")
    assert tensor_sha256(banks) == tensor_sha256(repeated)
    np.testing.assert_array_equal(offsets, repeated_offsets)
    norms = banks.norm(dim=-1)
    torch.testing.assert_close(norms.sum(dim=-1), torch.full((2, 8), 10.))
    torch.testing.assert_close(banks.sum(dim=2).norm(dim=-1), torch.full((2, 8), 10.))
    assert torch.all(norms <= 1 + 1e-7)
    assert torch.all((norms > 0).sum(dim=-1) == 10)
    # Steering changes directional impulse; it is not hidden as a timing-only bank.
    assert banks.sum(dim=2)[..., 0].abs().max() > 4.9
    for agent in range(2):
        assert len(set(offsets[agent])) == 8
    p = optimal_permutation(overlap_scores(banks))
    np.testing.assert_array_equal(offsets[0], offsets[1, p])
