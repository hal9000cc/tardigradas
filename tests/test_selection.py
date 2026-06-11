from __future__ import annotations

import numpy as np
import pytest

from tardigradas.operators import rank, select_parents


def test_rank_returns_normalized_expectation() -> None:
    expectation = rank([1.0, 2.0, 3.0], alpha=0.5, uniform_mix=0.0)

    assert expectation.shape == (3,)
    assert np.isclose(expectation.sum(), 1.0)


def test_rank_gives_highest_weight_to_best_score() -> None:
    scores = np.array([0.5, 3.0, 1.5], dtype=float)

    expectation = rank(scores, alpha=0.5, uniform_mix=0.0)

    assert expectation[1] == expectation.max()


def test_rank_returns_empty_array_for_empty_scores() -> None:
    expectation = rank([], alpha=0.5, uniform_mix=0.0)

    assert expectation.shape == (0,)


def test_rank_with_zero_uniform_mix_preserves_previous_distribution() -> None:
    expectation = rank([1.0, 2.0, 3.0], alpha=0.5, uniform_mix=0.0)
    expected = np.array([1.0 / np.sqrt(3.0), 1.0 / np.sqrt(2.0), 1.0], dtype=float)
    expected /= expected.sum()

    np.testing.assert_allclose(expectation, expected)


def test_rank_uniform_mix_blends_with_uniform_distribution() -> None:
    base = rank([1.0, 2.0, 3.0], alpha=0.5, uniform_mix=0.0)
    mixed = rank([1.0, 2.0, 3.0], alpha=0.5, uniform_mix=0.2)
    expected = 0.8 * base + np.full(3, 0.2 / 3.0, dtype=float)

    np.testing.assert_allclose(mixed, expected)


def test_rank_alpha_controls_selection_pressure() -> None:
    flat = rank([1.0, 2.0, 3.0], alpha=0.0, uniform_mix=0.0)
    steep = rank([1.0, 2.0, 3.0], alpha=1.0, uniform_mix=0.0)

    np.testing.assert_allclose(flat, np.full(3, 1.0 / 3.0, dtype=float))
    assert steep[2] > flat[2]
    assert steep[0] < flat[0]


@pytest.mark.parametrize(
    ("alpha", "uniform_mix"),
    [
        (-0.1, 0.0),
        (float("nan"), 0.0),
        (0.5, -0.1),
        (0.5, 1.1),
        (0.5, float("inf")),
    ],
)
def test_rank_validates_selection_parameters(alpha: float, uniform_mix: float) -> None:
    with pytest.raises(ValueError):
        rank([1.0, 2.0, 3.0], alpha=alpha, uniform_mix=uniform_mix)


def test_select_parents_returns_empty_array_for_zero_count() -> None:
    parents = select_parents(np.array([0.6, 0.4], dtype=float), 0)

    assert parents.shape == (0,)


def test_select_parents_returns_valid_indices_for_seeded_randomness() -> None:
    expectation = np.array([0.5, 0.3, 0.2], dtype=float)

    parents = select_parents(expectation, 5)

    assert parents.shape == (5,)
    assert parents.min() >= 0
    assert parents.max() < len(expectation)