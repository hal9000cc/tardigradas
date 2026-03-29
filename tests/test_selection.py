from __future__ import annotations

import numpy as np

from tardigradas.operators import rank, select_parents


def test_rank_returns_normalized_expectation() -> None:
    expectation = rank([1.0, 2.0, 3.0])

    assert expectation.shape == (3,)
    assert np.isclose(expectation.sum(), 1.0)


def test_rank_gives_highest_weight_to_best_score() -> None:
    scores = np.array([0.5, 3.0, 1.5], dtype=float)

    expectation = rank(scores)

    assert expectation[1] == expectation.max()


def test_select_parents_returns_empty_array_for_zero_count() -> None:
    parents = select_parents(np.array([0.6, 0.4], dtype=float), 0)

    assert parents.shape == (0,)


def test_select_parents_returns_valid_indices_for_seeded_randomness() -> None:
    expectation = np.array([0.5, 0.3, 0.2], dtype=float)

    parents = select_parents(expectation, 5)

    assert parents.shape == (5,)
    assert parents.min() >= 0
    assert parents.max() < len(expectation)