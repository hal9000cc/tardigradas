from __future__ import annotations

import numpy as np

from tardigradas import GenType
from tardigradas.operators import mutate_chromosome, mutation_gauss


def test_mutation_gauss_returns_same_value_for_fixed_gene() -> None:
    assert mutation_gauss(3.0, 3.0, 3.0) == 3.0


def test_mutation_gauss_clips_to_bounds(monkeypatch) -> None:
    monkeypatch.setattr(np.random, "normal", lambda mean, scale: 1000.0)

    mutated = mutation_gauss(0.0, -1.0, 1.0)

    assert mutated == 1.0


def test_mutate_chromosome_changes_only_mutable_positions_and_preserves_types(monkeypatch) -> None:
    monkeypatch.setattr(np.random, "choice", lambda values, size, replace=False: np.array([0, 1, 2], dtype=int))
    monkeypatch.setattr(np.random, "normal", lambda mean, scale: mean + scale)

    child = mutate_chromosome(
        parent_chromo=np.array([0.0, 2.0, 0.5], dtype=float),
        gen_types=np.array([GenType.bit.value, GenType.int.value, GenType.float.value], dtype=int),
        bounds_min=np.array([0.0, 0.0, -1.0], dtype=float),
        bounds_max=np.array([1.0, 5.0, 1.0], dtype=float),
        mutable_positions=np.array([0, 1, 2], dtype=int),
        n_mutation=3,
    )

    assert child[0] == 1.0
    assert child[1] == 3.0
    assert float(child[1]).is_integer()
    assert child[2] == 1.0


def test_mutate_chromosome_leaves_immutable_positions_untouched(monkeypatch) -> None:
    monkeypatch.setattr(np.random, "choice", lambda values, size, replace=False: np.array([2], dtype=int))
    monkeypatch.setattr(np.random, "normal", lambda mean, scale: mean - scale)

    child = mutate_chromosome(
        parent_chromo=np.array([1.0, 4.0, 0.5], dtype=float),
        gen_types=np.array([GenType.bit.value, GenType.int.value, GenType.float.value], dtype=int),
        bounds_min=np.array([0.0, 0.0, -1.0], dtype=float),
        bounds_max=np.array([1.0, 5.0, 1.0], dtype=float),
        mutable_positions=np.array([2], dtype=int),
        n_mutation=1,
    )

    assert np.array_equal(child[:2], np.array([1.0, 4.0], dtype=float))
    assert -1.0 <= child[2] <= 1.0