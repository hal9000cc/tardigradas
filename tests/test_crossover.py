from __future__ import annotations

import numpy as np

from tardigradas.operators import (
    crossover_arithmetic,
    crossover_blx,
    crossover_one_point,
    crossover_two_point,
    crossover_uniform,
)


def test_crossover_uniform_uses_only_parent_genes_and_keeps_group_together(monkeypatch) -> None:
    calls = iter(
        [
            np.array([0.8, 0.2], dtype=float),
            np.array([0.9], dtype=float),
        ]
    )

    monkeypatch.setattr(np.random, "random", lambda size=None: next(calls))

    kid = np.zeros(4, dtype=float)
    result = crossover_uniform(
        kid=kid,
        parent1=np.array([10.0, 11.0, 20.0, 21.0], dtype=float),
        parent2=np.array([100.0, 101.0, 200.0, 201.0], dtype=float),
        gene_groups=np.array([0, 0, 1, 1], dtype=int),
        gene_mask=np.array([True, True, True, True]),
    )

    assert np.array_equal(result, np.array([100.0, 11.0, 200.0, 201.0], dtype=float))


def test_crossover_blx_respects_bounds_and_gene_mask() -> None:
    kid = np.array([5.0, -5.0, 42.0], dtype=float)
    mask = np.array([True, True, False])

    result = crossover_blx(
        kid=kid,
        parent1=np.array([0.0, 1.0, 10.0], dtype=float),
        parent2=np.array([2.0, 3.0, 20.0], dtype=float),
        gene_mask=mask,
        bounds_min=np.array([-1.0, 0.0, 0.0], dtype=float),
        bounds_max=np.array([3.0, 4.0, 100.0], dtype=float),
        alpha=0.5,
    )

    assert -1.0 <= result[0] <= 3.0
    assert 0.0 <= result[1] <= 4.0
    assert result[2] == 42.0


def test_crossover_one_point_keeps_group_together(monkeypatch) -> None:
    randint_values = iter([2])
    monkeypatch.setattr(np.random, "random", lambda size=None: 0.1)
    monkeypatch.setattr(np.random, "randint", lambda low, high=None, size=None: next(randint_values))

    kid = np.zeros(4, dtype=float)
    result = crossover_one_point(
        kid=kid,
        parent1=np.array([10.0, 11.0, 12.0, 13.0], dtype=float),
        parent2=np.array([100.0, 101.0, 102.0, 103.0], dtype=float),
        gene_groups=np.array([0, 1, 1, 0], dtype=int),
        gene_mask=np.array([True, True, True, True]),
    )

    assert np.array_equal(result, np.array([10.0, 11.0, 12.0, 103.0], dtype=float))


def test_crossover_two_point_keeps_group_together(monkeypatch) -> None:
    randint_values = iter([1, 3])
    monkeypatch.setattr(np.random, "random", lambda size=None: 0.1)
    monkeypatch.setattr(np.random, "randint", lambda low, high=None, size=None: next(randint_values))

    kid = np.zeros(5, dtype=float)
    result = crossover_two_point(
        kid=kid,
        parent1=np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float),
        parent2=np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=float),
        gene_groups=np.array([0, 1, 1, 0, 0], dtype=int),
        gene_mask=np.array([True, True, True, True, True]),
    )

    assert np.array_equal(result, np.array([10.0, 101.0, 102.0, 103.0, 14.0], dtype=float))


def test_crossover_arithmetic_blends_values_within_bounds() -> None:
    kid = np.array([99.0, -99.0, 42.0], dtype=float)
    mask = np.array([True, True, False])

    result = crossover_arithmetic(
        kid=kid,
        parent1=np.array([0.0, 4.0, 10.0], dtype=float),
        parent2=np.array([2.0, 8.0, 20.0], dtype=float),
        gene_mask=mask,
        bounds_min=np.array([-1.0, 0.0, 0.0], dtype=float),
        bounds_max=np.array([3.0, 10.0, 100.0], dtype=float),
        alpha=0.25,
    )

    np.testing.assert_allclose(result[:2], np.array([1.5, 7.0], dtype=float))
    assert result[2] == 42.0