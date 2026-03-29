from __future__ import annotations

import numpy as np

from tardigradas.operators import crossover_blx, crossover_uniform


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