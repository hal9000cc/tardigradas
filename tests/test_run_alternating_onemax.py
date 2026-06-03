from __future__ import annotations

import numpy as np

from benchmarks.run_alternating_onemax import build_alternating_onemax_problem
from tests.helpers import create_engine


def test_alternating_onemax_fitness_inverts_every_second_bit() -> None:
    problem = build_alternating_onemax_problem(
        chromo_size=6,
        use_all_ones_initialization=False,
    )
    engine = create_engine(problem=problem)
    individual = engine.create_individual(chromo=[1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    assert problem.fitness(individual) == 4.0


def test_alternating_onemax_can_initialize_with_all_ones() -> None:
    problem = build_alternating_onemax_problem(
        chromo_size=6,
        use_all_ones_initialization=True,
    )
    engine = create_engine(problem=problem)
    individual = engine.create_individual(use_defaults=True)

    assert np.array_equal(individual.chromo, np.ones(6, dtype=float))