from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import EmptyFitnessProblem, VariableLengthProblem, VectorFitnessProblem, create_engine


def test_getitem_returns_expected_python_types(engine) -> None:
    individual = engine.create_individual(chromo=[1.0, 3.0, 0.25])

    assert individual[0] == 1
    assert isinstance(individual[0], int)
    assert individual[1] == 3
    assert isinstance(individual[1], int)
    assert individual[2] == 0.25
    assert isinstance(individual[2], float)


def test_getitem_rejects_non_integer_index(engine) -> None:
    individual = engine.create_individual(chromo=[1.0, 3.0, 0.25])

    with pytest.raises(TypeError, match="gene index must be int"):
        _ = individual["0"]


def test_fitness_wraps_scalar_result_into_array(engine) -> None:
    individual = engine.create_individual(chromo=[1.0, 3.0, 0.25])

    fitness = individual.fitness()

    assert np.array_equal(fitness, np.array([4.25], dtype=float))


def test_fitness_accepts_vector_result() -> None:
    engine = create_engine(problem=VectorFitnessProblem)
    individual = engine.create_individual(chromo=[1.0, 3.0, 0.25])

    fitness = individual.fitness()

    assert np.array_equal(fitness, np.array([4.0, 0.25], dtype=float))


def test_fitness_rejects_empty_result() -> None:
    engine = create_engine(problem=EmptyFitnessProblem)
    individual = engine.create_individual(chromo=[1.0, 3.0, 0.25])

    with pytest.raises(ValueError, match="fitness must return at least one numeric value"):
        individual.fitness()


def test_random_chromosome_respects_gene_bounds_and_types(engine) -> None:
    individual = engine.create_individual()

    assert individual[0] in {0, 1}
    assert 0 <= individual[1] <= 5
    assert float(individual[1]).is_integer()
    assert -1.0 <= individual[2] <= 1.0


def test_random_chromosome_can_apply_defaults(defaults_engine) -> None:
    individual = defaults_engine.create_individual(use_defaults=True)

    assert np.array_equal(individual.chromo, np.array([1.0, 2.0, 0.25], dtype=float))


def test_random_chromosome_can_use_problem_defined_length() -> None:
    engine = create_engine(problem=VariableLengthProblem)

    individual = engine.create_individual()

    assert len(individual.chromo) == 2