from __future__ import annotations

import numpy as np

from tests.helpers import DummyProblem, TaggedIndividual, TaggedProblem, create_engine


def test_problem_is_equal_supports_arrays_and_individuals(engine) -> None:
    chromo = np.array([1.0, 2.0, 0.5], dtype=float)
    individual = engine.create_individual(chromo=chromo)

    assert DummyProblem.is_equal(chromo, chromo.copy())
    assert DummyProblem.is_equal(individual, chromo)
    assert not DummyProblem.is_equal(individual, np.array([0.0, 2.0, 0.5], dtype=float))


def test_problem_create_individual_uses_custom_individual_class() -> None:
    engine = create_engine(problem=TaggedProblem)

    individual = engine.create_individual(chromo=[1.0, 3.0, 0.25])

    assert isinstance(individual, TaggedIndividual)
    assert individual.tag == "custom"