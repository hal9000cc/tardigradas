from __future__ import annotations

import numpy as np
import pytest

from tardigradas import Tardigradas, TardigradasException
from tests.helpers import DummyProblem, RejectAllProblem, create_engine


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": 0},
        {"population_size": -1},
        {"crossover_fraction": -0.1},
        {"fresh_blood_fraction": -0.1},
        {"gen_mutation_fraction": -0.1},
        {"crossover_fraction": 0.8, "fresh_blood_fraction": 0.3},
        {"n_elits": -1},
        {"n_elits": 6},
    ],
)
def test_engine_init_validates_parameters(kwargs: dict[str, float]) -> None:
    params = {
        "problem": DummyProblem,
        "population_size": 6,
        "crossover_fraction": 0.5,
        "fresh_blood_fraction": 0.0,
        "gen_mutation_fraction": 0.25,
        "n_elits": 1,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        Tardigradas(**params)


def test_population_init_creates_population_and_resets_runtime_state(engine) -> None:
    engine.iterations = 99
    engine.scores_history = [1.0]
    engine.custom_scores_history = [np.array([1.0], dtype=float)]
    engine.best_score = 5.0
    engine.best_iteration = 10
    engine.best_individual = engine.create_individual(chromo=[1.0, 2.0, 0.5])
    engine.step_best_individual = engine.best_individual
    engine.step_score = 5.0
    engine.step_custom_score = np.array([5.0], dtype=float)
    engine.scores = np.array([5.0], dtype=float)
    engine.full_scores = np.array([[5.0]], dtype=float)

    engine.population_init()

    assert len(engine.population) == engine.population_size
    assert engine.iterations == 0
    assert engine.scores_history == []
    assert engine.custom_scores_history == []
    assert engine.best_score is None
    assert engine.best_individual is None
    assert engine.step_best_individual is None
    assert engine.step_score is None
    assert engine.step_custom_score is None
    assert engine.scores.shape == (0,)
    assert engine.full_scores.shape == (0, 1)


def test_new_valid_individual_returns_valid_individual(engine) -> None:
    individual = engine.new_valid_individual()

    assert individual.chromo_valid()


def test_new_valid_individual_raises_for_impossible_problem() -> None:
    engine = create_engine(problem=RejectAllProblem)

    with pytest.raises(TardigradasException, match="can't create a new random chromosome"):
        engine.new_valid_individual()


def test_kill_doubles_replaces_duplicates_and_counts_them(engine, monkeypatch) -> None:
    engine.population = [
        engine.create_individual(chromo=[1.0, 2.0, 0.1]),
        engine.create_individual(chromo=[1.0, 2.0, 0.1]),
        engine.create_individual(chromo=[0.0, 4.0, -0.3]),
    ]

    replacement = engine.create_individual(chromo=[0.0, 1.0, 0.9])
    monkeypatch.setattr(engine, "new_valid_individual", lambda use_defaults=False: replacement)

    engine.kill_doubles()

    chromosomes = {individual.chromo.tobytes() for individual in engine.population}

    assert len(chromosomes) == 3
    assert engine.n_killed_doubles == 1


def test_population_chromosomes_returns_expected_shape(engine) -> None:
    assert engine.population_chromosomes.shape == (0, engine.chromo_size)

    engine.population = [
        engine.create_individual(chromo=[1.0, 2.0, 0.0]),
        engine.create_individual(chromo=[0.0, 3.0, 0.5]),
    ]

    chromosomes = engine.population_chromosomes

    assert chromosomes.shape == (2, engine.chromo_size)