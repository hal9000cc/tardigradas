from __future__ import annotations

import numpy as np

from tests.helpers import create_engine


def test_state_dict_contains_critical_runtime_fields(engine) -> None:
    engine.population_init()
    engine.step()

    state = engine.state_dict()

    assert state["iterations"] == engine.iterations
    assert state["best_score"] == engine.best_score
    assert len(state["population"]) == engine.population_size
    assert np.array_equal(state["scores"], engine.scores)
    assert np.array_equal(state["step_best_individual"], engine.step_best_individual.chromo)


def test_restore_from_dict_restores_population_and_best_state(engine) -> None:
    engine.population_init()
    engine.step()
    state = engine.state_dict()

    restored = create_engine()
    restored.restore_from_dict(state)

    assert restored.iterations == engine.iterations
    assert restored.best_score == engine.best_score
    assert restored.best_iteration == engine.best_iteration
    assert restored.population_size == engine.population_size
    assert len(restored.population) == len(engine.population)
    assert np.array_equal(restored.best_individual.chromo, engine.best_individual.chromo)
    assert np.array_equal(restored.step_best_individual.chromo, engine.step_best_individual.chromo)
    assert np.array_equal(restored.scores, engine.scores)


def test_save_to_file_and_restore_from_file_round_trip(engine, tmp_path) -> None:
    engine.population_init()
    engine.step()
    file_name = tmp_path / "state.pkl"

    engine.save_to_file(str(file_name))

    restored = create_engine()
    restored.restore_from_file(str(file_name))

    assert restored.iterations == engine.iterations
    assert restored.best_score == engine.best_score
    assert np.array_equal(restored.population_chromosomes, engine.population_chromosomes)