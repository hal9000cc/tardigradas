from __future__ import annotations

import numpy as np

from tardigradas import CrossoverBitType, CrossoverFloatType, CrossoverPolicy
from tests.helpers import create_engine


def test_state_dict_contains_critical_runtime_fields(engine) -> None:
    engine.population_init()
    engine.step()
    assert engine.step_best_individual is not None

    state = engine.state_dict()

    assert state["iterations"] == engine.iterations
    assert state["best_score"] == engine.best_score
    assert len(state["population"]) == engine.population_size
    assert np.array_equal(state["scores"], engine.scores)
    assert np.array_equal(state["step_best_individual"], engine.step_best_individual.chromo)
    assert state["step_validate_score"] == engine.step_validate_score
    assert state["validate_scores_history"] == engine.validate_scores_history


def test_restore_from_dict_restores_population_and_best_state(engine) -> None:
    engine.population_init()
    engine.step()
    assert engine.best_individual is not None
    assert engine.step_best_individual is not None
    state = engine.state_dict()

    restored = create_engine()
    restored.restore_from_dict(state)
    assert restored.best_individual is not None
    assert restored.step_best_individual is not None

    assert restored.iterations == engine.iterations
    assert restored.best_score == engine.best_score
    assert restored.best_iteration == engine.best_iteration
    assert restored.population_size == engine.population_size
    assert len(restored.population) == len(engine.population)
    assert np.array_equal(restored.best_individual.chromo, engine.best_individual.chromo)
    assert np.array_equal(restored.step_best_individual.chromo, engine.step_best_individual.chromo)
    assert np.array_equal(restored.scores, engine.scores)
    assert restored.step_validate_score == engine.step_validate_score
    assert restored.validate_scores_history == engine.validate_scores_history


def test_save_to_file_and_restore_from_file_round_trip(engine, tmp_path) -> None:
    engine.population_init()
    engine.step()
    file_name = tmp_path / "state.pkl"

    engine.save_to_file(str(file_name))

    restored = create_engine()
    restored.restore_from_file(str(file_name))

    assert restored.iterations == engine.iterations
    assert restored.best_score == engine.best_score
    assert len(restored.population_chromosomes) == len(engine.population_chromosomes)
    for restored_chromo, original_chromo in zip(restored.population_chromosomes, engine.population_chromosomes):
        assert np.array_equal(restored_chromo, original_chromo)


def test_restore_from_dict_preserves_crossover_policy_and_adaptive_state() -> None:
    engine = create_engine(
        crossover_policy=CrossoverPolicy.adaptive(
            bit_candidates=[CrossoverBitType.uniform, CrossoverBitType.one_point],
            float_candidates=[CrossoverFloatType.uniform, CrossoverFloatType.arithmetic],
            reward="elite_survival",
        )
    )
    engine.population_init()
    engine._adaptive_bit_epoch_uses[CrossoverBitType.one_point] = 3
    engine._adaptive_bit_epoch_successes[CrossoverBitType.one_point] = 2
    engine._adaptive_bit_scores[CrossoverBitType.one_point] = 0.75
    engine.population_origins[0] = {
        "source": "crossover",
        "bit_operator": CrossoverBitType.one_point,
        "float_operator": CrossoverFloatType.arithmetic,
        "eligible_for_reward": True,
    }

    restored = create_engine()
    restored.restore_from_dict(engine.state_dict())

    assert restored.crossover_policy == engine.crossover_policy
    assert restored._adaptive_bit_epoch_uses == engine._adaptive_bit_epoch_uses
    assert restored._adaptive_bit_epoch_successes == engine._adaptive_bit_epoch_successes
    assert restored._adaptive_bit_scores == engine._adaptive_bit_scores
    assert restored.population_origins == engine.population_origins
