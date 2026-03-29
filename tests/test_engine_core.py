from __future__ import annotations

import tardigradas.engine as engine_module
import numpy as np
import pytest

from tardigradas import CrossoverBitType, CrossoverFloatType, CrossoverPolicy, Tardigradas, TardigradasException
from tests.helpers import (
    DummyProblem,
    FixedGenesProblem,
    NonNegativeFloatProblem,
    RejectAllProblem,
    VectorFitnessProblem,
    build_population,
    create_engine,
)


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
        {"crossover_policy": object()},
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
    assert len(engine.population_origins) == engine.population_size


def test_engine_uses_uniform_policy_by_default() -> None:
    engine = create_engine()

    assert engine.crossover_policy == CrossoverPolicy.explicit(
        bit=CrossoverBitType.uniform,
        float=CrossoverFloatType.uniform,
    )


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


def test_step_raises_when_population_is_not_initialized(engine) -> None:
    with pytest.raises(TardigradasException, match="population is not initialized"):
        engine.step()


def test_mutation_raises_when_all_genes_are_fixed() -> None:
    engine = create_engine(problem=FixedGenesProblem, population_size=2)

    with pytest.raises(TardigradasException, match="all genes are fixed"):
        engine.mutation(np.array([0], dtype=int))


def test_crossover_falls_back_to_mutation_for_equal_parents(engine, monkeypatch) -> None:
    engine.population = build_population(
        engine,
        [
            [1.0, 2.0, 0.5],
            [1.0, 2.0, 0.5],
        ],
    )

    mutation_child = engine.create_individual(chromo=[0.0, 4.0, 0.1])
    captured: dict[str, np.ndarray] = {}

    def fake_mutation(parent_indices: np.ndarray) -> list:
        captured["parent_indices"] = np.array(parent_indices, copy=True)
        return [mutation_child]

    monkeypatch.setattr(engine, "mutation", fake_mutation)

    kids = engine.crossover(np.array([0, 1], dtype=int))

    assert kids == [mutation_child]
    np.testing.assert_array_equal(captured["parent_indices"], np.array([0], dtype=int))


def test_crossover_uses_explicit_policy_for_bit_and_float_branches(monkeypatch) -> None:
    engine = create_engine(
        population_size=2,
        crossover_policy=CrossoverPolicy.explicit(
            bit=CrossoverBitType.two_point,
            float=CrossoverFloatType.BLX,
        ),
    )
    engine.population = build_population(
        engine,
        [
            [1.0, 1.0, -0.2],
            [0.0, 4.0, 0.8],
        ],
    )

    captured: dict[str, np.ndarray] = {}

    def fake_two_point(kid, parent1, parent2, gene_groups, gene_mask):
        captured["bit_mask"] = np.array(gene_mask, copy=True)
        kid[gene_mask] = parent2[gene_mask]
        return kid

    def fake_blx(kid, parent1, parent2, gene_mask, bounds_min, bounds_max, alpha=0.5):
        captured["float_mask"] = np.array(gene_mask, copy=True)
        kid[gene_mask] = np.array([1.7, 0.25], dtype=float)
        return kid

    monkeypatch.setattr(engine_module, "crossover_two_point", fake_two_point)
    monkeypatch.setattr(engine_module, "crossover_blx", fake_blx)

    kid = engine.crossover(np.array([0, 1], dtype=int))[0]

    np.testing.assert_array_equal(captured["bit_mask"], np.array([True, False, False]))
    np.testing.assert_array_equal(captured["float_mask"], np.array([False, True, True]))
    assert kid.chromo.tolist() == [0.0, 2.0, 0.25]


def test_adaptive_policy_rewards_only_elite_crossover_children() -> None:
    engine = create_engine(
        population_size=2,
        crossover_policy=CrossoverPolicy.adaptive(
            bit_candidates=[CrossoverBitType.uniform, CrossoverBitType.one_point],
            float_candidates=[CrossoverFloatType.uniform, CrossoverFloatType.arithmetic],
            reward="elite_survival",
        ),
    )
    engine.population = build_population(
        engine,
        [
            [1.0, 1.0, 0.4],
            [0.0, 0.0, -0.4],
        ],
    )
    engine.population_origins = [
        {
            "source": "crossover",
            "bit_operator": CrossoverBitType.one_point,
            "float_operator": CrossoverFloatType.arithmetic,
            "eligible_for_reward": True,
        },
        {
            "source": "mutation",
            "bit_operator": None,
            "float_operator": None,
            "eligible_for_reward": False,
        },
    ]

    engine._update_adaptive_crossover_statistics(np.array([0], dtype=int))

    assert engine._adaptive_bit_successes[CrossoverBitType.one_point] == 1
    assert engine._adaptive_float_successes[CrossoverFloatType.arithmetic] == 1
    assert engine.population_origins[0]["eligible_for_reward"] is False


def test_adaptive_policy_biases_selection_towards_more_successful_operator(monkeypatch) -> None:
    engine = create_engine(
        crossover_policy=CrossoverPolicy.adaptive(
            bit_candidates=[CrossoverBitType.uniform, CrossoverBitType.one_point],
            float_candidates=[CrossoverFloatType.uniform],
            reward="elite_survival",
        ),
    )
    engine._adaptive_bit_uses[CrossoverBitType.uniform] = 6
    engine._adaptive_bit_successes[CrossoverBitType.uniform] = 0
    engine._adaptive_bit_uses[CrossoverBitType.one_point] = 2
    engine._adaptive_bit_successes[CrossoverBitType.one_point] = 2

    captured: dict[str, np.ndarray] = {}

    def fake_choice(options, p):
        captured["p"] = np.array(p, copy=True)
        return 1

    monkeypatch.setattr(np.random, "choice", fake_choice)

    operator = engine._select_bit_crossover_operator()

    assert operator == CrossoverBitType.one_point
    assert captured["p"][1] > captured["p"][0]


def test_step_updates_best_iteration_only_on_improvement(monkeypatch) -> None:
    engine = create_engine(population_size=2, crossover_fraction=0.0, n_elits=1)
    engine.population = build_population(
        engine,
        [
            [1.0, 5.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
    )

    score_batches = iter(
        [
            np.array([[1.0], [0.0]], dtype=float),
            np.array([[2.0], [0.0]], dtype=float),
            np.array([[1.5], [0.0]], dtype=float),
        ]
    )

    def fake_estimate_population() -> None:
        full_scores = next(score_batches)
        engine.full_scores = full_scores
        engine.scores = full_scores[:, 0]

    monkeypatch.setattr(engine, "estimate_population", fake_estimate_population)
    monkeypatch.setattr(
        engine,
        "mutation",
        lambda parent_indices: [
            engine.create_individual(chromo=[0.0, float(i), 0.0]) for i, _ in enumerate(parent_indices)
        ],
    )
    monkeypatch.setattr(engine, "kill_doubles", lambda: None)

    for _ in range(3):
        engine.step()

    assert engine.best_score == pytest.approx(2.0)
    assert engine.best_iteration == 1
    assert engine.scores_history == [1.0, 2.0, 1.5]


def test_step_keeps_scores_histories_in_sync(engine) -> None:
    engine.population_init()

    for _ in range(3):
        engine.step()

    assert len(engine.scores_history) == len(engine.custom_scores_history) == 3
    for score, custom_score in zip(engine.scores_history, engine.custom_scores_history):
        assert custom_score.shape == (1,)
        assert score == pytest.approx(custom_score[0])


def test_step_tracks_vector_fitness_scores() -> None:
    engine = create_engine(problem=VectorFitnessProblem)
    engine.population_init()

    engine.step()

    assert engine.full_scores.shape == (engine.population_size, 2)
    assert engine.step_custom_score is not None
    assert engine.step_custom_score.shape == (2,)
    assert engine.step_score == pytest.approx(engine.step_custom_score[0])
    np.testing.assert_allclose(engine.custom_scores_history[0], engine.step_custom_score)


def test_step_uses_expected_number_of_parents(monkeypatch) -> None:
    engine = create_engine(
        population_size=6,
        crossover_fraction=0.4,
        fresh_blood_fraction=0.2,
        n_elits=1,
    )
    engine.population = build_population(
        engine,
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0],
            [1.0, 5.0, 1.0],
        ],
    )

    captured: dict[str, np.ndarray | int] = {}
    fresh_blood_calls = {"count": 0}

    def fake_select_parents(expectation: np.ndarray, count: int) -> np.ndarray:
        captured["expectation"] = np.array(expectation, copy=True)
        captured["count"] = count
        return np.arange(count, dtype=int) % len(expectation)

    def fake_new_valid_individual(use_defaults: bool = False):
        fresh_blood_calls["count"] += 1
        return engine.create_individual(chromo=[1.0, 5.0, 0.7])

    monkeypatch.setattr(engine_module, "select_parents", fake_select_parents)
    monkeypatch.setattr(np.random, "permutation", lambda values: np.array(values, copy=True))
    monkeypatch.setattr(
        engine,
        "crossover",
        lambda parent_indices: [
            engine.create_individual(chromo=[1.0, float(i), 0.1 * (i + 1)])
            for i in range(len(parent_indices) // 2)
        ],
    )
    monkeypatch.setattr(
        engine,
        "mutation",
        lambda parent_indices: [
            engine.create_individual(chromo=[0.0, float(i), -0.1 * (i + 1)])
            for i, _ in enumerate(parent_indices)
        ],
    )
    monkeypatch.setattr(engine, "new_valid_individual", fake_new_valid_individual)
    monkeypatch.setattr(engine, "kill_doubles", lambda: None)

    engine.step()

    assert captured["count"] == 6
    assert captured["expectation"].shape == (engine.population_size,)
    assert fresh_blood_calls["count"] == 1


def test_step_preserves_population_size_across_elitism_crossover_mutation_and_fresh_blood(monkeypatch) -> None:
    engine = create_engine(
        population_size=6,
        crossover_fraction=0.4,
        fresh_blood_fraction=0.2,
        n_elits=1,
    )
    engine.population = build_population(
        engine,
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0],
            [1.0, 5.0, 1.0],
        ],
    )

    crossover_children = build_population(
        engine,
        [
            [1.0, 0.0, 0.1],
            [1.0, 1.0, 0.2],
        ],
    )
    mutation_children = build_population(
        engine,
        [
            [0.0, 5.0, -0.1],
            [1.0, 2.0, 0.3],
        ],
    )
    fresh_children = iter(build_population(engine, [[0.0, 5.0, 0.9]]))

    monkeypatch.setattr(engine, "crossover", lambda parent_indices: crossover_children)
    monkeypatch.setattr(engine, "mutation", lambda parent_indices: mutation_children)
    monkeypatch.setattr(engine, "new_valid_individual", lambda use_defaults=False: next(fresh_children))
    monkeypatch.setattr(engine, "kill_doubles", lambda: None)

    engine.step()

    assert len(engine.population) == engine.population_size
    assert [individual.chromo.tolist() for individual in engine.population] == [
        [1.0, 5.0, 1.0],
        [1.0, 0.0, 0.1],
        [1.0, 1.0, 0.2],
        [0.0, 5.0, -0.1],
        [1.0, 2.0, 0.3],
        [0.0, 5.0, 0.9],
    ]


def test_step_replaces_invalid_offspring_after_generation(monkeypatch) -> None:
    engine = create_engine(
        problem=NonNegativeFloatProblem,
        population_size=4,
        crossover_fraction=0.34,
        n_elits=1,
    )
    engine.population = build_population(
        engine,
        [
            [0.0, 0.0, 0.1],
            [0.0, 1.0, 0.2],
            [0.0, 2.0, 0.3],
            [1.0, 5.0, 1.0],
        ],
    )

    invalid_crossover = build_population(engine, [[0.0, 3.0, -0.5]])
    invalid_mutation = build_population(
        engine,
        [
            [0.0, 4.0, -0.6],
            [1.0, 2.0, -0.7],
        ],
    )
    replacements = iter(
        build_population(
            engine,
            [
                [1.0, 0.0, 0.5],
                [1.0, 1.0, 0.6],
                [1.0, 2.0, 0.7],
            ],
        )
    )
    replacement_calls = {"count": 0}

    def fake_new_valid_individual(use_defaults: bool = False):
        replacement_calls["count"] += 1
        return next(replacements)

    monkeypatch.setattr(engine, "crossover", lambda parent_indices: invalid_crossover)
    monkeypatch.setattr(engine, "mutation", lambda parent_indices: invalid_mutation)
    monkeypatch.setattr(engine, "new_valid_individual", fake_new_valid_individual)
    monkeypatch.setattr(engine, "kill_doubles", lambda: None)

    engine.step()

    assert replacement_calls["count"] == 3
    assert all(individual.chromo_valid() for individual in engine.population)
    assert [individual.chromo.tolist() for individual in engine.population[1:]] == [
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 0.6],
        [1.0, 2.0, 0.7],
    ]


def test_step_kill_doubles_is_applied_after_generation(monkeypatch) -> None:
    engine = create_engine(
        population_size=4,
        crossover_fraction=0.34,
        n_elits=1,
    )
    engine.population = build_population(
        engine,
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.1],
            [0.0, 2.0, 0.2],
            [1.0, 5.0, 1.0],
        ],
    )

    duplicate_chromosome = [0.0, 1.0, 0.5]
    replacements = iter(
        build_population(
            engine,
            [
                [1.0, 0.0, 0.5],
                [1.0, 1.0, 0.6],
            ],
        )
    )
    kill_doubles_called = {"value": False}
    original_kill_doubles = engine.kill_doubles

    def fake_new_valid_individual(use_defaults: bool = False):
        return next(replacements)

    def wrapped_kill_doubles() -> None:
        kill_doubles_called["value"] = True
        original_kill_doubles()

    monkeypatch.setattr(
        engine,
        "crossover",
        lambda parent_indices: build_population(engine, [duplicate_chromosome]),
    )
    monkeypatch.setattr(
        engine,
        "mutation",
        lambda parent_indices: build_population(engine, [duplicate_chromosome, duplicate_chromosome]),
    )
    monkeypatch.setattr(engine, "new_valid_individual", fake_new_valid_individual)
    monkeypatch.setattr(engine, "kill_doubles", wrapped_kill_doubles)

    engine.step()

    chromosomes = [individual.chromo.tobytes() for individual in engine.population]

    assert kill_doubles_called["value"] is True
    assert len(set(chromosomes)) == engine.population_size
    assert engine.n_killed_doubles == 2