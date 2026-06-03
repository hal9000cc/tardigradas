from __future__ import annotations

import time
from typing import Any, cast

import numpy as np
import pytest

import tardigradas.evaluation as evaluation_module
from tardigradas import (
    ChromosomeSchema,
    EvaluationConfig,
    GenType,
    IncompleteEpochError,
    Individual,
    Problem,
    Tardigradas,
    TardigradasException,
)
from tests.helpers import build_population, create_engine


class ImportableEvaluationProblem(Problem):
    @staticmethod
    def init_environment(tardigradas: Tardigradas) -> None:
        return None

    @staticmethod
    def gen_info(tardigradas: Tardigradas) -> ChromosomeSchema:
        return ChromosomeSchema(
            gen_types=[GenType.float],
            bounds=([-10.0], [10.0]),
        )

    @staticmethod
    def fitness(individual: Individual) -> list[float]:
        context = individual.evaluation_context
        if context is None:
            return [float(individual.chromo[0])]
        return [float(individual.chromo[0]), float(context.individual_index), float(context.attempt)]


class RetryEvaluationProblem(ImportableEvaluationProblem):
    @staticmethod
    def fitness(individual: Individual) -> float:
        context = individual.evaluation_context
        if context is not None and context.attempt < 3:
            raise RuntimeError("temporary failure")
        return float(individual.chromo[0])


class MissingEvaluationProblem(ImportableEvaluationProblem):
    @staticmethod
    def fitness(individual: Individual) -> float:
        context = individual.evaluation_context
        if context is not None and context.individual_index == 1:
            raise RuntimeError("permanent failure")
        return float(individual.chromo[0])


def test_evaluation_config_validates_values() -> None:
    with pytest.raises(ValueError, match="workers"):
        EvaluationConfig(workers=0)

    with pytest.raises(ValueError, match="max_attempts"):
        EvaluationConfig(max_attempts=0)


def test_engine_accepts_evaluation_config() -> None:
    config = EvaluationConfig(workers=2, max_attempts=3)
    engine = create_engine(evaluation=config)

    assert engine.evaluation == config


def test_parallel_evaluation_matches_expected_scores() -> None:
    engine = create_engine(
        problem=ImportableEvaluationProblem,
        population_size=3,
        n_elits=1,
        evaluation=EvaluationConfig(workers=2),
    )
    engine.population = build_population(engine, [[1.0], [2.0], [3.0]])

    engine.estimate_population()

    np.testing.assert_allclose(engine.full_scores[:, 0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(engine.full_scores[:, 1], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(engine.full_scores[:, 2], np.array([1.0, 1.0, 1.0]))
    assert engine.evaluation_state is None


def test_parallel_evaluation_retries_failed_workers() -> None:
    engine = create_engine(
        problem=RetryEvaluationProblem,
        population_size=2,
        n_elits=1,
        evaluation=EvaluationConfig(workers=2, max_attempts=3),
    )
    engine.population = build_population(engine, [[1.0], [2.0]])

    engine.estimate_population()

    np.testing.assert_allclose(engine.scores, np.array([1.0, 2.0]))
    assert engine.evaluation_state is None


def test_parallel_evaluation_starts_next_task_when_one_worker_finishes(monkeypatch) -> None:
    engine = create_engine(
        problem=ImportableEvaluationProblem,
        population_size=3,
        n_elits=1,
        evaluation=EvaluationConfig(workers=2),
    )
    engine.population = build_population(engine, [[1.0], [2.0], [3.0]])

    class FakeProcess:
        def __init__(self, duration: float) -> None:
            self.started_at = time.perf_counter()
            self.duration = duration
            self.returncode: int | None = None

        def poll(self) -> int | None:
            if self.returncode is None and time.perf_counter() - self.started_at >= self.duration:
                self.returncode = 0
            return self.returncode

    durations = {0: 0.20, 1: 0.02, 2: 0.20}
    start_times: dict[int, float] = {}

    def fake_start_worker(temp_dir, current_engine, index, attempt, problem_module, problem_qualified_name):
        start_times[index] = time.perf_counter()
        return evaluation_module._RunningTask(
            process=cast(Any, FakeProcess(durations[index])),
            index=index,
            attempt=attempt,
            response_path=temp_dir / f"response_{index}_{attempt}.pkl",
        )

    monkeypatch.setattr(evaluation_module, "_start_worker", fake_start_worker)
    monkeypatch.setattr(evaluation_module, "_read_worker_score", lambda task: [float(task.index)])

    engine.estimate_population()

    assert start_times[2] < start_times[0] + durations[0]
    np.testing.assert_allclose(engine.scores, np.array([0.0, 1.0, 2.0]))


def test_parallel_evaluation_reports_incomplete_epoch_after_all_possible_work() -> None:
    engine = create_engine(
        problem=MissingEvaluationProblem,
        population_size=3,
        n_elits=1,
        evaluation=EvaluationConfig(workers=2, max_attempts=2),
    )
    engine.population = build_population(engine, [[1.0], [2.0], [3.0]])

    with pytest.raises(IncompleteEpochError) as error:
        engine.estimate_population()

    assert error.value.missing_indices == [1]
    assert engine.evaluation_state is not None
    evaluation_state = cast(dict[str, Any], engine.evaluation_state)
    scores = cast(list[Any], evaluation_state["scores"])
    assert evaluation_state["phase"] == "incomplete_population"
    assert scores[0] == [1.0]
    assert scores[1] is None
    assert scores[2] == [3.0]
    assert evaluation_state["attempts"] == [1, 2, 1]


def test_parallel_evaluation_restores_partial_state_and_only_calculates_missing_scores() -> None:
    engine = create_engine(
        problem=ImportableEvaluationProblem,
        population_size=3,
        n_elits=1,
        evaluation=EvaluationConfig(workers=2),
    )
    engine.population = build_population(engine, [[1.0], [2.0], [3.0]])
    engine.evaluation_state = {
        "phase": "evaluating_population",
        "generation": engine.iterations,
        "population_signatures": [individual.chromo.tobytes() for individual in engine.population],
        "scores": [[99.0, 99.0, 99.0], None, None],
        "attempts": [1, 0, 0],
        "max_attempts": 3,
        "missing_indices": [],
    }

    state = engine.state_dict()
    restored = create_engine(
        problem=ImportableEvaluationProblem,
        population_size=3,
        n_elits=1,
        evaluation=EvaluationConfig(workers=2),
    )
    restored.restore_from_dict(state)

    restored.estimate_population()

    np.testing.assert_allclose(restored.full_scores[0], np.array([99.0, 99.0, 99.0]))
    np.testing.assert_allclose(restored.full_scores[1], np.array([2.0, 1.0, 1.0]))
    np.testing.assert_allclose(restored.full_scores[2], np.array([3.0, 2.0, 1.0]))


def test_subprocess_evaluation_rejects_local_problem_class() -> None:
    class LocalProblem(ImportableEvaluationProblem):
        pass

    engine = create_engine(
        problem=LocalProblem,
        population_size=2,
        n_elits=1,
        evaluation=EvaluationConfig(workers=2),
    )
    engine.population = build_population(engine, [[1.0], [2.0]])

    with pytest.raises(TardigradasException, match="importable class"):
        engine.estimate_population()
