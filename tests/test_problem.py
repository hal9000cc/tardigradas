from __future__ import annotations

import numpy as np
import pytest

from tardigradas import (
    EvaluationTaskResult,
    EvaluationTaskSpec,
    IndividualTaskState,
    TaskEvaluationContext,
    TaskSchedulingDecision,
)
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


def test_task_evaluation_types_are_publicly_available() -> None:
    context = TaskEvaluationContext(
        generation=3,
        individual_index=4,
        population_size=10,
        evaluation_number=2,
        attempt=5,
        task_id="task-1",
        task_number=7,
    )
    task = EvaluationTaskSpec(
        task_id="task-1",
        individual_index=4,
        generation=3,
        task_number=7,
        payload={"segment": 2},
    )
    result = EvaluationTaskResult(
        task_id="task-1",
        individual_index=4,
        ok=True,
        score=[1.5],
        payload={"status": "done"},
    )
    state = IndividualTaskState(
        individual_index=4,
        payload={"seed": 42},
        scheduled_count=1,
        completed_count=1,
        results=[result],
    )
    decision = TaskSchedulingDecision(
        new_tasks=[task],
        stop_individual=True,
        ready_to_aggregate=True,
        individual_failed=False,
    )

    assert context.task_id == "task-1"
    assert task.payload == {"segment": 2}
    assert result.score == [1.5]
    assert state.results == [result]
    assert decision.new_tasks == [task]
    assert decision.ready_to_aggregate is True


def test_problem_task_hooks_are_disabled_by_default() -> None:
    engine = create_engine(problem=DummyProblem)
    individual = engine.create_individual(chromo=[1.0, 2.0, 0.25])
    context = TaskEvaluationContext(generation=0, individual_index=0, population_size=engine.population_size)
    state = IndividualTaskState(individual_index=0)
    task = EvaluationTaskSpec(task_id="task-1", individual_index=0, generation=0, task_number=1)
    result = EvaluationTaskResult(task_id="task-1", individual_index=0, ok=True, payload={"ok": True})

    assert DummyProblem.has_evaluation_tasks() is False
    with pytest.raises(NotImplementedError):
        DummyProblem.init_task_state(individual, context)
    with pytest.raises(NotImplementedError):
        DummyProblem.initial_evaluation_tasks(individual, context, state)
    with pytest.raises(NotImplementedError):
        DummyProblem.evaluate_task(individual, task, context)
    with pytest.raises(NotImplementedError):
        DummyProblem.update_task_state(individual, context, state, result)
    with pytest.raises(NotImplementedError):
        DummyProblem.aggregate_task_results(individual, context, state)