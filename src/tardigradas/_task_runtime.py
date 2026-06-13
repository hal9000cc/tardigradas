from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .evaluation import (
    EvaluationConfig,
    _RunningTask,
    _subprocess_environment,
    collect_scores_from_state,
    normalize_fitness_score,
    population_signatures,
    problem_import_path,
)
from .exceptions import EvaluationFailure, IncompleteEpochError, TardigradasException
from .task_evaluation import EvaluationTaskResult, EvaluationTaskSpec, IndividualTaskState, TaskEvaluationContext, TaskSchedulingDecision

if TYPE_CHECKING:
    from .engine import Tardigradas


def _base_context(engine: Tardigradas, individual_index: int) -> TaskEvaluationContext:
    return TaskEvaluationContext(
        generation=int(engine.iterations),
        individual_index=int(individual_index),
        population_size=int(engine.population_size),
    )


def _task_context(engine: Tardigradas, task: EvaluationTaskSpec, attempt: int) -> TaskEvaluationContext:
    return TaskEvaluationContext(
        generation=int(engine.iterations),
        individual_index=int(task.individual_index),
        population_size=int(engine.population_size),
        evaluation_number=1,
        attempt=int(attempt),
        task_id=task.task_id,
        task_number=int(task.task_number),
    )


def _ensure_task_state(state: object, individual_index: int) -> IndividualTaskState:
    if not isinstance(state, IndividualTaskState):
        raise TypeError("task evaluation hooks must return IndividualTaskState instances")
    if int(state.individual_index) != int(individual_index):
        raise TardigradasException("IndividualTaskState.individual_index does not match the population index")
    return state


def _validate_task_spec(engine: Tardigradas, task: EvaluationTaskSpec) -> None:
    if not isinstance(task, EvaluationTaskSpec):
        raise TypeError("task evaluation hooks must return EvaluationTaskSpec instances")
    if not isinstance(task.task_id, str) or not task.task_id:
        raise TardigradasException("task_id must be a non-empty string")
    if not isinstance(task.payload, dict):
        raise TardigradasException("task payload must be a dictionary")
    if int(task.individual_index) < 0 or int(task.individual_index) >= len(engine.population):
        raise TardigradasException("task individual_index is outside the current population")
    if int(task.generation) != int(engine.iterations):
        raise TardigradasException("task generation does not match the current engine iteration")


def _validate_task_decision(decision: TaskSchedulingDecision) -> None:
    if not isinstance(decision, TaskSchedulingDecision):
        raise TypeError("update_task_state() must return TaskSchedulingDecision")
    if decision.new_tasks and (decision.ready_to_aggregate or decision.stop_individual or decision.individual_failed):
        raise TardigradasException("task scheduling decision cannot finalize an individual and schedule new tasks at once")


def _refresh_task_progress(state: dict[str, Any]) -> None:
    task_states = [task_state for task_state in state.get("individual_states", []) if isinstance(task_state, IndividualTaskState)]
    state["task_progress"] = {
        "completed_tasks": int(sum(task_state.completed_count for task_state in task_states)),
        "scheduled_tasks": int(sum(task_state.scheduled_count for task_state in task_states)),
        "running_tasks": int(len(state.get("in_flight", {}))),
        "completed_individuals": int(sum(score is not None for score in state.get("scores", []))),
        "population_size": int(len(state.get("scores", []))),
    }


def _notify_progress(engine: Tardigradas, state: dict[str, Any]) -> None:
    if engine.fitness_progress_fun is None:
        return
    population_size = max(1, int(len(state.get("scores", []))))
    completed_individuals = int(sum(score is not None for score in state.get("scores", [])))
    missing_individuals = len({int(index) for index in state.get("missing_indices", [])})
    engine.fitness_progress_fun(engine, (completed_individuals + missing_individuals) / population_size)


def _record_task_error(
    state: dict[str, Any],
    task: EvaluationTaskSpec,
    result: EvaluationTaskResult,
    attempt: int,
) -> None:
    state.setdefault("last_errors", {})[task.task_id] = {
        "task_id": task.task_id,
        "individual_index": int(task.individual_index),
        "retryable": bool(result.retryable),
        "failure_kind": result.failure_kind,
        "error_message": result.error_message,
        "attempt": int(attempt),
    }


def _mark_missing(
    engine: Tardigradas,
    state: dict[str, Any],
    individual_index: int,
    failure_message: str | None = None,
) -> None:
    missing_indices = state.setdefault("missing_indices", [])
    if int(individual_index) not in [int(index) for index in missing_indices]:
        missing_indices.append(int(individual_index))
    if failure_message is not None:
        state.setdefault("last_errors", {})[f"individual:{int(individual_index)}"] = {
            "individual_index": int(individual_index),
            "error_message": failure_message,
        }
    _refresh_task_progress(state)
    _notify_progress(engine, state)


def _enqueue_task(
    engine: Tardigradas,
    state: dict[str, Any],
    task: EvaluationTaskSpec,
) -> None:
    _validate_task_spec(engine, task)
    attempts = state.setdefault("attempts", {})
    if task.task_id in attempts:
        raise TardigradasException(f"duplicate task_id in task evaluation: {task.task_id}")
    attempts[task.task_id] = 0
    task_state = _ensure_task_state(state["individual_states"][task.individual_index], task.individual_index)
    task_state.scheduled_count += 1
    state.setdefault("ready_tasks", []).append(task)
    _refresh_task_progress(state)


def _aggregate_individual(engine: Tardigradas, state: dict[str, Any], individual_index: int) -> None:
    if state["scores"][individual_index] is not None:
        return
    individual = engine.population[individual_index]
    task_state = _ensure_task_state(state["individual_states"][individual_index], individual_index)
    context = _base_context(engine, individual_index)
    try:
        score = normalize_fitness_score(engine.problem.aggregate_task_results(individual, context, task_state)).tolist()
    except Exception as exc:
        state.setdefault("last_errors", {})[f"aggregate:{int(individual_index)}"] = {
            "individual_index": int(individual_index),
            "failure_kind": "aggregate_exception",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        _mark_missing(engine, state, individual_index, str(exc))
        return
    state["scores"][individual_index] = score
    _refresh_task_progress(state)
    _notify_progress(engine, state)


def _task_result_from_exception(task: EvaluationTaskSpec, exc: Exception) -> EvaluationTaskResult:
    if isinstance(exc, EvaluationFailure):
        return EvaluationTaskResult(
            task_id=task.task_id,
            individual_index=int(task.individual_index),
            ok=False,
            payload=dict(exc.details),
            retryable=bool(exc.retryable),
            failure_kind=exc.failure_kind,
            error_message=exc.error_message if exc.error_message is not None else str(exc),
        )
    return EvaluationTaskResult(
        task_id=task.task_id,
        individual_index=int(task.individual_index),
        ok=False,
        payload={"error_type": type(exc).__name__},
        retryable=False,
        failure_kind="exception",
        error_message=str(exc),
    )


def create_task_evaluation_state(engine: Tardigradas, max_attempts: int) -> dict[str, Any]:
    state: dict[str, Any] = {
        "phase": "evaluating_task_population",
        "generation": int(engine.iterations),
        "population_signatures": population_signatures(engine.population),
        "scores": [None for _ in engine.population],
        "max_attempts": int(max_attempts),
        "attempts": {},
        "individual_states": [None for _ in engine.population],
        "ready_tasks": [],
        "in_flight": {},
        "missing_indices": [],
        "last_errors": {},
        "task_progress": {},
    }
    for individual_index, individual in enumerate(engine.population):
        context = _base_context(engine, individual_index)
        task_state = _ensure_task_state(engine.problem.init_task_state(individual, context), individual_index)
        state["individual_states"][individual_index] = task_state
        initial_tasks = engine.problem.initial_evaluation_tasks(individual, context, task_state)
        for task in initial_tasks:
            _enqueue_task(engine, state, task)
        if not initial_tasks:
            _aggregate_individual(engine, state, individual_index)
    _refresh_task_progress(state)
    return state


def prepare_task_evaluation_state(engine: Tardigradas, max_attempts: int) -> dict[str, Any]:
    current_signatures = population_signatures(engine.population)
    state = getattr(engine, "evaluation_state", None)
    if state is None:
        state = create_task_evaluation_state(engine, max_attempts)
        engine.evaluation_state = state
        return state

    if int(state.get("generation", -1)) != int(engine.iterations):
        raise TardigradasException("restored task evaluation_state generation does not match current iteration")
    if list(state.get("population_signatures", [])) != current_signatures:
        raise TardigradasException("restored task evaluation_state population does not match current population")

    scores = list(state.get("scores", []))
    individual_states = list(state.get("individual_states", []))
    if len(scores) != len(engine.population) or len(individual_states) != len(engine.population):
        raise TardigradasException("restored task evaluation_state has invalid population size")

    state["scores"] = scores
    state["individual_states"] = [_ensure_task_state(task_state, index) for index, task_state in enumerate(individual_states)]
    state["max_attempts"] = int(max_attempts)
    state.setdefault("attempts", {})
    state.setdefault("ready_tasks", [])
    state.setdefault("in_flight", {})
    state.setdefault("missing_indices", [])
    state.setdefault("last_errors", {})
    state.setdefault("task_progress", {})

    original_ready_tasks = list(state.get("ready_tasks", []))
    original_in_flight = dict(state.get("in_flight", {}))

    normalized_ready_tasks: list[EvaluationTaskSpec] = []
    ready_task_ids: set[str] = set()
    for task in original_ready_tasks:
        _validate_task_spec(engine, task)
        if task.task_id not in ready_task_ids:
            normalized_ready_tasks.append(task)
            ready_task_ids.add(task.task_id)
    state["ready_tasks"] = normalized_ready_tasks

    in_flight_tasks: list[EvaluationTaskSpec] = []
    for task_id, payload in original_in_flight.items():
        if not isinstance(payload, dict):
            raise TardigradasException("restored task evaluation_state has invalid in_flight payload")
        task_payload = payload.get("task")
        if not isinstance(task_payload, EvaluationTaskSpec):
            raise TardigradasException("restored task evaluation_state has invalid in_flight task payload")
        task = task_payload
        _validate_task_spec(engine, task)
        in_flight_tasks.append(task)
        if task.task_id not in ready_task_ids:
            state["ready_tasks"].append(task)
            ready_task_ids.add(task.task_id)
    state["in_flight"] = {}

    if state.get("phase") == "incomplete_task_population":
        missing_indices = [int(index) for index in state.get("missing_indices", [])]
        state["missing_indices"] = []
        removed_task_ids = {
            task.task_id
            for task in [*normalized_ready_tasks, *in_flight_tasks]
            if int(task.individual_index) in missing_indices
        }
        state["ready_tasks"] = [
            task
            for task in state["ready_tasks"]
            if isinstance(task, EvaluationTaskSpec) and int(task.individual_index) not in missing_indices
        ]
        attempts = state.setdefault("attempts", {})
        preserved_attempts = {task_id: value for task_id, value in attempts.items() if task_id not in removed_task_ids}
        state["attempts"] = preserved_attempts
        for individual_index in missing_indices:
            individual = engine.population[individual_index]
            context = _base_context(engine, individual_index)
            task_state = _ensure_task_state(engine.problem.init_task_state(individual, context), individual_index)
            state["individual_states"][individual_index] = task_state
            state["scores"][individual_index] = None
            initial_tasks = engine.problem.initial_evaluation_tasks(individual, context, task_state)
            for task in initial_tasks:
                _enqueue_task(engine, state, task)
            if not initial_tasks:
                _aggregate_individual(engine, state, individual_index)
        state["phase"] = "evaluating_task_population"

    _refresh_task_progress(state)
    engine.evaluation_state = state
    return state


def estimate_task_population_sequential(engine: Tardigradas, config: Optional[EvaluationConfig]) -> None:
    max_attempts = 1 if config is None else max(1, int(config.max_attempts))
    state = prepare_task_evaluation_state(engine, max_attempts)

    while state["ready_tasks"]:
        task = state["ready_tasks"].pop(0)
        if not isinstance(task, EvaluationTaskSpec):
            raise TypeError("task evaluation state contains a non-EvaluationTaskSpec task")
        if state["scores"][task.individual_index] is not None:
            continue
        if int(task.individual_index) in {int(index) for index in state.get("missing_indices", [])}:
            continue

        attempt = int(state["attempts"].get(task.task_id, 0)) + 1
        state["attempts"][task.task_id] = attempt
        state["in_flight"] = {task.task_id: {"task": task, "attempt": attempt}}
        _refresh_task_progress(state)
        individual = engine.population[task.individual_index]
        context = _task_context(engine, task, attempt)
        previous_context = individual.evaluation_context
        individual.evaluation_context = context
        try:
            task_result = engine.problem.evaluate_task(individual, task, context)
        except Exception as exc:
            task_result = _task_result_from_exception(task, exc)
        finally:
            individual.evaluation_context = previous_context
        state["in_flight"] = {}
        _refresh_task_progress(state)

        if not isinstance(task_result, EvaluationTaskResult):
            raise TypeError("evaluate_task() must return EvaluationTaskResult")
        if task_result.task_id != task.task_id or int(task_result.individual_index) != int(task.individual_index):
            raise TardigradasException("evaluate_task() returned a result for a different task or individual")

        if not task_result.ok and task_result.retryable and attempt < max_attempts:
            _record_task_error(state, task, task_result, attempt)
            state["ready_tasks"].append(task)
            continue

        if not task_result.ok:
            _record_task_error(state, task, task_result, attempt)

        task_state = _ensure_task_state(state["individual_states"][task.individual_index], task.individual_index)
        task_state.completed_count += 1
        task_state.results.append(task_result)
        _refresh_task_progress(state)

        decision = engine.problem.update_task_state(individual, context, task_state, task_result)
        _validate_task_decision(decision)
        if decision.stop_individual:
            task_state.stopped = True
        if decision.individual_failed:
            _mark_missing(
                engine,
                state,
                task.individual_index,
                decision.failure_message if decision.failure_message is not None else task_result.error_message,
            )
            continue

        for new_task in decision.new_tasks:
            _enqueue_task(engine, state, new_task)

        if decision.ready_to_aggregate or decision.stop_individual:
            _aggregate_individual(engine, state, task.individual_index)
            continue

        if not task_result.ok and not decision.new_tasks:
            _mark_missing(engine, state, task.individual_index, task_result.error_message)

    missing = [index for index, score in enumerate(state["scores"]) if score is None]
    if missing:
        state["missing_indices"] = missing
        state["phase"] = "incomplete_task_population"
        _refresh_task_progress(state)
        raise IncompleteEpochError(missing)

    full_scores, scores = collect_scores_from_state(state)
    engine.full_scores = full_scores
    engine.scores = scores
    engine.evaluation_state = None


def estimate_task_population_subprocess(engine: Tardigradas, config: EvaluationConfig) -> None:
    state = prepare_task_evaluation_state(engine, config.max_attempts)
    problem_module, problem_qualified_name = problem_import_path(engine.problem)
    ready_tasks = state.setdefault("ready_tasks", [])
    running: dict[str, tuple[EvaluationTaskSpec, _RunningTask]] = {}

    with tempfile.TemporaryDirectory(prefix="tardigradas-task-eval-") as temp_name:
        temp_dir = Path(temp_name)
        while ready_tasks or running:
            while len(running) < config.workers and ready_tasks:
                task = ready_tasks.pop(0)
                if state["scores"][task.individual_index] is not None:
                    continue
                if int(task.individual_index) in {int(index) for index in state.get("missing_indices", [])}:
                    continue

                attempt = int(state["attempts"].get(task.task_id, 0)) + 1
                state["attempts"][task.task_id] = attempt
                worker_task = _start_task_worker(
                    temp_dir,
                    engine,
                    task,
                    attempt,
                    problem_module,
                    problem_qualified_name,
                )
                running[task.task_id] = (task, worker_task)
                state["in_flight"][task.task_id] = {"task": task, "attempt": attempt}
                _refresh_task_progress(state)

            completed_task_ids: list[str] = []
            for task_id, (_, worker_task) in running.items():
                if worker_task.process.poll() is not None:
                    completed_task_ids.append(task_id)

            if not completed_task_ids:
                time.sleep(0.01)
                continue

            for task_id in completed_task_ids:
                task, worker_task = running.pop(task_id)
                attempt = int(state.get("in_flight", {}).get(task_id, {}).get("attempt", state["attempts"].get(task_id, 0)))
                state.get("in_flight", {}).pop(task_id, None)
                task_result, transport_error = _read_task_worker_result(worker_task, task)
                _refresh_task_progress(state)

                if transport_error is not None:
                    state.setdefault("last_errors", {})[task.task_id] = transport_error
                    if attempt < config.max_attempts:
                        ready_tasks.append(task)
                        state["in_flight"] = {}
                        _refresh_task_progress(state)
                    else:
                        _mark_missing(engine, state, task.individual_index, str(transport_error.get("error_message")))
                    continue

                if task_result is None:
                    raise TardigradasException("task worker did not return a task result")

                if not task_result.ok and task_result.retryable and attempt < config.max_attempts:
                    _record_task_error(state, task, task_result, attempt)
                    ready_tasks.append(task)
                    continue

                if not task_result.ok:
                    _record_task_error(state, task, task_result, attempt)

                individual = engine.population[task.individual_index]
                context = _task_context(engine, task, attempt)
                task_state = _ensure_task_state(state["individual_states"][task.individual_index], task.individual_index)
                task_state.completed_count += 1
                task_state.results.append(task_result)
                _refresh_task_progress(state)

                decision = engine.problem.update_task_state(individual, context, task_state, task_result)
                _validate_task_decision(decision)
                if decision.stop_individual:
                    task_state.stopped = True
                if decision.individual_failed:
                    _mark_missing(
                        engine,
                        state,
                        task.individual_index,
                        decision.failure_message if decision.failure_message is not None else task_result.error_message,
                    )
                    continue

                for new_task in decision.new_tasks:
                    _enqueue_task(engine, state, new_task)

                if decision.ready_to_aggregate or decision.stop_individual:
                    _aggregate_individual(engine, state, task.individual_index)
                    continue

                if not task_result.ok and not decision.new_tasks:
                    _mark_missing(engine, state, task.individual_index, task_result.error_message)

    missing = [index for index, score in enumerate(state["scores"]) if score is None]
    if missing:
        state["missing_indices"] = missing
        state["phase"] = "incomplete_task_population"
        _refresh_task_progress(state)
        raise IncompleteEpochError(missing)

    full_scores, scores = collect_scores_from_state(state)
    engine.full_scores = full_scores
    engine.scores = scores
    engine.evaluation_state = None


def _write_task_worker_request(
    request_path: Path,
    response_path: Path,
    engine: Tardigradas,
    task: EvaluationTaskSpec,
    attempt: int,
    problem_module: str,
    problem_qualified_name: str,
) -> None:
    individual = engine.population[task.individual_index]
    request = {
        "mode": "task",
        "problem_module": problem_module,
        "problem_qualified_name": problem_qualified_name,
        "population_size": int(engine.population_size),
        "crossover_fraction": float(engine.crossover_fraction),
        "fresh_blood_fraction": float(engine.fresh_blood_fraction),
        "gen_mutation_fraction": float(engine.gen_mutation_fraction),
        "n_elits": int(engine.n_elits),
        "elit_estimates_count": int(getattr(engine, "elit_estimates_count", 1)),
        "generation": int(engine.iterations),
        "individual_index": int(task.individual_index),
        "attempt": int(attempt),
        "chromo": individual.chromo,
        "task": {
            "task_id": task.task_id,
            "individual_index": int(task.individual_index),
            "generation": int(task.generation),
            "task_number": int(task.task_number),
            "payload": dict(task.payload),
        },
        "response_path": str(response_path),
    }
    with request_path.open("wb") as file:
        pickle.dump(request, file)


def _start_task_worker(
    temp_dir: Path,
    engine: Tardigradas,
    task: EvaluationTaskSpec,
    attempt: int,
    problem_module: str,
    problem_qualified_name: str,
) -> _RunningTask:
    request_path = temp_dir / f"task_request_{task.individual_index}_{task.task_number}_{attempt}.pkl"
    response_path = temp_dir / f"task_response_{task.individual_index}_{task.task_number}_{attempt}.pkl"
    _write_task_worker_request(
        request_path,
        response_path,
        engine,
        task,
        attempt,
        problem_module,
        problem_qualified_name,
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "tardigradas._evaluation_worker", str(request_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_subprocess_environment(),
    )
    return _RunningTask(process=process, index=int(task.individual_index), attempt=int(attempt), response_path=response_path)


def _read_task_worker_result(
    worker_task: _RunningTask,
    expected_task: EvaluationTaskSpec,
) -> tuple[EvaluationTaskResult | None, dict[str, Any] | None]:
    if not worker_task.response_path.exists():
        return None, {
            "failure_kind": "worker_failed",
            "error_message": "worker did not write a response",
        }

    try:
        with worker_task.response_path.open("rb") as file:
            response = pickle.load(file)
    except Exception as exc:
        return None, {
            "failure_kind": "invalid_response",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    if not isinstance(response, dict) or not bool(response.get("ok", False)):
        return None, {
            "failure_kind": "invalid_response",
            "error_message": "task worker response is not a successful dictionary payload",
        }

    task_result_payload = response.get("task_result")
    if not isinstance(task_result_payload, dict):
        return None, {
            "failure_kind": "invalid_response",
            "error_message": "task worker response does not contain task_result",
        }

    try:
        task_result = EvaluationTaskResult(
            task_id=str(task_result_payload["task_id"]),
            individual_index=int(task_result_payload["individual_index"]),
            ok=bool(task_result_payload["ok"]),
            score=task_result_payload.get("score"),
            payload=task_result_payload.get("payload"),
            retryable=bool(task_result_payload.get("retryable", False)),
            failure_kind=task_result_payload.get("failure_kind"),
            error_message=task_result_payload.get("error_message"),
        )
    except Exception as exc:
        return None, {
            "failure_kind": "invalid_response",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    if task_result.task_id != expected_task.task_id or int(task_result.individual_index) != int(expected_task.individual_index):
        return None, {
            "failure_kind": "invalid_response",
            "error_message": "task worker returned a result for a different task or individual",
        }

    return task_result, None
