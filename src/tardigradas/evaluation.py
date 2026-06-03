from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .exceptions import IncompleteEpochError, TardigradasException

if TYPE_CHECKING:
    from .engine import Tardigradas


@dataclass(frozen=True)
class EvaluationConfig:
    workers: int = 1
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("evaluation.workers must be positive")
        if self.max_attempts < 1:
            raise ValueError("evaluation.max_attempts must be positive")


@dataclass(frozen=True)
class EvaluationContext:
    generation: int
    individual_index: int
    attempt: int


def normalize_fitness_score(raw_score: Any) -> np.ndarray:
    if np.isscalar(raw_score):
        return np.asarray([raw_score], dtype=float).reshape(-1)

    scores = np.array(raw_score, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError("fitness must return at least one numeric value")
    return scores


def population_signatures(population: list[Any]) -> list[bytes]:
    return [np.asarray(individual.chromo, dtype=float).reshape(-1).tobytes() for individual in population]


def problem_import_path(problem: type[Any]) -> tuple[str, str]:
    module_name = problem.__module__
    qualified_name = problem.__qualname__
    if module_name == "__main__" or "<locals>" in qualified_name:
        raise TardigradasException(
            "subprocess evaluation requires problem to be an importable class, "
            "not a __main__ or local class"
        )
    return module_name, qualified_name


def create_evaluation_state(engine: Tardigradas, max_attempts: int) -> dict[str, Any]:
    return {
        "phase": "evaluating_population",
        "generation": int(engine.iterations),
        "population_signatures": population_signatures(engine.population),
        "scores": [None for _ in engine.population],
        "attempts": [0 for _ in engine.population],
        "max_attempts": int(max_attempts),
        "missing_indices": [],
    }


def prepare_evaluation_state(engine: Tardigradas, max_attempts: int) -> dict[str, Any]:
    current_signatures = population_signatures(engine.population)
    state = getattr(engine, "evaluation_state", None)
    if state is None:
        state = create_evaluation_state(engine, max_attempts)
        engine.evaluation_state = state
        return state

    if int(state.get("generation", -1)) != int(engine.iterations):
        raise TardigradasException("restored evaluation_state generation does not match current iteration")
    if list(state.get("population_signatures", [])) != current_signatures:
        raise TardigradasException("restored evaluation_state population does not match current population")

    scores = list(state.get("scores", []))
    attempts = list(state.get("attempts", []))
    if len(scores) != len(engine.population) or len(attempts) != len(engine.population):
        raise TardigradasException("restored evaluation_state has invalid population size")

    state["scores"] = scores
    state["attempts"] = [int(attempt) for attempt in attempts]
    state["max_attempts"] = int(max_attempts)
    state.setdefault("missing_indices", [])

    if state.get("phase") == "incomplete_population":
        for index in state["missing_indices"]:
            state["attempts"][int(index)] = 0
        state["missing_indices"] = []
        state["phase"] = "evaluating_population"

    engine.evaluation_state = state
    return state


def collect_scores_from_state(state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    score_rows = state.get("scores", [])
    if any(score is None for score in score_rows):
        missing = [index for index, score in enumerate(score_rows) if score is None]
        raise IncompleteEpochError(missing)

    full_scores = np.vstack([normalize_fitness_score(score) for score in score_rows])
    return full_scores, full_scores[:, 0]


def estimate_population_sequential(engine: Tardigradas) -> None:
    scores = []
    for i, individual in enumerate(engine.population):
        scores.append(individual.fitness())
        if engine.fitness_progress_fun:
            engine.fitness_progress_fun(engine, i / engine.population_size)

    engine.full_scores = np.vstack(scores)
    engine.scores = engine.full_scores[:, 0]
    engine.evaluation_state = None


@dataclass
class _RunningTask:
    process: subprocess.Popen[Any]
    index: int
    attempt: int
    response_path: Path


def _write_worker_request(
    request_path: Path,
    response_path: Path,
    engine: Tardigradas,
    index: int,
    attempt: int,
    problem_module: str,
    problem_qualified_name: str,
) -> None:
    individual = engine.population[index]
    request = {
        "problem_module": problem_module,
        "problem_qualified_name": problem_qualified_name,
        "population_size": int(engine.population_size),
        "crossover_fraction": float(engine.crossover_fraction),
        "fresh_blood_fraction": float(engine.fresh_blood_fraction),
        "gen_mutation_fraction": float(engine.gen_mutation_fraction),
        "n_elits": int(engine.n_elits),
        "generation": int(engine.iterations),
        "individual_index": int(index),
        "attempt": int(attempt),
        "chromo": np.asarray(individual.chromo, dtype=float),
        "response_path": str(response_path),
    }
    with request_path.open("wb") as file:
        pickle.dump(request, file)


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    python_path_parts = [path for path in sys.path if path]
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_path_parts.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_path_parts))
    return environment


def _start_worker(
    temp_dir: Path,
    engine: Tardigradas,
    index: int,
    attempt: int,
    problem_module: str,
    problem_qualified_name: str,
) -> _RunningTask:
    request_path = temp_dir / f"request_{index}_{attempt}.pkl"
    response_path = temp_dir / f"response_{index}_{attempt}.pkl"
    _write_worker_request(
        request_path,
        response_path,
        engine,
        index,
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
    return _RunningTask(process=process, index=index, attempt=attempt, response_path=response_path)


def _read_worker_score(task: _RunningTask) -> list[float] | None:
    if task.process.returncode != 0 or not task.response_path.exists():
        return None

    try:
        with task.response_path.open("rb") as file:
            response = pickle.load(file)
    except Exception:
        return None


    if not isinstance(response, dict) or not response.get("ok", False):
        return None
    return normalize_fitness_score(response.get("score")).tolist()


def _completed_count(state: dict[str, Any]) -> int:
    scores_done = sum(score is not None for score in state.get("scores", []))
    missing_done = len(set(int(index) for index in state.get("missing_indices", [])))
    return scores_done + missing_done


def _notify_progress(engine: Tardigradas, state: dict[str, Any]) -> None:
    if engine.fitness_progress_fun is None:
        return
    progress = _completed_count(state) / engine.population_size
    engine.fitness_progress_fun(engine, progress)


def estimate_population_subprocess(engine: Tardigradas, config: EvaluationConfig) -> None:
    state = prepare_evaluation_state(engine, config.max_attempts)
    problem_module, problem_qualified_name = problem_import_path(engine.problem)
    pending = deque(index for index, score in enumerate(state["scores"]) if score is None)
    in_flight: list[_RunningTask] = []

    with tempfile.TemporaryDirectory(prefix="tardigradas-eval-") as temp_name:
        temp_dir = Path(temp_name)
        while pending or in_flight:
            while pending and len(in_flight) < config.workers:
                index = int(pending.popleft())
                if state["scores"][index] is not None:
                    continue
                if int(state["attempts"][index]) >= config.max_attempts:
                    if index not in state["missing_indices"]:
                        state["missing_indices"].append(index)
                        _notify_progress(engine, state)
                    continue

                attempt = int(state["attempts"][index]) + 1
                state["attempts"][index] = attempt
                in_flight.append(
                    _start_worker(
                        temp_dir,
                        engine,
                        index,
                        attempt,
                        problem_module,
                        problem_qualified_name,
                    )
                )

            completed: list[_RunningTask] = []
            for task in in_flight:
                if task.process.poll() is not None:
                    completed.append(task)

            if not completed:
                time.sleep(0.01)
                continue

            for task in completed:
                in_flight.remove(task)
                score = _read_worker_score(task)
                if score is not None:
                    state["scores"][task.index] = score
                    _notify_progress(engine, state)
                    continue

                if int(state["attempts"][task.index]) < config.max_attempts:
                    pending.append(task.index)
                elif task.index not in state["missing_indices"]:
                    state["missing_indices"].append(task.index)
                    _notify_progress(engine, state)

    missing = [index for index, score in enumerate(state["scores"]) if score is None]
    if missing:
        state["missing_indices"] = missing
        state["phase"] = "incomplete_population"
        raise IncompleteEpochError(missing)

    full_scores, scores = collect_scores_from_state(state)
    engine.full_scores = full_scores
    engine.scores = scores
    engine.evaluation_state = None


def estimate_population(engine: Tardigradas, config: Optional[EvaluationConfig]) -> None:
    if config is None or config.workers <= 1:
        estimate_population_sequential(engine)
        return
    estimate_population_subprocess(engine, config)
