from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from enum import Enum
from time import perf_counter
from typing import Any

import numpy as np

from ._paths import ensure_project_paths


ensure_project_paths()


from tardigradas import CrossoverPolicy, Problem, Tardigradas


class _FitnessEvaluationProgress:
    def __init__(self) -> None:
        self._last_width = 0

    def __call__(self, engine: Tardigradas, progress: float) -> None:
        total = len(engine.population)
        if total <= 0:
            return

        evaluated = int(np.clip(np.floor(progress * total) + 1, 1, total))
        message = f"evaluated: {evaluated}/{total}..."
        self._last_width = max(self._last_width, len(message))
        sys.stdout.write(f"\r{message}")
        sys.stdout.flush()

    def clear(self) -> None:
        if self._last_width == 0:
            return

        sys.stdout.write("\r" + (" " * self._last_width) + "\r")
        sys.stdout.flush()
        self._last_width = 0


def create_benchmark_engine(
    *,
    problem: type[Problem],
    population_size: int,
    crossover_fraction: float,
    fresh_blood_fraction: float,
    gen_mutation_fraction: float,
    n_elits: int,
    crossover_policy: CrossoverPolicy | None = None,
) -> Tardigradas:
    return Tardigradas(
        problem=problem,
        population_size=population_size,
        crossover_fraction=crossover_fraction,
        fresh_blood_fraction=fresh_blood_fraction,
        gen_mutation_fraction=gen_mutation_fraction,
        n_elits=n_elits,
        crossover_policy=crossover_policy,
    )


def run_benchmark(
    problem: type[Problem],
    *,
    population_size: int,
    crossover_fraction: float,
    gen_mutation_fraction: float,
    n_elits: int,
    max_iterations: int,
    fresh_blood_fraction: float = 0.0,
    crossover_policy: CrossoverPolicy | None = None,
    engine_factory: Callable[..., Tardigradas] | None = None,
    show_epoch_progress: bool = True,
) -> tuple[Tardigradas, float]:
    create_engine = create_benchmark_engine if engine_factory is None else engine_factory
    engine = create_engine(
        problem=problem,
        population_size=population_size,
        crossover_fraction=crossover_fraction,
        fresh_blood_fraction=fresh_blood_fraction,
        gen_mutation_fraction=gen_mutation_fraction,
        n_elits=n_elits,
        crossover_policy=crossover_policy,
    )
    progress = _FitnessEvaluationProgress()

    engine.population_init()

    try:
        engine.fitness_progress_fun = progress
        engine.estimate_population()
        progress.clear()
        initial_best_score = float(np.max(engine.scores))
        benchmark_started_at = perf_counter()

        def log_epoch(current_engine: Tardigradas) -> bool:
            progress.clear()
            if show_epoch_progress:
                print_benchmark_epoch(
                    current_engine,
                    initial_best_score,
                    elapsed_time_seconds=perf_counter() - benchmark_started_at,
                )
            return False

        engine.loop(
            max_iterations=max_iterations,
            epoch_without_improve=max_iterations,
            loop_fun=log_epoch,
            fitness_progress_fun=progress,
        )
    finally:
        progress.clear()
        engine.fitness_progress_fun = None

    return engine, initial_best_score


def _format_crossover_policy(policy: CrossoverPolicy) -> str:
    if policy.is_explicit:
        bit_name = policy.bit.name if policy.bit is not None else "none"
        float_name = policy.float.name if policy.float is not None else "none"
        return f"explicit(bit={bit_name}, float={float_name})"

    bit_candidates = ", ".join(candidate.name for candidate in policy.bit_candidates)
    float_candidates = ", ".join(candidate.name for candidate in policy.float_candidates)
    return (
        "adaptive("
        f"bit_candidates=[{bit_candidates}], "
        f"float_candidates=[{float_candidates}], "
        f"reward={policy.reward}, "
        f"min_probability={policy.min_probability}, "
        f"period={policy.period}, "
        f"alpha={policy.alpha:.6f}"
        ")"
    )


def _format_benchmark_value(value: Any) -> Any:
    if isinstance(value, CrossoverPolicy):
        return _format_crossover_policy(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{key}: {_format_benchmark_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(_format_benchmark_value(item)) for item in value) + "]"
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_benchmark_epoch_metrics(
    engine: Tardigradas,
    initial_best_score: float,
    *,
    elapsed_time_seconds: float,
) -> dict[str, Any]:
    population_mean_score = float(np.mean(engine.scores)) if engine.scores.size else None
    population_max_score = float(np.max(engine.scores)) if engine.scores.size else None
    score_improvement = None
    best_epoch = None
    if engine.best_score is not None:
        score_improvement = float(engine.best_score - initial_best_score)
        best_epoch = int(engine.best_iteration + 1)

    metrics: dict[str, Any] = {
        "step_score": engine.step_score,
        "best_score": engine.best_score,
        "best_epoch": best_epoch,
        "score_improvement": score_improvement,
        "population_mean_score": population_mean_score,
        "population_max_score": population_max_score,
        "killed_doubles": engine.n_killed_doubles,
        "elapsed_time_sec": round(float(elapsed_time_seconds), 3),
    }

    adaptive_state = engine.adaptive_crossover_state()
    if adaptive_state.get("mode") == "adaptive":
        metrics.update(
            {
                "adaptive_reward": adaptive_state["reward"],
                "adaptive_min_probability": adaptive_state["min_probability"],
                "adaptive_period": adaptive_state["period"],
                "adaptive_alpha": adaptive_state["alpha"],
                "adaptive_bit_candidates": adaptive_state["bit_candidates"],
                "adaptive_float_candidates": adaptive_state["float_candidates"],
                "adaptive_bit_epoch_uses": adaptive_state["bit_epoch_uses"],
                "adaptive_bit_epoch_successes": adaptive_state["bit_epoch_successes"],
                "adaptive_bit_instant_scores": adaptive_state["bit_instant_scores"],
                "adaptive_bit_scores": adaptive_state["bit_scores"],
                "adaptive_bit_probabilities": adaptive_state["bit_probabilities"],
                "adaptive_float_epoch_uses": adaptive_state["float_epoch_uses"],
                "adaptive_float_epoch_successes": adaptive_state["float_epoch_successes"],
                "adaptive_float_instant_scores": adaptive_state["float_instant_scores"],
                "adaptive_float_scores": adaptive_state["float_scores"],
                "adaptive_float_probabilities": adaptive_state["float_probabilities"],
            }
        )

    return metrics


def print_benchmark_epoch(
    engine: Tardigradas,
    initial_best_score: float,
    *,
    elapsed_time_seconds: float,
) -> None:
    print(f"Epoch {engine.iterations}:")
    metrics = _build_benchmark_epoch_metrics(
        engine,
        initial_best_score,
        elapsed_time_seconds=elapsed_time_seconds,
    )
    for key, value in metrics.items():
        print(f"  - {key}: {_format_benchmark_value(value)}")


def print_benchmark_configuration(
    benchmark_name: str,
    *,
    problem: type[Problem],
    config: Mapping[str, Any],
) -> None:
    print(f"Benchmark: {benchmark_name}")
    print(f"Problem: {problem.__name__}")
    print("Parameters:")
    for key, value in config.items():
        print(f"  - {key}: {_format_benchmark_value(value)}")


def print_benchmark_summary(
    engine: Tardigradas,
    initial_best_score: float,
    *,
    extra_metrics: Mapping[str, Any] | None = None,
    show_best_chromosome: bool = True,
) -> None:
    score_improvement = None
    if engine.best_score is not None:
        score_improvement = float(engine.best_score - initial_best_score)

    print("Result:")
    print(f"  - iterations: {engine.iterations}")
    print(f"  - initial_best_score: {initial_best_score}")
    print(f"  - best_score: {engine.best_score}")
    print(f"  - best_iteration: {engine.best_iteration}")
    print(f"  - score_improvement: {score_improvement}")
    print(f"  - killed_doubles: {engine.n_killed_doubles}")

    if show_best_chromosome and engine.best_individual is not None:
        print(
            "  - best_chromosome: "
            f"{np.array2string(engine.best_individual.chromo, precision=6, separator=', ')}"
        )

    if extra_metrics is None:
        return

    for key, value in extra_metrics.items():
        print(f"  - {key}: {value}")