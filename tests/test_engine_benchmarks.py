from __future__ import annotations

import numpy as np

from benchmarks.common import run_benchmark
from benchmarks.problems import (
    AckleyProblem,
    OneMaxProblem,
    RastriginProblem,
    RosenbrockProblem,
    RoyalRoadProblem,
    SphereProblem,
)


def test_onemax_reaches_near_optimal_bitstring() -> None:
    engine, initial_best_score = run_benchmark(
        OneMaxProblem,
        population_size=40,
        crossover_fraction=0.6,
        gen_mutation_fraction=0.12,
        n_elits=2,
        max_iterations=80,
    )

    assert engine.best_individual is not None
    assert engine.best_score is not None
    assert initial_best_score < float(OneMaxProblem.n_bits)
    assert engine.best_score >= float(OneMaxProblem.n_bits - 1)
    assert int(np.sum(engine.best_individual.chromo)) >= OneMaxProblem.n_bits


def test_sphere_converges_close_to_zero() -> None:
    engine, initial_best_score = run_benchmark(
        SphereProblem,
        population_size=50,
        crossover_fraction=0.55,
        gen_mutation_fraction=0.18,
        n_elits=2,
        max_iterations=120,
    )

    assert engine.best_score is not None
    assert engine.best_individual is not None
    assert engine.best_score >= initial_best_score
    assert engine.best_score >= -0.25
    assert float(np.linalg.norm(engine.best_individual.chromo)) <= 0.02


def test_rastrigin_reaches_global_search_threshold() -> None:
    engine, initial_best_score = run_benchmark(
        RastriginProblem,
        population_size=200,
        crossover_fraction=0.55,
        fresh_blood_fraction=0.05,
        gen_mutation_fraction=0.24,
        n_elits=2,
        max_iterations=50,
    )

    print(f"Best score: {engine.best_score}, initial best score: {initial_best_score}")
    assert engine.best_score is not None
    assert engine.best_score >= initial_best_score
    assert engine.best_score >= -0.1


def test_royal_road_preserves_building_blocks() -> None:
    engine, initial_best_score = run_benchmark(
        RoyalRoadProblem,
        population_size=48,
        crossover_fraction=0.65,
        gen_mutation_fraction=0.08,
        n_elits=2,
        max_iterations=50,
    )

    assert engine.best_individual is not None
    assert engine.best_score is not None
    assert engine.best_score >= initial_best_score
    assert engine.best_score >= 24


def test_rosenbrock_handles_curved_valley_landscape() -> None:
    engine, initial_best_score = run_benchmark(
        RosenbrockProblem,
        population_size=80,
        crossover_fraction=0.45,
        fresh_blood_fraction=0.05,
        gen_mutation_fraction=0.22,
        n_elits=3,
        max_iterations=50,
    )

    print(f"Best score: {engine.best_score}, initial best score: {initial_best_score}")
    assert engine.best_score is not None
    assert engine.best_individual is not None
    assert engine.best_score >= initial_best_score
    assert engine.best_score >= -0.05


def test_ackley_is_sensitive_but_still_reaches_threshold() -> None:
    engine, initial_best_score = run_benchmark(
        AckleyProblem,
        population_size=100,
        crossover_fraction=0.45,
        fresh_blood_fraction=0.05,
        gen_mutation_fraction=0.3,
        n_elits=2,
        max_iterations=100,
    )

    print(f"Best score: {engine.best_score}, initial best score: {initial_best_score}")
    assert engine.best_score is not None
    assert engine.best_individual is not None
    assert engine.best_score >= initial_best_score
    assert engine.best_score >= -0.1