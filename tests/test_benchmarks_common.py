from __future__ import annotations

import numpy as np

from benchmarks.common import print_benchmark_configuration, print_benchmark_epoch, run_benchmark
from benchmarks.problems import OneMaxProblem
from tardigradas import CrossoverBitType, CrossoverFloatType, CrossoverPolicy
from tests.helpers import DummyProblem, create_engine


def test_print_benchmark_configuration_formats_explicit_crossover_policy(capsys) -> None:
    policy = CrossoverPolicy.explicit(
        bit=CrossoverBitType.two_point,
        float=CrossoverFloatType.BLX,
    )

    print_benchmark_configuration(
        "OneMax",
        problem=OneMaxProblem,
        config={"crossover_policy": policy},
    )

    captured = capsys.readouterr()
    assert "crossover_policy: explicit(bit=two_point, float=BLX)" in captured.out


def test_print_benchmark_configuration_formats_adaptive_crossover_policy(capsys) -> None:
    policy = CrossoverPolicy.adaptive(
        bit_candidates=[CrossoverBitType.uniform, CrossoverBitType.one_point],
        float_candidates=[CrossoverFloatType.uniform, CrossoverFloatType.arithmetic],
        min_probability=0.1,
    )

    print_benchmark_configuration(
        "OneMax",
        problem=OneMaxProblem,
        config={"crossover_policy": policy},
    )

    captured = capsys.readouterr()
    assert (
        "crossover_policy: adaptive(bit_candidates=[uniform, one_point], "
        "float_candidates=[uniform, arithmetic], reward=elite_survival, min_probability=0.1, "
        "period=20, alpha=0.095238)"
        in captured.out
    )


def test_run_benchmark_passes_crossover_policy_to_engine() -> None:
    policy = CrossoverPolicy.explicit(
        bit=CrossoverBitType.one_point,
        float=CrossoverFloatType.arithmetic,
    )

    engine, _ = run_benchmark(
        OneMaxProblem,
        population_size=12,
        crossover_fraction=0.5,
        gen_mutation_fraction=0.1,
        n_elits=1,
        max_iterations=1,
        crossover_policy=policy,
    )

    assert engine.crossover_policy == policy


def test_run_benchmark_prints_epoch_progress(capsys) -> None:
    run_benchmark(
        OneMaxProblem,
        population_size=12,
        crossover_fraction=0.5,
        gen_mutation_fraction=0.1,
        n_elits=1,
        max_iterations=1,
    )

    captured = capsys.readouterr()
    assert "Epoch 1:" in captured.out
    assert "step_score:" in captured.out
    assert "population_mean_score:" in captured.out
    assert "elapsed_time_sec:" in captured.out


def test_run_benchmark_prints_single_line_fitness_progress(capsys) -> None:
    run_benchmark(
        OneMaxProblem,
        population_size=12,
        crossover_fraction=0.5,
        gen_mutation_fraction=0.1,
        n_elits=1,
        max_iterations=1,
        show_epoch_progress=False,
    )

    captured = capsys.readouterr()
    assert "\revaluated: 1/12..." in captured.out
    assert "\revaluated: 12/12..." in captured.out
    assert ("\r" + (" " * len("evaluated: 12/12...")) + "\r") in captured.out


def test_print_benchmark_epoch_includes_adaptive_policy_details(capsys) -> None:
    policy = CrossoverPolicy.adaptive(
        bit_candidates=[CrossoverBitType.uniform, CrossoverBitType.one_point],
        float_candidates=[CrossoverFloatType.uniform, CrossoverFloatType.arithmetic],
        min_probability=0.1,
    )
    engine = create_engine(
        problem=DummyProblem,
        population_size=6,
        crossover_policy=policy,
    )
    engine.iterations = 2
    engine.step_score = 1.25
    engine.best_score = 2.5
    engine.best_iteration = 1
    engine.scores = np.array([1.0, 2.0, 3.0], dtype=float)
    engine.n_killed_doubles = 1
    engine._adaptive_last_bit_epoch_uses[CrossoverBitType.uniform] = 3
    engine._adaptive_last_bit_epoch_uses[CrossoverBitType.one_point] = 1
    engine._adaptive_last_bit_epoch_successes[CrossoverBitType.uniform] = 1
    engine._adaptive_last_bit_epoch_successes[CrossoverBitType.one_point] = 1
    engine._adaptive_last_float_epoch_uses[CrossoverFloatType.uniform] = 2
    engine._adaptive_last_float_epoch_uses[CrossoverFloatType.arithmetic] = 2
    engine._adaptive_last_float_epoch_successes[CrossoverFloatType.uniform] = 1
    engine._adaptive_last_float_epoch_successes[CrossoverFloatType.arithmetic] = 0

    print_benchmark_epoch(engine, initial_best_score=0.5, elapsed_time_seconds=0.1234)

    captured = capsys.readouterr()
    assert "Epoch 2:" in captured.out
    assert "adaptive_reward: elite_survival" in captured.out
    assert "adaptive_min_probability: 0.1" in captured.out
    assert "adaptive_bit_candidates: [uniform, one_point]" in captured.out
    assert "adaptive_bit_epoch_uses: {uniform: 3, one_point: 1}" in captured.out
    assert "adaptive_bit_epoch_successes: {uniform: 1, one_point: 1}" in captured.out
    assert "adaptive_bit_probabilities:" in captured.out
    assert "adaptive_float_candidates: [uniform, arithmetic]" in captured.out
    assert "adaptive_float_epoch_uses: {uniform: 2, arithmetic: 2}" in captured.out
    assert "adaptive_float_epoch_successes: {uniform: 1, arithmetic: 0}" in captured.out
    assert "adaptive_float_probabilities:" in captured.out
