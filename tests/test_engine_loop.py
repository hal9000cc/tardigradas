from __future__ import annotations

import pytest

from tests.helpers import ConstantFitnessProblem, create_engine


def test_loop_stops_when_epoch_without_improve_is_reached() -> None:
    engine = create_engine(problem=ConstantFitnessProblem, population_size=4)
    engine.population_init()

    engine.loop(max_iterations=50, epoch_without_improve=2, loop_fun=lambda _: False)

    assert engine.iterations == 3
    assert engine.best_iteration == 0
    assert len(engine.scores_history) == 3


def test_loop_stops_when_loop_fun_returns_true() -> None:
    engine = create_engine(problem=ConstantFitnessProblem, population_size=4)
    engine.population_init()

    calls: list[int] = []

    def stop_after_second_iteration(tardigradas) -> bool:
        calls.append(tardigradas.iterations)
        return tardigradas.iterations >= 2

    engine.loop(max_iterations=10, epoch_without_improve=10, loop_fun=stop_after_second_iteration)

    assert engine.iterations == 2
    assert calls == [1, 2]


def test_loop_evaluates_loop_fun_before_other_stop_conditions() -> None:
    engine = create_engine(problem=ConstantFitnessProblem, population_size=4)
    engine.population_init()

    calls: list[int] = []

    def stop_immediately(tardigradas) -> bool:
        calls.append(tardigradas.iterations)
        return True

    engine.loop(max_iterations=1, epoch_without_improve=0, loop_fun=stop_immediately)

    assert engine.iterations == 1
    assert calls == [1]


def test_loop_propagates_fitness_progress_callback_during_population_estimation() -> None:
    engine = create_engine(problem=ConstantFitnessProblem, population_size=4)
    engine.population_init()

    progress_updates: list[float] = []

    engine.loop(
        max_iterations=1,
        epoch_without_improve=10,
        loop_fun=lambda _: False,
        fitness_progress_fun=lambda _, progress: progress_updates.append(progress),
    )

    expected = [i / engine.population_size for i in range(engine.population_size)]

    assert progress_updates == pytest.approx(expected)
