from __future__ import annotations


def test_step_updates_runtime_state(engine) -> None:
    engine.population_init()

    engine.step()

    assert engine.iterations == 1
    assert len(engine.population) == engine.population_size
    assert len(engine.scores_history) == 1
    assert len(engine.custom_scores_history) == 1
    assert engine.step_best_individual is not None
    assert engine.best_individual is not None
    assert engine.step_score is not None


def test_loop_stops_at_max_iterations(engine) -> None:
    engine.population_init()

    engine.loop(max_iterations=2, epoch_without_improve=100, loop_fun=lambda _: False)

    assert engine.iterations == 2