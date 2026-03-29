from __future__ import annotations

import numpy as np

import tardigradas.progress_panel as progress_panel_module
from tardigradas import CrossoverBitType, CrossoverFloatType, CrossoverPolicy, create_progress_panel
from tests.helpers import VectorFitnessProblem, create_engine


def test_create_progress_panel_falls_back_without_matplotlib(monkeypatch) -> None:
    def fake_import_module(name: str):
        raise ImportError(name)

    monkeypatch.setattr(progress_panel_module.importlib, "import_module", fake_import_module)

    panel = create_progress_panel(title="test", prefer_matplotlib=True)

    assert panel.rendering_available is False
    panel.show(block=False)
    panel.close()


def test_progress_panel_builds_sorted_population_bars_with_origin_colors() -> None:
    engine = create_engine(population_size=4)
    engine.iterations = 3
    engine.best_score = 2.0
    engine.scores = np.array([0.5, 2.0, 1.25, 1.75], dtype=float)
    engine.step_custom_score = np.array([2.0, 0.25], dtype=float)
    engine.n_killed_doubles = 2
    engine.step_population_origins = [
        {"source": "fresh"},
        {"source": "elite"},
        {"source": "mutation"},
        {"source": "crossover"},
    ]

    panel = create_progress_panel(prefer_matplotlib=False)
    panel.initial_best_score = 1.0

    snapshot = panel.build_snapshot(engine)

    assert snapshot.iteration == 3
    assert snapshot.population_mean_score == np.mean(engine.scores)
    assert snapshot.population_max_score == np.max(engine.scores)
    assert snapshot.custom_score == 0.25
    assert snapshot.score_improvement == 1.0
    assert snapshot.killed_doubles == 2
    assert [bar.source for bar in snapshot.population_bars] == ["elite", "crossover", "mutation", "fresh"]
    assert [bar.color for bar in snapshot.population_bars] == ["red", "green", "yellow", "blue"]
    assert [bar.score for bar in snapshot.population_bars] == [2.0, 1.75, 1.25, 0.5]


def test_progress_panel_omits_custom_score_when_fitness_has_single_component() -> None:
    engine = create_engine(population_size=2)
    engine.iterations = 1
    engine.best_score = 1.0
    engine.scores = np.array([0.5, 1.0], dtype=float)
    engine.step_custom_score = np.array([1.0], dtype=float)

    panel = create_progress_panel(prefer_matplotlib=False)
    snapshot = panel.build_snapshot(engine)

    assert snapshot.custom_score is None


def test_progress_panel_uses_second_fitness_component_as_custom_score() -> None:
    engine = create_engine(problem=VectorFitnessProblem, population_size=3)
    engine.population_init()
    engine.step()

    panel = create_progress_panel(prefer_matplotlib=False)
    snapshot = panel.build_snapshot(engine)

    assert engine.step_custom_score is not None
    assert snapshot.custom_score == float(engine.step_custom_score[1])


def test_progress_panel_exposes_adaptive_probabilities() -> None:
    engine = create_engine(
        crossover_policy=CrossoverPolicy.adaptive(
            bit_candidates=[CrossoverBitType.uniform, CrossoverBitType.one_point],
            float_candidates=[CrossoverFloatType.uniform, CrossoverFloatType.arithmetic],
        )
    )
    engine.iterations = 1
    engine.best_score = 1.5
    engine.scores = np.array([1.25, 1.5], dtype=float)
    engine.step_population_origins = [{"source": "initial"}, {"source": "initial"}]
    engine._adaptive_bit_probabilities[CrossoverBitType.uniform] = 0.7
    engine._adaptive_bit_probabilities[CrossoverBitType.one_point] = 0.3
    engine._adaptive_float_probabilities[CrossoverFloatType.uniform] = 0.4
    engine._adaptive_float_probabilities[CrossoverFloatType.arithmetic] = 0.6

    panel = create_progress_panel(prefer_matplotlib=False)
    snapshot = panel.build_snapshot(engine)

    assert snapshot.adaptive_mode is True
    assert snapshot.adaptive_bit_probabilities == {"uniform": 0.7, "one_point": 0.3}
    assert snapshot.adaptive_float_probabilities == {"uniform": 0.4, "arithmetic": 0.6}
