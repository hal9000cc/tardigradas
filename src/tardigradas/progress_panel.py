from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np

from .engine import Tardigradas


SOURCE_COLORS: dict[str, str] = {
    "elite": "red",
    "crossover": "green",
    "mutation": "yellow",
    "fresh": "blue",
    "initial": "blue",
    "restored": "blue",
}
FALLBACK_SOURCE_COLOR = "gray"


@dataclass(frozen=True)
class PopulationBarEntry:
    rank: int
    score: float
    source: str
    color: str


@dataclass(frozen=True)
class ProgressSnapshot:
    iteration: int
    population_mean_score: float | None
    population_max_score: float | None
    custom_score: float | None
    score_improvement: float | None
    killed_doubles: int
    population_bars: tuple[PopulationBarEntry, ...]
    adaptive_mode: bool
    adaptive_bit_probabilities: dict[str, float]
    adaptive_float_probabilities: dict[str, float]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _best_score_from_engine(engine: Tardigradas) -> float | None:
    if engine.best_score is not None:
        return float(engine.best_score)
    if engine.step_score is not None:
        return float(engine.step_score)
    if engine.scores.size:
        return float(np.max(engine.scores))
    return None


def _custom_score_from_vector(value: Any) -> float | None:
    if value is None:
        return None

    custom_scores = np.asarray(value, dtype=float).reshape(-1)
    if custom_scores.size < 2:
        return None
    return float(custom_scores[1])


def _score_aligned_origins(engine: Tardigradas) -> list[dict[str, object]]:
    score_count = int(engine.scores.size)
    if score_count == 0:
        return []

    step_origins = getattr(engine, "step_population_origins", [])
    if len(step_origins) == score_count:
        return [dict(origin) for origin in step_origins]

    if len(engine.population_origins) == score_count:
        return [dict(origin) for origin in engine.population_origins]

    return [{"source": "unknown"} for _ in range(score_count)]


def _population_source(origin: dict[str, object]) -> str:
    return str(origin.get("source", "unknown")).lower()


def _build_population_bars(engine: Tardigradas) -> tuple[PopulationBarEntry, ...]:
    if engine.scores.size == 0:
        return ()

    origins = _score_aligned_origins(engine)
    order = np.argsort(-engine.scores)
    bars: list[PopulationBarEntry] = []

    for rank, index in enumerate(order, start=1):
        source = _population_source(origins[int(index)])
        bars.append(
            PopulationBarEntry(
                rank=rank,
                score=float(engine.scores[int(index)]),
                source=source,
                color=SOURCE_COLORS.get(source, FALLBACK_SOURCE_COLOR),
            )
        )

    return tuple(bars)


def _adaptive_probabilities(engine: Tardigradas) -> tuple[bool, dict[str, float], dict[str, float]]:
    adaptive_state = engine.adaptive_crossover_state()
    if adaptive_state.get("mode") != "adaptive":
        return False, {}, {}

    raw_bit_probabilities = adaptive_state.get("bit_probabilities", {})
    raw_float_probabilities = adaptive_state.get("float_probabilities", {})
    bit_mapping = cast(Mapping[str, Any], raw_bit_probabilities) if isinstance(raw_bit_probabilities, Mapping) else {}
    float_mapping = (
        cast(Mapping[str, Any], raw_float_probabilities) if isinstance(raw_float_probabilities, Mapping) else {}
    )

    bit_probabilities = {
        str(name): float(value)
        for name, value in bit_mapping.items()
    }
    float_probabilities = {
        str(name): float(value)
        for name, value in float_mapping.items()
    }
    return True, bit_probabilities, float_probabilities


def _history_series(
    history: list[ProgressSnapshot],
    extractor: Callable[[ProgressSnapshot], float | None],
) -> np.ndarray:
    values: list[float] = []
    for snapshot in history:
        extracted = extractor(snapshot)
        values.append(np.nan if extracted is None else float(extracted))
    return np.array(values, dtype=float)


class _MatplotlibProgressRenderer:
    def __init__(self, *, pyplot: Any, patches: Any, title: str) -> None:
        self._pyplot = pyplot
        self._patches = patches
        self._title = title

        self._pyplot.ion()
        self._figure = self._pyplot.figure(figsize=(15, 10))
        grid = self._figure.add_gridspec(6, 2, hspace=0.9, wspace=0.35)
        self._score_axis = self._figure.add_subplot(grid[:3, 0])
        self._custom_score_axis = self._figure.add_subplot(grid[3:, 0])
        self._population_axis = self._figure.add_subplot(grid[:2, 1])
        self._improved_axis = self._figure.add_subplot(grid[2:4, 1])
        self._adaptive_axis = self._figure.add_subplot(grid[4:, 1])
        self._improved_secondary_axis = self._improved_axis.twinx()

    def render(self, history: list[ProgressSnapshot]) -> None:
        if not history:
            return

        latest = history[-1]
        iterations = [snapshot.iteration for snapshot in history]

        self._score_axis.clear()
        self._custom_score_axis.clear()
        self._population_axis.clear()
        self._improved_axis.clear()
        self._improved_secondary_axis.clear()
        self._adaptive_axis.clear()

        self._plot_scores(iterations, history)
        self._plot_custom_score(iterations, history)
        self._plot_population(latest)
        self._plot_improved(iterations, history)
        self._plot_adaptive(iterations, history, latest)

        self._figure.suptitle(f"{self._title} · epoch {latest.iteration}")
        self._figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        self._figure.canvas.draw_idle()
        if hasattr(self._figure.canvas, "flush_events"):
            self._figure.canvas.flush_events()
        self._pyplot.pause(0.001)

    def _plot_scores(self, iterations: list[int], history: list[ProgressSnapshot]) -> None:
        series = [
            ("population_mean_score", lambda item: item.population_mean_score),
            ("population_max_score", lambda item: item.population_max_score),
        ]
        for label, extractor in series:
            values = _history_series(history, extractor)
            if np.isnan(values).all():
                continue
            self._score_axis.plot(iterations, values, label=label)

        self._score_axis.set_title("Score trends")
        self._score_axis.set_xlabel("Epoch")
        self._score_axis.set_ylabel("Score")
        if self._score_axis.lines:
            self._score_axis.legend(loc="best")

    def _plot_custom_score(self, iterations: list[int], history: list[ProgressSnapshot]) -> None:
        custom_score_values = _history_series(history, lambda item: item.custom_score)
        if np.isnan(custom_score_values).all():
            self._custom_score_axis.text(
                0.5,
                0.5,
                "Custom score is unavailable",
                ha="center",
                va="center",
                transform=self._custom_score_axis.transAxes,
            )
            self._custom_score_axis.set_title("Custom score")
            return

        self._custom_score_axis.plot(iterations, custom_score_values, label="custom_score", color="tab:orange")
        self._custom_score_axis.set_title("Custom score")
        self._custom_score_axis.set_xlabel("Epoch")
        self._custom_score_axis.set_ylabel("Custom score")
        self._custom_score_axis.legend(loc="best")

    def _plot_improved(self, iterations: list[int], history: list[ProgressSnapshot]) -> None:
        improvement_values = _history_series(history, lambda item: item.score_improvement)
        killed_doubles_values = _history_series(history, lambda item: float(item.killed_doubles))

        if not np.isnan(improvement_values).all():
            self._improved_axis.plot(iterations, improvement_values, label="score_improvement", color="tab:blue")
        if not np.isnan(killed_doubles_values).all():
            self._improved_secondary_axis.plot(
                iterations,
                killed_doubles_values,
                label="killed_doubles",
                color="tab:purple",
            )

        self._improved_axis.set_title("Improved")
        self._improved_axis.set_xlabel("Epoch")
        self._improved_axis.set_ylabel("Score improvement")
        self._improved_secondary_axis.set_ylabel("Killed doubles")

        handles = list(self._improved_axis.lines) + list(self._improved_secondary_axis.lines)
        if handles:
            labels = [line.get_label() for line in handles]
            self._improved_axis.legend(handles, labels, loc="best")

    def _plot_population(self, latest: ProgressSnapshot) -> None:
        if not latest.population_bars:
            self._population_axis.text(
                0.5,
                0.5,
                "Population scores are unavailable",
                ha="center",
                va="center",
                transform=self._population_axis.transAxes,
            )
            self._population_axis.set_title("Population scores")
            return

        ranks = [bar.rank for bar in latest.population_bars]
        scores = [bar.score for bar in latest.population_bars]
        colors = [bar.color for bar in latest.population_bars]
        self._population_axis.bar(ranks, scores, color=colors, edgecolor="black", linewidth=0.5)
        self._population_axis.set_title("Population scores by origin")
        self._population_axis.set_xlabel("Rank (descending score)")
        self._population_axis.set_ylabel("Score")

        sources_in_plot: list[str] = []
        for bar in latest.population_bars:
            if bar.source not in sources_in_plot:
                sources_in_plot.append(bar.source)

        legend_handles = [
            self._patches.Patch(
                facecolor=SOURCE_COLORS.get(source, FALLBACK_SOURCE_COLOR),
                edgecolor="black",
                label=source,
            )
            for source in sources_in_plot
        ]
        if legend_handles:
            self._population_axis.legend(handles=legend_handles, loc="best")

    def _plot_adaptive(
        self,
        iterations: list[int],
        history: list[ProgressSnapshot],
        latest: ProgressSnapshot,
    ) -> None:
        if not latest.adaptive_mode:
            self._adaptive_axis.text(
                0.5,
                0.5,
                "Adaptive crossover is disabled",
                ha="center",
                va="center",
                transform=self._adaptive_axis.transAxes,
            )
            self._adaptive_axis.set_title("Adaptive crossover")
            return

        bit_names = sorted({name for snapshot in history for name in snapshot.adaptive_bit_probabilities})
        float_names = sorted({name for snapshot in history for name in snapshot.adaptive_float_probabilities})

        for name in bit_names:
            values = np.array(
                [snapshot.adaptive_bit_probabilities.get(name, np.nan) for snapshot in history],
                dtype=float,
            )
            self._adaptive_axis.plot(iterations, values, label=f"bit:{name}")

        for name in float_names:
            values = np.array(
                [snapshot.adaptive_float_probabilities.get(name, np.nan) for snapshot in history],
                dtype=float,
            )
            self._adaptive_axis.plot(iterations, values, label=f"float:{name}", linestyle="--")

        self._adaptive_axis.set_title("Adaptive crossover probabilities")
        self._adaptive_axis.set_xlabel("Epoch")
        self._adaptive_axis.set_ylabel("Probability")
        if self._adaptive_axis.lines:
            self._adaptive_axis.legend(loc="best")

    def show(self, *, block: bool) -> None:
        self._pyplot.show(block=block)

    def close(self) -> None:
        self._pyplot.close(self._figure)


class ProgressPanel:
    def __init__(self, *, renderer: _MatplotlibProgressRenderer | None = None) -> None:
        self.history: list[ProgressSnapshot] = []
        self.initial_best_score: float | None = None
        self._renderer = renderer

    @property
    def rendering_available(self) -> bool:
        return self._renderer is not None

    def capture_initial_state(self, engine: Tardigradas) -> None:
        initial_best_score = _best_score_from_engine(engine)
        if initial_best_score is not None:
            self.initial_best_score = initial_best_score

    def build_snapshot(self, engine: Tardigradas) -> ProgressSnapshot:
        population_mean_score = float(np.mean(engine.scores)) if engine.scores.size else None
        population_max_score = float(np.max(engine.scores)) if engine.scores.size else None
        custom_score = _custom_score_from_vector(engine.step_custom_score)
        current_best_score = _optional_float(engine.best_score)
        if current_best_score is None:
            current_best_score = _optional_float(engine.step_score)
        if current_best_score is None:
            current_best_score = population_max_score

        if self.initial_best_score is None and current_best_score is not None:
            self.initial_best_score = current_best_score

        score_improvement = None
        if current_best_score is not None and self.initial_best_score is not None:
            score_improvement = float(current_best_score - self.initial_best_score)

        adaptive_mode, adaptive_bit_probabilities, adaptive_float_probabilities = _adaptive_probabilities(engine)

        return ProgressSnapshot(
            iteration=int(engine.iterations),
            population_mean_score=population_mean_score,
            population_max_score=population_max_score,
            custom_score=custom_score,
            score_improvement=score_improvement,
            killed_doubles=int(engine.n_killed_doubles),
            population_bars=_build_population_bars(engine),
            adaptive_mode=adaptive_mode,
            adaptive_bit_probabilities=adaptive_bit_probabilities,
            adaptive_float_probabilities=adaptive_float_probabilities,
        )

    def update(self, engine: Tardigradas) -> bool:
        snapshot = self.build_snapshot(engine)
        self.history.append(snapshot)
        if self._renderer is not None:
            self._renderer.render(self.history)
        return False

    def loop_callback(self) -> Callable[[Tardigradas], bool]:
        return self.update

    def show(self, *, block: bool = True) -> None:
        if self._renderer is None:
            return
        self._renderer.show(block=block)

    def close(self) -> None:
        if self._renderer is None:
            return
        self._renderer.close()

    def __call__(self, engine: Tardigradas) -> bool:
        return self.update(engine)


def _load_matplotlib_renderer(title: str) -> _MatplotlibProgressRenderer:
    pyplot = importlib.import_module("matplotlib.pyplot")
    patches = importlib.import_module("matplotlib.patches")
    return _MatplotlibProgressRenderer(pyplot=pyplot, patches=patches, title=title)


def create_progress_panel(
    *,
    title: str = "Tardigradas progress",
    prefer_matplotlib: bool = True,
) -> ProgressPanel:
    renderer = None
    if prefer_matplotlib:
        try:
            renderer = _load_matplotlib_renderer(title)
        except Exception:
            renderer = None

    return ProgressPanel(renderer=renderer)