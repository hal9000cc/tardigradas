from __future__ import annotations

import pickle
from typing import Any

import numpy as np


def state_dict(engine: Any) -> dict[str, Any]:
    best_individual = None if engine.best_individual is None else engine.best_individual.chromo
    step_best_individual = None if engine.step_best_individual is None else engine.step_best_individual.chromo

    return {
        "epoch_score": engine.step_score,
        "step_score": engine.step_score,
        "step_custom_score": engine.step_custom_score,
        "iterations": engine.iterations,
        "scores_epoch": engine.scores_history,
        "scores_history": engine.scores_history,
        "custom_scores_epoch": engine.custom_scores_history,
        "custom_scores_history": engine.custom_scores_history,
        "crossover_fraction": engine.crossover_fraction,
        "fresh_blood_fraction": engine.fresh_blood_fraction,
        "gen_mutation_fraction": engine.gen_mutation_fraction,
        "chromo_bounds_max": engine.chromo_bounds_max,
        "chromo_bounds_min": engine.chromo_bounds_min,
        "chromo_gen_groups": engine.chromo_gen_groups,
        "chromo_defaults": engine.chromo_defaults,
        "chromo_defaults_probability": engine.chromo_defaults_probability,
        "gen_types": engine.gen_types,
        "chromo_size": engine.chromo_size,
        "gen_comments": engine.gen_comments,
        "n_elits": engine.n_elits,
        "population_size": engine.population_size,
        "scores": engine.scores,
        "best_score": engine.best_score,
        "best_iteration": engine.best_iteration,
        "best_resolve": best_individual,
        "best_individual": best_individual,
        "step_best_individual": step_best_individual,
        "population": [individual.chromo for individual in engine.population],
    }


def restore_from_dict(engine: Any, state: dict[str, Any]) -> None:
    engine.step_score = state.get("step_score", state.get("epoch_score"))
    engine.step_custom_score = state.get("step_custom_score")
    engine.iterations = int(state.get("iterations", 0))
    engine.scores_history = list(state.get("scores_history", state.get("scores_epoch", [])))
    engine.custom_scores_history = list(state.get("custom_scores_history", state.get("custom_scores_epoch", [])))
    engine.crossover_fraction = float(state.get("crossover_fraction", engine.crossover_fraction))
    engine.fresh_blood_fraction = float(state.get("fresh_blood_fraction", engine.fresh_blood_fraction))
    engine.gen_mutation_fraction = float(state.get("gen_mutation_fraction", engine.gen_mutation_fraction))
    engine.n_elits = int(state.get("n_elits", engine.n_elits))
    engine.population_size = int(state.get("population_size", engine.population_size))
    engine.scores = np.array(state.get("scores", []), dtype=float)
    engine.best_score = state.get("best_score")
    engine.best_iteration = int(state.get("best_iteration", 0))

    population = state.get("population", [])
    engine.population = [engine.create_individual(chromo=chromo) for chromo in population]

    best_individual = state.get("best_individual", state.get("best_resolve"))
    engine.best_individual = engine.create_individual(chromo=best_individual) if best_individual is not None else None

    step_best_individual = state.get("step_best_individual", best_individual)
    engine.step_best_individual = (
        engine.create_individual(chromo=step_best_individual) if step_best_individual is not None else engine.best_individual
    )

    engine.full_scores = np.zeros((0, 1), dtype=float)
    engine.fitness_progress_fun = None


def save_to_file(engine: Any, file_name: str) -> None:
    with open(file_name, "wb") as file:
        pickle.dump(state_dict(engine), file)


def restore_from_file(engine: Any, file_name: str) -> None:
    with open(file_name, "rb") as file:
        state = pickle.load(file)

    restore_from_dict(engine, state)