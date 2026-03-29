from __future__ import annotations

import datetime as dt
import gc
from typing import Callable, Optional, Sequence, Union

import numpy as np

from .exceptions import TardigradasException
from .gen_types import GenType
from .individual import Individual
from .operators import crossover_uniform, mutate_chromosome, rank, select_parents
from .problem import Problem
from .schema import ChromosomeSchema
from .serialization import restore_from_dict, restore_from_file, save_to_file, state_dict


class Tardigradas:
    def __init__(
        self,
        problem: type[Problem],
        population_size: int,
        crossover_fraction: float = 0.5,
        fresh_blood_fraction: float = 0.0,
        gen_mutation_fraction: float = 0.1,
        fitness_environment: Optional[object] = None,
        n_elits: Optional[int] = None,
    ) -> None:
        if not issubclass(problem, Problem):
            raise TypeError("problem must be a Problem subclass")
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if crossover_fraction < 0 or fresh_blood_fraction < 0 or gen_mutation_fraction < 0:
            raise ValueError("fractions must be non-negative")
        if crossover_fraction + fresh_blood_fraction > 1:
            raise ValueError("crossover_fraction + fresh_blood_fraction must be <= 1")

        self.problem = problem
        self.environment = fitness_environment
        self.population_size = int(population_size)
        self.crossover_fraction = float(crossover_fraction)
        self.fresh_blood_fraction = float(fresh_blood_fraction)
        self.gen_mutation_fraction = float(gen_mutation_fraction)
        self.n_elits = 1 if n_elits is None else int(n_elits)

        if self.n_elits < 0 or self.n_elits >= self.population_size:
            raise ValueError("n_elits must be in range [0, population_size)")

        problem.init_environment(self)
        schema = problem.gen_info(self)
        if not isinstance(schema, ChromosomeSchema):
            raise TypeError("problem.gen_info() must return ChromosomeSchema")

        self.schema = schema
        self.chromo_size = schema.chromo_size
        self.gen_comments = list(schema.comments)
        self.gen_types = np.array([gen_type.value for gen_type in schema.gen_types], dtype=int)
        self.chromo_bounds_min = np.array(schema.bounds[0], dtype=float)
        self.chromo_bounds_max = np.array(schema.bounds[1], dtype=float)
        self.chromo_gen_groups = np.array(schema.groups, dtype=int)
        self.chromo_defaults = np.array(schema.defaults, dtype=float)
        self.chromo_defaults_probability = np.array(schema.defaults_probability, dtype=float)
        self.mutable_positions = np.nonzero(self.chromo_bounds_min != self.chromo_bounds_max)[0]

        self.population: list[Individual] = []
        self.iterations = 0
        self.scores_history: list[float] = []
        self.custom_scores_history: list[np.ndarray] = []
        self.best_score: Optional[float] = None
        self.best_iteration = 0
        self.best_individual: Optional[Individual] = None
        self.step_best_individual: Optional[Individual] = None
        self.step_score: Optional[float] = None
        self.step_custom_score: Optional[np.ndarray] = None
        self.fitness_progress_fun: Optional[Callable[[Tardigradas, float], object]] = None
        self.scores = np.zeros(0, dtype=float)
        self.full_scores = np.zeros((0, 1), dtype=float)
        self.n_killed_doubles = 0

    def show_progress(self, *_: object) -> bool:
        time = dt.datetime.now()
        print(f"{time}: {self.iterations=}, {self.step_score=}")
        return False

    @property
    def population_chromosomes(self) -> np.ndarray:
        if not self.population:
            return np.zeros((0, self.chromo_size), dtype=float)
        return np.vstack([individual.chromo for individual in self.population])

    def create_individual(
        self,
        chromo: Optional[Union[Sequence[float], np.ndarray]] = None,
        use_defaults: bool = False,
    ) -> Individual:
        return self.problem.create_individual(self, chromo=chromo, use_defaults=use_defaults)

    @property
    def best_resolve(self) -> Optional[Individual]:
        return self.best_individual

    @property
    def step_best_resolve(self) -> Optional[Individual]:
        return self.step_best_individual

    def new_valid_individual(self, use_defaults: bool = False) -> Individual:
        n_attempts = 200
        for _ in range(n_attempts):
            random_individual = self.create_individual(use_defaults=use_defaults)
            if random_individual.chromo_valid():
                return random_individual

        raise TardigradasException(f"can't create a new random chromosome in {n_attempts} attempts")

    def population_init(self) -> None:
        self.population = [self.new_valid_individual(use_defaults=True) for _ in range(self.population_size)]
        self.iterations = 0
        self.scores_history = []
        self.custom_scores_history = []
        self.best_score = None
        self.best_iteration = 0
        self.best_individual = None
        self.step_best_individual = None
        self.step_score = None
        self.step_custom_score = None
        self.fitness_progress_fun = None
        self.scores = np.zeros(0, dtype=float)
        self.full_scores = np.zeros((0, 1), dtype=float)

    def crossover(self, parent_indices: np.ndarray) -> list[Individual]:
        kids: list[Individual] = []
        n_kids = len(parent_indices) // 2

        for i in range(n_kids):
            parent1 = self.population[parent_indices[i]]
            parent2 = self.population[parent_indices[i + n_kids]]

            if self.problem.is_equal(parent1.chromo, parent2.chromo):
                kids.extend(self.mutation(np.array([parent_indices[i]], dtype=int)))
                continue

            kid_chromo = np.zeros(self.chromo_size, dtype=float)
            gene_mask = np.ones(self.chromo_size, dtype=bool)
            crossover_uniform(kid_chromo, parent1.chromo, parent2.chromo, self.chromo_gen_groups, gene_mask)

            int_gen_indices = np.nonzero(self.gen_types == GenType.int.value)[0]
            for i_int_gen in int_gen_indices:
                kid_chromo[i_int_gen] = round(kid_chromo[i_int_gen])

            kids.append(self.create_individual(chromo=kid_chromo))

        return kids

    def mutation(self, parent_indices: np.ndarray) -> list[Individual]:
        if len(parent_indices) == 0:
            return []
        if len(self.mutable_positions) == 0:
            raise TardigradasException("can't mutate chromosome because all genes are fixed")

        n_mutation = int(round(abs(np.random.normal(0, self.chromo_size * self.gen_mutation_fraction))))
        n_mutation = int(np.clip(n_mutation, 1, max(1, len(self.mutable_positions))))

        kids: list[Individual] = []
        for i_parent in parent_indices:
            n_attempts = 200
            for _ in range(n_attempts):
                kid_chromo = mutate_chromosome(
                    parent_chromo=self.population[i_parent].chromo,
                    gen_types=self.gen_types,
                    bounds_min=self.chromo_bounds_min,
                    bounds_max=self.chromo_bounds_max,
                    mutable_positions=self.mutable_positions,
                    n_mutation=n_mutation,
                )
                kid = self.create_individual(chromo=kid_chromo)
                if kid.chromo_valid() and not self.problem.is_equal(kid.chromo, self.population[i_parent].chromo):
                    kids.append(kid)
                    break
            else:
                raise TardigradasException(f"can't create a mutated chromosome in {n_attempts} attempts")

        return kids

    def estimate_population(self) -> None:
        scores = []
        for i, individual in enumerate(self.population):
            scores.append(individual.fitness())
            if self.fitness_progress_fun:
                self.fitness_progress_fun(self, i / self.population_size)

        self.full_scores = np.vstack(scores)
        self.scores = self.full_scores[:, 0]

    def kill_doubles(self) -> None:
        self.n_killed_doubles = 0
        seen: set[bytes] = set()
        n_attempts = 200

        for index, individual in enumerate(self.population):
            signature = individual.chromo.tobytes()
            if signature not in seen:
                seen.add(signature)
                continue

            for _ in range(n_attempts):
                replacement = self.new_valid_individual()
                signature = replacement.chromo.tobytes()
                if signature not in seen:
                    self.population[index] = replacement
                    seen.add(signature)
                    self.n_killed_doubles += 1
                    break
            else:
                raise TardigradasException(f"can't replace duplicate chromosome in {n_attempts} attempts")

    def state_dict(self) -> dict[str, object]:
        return state_dict(self)

    def restore_from_dict(self, state: dict[str, object]) -> None:
        restore_from_dict(self, state)

    def save_to_file(self, file_name: str) -> None:
        save_to_file(self, file_name)

    def restore_from_file(self, file_name: str) -> None:
        restore_from_file(self, file_name)

    def step(self) -> None:
        if not self.population:
            raise TardigradasException("population is not initialized, call population_init() first")

        self.estimate_population()

        n_generation_slots = self.population_size - self.n_elits
        n_crossover = int(np.floor(n_generation_slots * self.crossover_fraction))
        n_fresh_blood = int(np.floor(n_generation_slots * self.fresh_blood_fraction))
        n_mutation = n_generation_slots - n_crossover - n_fresh_blood
        n_parents = n_crossover * 2 + n_mutation

        expectation = rank(self.scores)
        parent_indices = select_parents(expectation, n_parents)
        parent_indices = np.random.permutation(parent_indices)

        ix_best = np.argsort(-self.scores)
        elite_indices = ix_best[: self.n_elits]
        kids_elit = [self.population[index] for index in elite_indices]

        self.step_score = float(self.scores[ix_best[0]])
        self.step_custom_score = self.full_scores[ix_best[0]]
        self.step_best_individual = self.population[ix_best[0]]

        if self.best_score is None or self.step_score > self.best_score:
            self.best_score = self.step_score
            self.best_iteration = self.iterations
            self.best_individual = self.population[ix_best[0]]

        self.scores_history.append(self.step_score)
        self.custom_scores_history.append(self.step_custom_score)

        kids_crossover = self.crossover(parent_indices[: 2 * n_crossover]) if n_crossover else []
        kids_mutation = self.mutation(parent_indices[2 * n_crossover :]) if n_mutation else []
        kids_new = [self.new_valid_individual(use_defaults=True) for _ in range(n_fresh_blood)]

        self.population = kids_elit + kids_crossover + kids_mutation + kids_new

        for i_individual, individual in enumerate(self.population):
            if not individual.chromo_valid():
                self.population[i_individual] = self.new_valid_individual()

        self.kill_doubles()
        self.iterations += 1

    def loop(
        self,
        max_iterations: Optional[int] = None,
        epoch_without_improve: int = 50,
        loop_fun: Optional[Callable[[Tardigradas], bool]] = None,
        fitness_progress_fun: Optional[Callable[[Tardigradas, float], object]] = None,
    ) -> None:
        self.fitness_progress_fun = fitness_progress_fun
        loop_fun = self.show_progress if loop_fun is None else loop_fun

        while True:
            self.step()
            gc.collect()

            if loop_fun(self):
                break
            if max_iterations is not None and self.iterations >= max_iterations:
                break
            if self.iterations - self.best_iteration > epoch_without_improve:
                break