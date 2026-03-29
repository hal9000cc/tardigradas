from __future__ import annotations

import datetime as dt
import gc
from collections.abc import Sequence as CollectionSequence
from typing import Callable, Optional, Sequence, TypeVar, Union

import numpy as np

from .crossover_policy import CrossoverPolicy
from .exceptions import TardigradasException
from .gen_types import CrossoverBitType, CrossoverFloatType, GenType
from .individual import Individual
from .operators import (
    crossover_arithmetic,
    crossover_blx,
    crossover_one_point,
    crossover_two_point,
    crossover_uniform,
    mutate_chromosome,
    rank,
    select_parents,
)
from .problem import Problem
from .schema import ChromosomeSchema
from .serialization import restore_from_dict, restore_from_file, save_to_file, state_dict


TCrossoverOperator = TypeVar("TCrossoverOperator", CrossoverBitType, CrossoverFloatType)


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
        crossover_policy: Optional[CrossoverPolicy] = None,
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
        if crossover_policy is not None and not isinstance(crossover_policy, CrossoverPolicy):
            raise TypeError("crossover_policy must be CrossoverPolicy or None")
        self.crossover_policy = CrossoverPolicy.default() if crossover_policy is None else crossover_policy

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
        self.bit_gene_mask = self.gen_types == GenType.bit.value
        self.float_gene_mask = self.gen_types != GenType.bit.value
        self.int_gene_indices = np.nonzero(self.gen_types == GenType.int.value)[0]
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
        self.population_origins: list[dict[str, object]] = []
        self._last_crossover_origins: list[dict[str, object]] = []
        self._last_mutation_origins: list[dict[str, object]] = []
        # EMA scores and selection probabilities
        self._adaptive_bit_scores: dict[CrossoverBitType, float] = {}
        self._adaptive_bit_probabilities: dict[CrossoverBitType, float] = {}
        self._adaptive_float_scores: dict[CrossoverFloatType, float] = {}
        self._adaptive_float_probabilities: dict[CrossoverFloatType, float] = {}
        # Within-epoch counters (reset each epoch after statistics update)
        self._adaptive_bit_epoch_uses: dict[CrossoverBitType, int] = {}
        self._adaptive_bit_epoch_successes: dict[CrossoverBitType, int] = {}
        self._adaptive_float_epoch_uses: dict[CrossoverFloatType, int] = {}
        self._adaptive_float_epoch_successes: dict[CrossoverFloatType, int] = {}
        # Snapshot of the previous epoch (used for reporting only)
        self._adaptive_last_bit_epoch_uses: dict[CrossoverBitType, int] = {}
        self._adaptive_last_bit_epoch_successes: dict[CrossoverBitType, int] = {}
        self._adaptive_last_bit_instant_scores: dict[CrossoverBitType, Optional[float]] = {}
        self._adaptive_last_float_epoch_uses: dict[CrossoverFloatType, int] = {}
        self._adaptive_last_float_epoch_successes: dict[CrossoverFloatType, int] = {}
        self._adaptive_last_float_instant_scores: dict[CrossoverFloatType, Optional[float]] = {}
        self._reset_crossover_runtime_state()

    # ------------------------------------------------------------------
    # Population origin helpers
    # ------------------------------------------------------------------

    def _default_population_origin(self, source: str) -> dict[str, object]:
        return {
            "source": source,
            "bit_operator": None,
            "float_operator": None,
            "eligible_for_reward": False,
        }

    def _clone_population_origin(self, origin: dict[str, object]) -> dict[str, object]:
        return {
            "source": origin.get("source", "unknown"),
            "bit_operator": origin.get("bit_operator"),
            "float_operator": origin.get("float_operator"),
            "eligible_for_reward": bool(origin.get("eligible_for_reward", False)),
        }

    def _copy_origin_for_elitism(self, origin: dict[str, object]) -> dict[str, object]:
        copied_origin = self._clone_population_origin(origin)
        copied_origin["source"] = "elite"
        copied_origin["eligible_for_reward"] = False
        return copied_origin

    def _ensure_population_origins(self) -> None:
        if len(self.population_origins) == len(self.population):
            return
        self.population_origins = [self._default_population_origin("unknown") for _ in self.population]

    def _take_last_generated_origins(self, attr_name: str, expected_count: int, source: str) -> list[dict[str, object]]:
        generated_origins = [self._clone_population_origin(origin) for origin in getattr(self, attr_name, [])]
        setattr(self, attr_name, [])
        if len(generated_origins) != expected_count:
            return [self._default_population_origin(source) for _ in range(expected_count)]
        return generated_origins

    # ------------------------------------------------------------------
    # Adaptive crossover: state initialization
    # ------------------------------------------------------------------

    def _reset_crossover_runtime_state(self) -> None:
        self.population_origins = []
        self._last_crossover_origins = []
        self._last_mutation_origins = []
        self._adaptive_bit_scores = {}
        self._adaptive_bit_probabilities = {}
        self._adaptive_float_scores = {}
        self._adaptive_float_probabilities = {}
        self._adaptive_bit_epoch_uses = {}
        self._adaptive_bit_epoch_successes = {}
        self._adaptive_float_epoch_uses = {}
        self._adaptive_float_epoch_successes = {}
        self._adaptive_last_bit_epoch_uses = {}
        self._adaptive_last_bit_epoch_successes = {}
        self._adaptive_last_bit_instant_scores = {}
        self._adaptive_last_float_epoch_uses = {}
        self._adaptive_last_float_epoch_successes = {}
        self._adaptive_last_float_instant_scores = {}

        if not self.crossover_policy.is_adaptive:
            return

        initial_bit_probs = self._normalized_probabilities(
            np.ones(len(self.crossover_policy.bit_candidates), dtype=float)
        )
        initial_float_probs = self._normalized_probabilities(
            np.ones(len(self.crossover_policy.float_candidates), dtype=float)
        )

        for index, operator in enumerate(self.crossover_policy.bit_candidates):
            self._adaptive_bit_scores[operator] = 0.5
            self._adaptive_bit_probabilities[operator] = float(initial_bit_probs[index])
            self._adaptive_bit_epoch_uses[operator] = 0
            self._adaptive_bit_epoch_successes[operator] = 0
            self._adaptive_last_bit_epoch_uses[operator] = 0
            self._adaptive_last_bit_epoch_successes[operator] = 0
            self._adaptive_last_bit_instant_scores[operator] = None

        for index, operator in enumerate(self.crossover_policy.float_candidates):
            self._adaptive_float_scores[operator] = 0.5
            self._adaptive_float_probabilities[operator] = float(initial_float_probs[index])
            self._adaptive_float_epoch_uses[operator] = 0
            self._adaptive_float_epoch_successes[operator] = 0
            self._adaptive_last_float_epoch_uses[operator] = 0
            self._adaptive_last_float_epoch_successes[operator] = 0
            self._adaptive_last_float_instant_scores[operator] = None

    # ------------------------------------------------------------------
    # Adaptive crossover: probability helpers
    # ------------------------------------------------------------------

    def _normalized_probabilities(self, weights: np.ndarray) -> np.ndarray:
        probabilities = np.array(weights, dtype=float)
        if probabilities.size == 0:
            return probabilities

        total = float(probabilities.sum())
        if total <= 0.0:
            probabilities = np.ones(probabilities.size, dtype=float) / probabilities.size
        else:
            probabilities /= total

        if self.crossover_policy.min_probability > 0.0:
            floor = self.crossover_policy.min_probability
            probabilities = probabilities * (1.0 - probabilities.size * floor) + floor

        probabilities /= probabilities.sum()
        return probabilities

    def _adaptive_probabilities_from_scores(
        self,
        candidates: Sequence[TCrossoverOperator],
        scores: dict[TCrossoverOperator, float],
    ) -> np.ndarray:
        return self._normalized_probabilities(
            np.array([scores[candidate] for candidate in candidates], dtype=float)
        )

    @staticmethod
    def _adaptive_instant_score(uses: int, successes: int) -> float:
        return (successes + 1.0) / (uses + 2.0)

    def _update_adaptive_operator_statistics(
        self,
        candidates: Sequence[TCrossoverOperator],
        epoch_uses: dict[TCrossoverOperator, int],
        epoch_successes: dict[TCrossoverOperator, int],
        scores: dict[TCrossoverOperator, float],
        probabilities: dict[TCrossoverOperator, float],
        last_epoch_uses: dict[TCrossoverOperator, int],
        last_epoch_successes: dict[TCrossoverOperator, int],
        last_epoch_instant_scores: dict[TCrossoverOperator, Optional[float]],
    ) -> None:
        alpha = self.crossover_policy.alpha

        for candidate in candidates:
            uses = int(epoch_uses[candidate])
            successes = int(epoch_successes[candidate])
            last_epoch_uses[candidate] = uses
            last_epoch_successes[candidate] = successes

            if uses > 0:
                instant_score = self._adaptive_instant_score(uses, successes)
                scores[candidate] = (1.0 - alpha) * scores[candidate] + alpha * instant_score
                last_epoch_instant_scores[candidate] = instant_score
            else:
                last_epoch_instant_scores[candidate] = None

            epoch_uses[candidate] = 0
            epoch_successes[candidate] = 0

        updated_probabilities = self._adaptive_probabilities_from_scores(candidates, scores)
        for index, candidate in enumerate(candidates):
            probabilities[candidate] = float(updated_probabilities[index])

    # ------------------------------------------------------------------
    # Adaptive crossover: operator selection
    # ------------------------------------------------------------------

    def _select_adaptive_operator(
        self,
        candidates: Sequence[TCrossoverOperator],
        probabilities: dict[TCrossoverOperator, float],
        epoch_uses: dict[TCrossoverOperator, int],
    ) -> TCrossoverOperator:
        distribution = np.array([probabilities[candidate] for candidate in candidates], dtype=float)
        distribution /= distribution.sum()
        selected_index = int(np.random.choice(len(candidates), p=distribution))
        operator = candidates[selected_index]
        epoch_uses[operator] += 1
        return operator

    def _select_bit_crossover_operator(self) -> Optional[CrossoverBitType]:
        if not self.bit_gene_mask.any():
            return None
        if self.crossover_policy.is_explicit:
            return self.crossover_policy.bit
        return self._select_adaptive_operator(
            self.crossover_policy.bit_candidates,
            self._adaptive_bit_probabilities,
            self._adaptive_bit_epoch_uses,
        )

    def _select_float_crossover_operator(self) -> Optional[CrossoverFloatType]:
        if not self.float_gene_mask.any():
            return None
        if self.crossover_policy.is_explicit:
            return self.crossover_policy.float
        return self._select_adaptive_operator(
            self.crossover_policy.float_candidates,
            self._adaptive_float_probabilities,
            self._adaptive_float_epoch_uses,
        )

    # ------------------------------------------------------------------
    # Crossover and mutation operators
    # ------------------------------------------------------------------

    def _apply_bit_crossover(
        self,
        kid: np.ndarray,
        parent1: np.ndarray,
        parent2: np.ndarray,
        operator: CrossoverBitType,
    ) -> np.ndarray:
        if operator == CrossoverBitType.uniform:
            return crossover_uniform(kid, parent1, parent2, self.chromo_gen_groups, self.bit_gene_mask)
        if operator == CrossoverBitType.one_point:
            return crossover_one_point(kid, parent1, parent2, self.chromo_gen_groups, self.bit_gene_mask)
        if operator == CrossoverBitType.two_point:
            return crossover_two_point(kid, parent1, parent2, self.chromo_gen_groups, self.bit_gene_mask)
        raise TardigradasException(f"unsupported bit crossover operator: {operator}")

    def _apply_float_crossover(
        self,
        kid: np.ndarray,
        parent1: np.ndarray,
        parent2: np.ndarray,
        operator: CrossoverFloatType,
    ) -> np.ndarray:
        if operator == CrossoverFloatType.uniform:
            return crossover_uniform(kid, parent1, parent2, self.chromo_gen_groups, self.float_gene_mask)
        if operator == CrossoverFloatType.arithmetic:
            return crossover_arithmetic(
                kid,
                parent1,
                parent2,
                self.float_gene_mask,
                self.chromo_bounds_min,
                self.chromo_bounds_max,
            )
        if operator == CrossoverFloatType.BLX:
            return crossover_blx(
                kid,
                parent1,
                parent2,
                self.float_gene_mask,
                self.chromo_bounds_min,
                self.chromo_bounds_max,
            )
        raise TardigradasException(f"unsupported float crossover operator: {operator}")

    # ------------------------------------------------------------------
    # Adaptive crossover: statistics update (called once per step)
    # ------------------------------------------------------------------

    def _update_adaptive_crossover_statistics(self, elite_indices: np.ndarray) -> None:
        if not self.crossover_policy.is_adaptive:
            return

        self._ensure_population_origins()
        elite_index_set = {int(index) for index in elite_indices}

        for index, origin in enumerate(self.population_origins):
            if not bool(origin.get("eligible_for_reward", False)):
                continue

            if index in elite_index_set and origin.get("source") == "crossover":
                bit_operator = origin.get("bit_operator")
                if isinstance(bit_operator, CrossoverBitType) and bit_operator in self._adaptive_bit_epoch_successes:
                    self._adaptive_bit_epoch_successes[bit_operator] += 1

                float_operator = origin.get("float_operator")
                if isinstance(float_operator, CrossoverFloatType) and float_operator in self._adaptive_float_epoch_successes:
                    self._adaptive_float_epoch_successes[float_operator] += 1

            origin["eligible_for_reward"] = False

        self._update_adaptive_operator_statistics(
            self.crossover_policy.bit_candidates,
            self._adaptive_bit_epoch_uses,
            self._adaptive_bit_epoch_successes,
            self._adaptive_bit_scores,
            self._adaptive_bit_probabilities,
            self._adaptive_last_bit_epoch_uses,
            self._adaptive_last_bit_epoch_successes,
            self._adaptive_last_bit_instant_scores,
        )
        self._update_adaptive_operator_statistics(
            self.crossover_policy.float_candidates,
            self._adaptive_float_epoch_uses,
            self._adaptive_float_epoch_successes,
            self._adaptive_float_scores,
            self._adaptive_float_probabilities,
            self._adaptive_last_float_epoch_uses,
            self._adaptive_last_float_epoch_successes,
            self._adaptive_last_float_instant_scores,
        )

    # ------------------------------------------------------------------
    # Adaptive crossover: reporting
    # ------------------------------------------------------------------

    def adaptive_crossover_state(self) -> dict[str, object]:
        if not self.crossover_policy.is_adaptive:
            return {
                "mode": self.crossover_policy.mode,
                "bit": self.crossover_policy.bit.name if self.crossover_policy.bit is not None else None,
                "float": self.crossover_policy.float.name if self.crossover_policy.float is not None else None,
            }

        def build_operator_state(
            candidates: CollectionSequence[TCrossoverOperator],
            epoch_uses: dict[TCrossoverOperator, int],
            epoch_successes: dict[TCrossoverOperator, int],
            instant_scores: dict[TCrossoverOperator, Optional[float]],
            scores: dict[TCrossoverOperator, float],
            probabilities: dict[TCrossoverOperator, float],
        ) -> tuple[
            list[str],
            dict[str, int],
            dict[str, int],
            dict[str, Optional[float]],
            dict[str, float],
            dict[str, float],
        ]:
            candidate_names = [candidate.name for candidate in candidates]
            named_epoch_uses = {candidate.name: int(epoch_uses[candidate]) for candidate in candidates}
            named_epoch_successes = {candidate.name: int(epoch_successes[candidate]) for candidate in candidates}
            named_instant_scores = {
                candidate.name: None if instant_scores[candidate] is None else float(instant_scores[candidate])
                for candidate in candidates
            }
            named_scores = {candidate.name: float(scores[candidate]) for candidate in candidates}
            named_probabilities = {candidate.name: float(probabilities[candidate]) for candidate in candidates}
            return (
                candidate_names,
                named_epoch_uses,
                named_epoch_successes,
                named_instant_scores,
                named_scores,
                named_probabilities,
            )

        (
            bit_candidates,
            bit_epoch_uses,
            bit_epoch_successes,
            bit_instant_scores,
            bit_scores,
            bit_probabilities,
        ) = build_operator_state(
            self.crossover_policy.bit_candidates,
            self._adaptive_last_bit_epoch_uses,
            self._adaptive_last_bit_epoch_successes,
            self._adaptive_last_bit_instant_scores,
            self._adaptive_bit_scores,
            self._adaptive_bit_probabilities,
        )
        (
            float_candidates,
            float_epoch_uses,
            float_epoch_successes,
            float_instant_scores,
            float_scores,
            float_probabilities,
        ) = build_operator_state(
            self.crossover_policy.float_candidates,
            self._adaptive_last_float_epoch_uses,
            self._adaptive_last_float_epoch_successes,
            self._adaptive_last_float_instant_scores,
            self._adaptive_float_scores,
            self._adaptive_float_probabilities,
        )

        return {
            "mode": self.crossover_policy.mode,
            "reward": self.crossover_policy.reward,
            "min_probability": self.crossover_policy.min_probability,
            "period": self.crossover_policy.period,
            "alpha": self.crossover_policy.alpha,
            "bit_candidates": bit_candidates,
            "float_candidates": float_candidates,
            "bit_epoch_uses": bit_epoch_uses,
            "bit_epoch_successes": bit_epoch_successes,
            "bit_instant_scores": bit_instant_scores,
            "bit_scores": bit_scores,
            "bit_probabilities": bit_probabilities,
            "float_epoch_uses": float_epoch_uses,
            "float_epoch_successes": float_epoch_successes,
            "float_instant_scores": float_instant_scores,
            "float_scores": float_scores,
            "float_probabilities": float_probabilities,
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def population_init(self) -> None:
        self._reset_crossover_runtime_state()
        self.population = [self.new_valid_individual(use_defaults=True) for _ in range(self.population_size)]
        self.population_origins = [self._default_population_origin("initial") for _ in range(self.population_size)]
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
        origins: list[dict[str, object]] = []
        n_kids = len(parent_indices) // 2

        for i in range(n_kids):
            parent1 = self.population[parent_indices[i]]
            parent2 = self.population[parent_indices[i + n_kids]]

            if self.problem.is_equal(parent1.chromo, parent2.chromo):
                fallback_kids = self.mutation(np.array([parent_indices[i]], dtype=int))
                kids.extend(fallback_kids)
                origins.extend(self._take_last_generated_origins("_last_mutation_origins", len(fallback_kids), "mutation"))
                continue

            kid_chromo = np.zeros(self.chromo_size, dtype=float)
            bit_operator = self._select_bit_crossover_operator()
            float_operator = self._select_float_crossover_operator()

            if bit_operator is not None:
                self._apply_bit_crossover(kid_chromo, parent1.chromo, parent2.chromo, bit_operator)
            if float_operator is not None:
                self._apply_float_crossover(kid_chromo, parent1.chromo, parent2.chromo, float_operator)

            for i_int_gen in self.int_gene_indices:
                kid_chromo[i_int_gen] = float(round(float(kid_chromo[i_int_gen])))

            kids.append(self.create_individual(chromo=kid_chromo))

            origins.append(
                {
                    "source": "crossover",
                    "bit_operator": bit_operator,
                    "float_operator": float_operator,
                    "eligible_for_reward": self.crossover_policy.is_adaptive,
                }
            )

        self._last_crossover_origins = origins
        return kids

    def mutation(self, parent_indices: np.ndarray) -> list[Individual]:
        self._last_mutation_origins = []
        if len(parent_indices) == 0:
            return []
        if len(self.mutable_positions) == 0:
            raise TardigradasException("can't mutate chromosome because all genes are fixed")

        n_mutation = int(round(abs(np.random.normal(0, self.chromo_size * self.gen_mutation_fraction))))
        n_mutation = int(np.clip(n_mutation, 1, max(1, len(self.mutable_positions))))

        kids: list[Individual] = []
        origins: list[dict[str, object]] = []
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
                    origins.append(self._default_population_origin("mutation"))
                    break
            else:
                raise TardigradasException(f"can't create a mutated chromosome in {n_attempts} attempts")

        self._last_mutation_origins = origins
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
        self._ensure_population_origins()
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
                    self.population_origins[index] = self._default_population_origin("fresh")
                    seen.add(signature)
                    self.n_killed_doubles += 1
                    break
            else:
                raise TardigradasException(f"can't replace duplicate chromosome in {n_attempts} attempts")

    # ------------------------------------------------------------------
    # Serialization delegation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, object]:
        return state_dict(self)

    def restore_from_dict(self, state: dict[str, object]) -> None:
        restore_from_dict(self, state)

    def save_to_file(self, file_name: str) -> None:
        save_to_file(self, file_name)

    def restore_from_file(self, file_name: str) -> None:
        restore_from_file(self, file_name)

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------

    def step(self) -> None:
        if not self.population:
            raise TardigradasException("population is not initialized, call population_init() first")

        self._ensure_population_origins()
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
        self._update_adaptive_crossover_statistics(elite_indices)
        kids_elit = [self.population[index] for index in elite_indices]
        elite_origins = [self._copy_origin_for_elitism(self.population_origins[index]) for index in elite_indices]

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
        kids_crossover_origins = self._take_last_generated_origins("_last_crossover_origins", len(kids_crossover), "crossover")
        kids_mutation = self.mutation(parent_indices[2 * n_crossover :]) if n_mutation else []
        kids_mutation_origins = self._take_last_generated_origins("_last_mutation_origins", len(kids_mutation), "mutation")
        kids_new = [self.new_valid_individual(use_defaults=True) for _ in range(n_fresh_blood)]
        kids_new_origins = [self._default_population_origin("fresh") for _ in range(n_fresh_blood)]

        self.population = kids_elit + kids_crossover + kids_mutation + kids_new
        self.population_origins = elite_origins + kids_crossover_origins + kids_mutation_origins + kids_new_origins

        for i_individual, individual in enumerate(self.population):
            if not individual.chromo_valid():
                self.population[i_individual] = self.new_valid_individual()
                self.population_origins[i_individual] = self._default_population_origin("fresh")

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
