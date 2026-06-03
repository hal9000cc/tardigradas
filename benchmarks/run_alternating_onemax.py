from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from benchmarks.common import print_benchmark_configuration, print_benchmark_summary, run_benchmark
from benchmarks.problems import BitBenchmarkProblem, build_bit_benchmark_schema
from tardigradas import (
    ChromosomeSchema,
    CrossoverBitType,
    CrossoverFloatType,
    CrossoverPolicy,
    Individual,
    Tardigradas,
    create_progress_panel,
)


CHROMO_SIZE = 100000
USE_ALL_ONES_INITIALIZATION = True
POPULATION_SIZE = 12
CROSSOVER_FRACTION = 0.6
FRESH_BLOOD_FRACTION = 0.0
GEN_MUTATION_FRACTION = 0.2
N_ELITS = 2
MAX_ITERATIONS = 20
CROSSOVER_POLICY = CrossoverPolicy.explicit(
    bit=CrossoverBitType.uniform,
    float=CrossoverFloatType.uniform,
)
SHOW_PROGRESS_PANEL = True


def build_alternating_onemax_problem(
    *,
    chromo_size: int,
    use_all_ones_initialization: bool,
) -> type[BitBenchmarkProblem]:
    class ConfiguredAlternatingOneMaxProblem(BitBenchmarkProblem):
        n_bits = chromo_size

        @classmethod
        def gen_info(cls, tardigradas: Tardigradas) -> ChromosomeSchema:
            schema = build_bit_benchmark_schema(cls.n_bits)
            if use_all_ones_initialization:
                schema.defaults = [1.0] * cls.n_bits
                schema.defaults_probability = [1.0] * cls.n_bits
            return schema

        @staticmethod
        def fitness(individual: Individual) -> float:
            chromo = individual.chromo
            return float(np.sum(chromo[::2]) + np.sum(1.0 - chromo[1::2]))

    ConfiguredAlternatingOneMaxProblem.__name__ = "ConfiguredAlternatingOneMaxProblem"
    return ConfiguredAlternatingOneMaxProblem


ConfiguredAlternatingOneMaxProblem = build_alternating_onemax_problem(
    chromo_size=CHROMO_SIZE,
    use_all_ones_initialization=USE_ALL_ONES_INITIALIZATION,
)


def main() -> None:
    progress_panel = create_progress_panel(title="Alternating OneMax progress") if SHOW_PROGRESS_PANEL else None
    config = {
        "chromo_size": CHROMO_SIZE,
        "use_all_ones_initialization": USE_ALL_ONES_INITIALIZATION,
        "population_size": POPULATION_SIZE,
        "crossover_fraction": CROSSOVER_FRACTION,
        "fresh_blood_fraction": FRESH_BLOOD_FRACTION,
        "gen_mutation_fraction": GEN_MUTATION_FRACTION,
        "n_elits": N_ELITS,
        "max_iterations": MAX_ITERATIONS,
        "crossover_policy": CROSSOVER_POLICY,
    }
    print_benchmark_configuration(
        "Alternating OneMax",
        problem=ConfiguredAlternatingOneMaxProblem,
        config=config,
    )
    engine, initial_best_score = run_benchmark(
        ConfiguredAlternatingOneMaxProblem,
        population_size=POPULATION_SIZE,
        crossover_fraction=CROSSOVER_FRACTION,
        fresh_blood_fraction=FRESH_BLOOD_FRACTION,
        gen_mutation_fraction=GEN_MUTATION_FRACTION,
        n_elits=N_ELITS,
        max_iterations=MAX_ITERATIONS,
        crossover_policy=CROSSOVER_POLICY,
        progress_panel=progress_panel,
    )
    print_benchmark_summary(engine, initial_best_score)
    if progress_panel is not None:
        progress_panel.show(block=True)


if __name__ == "__main__":
    main()