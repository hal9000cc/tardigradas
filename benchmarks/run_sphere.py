from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from benchmarks.common import print_benchmark_configuration, print_benchmark_summary, run_benchmark
from benchmarks.problems import SphereProblem
from tardigradas import CrossoverBitType, CrossoverFloatType, CrossoverPolicy


POPULATION_SIZE = 50
CROSSOVER_FRACTION = 0.55
FRESH_BLOOD_FRACTION = 0.0
GEN_MUTATION_FRACTION = 0.18
N_ELITS = 2
MAX_ITERATIONS = 120
CROSSOVER_POLICY = CrossoverPolicy.explicit(
    bit=CrossoverBitType.uniform,
    float=CrossoverFloatType.uniform,
)


def main() -> None:
    config = {
        "population_size": POPULATION_SIZE,
        "crossover_fraction": CROSSOVER_FRACTION,
        "fresh_blood_fraction": FRESH_BLOOD_FRACTION,
        "gen_mutation_fraction": GEN_MUTATION_FRACTION,
        "n_elits": N_ELITS,
        "max_iterations": MAX_ITERATIONS,
        "crossover_policy": CROSSOVER_POLICY,
    }
    print_benchmark_configuration("Sphere", problem=SphereProblem, config=config)
    engine, initial_best_score = run_benchmark(
        SphereProblem,
        population_size=POPULATION_SIZE,
        crossover_fraction=CROSSOVER_FRACTION,
        fresh_blood_fraction=FRESH_BLOOD_FRACTION,
        gen_mutation_fraction=GEN_MUTATION_FRACTION,
        n_elits=N_ELITS,
        max_iterations=MAX_ITERATIONS,
        crossover_policy=CROSSOVER_POLICY,
    )
    print_benchmark_summary(engine, initial_best_score)


if __name__ == "__main__":
    main()