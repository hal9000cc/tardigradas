from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from benchmarks.common import print_benchmark_configuration, print_benchmark_summary, run_benchmark
from benchmarks.problems import OneMaxProblem
from tardigradas import CrossoverBitType, CrossoverFloatType, CrossoverPolicy, create_progress_panel


CHROMO_SIZE = 24
POPULATION_SIZE = 40
CROSSOVER_FRACTION = 0.6
FRESH_BLOOD_FRACTION = 0.0
GEN_MUTATION_FRACTION = 0.12
N_ELITS = 2
MAX_ITERATIONS = 80
CROSSOVER_POLICY = CrossoverPolicy.explicit(
    bit=CrossoverBitType.uniform,
    float=CrossoverFloatType.uniform,
)
SHOW_PROGRESS_PANEL = True


class ConfiguredOneMaxProblem(OneMaxProblem):
    n_bits = CHROMO_SIZE


def main() -> None:
    progress_panel = create_progress_panel(title="OneMax progress") if SHOW_PROGRESS_PANEL else None
    config = {
        "chromo_size": CHROMO_SIZE,
        "population_size": POPULATION_SIZE,
        "crossover_fraction": CROSSOVER_FRACTION,
        "fresh_blood_fraction": FRESH_BLOOD_FRACTION,
        "gen_mutation_fraction": GEN_MUTATION_FRACTION,
        "n_elits": N_ELITS,
        "max_iterations": MAX_ITERATIONS,
        "crossover_policy": CROSSOVER_POLICY,
    }
    print_benchmark_configuration("OneMax", problem=ConfiguredOneMaxProblem, config=config)
    engine, initial_best_score = run_benchmark(
        ConfiguredOneMaxProblem,
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