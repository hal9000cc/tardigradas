from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from benchmarks.common import print_benchmark_configuration, print_benchmark_summary, run_benchmark
from benchmarks.problems import RoyalRoadProblem
from tardigradas import CrossoverBitType, CrossoverFloatType, CrossoverPolicy, create_progress_panel


POPULATION_SIZE = 48
CROSSOVER_FRACTION = 0.65
FRESH_BLOOD_FRACTION = 0.0
GEN_MUTATION_FRACTION = 0.08
N_ELITS = 2
MAX_ITERATIONS = 50
CROSSOVER_POLICY = CrossoverPolicy.explicit(
    bit=CrossoverBitType.uniform,
    float=CrossoverFloatType.uniform,
)
SHOW_PROGRESS_PANEL = True


def main() -> None:
    progress_panel = create_progress_panel(title="Royal Road progress") if SHOW_PROGRESS_PANEL else None
    config = {
        "population_size": POPULATION_SIZE,
        "crossover_fraction": CROSSOVER_FRACTION,
        "fresh_blood_fraction": FRESH_BLOOD_FRACTION,
        "gen_mutation_fraction": GEN_MUTATION_FRACTION,
        "n_elits": N_ELITS,
        "max_iterations": MAX_ITERATIONS,
        "crossover_policy": CROSSOVER_POLICY,
    }
    print_benchmark_configuration("Royal Road", problem=RoyalRoadProblem, config=config)
    engine, initial_best_score = run_benchmark(
        RoyalRoadProblem,
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