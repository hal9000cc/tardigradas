from __future__ import annotations

from collections.abc import Sized
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from benchmarks.common import print_benchmark_configuration, print_benchmark_summary, run_benchmark
from benchmarks.mnist_helpers import (
    DEFAULT_BATCH_SIZE,
    MnistFullTrainConvProblem,
    benchmark_environment,
    create_mnist_benchmark_engine,
    evaluate_best_individual_on_test_split,
    resolve_mnist_root,
)
from tardigradas import CrossoverPolicy, create_progress_panel


POPULATION_SIZE = 50
CROSSOVER_FRACTION = 0.6
FRESH_BLOOD_FRACTION = 0.2
GEN_MUTATION_FRACTION = 0.02
N_ELITS = 2
MAX_ITERATIONS = 1000
BATCH_SIZE = DEFAULT_BATCH_SIZE
DATA_ROOT: str | None = None
REQUIRE_CUDA = True
CROSSOVER_POLICY = CrossoverPolicy.adaptive()
SHOW_PROGRESS_PANEL = True


class ScriptMnistProblem(MnistFullTrainConvProblem):
    batch_size = BATCH_SIZE
    data_root = DATA_ROOT
    require_cuda = REQUIRE_CUDA


def main() -> None:
    progress_panel = create_progress_panel(title="MNIST progress") if SHOW_PROGRESS_PANEL else None
    resolved_root = resolve_mnist_root(DATA_ROOT)
    config = {
        "population_size": POPULATION_SIZE,
        "crossover_fraction": CROSSOVER_FRACTION,
        "fresh_blood_fraction": FRESH_BLOOD_FRACTION,
        "gen_mutation_fraction": GEN_MUTATION_FRACTION,
        "n_elits": N_ELITS,
        "max_iterations": MAX_ITERATIONS,
        "batch_size": BATCH_SIZE,
        "data_root": resolved_root,
        "require_cuda": REQUIRE_CUDA,
        "crossover_policy": CROSSOVER_POLICY,
    }
    print_benchmark_configuration("MNIST", problem=ScriptMnistProblem, config=config)

    engine, initial_best_score = run_benchmark(
        ScriptMnistProblem,
        population_size=POPULATION_SIZE,
        crossover_fraction=CROSSOVER_FRACTION,
        fresh_blood_fraction=FRESH_BLOOD_FRACTION,
        gen_mutation_fraction=GEN_MUTATION_FRACTION,
        n_elits=N_ELITS,
        max_iterations=MAX_ITERATIONS,
        engine_factory=create_mnist_benchmark_engine,
        crossover_policy=CROSSOVER_POLICY,
        progress_panel=progress_panel,
    )
    environment = benchmark_environment(engine)
    best_test_metrics = evaluate_best_individual_on_test_split(engine)
    train_dataset = environment.train_loader.dataset
    test_dataset = environment.test_loader.dataset
    if not isinstance(train_dataset, Sized) or not isinstance(test_dataset, Sized):
        raise TypeError("MNIST benchmark datasets must implement __len__")

    extra_metrics = {
        "device": environment.device,
        "train_examples": len(train_dataset),
        "test_examples": len(test_dataset),
        "best_test_loss": best_test_metrics.loss,
        "best_test_accuracy": best_test_metrics.accuracy,
    }
    print_benchmark_summary(
        engine,
        initial_best_score,
        extra_metrics=extra_metrics,
        show_best_chromosome=False,
    )
    if progress_panel is not None:
        progress_panel.show(block=True)


if __name__ == "__main__":
    main()