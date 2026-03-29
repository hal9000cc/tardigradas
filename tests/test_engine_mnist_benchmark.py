from __future__ import annotations

from collections.abc import Sized
import os
from typing import cast

import numpy as np
import pytest


pytest.importorskip("torch")
pytest.importorskip("torchvision")


from tests.mnist_helpers import (
    benchmark_environment,
    create_mnist_benchmark_engine,
    evaluate_best_individual_on_test_split,
    get_mnist_benchmark_skip_reason,
)


pytestmark = [pytest.mark.slow, pytest.mark.gpu]


def test_mnist_full_train_cnn_benchmark_evaluates_full_splits() -> None:
    if os.getenv("TARDIGRADAS_RUN_MNIST_BENCHMARK") != "1":
        pytest.skip("MNIST benchmark is opt-in. Set TARDIGRADAS_RUN_MNIST_BENCHMARK=1 to enable it.")

    skip_reason = get_mnist_benchmark_skip_reason(require_cuda=True)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    engine = create_mnist_benchmark_engine(
        population_size=12,
        crossover_fraction=0.6,
        fresh_blood_fraction=0.2,
        gen_mutation_fraction=0.02,
        n_elits=2,
    )
    environment = benchmark_environment(engine)

    assert len(cast(Sized, environment.train_loader.dataset)) == 60_000
    assert len(cast(Sized, environment.test_loader.dataset)) == 10_000

    engine.population_init()
    engine.estimate_population()
    initial_best_score = float(np.max(engine.scores))

    engine.loop(
        max_iterations=4,
        epoch_without_improve=4,
        loop_fun=lambda _: False,
    )

    best_test_metrics = evaluate_best_individual_on_test_split(engine)

    assert engine.best_score is not None
    assert engine.best_individual is not None
    assert engine.best_score >= initial_best_score
    assert best_test_metrics.n_examples == 10_000
    assert best_test_metrics.loss < 2.5
    assert best_test_metrics.accuracy >= 0.10