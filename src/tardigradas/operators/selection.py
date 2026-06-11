from __future__ import annotations

from typing import Sequence, Union

import numpy as np


def select_parents(expectation: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=int)

    wheel = expectation.cumsum()
    parents = np.zeros(count, dtype=int)
    step_size = 1 / count
    position = np.random.random() * step_size
    lowest = 0

    for i in range(count):
        for j in range(lowest, len(wheel)):
            if position < wheel[j]:
                parents[i] = j
                lowest = j
                break
        position += step_size

    return parents


def rank(estimates: Union[Sequence[float], np.ndarray], alpha: float, uniform_mix: float) -> np.ndarray:
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a finite non-negative number")
    if not np.isfinite(uniform_mix) or uniform_mix < 0.0 or uniform_mix > 1.0:
        raise ValueError("uniform_mix must be a finite number in range [0, 1]")

    scores = np.asarray(estimates, dtype=float)
    n_scores = len(scores)
    if n_scores == 0:
        return np.zeros(0, dtype=float)

    ix = (-scores).argsort()
    ranks = np.empty(n_scores, dtype=float)
    ranks[ix] = np.arange(1, n_scores + 1, dtype=float)

    expectation = 1.0 / np.power(ranks, alpha)
    expectation /= expectation.sum()

    if uniform_mix > 0.0:
        expectation = (1.0 - uniform_mix) * expectation + uniform_mix / n_scores
        expectation /= expectation.sum()

    return expectation