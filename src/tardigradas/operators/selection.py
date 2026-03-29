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


def rank(estimates: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    expectation = np.zeros(len(estimates), dtype=float)
    ix = (-np.array(estimates, dtype=float)).argsort()
    expectation[ix] = 1.0 / np.power(np.arange(1, len(ix) + 1), 0.5)
    return expectation / np.sum(expectation)