from __future__ import annotations

import numpy as np

from ..gen_types import GenType


def mutation_gauss(value: float, value_min: float, value_max: float, std: float = 0.5) -> float:
    if value_min == value_max:
        return value

    scale = max(value - value_min, value_max - value)
    new_value = np.random.normal(value, scale * std)
    return float(np.clip(new_value, value_min, value_max))


def mutate_chromosome(
    parent_chromo: np.ndarray,
    gen_types: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    mutable_positions: np.ndarray,
    n_mutation: int,
) -> np.ndarray:
    kid_chromo = np.array(parent_chromo, dtype=float)
    replace = n_mutation > len(mutable_positions)
    points = np.random.choice(mutable_positions, size=n_mutation, replace=replace)

    for point in points:
        gen_type = GenType(gen_types[point])
        gen = kid_chromo[point]

        if gen_type == GenType.bit:
            new_gen = (gen + 1) % 2
        elif gen_type == GenType.int:
            new_gen = int(mutation_gauss(gen, bounds_min[point], bounds_max[point]))
            while new_gen == gen and bounds_min[point] < bounds_max[point]:
                new_gen = int(np.random.randint(int(bounds_min[point]), int(bounds_max[point]) + 1))
        else:
            new_gen = mutation_gauss(gen, bounds_min[point], bounds_max[point])

        kid_chromo[point] = new_gen

    return kid_chromo