from __future__ import annotations

from typing import Optional

import numpy as np


def _build_crossover_units(gene_groups: np.ndarray, gene_mask: np.ndarray) -> list[np.ndarray]:
    active_positions = np.flatnonzero(gene_mask)
    units: list[np.ndarray] = []
    seen_groups: set[int] = set()

    for position in active_positions:
        group = int(gene_groups[position])
        if group <= 0:
            units.append(np.array([position], dtype=int))
            continue
        if group in seen_groups:
            continue
        seen_groups.add(group)
        units.append(active_positions[gene_groups[active_positions] == group])

    return units


def _build_unit_mask(units: list[np.ndarray], start: int, stop: int, size: int) -> np.ndarray:
    unit_mask = np.zeros(size, dtype=bool)
    for unit in units[start:stop]:
        unit_mask[unit] = True
    return unit_mask


def crossover_uniform(
    kid: np.ndarray,
    parent1: np.ndarray,
    parent2: np.ndarray,
    gene_groups: np.ndarray,
    gene_mask: np.ndarray,
) -> np.ndarray:
    kid[gene_mask] = parent1[gene_mask]

    ix_no_group = gene_groups == 0
    ixb_move_no_group = np.random.random(int(ix_no_group.sum())) > 0.5

    groups = list(set(gene_groups[gene_groups > 0]))
    ixb_move_group = np.random.random(len(groups)) > 0.5

    ixb_move2 = np.zeros(len(gene_groups), dtype=bool)
    ixb_move2[ix_no_group] = ixb_move_no_group

    for i_group, group in enumerate(groups):
        ixb_move2[gene_groups == group] = ixb_move_group[i_group]

    kid[gene_mask & ixb_move2] = parent2[gene_mask & ixb_move2]
    return kid


def crossover_one_point(
    kid: np.ndarray,
    parent1: np.ndarray,
    parent2: np.ndarray,
    gene_groups: np.ndarray,
    gene_mask: np.ndarray,
) -> np.ndarray:
    units = _build_crossover_units(gene_groups, gene_mask)
    primary_parent, secondary_parent = (parent1, parent2) if np.random.random() < 0.5 else (parent2, parent1)
    kid[gene_mask] = primary_parent[gene_mask]

    if len(units) <= 1:
        return kid

    crossover_point = int(np.random.randint(1, len(units)))
    suffix_mask = _build_unit_mask(units, crossover_point, len(units), len(gene_mask))
    kid[suffix_mask] = secondary_parent[suffix_mask]
    return kid


def crossover_two_point(
    kid: np.ndarray,
    parent1: np.ndarray,
    parent2: np.ndarray,
    gene_groups: np.ndarray,
    gene_mask: np.ndarray,
) -> np.ndarray:
    units = _build_crossover_units(gene_groups, gene_mask)
    if len(units) < 3:
        return crossover_one_point(kid, parent1, parent2, gene_groups, gene_mask)

    primary_parent, secondary_parent = (parent1, parent2) if np.random.random() < 0.5 else (parent2, parent1)
    kid[gene_mask] = primary_parent[gene_mask]

    first_point = int(np.random.randint(1, len(units) - 1))
    second_point = int(np.random.randint(first_point + 1, len(units)))
    middle_mask = _build_unit_mask(units, first_point, second_point, len(gene_mask))
    kid[middle_mask] = secondary_parent[middle_mask]
    return kid


def crossover_arithmetic(
    kid: np.ndarray,
    parent1: np.ndarray,
    parent2: np.ndarray,
    gene_mask: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    alpha: Optional[float] = None,
) -> np.ndarray:
    blend_alpha = float(np.random.random()) if alpha is None else float(alpha)
    values = blend_alpha * parent1[gene_mask] + (1.0 - blend_alpha) * parent2[gene_mask]
    kid[gene_mask] = np.clip(values, bounds_min[gene_mask], bounds_max[gene_mask])
    return kid


def crossover_blx(
    kid: np.ndarray,
    parent1: np.ndarray,
    parent2: np.ndarray,
    gene_mask: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    parents = np.vstack((parent1[gene_mask], parent2[gene_mask]))
    interval = np.abs(parents[0] - parents[1]) * alpha
    lows = np.clip(parents.min(0) - interval, bounds_min[gene_mask], bounds_max[gene_mask])
    highs = np.clip(parents.max(0) + interval, bounds_min[gene_mask], bounds_max[gene_mask])
    kid[gene_mask] = lows + np.random.random(len(lows)) * (highs - lows)
    return kid