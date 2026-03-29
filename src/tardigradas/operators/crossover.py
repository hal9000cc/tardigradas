from __future__ import annotations

import numpy as np


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