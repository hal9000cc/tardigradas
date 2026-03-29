from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from .gen_types import GenType

if TYPE_CHECKING:
    from .engine import Tardigradas


ChromoInput = Union[Sequence[float], np.ndarray]


class Individual:
    def __init__(
        self,
        tardigradas: Tardigradas,
        chromo: Optional[ChromoInput] = None,
        use_defaults: bool = False,
    ) -> None:
        self.tardigradas = tardigradas

        if chromo is None:
            self.chromo_new(use_defaults=use_defaults)
        else:
            self.chromo_new(chromo=chromo)

    def chromo_new_random(self, use_defaults: bool = False) -> None:
        ixb_bits = self.tardigradas.gen_types == GenType.bit.value
        ixb_int = self.tardigradas.gen_types == GenType.int.value
        ixb_float = self.tardigradas.gen_types == GenType.float.value

        new_chromo = np.zeros(self.tardigradas.chromo_size, dtype=float)

        n_bits = int(ixb_bits.sum())
        if n_bits:
            new_chromo[ixb_bits] = np.random.random(n_bits) > 0.5

        n_float = int(ixb_float.sum())
        if n_float:
            new_chromo[ixb_float] = self.tardigradas.chromo_bounds_min[ixb_float] + np.random.random(n_float) * (
                self.tardigradas.chromo_bounds_max[ixb_float] - self.tardigradas.chromo_bounds_min[ixb_float]
            )

        if ixb_int.any():
            bounds_min = self.tardigradas.chromo_bounds_min[ixb_int].astype(int)
            bounds_max = self.tardigradas.chromo_bounds_max[ixb_int].astype(int)
            new_chromo[ixb_int] = np.random.randint(bounds_min, bounds_max + 1)

        if use_defaults:
            ixb_defaults = ~np.isnan(self.tardigradas.chromo_defaults)
            ixb_apply_defaults = np.random.random(len(ixb_defaults)) <= self.tardigradas.chromo_defaults_probability
            ixb_defaults &= ixb_apply_defaults
            new_chromo[ixb_defaults] = self.tardigradas.chromo_defaults[ixb_defaults]

        self.chromo = new_chromo

    def chromo_new(
        self,
        chromo: Optional[ChromoInput] = None,
        use_defaults: bool = False,
    ) -> None:
        if chromo is not None:
            self.chromo = np.array(chromo, dtype=float)
            return

        self.chromo_new_random(use_defaults=use_defaults)

    def __getitem__(self, item: int) -> Union[int, float]:
        if not isinstance(item, int):
            raise TypeError("gene index must be int")

        gen_value = float(self.chromo[item])
        gen_type = GenType(self.tardigradas.gen_types[item])
        if gen_type in (GenType.bit, GenType.int):
            return int(gen_value)
        return float(gen_value)

    def fitness(self) -> np.ndarray:
        raw_score = self.tardigradas.problem.fitness(self)
        if np.isscalar(raw_score):
            return np.array([float(raw_score)], dtype=float)

        scores = np.array(raw_score, dtype=float).reshape(-1)
        if scores.size == 0:
            raise ValueError("fitness must return at least one numeric value")
        return scores

    def chromo_valid(self) -> bool:
        return bool(self.tardigradas.problem.chromo_valid(self))