from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from .gen_types import GenType

if TYPE_CHECKING:
    from .engine import Tardigradas
    from .evaluation import EvaluationContext


ChromoInput = Union[Sequence[float], np.ndarray]


class Individual:
    def __init__(
        self,
        tardigradas: Tardigradas,
        chromo: Optional[ChromoInput] = None,
        use_defaults: bool = False,
    ) -> None:
        self.tardigradas = tardigradas
        self.evaluation_context: EvaluationContext | None = None

        if chromo is None:
            self.chromo_new(use_defaults=use_defaults)
        else:
            self.chromo_new(chromo=chromo)

    def chromo_new_random(self, use_defaults: bool = False) -> None:
        chromo_size = self.tardigradas.problem.random_chromo_size(self.tardigradas)
        chromo_size = self.tardigradas._validate_chromo_length(chromo_size)
        schema_prefix = self.tardigradas._schema_prefix(chromo_size)

        ixb_bits = schema_prefix["gen_types"] == GenType.bit.value
        ixb_int = schema_prefix["gen_types"] == GenType.int.value
        ixb_float = schema_prefix["gen_types"] == GenType.float.value

        new_chromo = np.zeros(chromo_size, dtype=float)

        n_bits = int(ixb_bits.sum())
        if n_bits:
            bit_lows = schema_prefix["bounds_min"][ixb_bits]
            bit_highs = schema_prefix["bounds_max"][ixb_bits]
            random_bits = (np.random.random(n_bits) > 0.5).astype(float)
            fixed_bits = bit_lows == bit_highs
            random_bits[fixed_bits] = bit_lows[fixed_bits]
            random_bits[bit_lows > 0.0] = 1.0
            random_bits[bit_highs < 1.0] = 0.0
            new_chromo[ixb_bits] = random_bits

        n_float = int(ixb_float.sum())
        if n_float:
            new_chromo[ixb_float] = schema_prefix["bounds_min"][ixb_float] + np.random.random(n_float) * (
                schema_prefix["bounds_max"][ixb_float] - schema_prefix["bounds_min"][ixb_float]
            )

        if ixb_int.any():
            bounds_min = schema_prefix["bounds_min"][ixb_int].astype(int)
            bounds_max = schema_prefix["bounds_max"][ixb_int].astype(int)
            new_chromo[ixb_int] = np.random.randint(bounds_min, bounds_max + 1)

        if use_defaults:
            defaults = self.tardigradas.chromo_defaults[:chromo_size]
            defaults_probability = self.tardigradas.chromo_defaults_probability[:chromo_size]
            ixb_defaults = ~np.isnan(defaults)
            ixb_apply_defaults = np.random.random(len(ixb_defaults)) <= defaults_probability
            ixb_defaults &= ixb_apply_defaults
            new_chromo[ixb_defaults] = defaults[ixb_defaults]

        if not self.tardigradas.validate_chromosome(new_chromo):
            raise ValueError("generated chromosome values must match ChromosomeSchema bounds and gene types")

        self.chromo = new_chromo

    def chromo_new(
        self,
        chromo: Optional[ChromoInput] = None,
        use_defaults: bool = False,
    ) -> None:
        if chromo is not None:
            self.chromo = self.tardigradas._validate_chromosome(chromo)
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
        from .evaluation import normalize_fitness_score

        return normalize_fitness_score(raw_score)

    def chromo_valid(self) -> bool:
        return bool(self.tardigradas.validate_chromosome(self.chromo) and self.tardigradas.problem.chromo_valid(self))