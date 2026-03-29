from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from .individual import Individual
from .schema import ChromosomeSchema

if TYPE_CHECKING:
    from .engine import Tardigradas


ChromoLike = Union[Sequence[float], np.ndarray, Individual]


class Problem(ABC):
    individual_class: type[Individual] = Individual

    @staticmethod
    @abstractmethod
    def init_environment(tardigradas: Tardigradas) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def gen_info(tardigradas: Tardigradas) -> ChromosomeSchema:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def fitness(individual: Individual) -> Union[Sequence[float], float]:
        raise NotImplementedError

    @staticmethod
    def chromo_valid(individual: Individual) -> bool:
        return True

    @staticmethod
    def _to_chromo_array(value: ChromoLike) -> np.ndarray:
        if isinstance(value, Individual):
            return np.asarray(value.chromo, dtype=float)
        return np.asarray(value, dtype=float)

    @staticmethod
    def is_equal(chromo1: ChromoLike, chromo2: ChromoLike) -> bool:
        return bool(np.array_equal(Problem._to_chromo_array(chromo1), Problem._to_chromo_array(chromo2)))

    @classmethod
    def create_individual(
        cls,
        tardigradas: Tardigradas,
        chromo: Optional[Union[Sequence[float], np.ndarray]] = None,
        use_defaults: bool = False,
    ) -> Individual:
        return cls.individual_class(tardigradas, chromo=chromo, use_defaults=use_defaults)