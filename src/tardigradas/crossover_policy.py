from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, TypeVar

from .gen_types import CrossoverBitType, CrossoverFloatType


_VALID_POLICY_MODES = {"explicit", "adaptive"}
_VALID_ADAPTIVE_REWARDS = {"elite_survival"}

TEnum = TypeVar("TEnum")


def _normalize_candidates(
    candidates: Optional[Iterable[TEnum]],
    *,
    enum_type: type[TEnum],
) -> tuple[TEnum, ...]:
    values = tuple(member for member in enum_type) if candidates is None else tuple(candidates)
    if not values:
        raise ValueError("adaptive crossover candidates must not be empty")

    unique_values: list[TEnum] = []
    for value in values:
        if not isinstance(value, enum_type):
            raise TypeError(f"adaptive crossover candidates must contain {enum_type.__name__} values")
        if value not in unique_values:
            unique_values.append(value)

    return tuple(unique_values)


def _validate_min_probability(min_probability: float, *, n_candidates: int) -> float:
    value = float(min_probability)
    if value < 0.0 or value > 1.0:
        raise ValueError("min_probability must be in range [0, 1]")
    if n_candidates * value > 1.0:
        raise ValueError("min_probability is too large for the number of adaptive candidates")
    return value


def _validate_period(period: int) -> int:
    value = int(period)
    if value <= 0:
        raise ValueError("period must be positive")
    return value


@dataclass(frozen=True)
class CrossoverPolicy:
    mode: str
    bit: Optional[CrossoverBitType] = None
    float: Optional[CrossoverFloatType] = None
    bit_candidates: tuple[CrossoverBitType, ...] = ()
    float_candidates: tuple[CrossoverFloatType, ...] = ()
    reward: str = "elite_survival"
    min_probability: float = 0.0
    period: int = 20

    def __post_init__(self) -> None:
        if self.mode not in _VALID_POLICY_MODES:
            raise ValueError(f"unsupported crossover policy mode: {self.mode}")

        if self.mode == "explicit":
            if not isinstance(self.bit, CrossoverBitType):
                raise TypeError("explicit crossover policy bit operator must be CrossoverBitType")
            if not isinstance(self.float, CrossoverFloatType):
                raise TypeError("explicit crossover policy float operator must be CrossoverFloatType")
            if self.bit_candidates or self.float_candidates:
                raise ValueError("explicit crossover policy must not define adaptive candidates")
            return

        if self.reward not in _VALID_ADAPTIVE_REWARDS:
            raise ValueError(f"unsupported adaptive crossover reward: {self.reward}")
        if not self.bit_candidates or not self.float_candidates:
            raise ValueError("adaptive crossover policy must define bit and float candidates")

        _validate_min_probability(self.min_probability, n_candidates=len(self.bit_candidates))
        _validate_min_probability(self.min_probability, n_candidates=len(self.float_candidates))
        _validate_period(self.period)

    @property
    def is_explicit(self) -> bool:
        return self.mode == "explicit"

    @property
    def is_adaptive(self) -> bool:
        return self.mode == "adaptive"

    @property
    def alpha(self) -> float:
        return 2.0 / (self.period + 1)

    @classmethod
    def default(cls) -> CrossoverPolicy:
        return cls.explicit(
            bit=CrossoverBitType.uniform,
            float=CrossoverFloatType.uniform,
        )

    @classmethod
    def explicit(
        cls,
        *,
        bit: CrossoverBitType = CrossoverBitType.uniform,
        float: CrossoverFloatType = CrossoverFloatType.uniform,
    ) -> CrossoverPolicy:
        return cls(mode="explicit", bit=bit, float=float)

    @classmethod
    def adaptive(
        cls,
        *,
        bit_candidates: Optional[Iterable[CrossoverBitType]] = None,
        float_candidates: Optional[Iterable[CrossoverFloatType]] = None,
        reward: str = "elite_survival",
        min_probability: float = 0.0,
        period: int = 20,
    ) -> CrossoverPolicy:
        normalized_bit_candidates = _normalize_candidates(bit_candidates, enum_type=CrossoverBitType)
        normalized_float_candidates = _normalize_candidates(float_candidates, enum_type=CrossoverFloatType)
        normalized_min_probability = _validate_min_probability(min_probability, n_candidates=len(normalized_bit_candidates))
        _validate_min_probability(normalized_min_probability, n_candidates=len(normalized_float_candidates))
        normalized_period = _validate_period(period)

        return cls(
            mode="adaptive",
            bit_candidates=normalized_bit_candidates,
            float_candidates=normalized_float_candidates,
            reward=reward,
            min_probability=normalized_min_probability,
            period=normalized_period,
        )