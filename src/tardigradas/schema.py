from __future__ import annotations

from dataclasses import dataclass, field
from math import nan

from .gen_types import GenType


@dataclass
class ChromosomeSchema:
    gen_types: list[GenType]
    bounds: tuple[list[float], list[float]]
    comments: list[str] = field(default_factory=list)
    groups: list[int] = field(default_factory=list)
    defaults: list[float] = field(default_factory=list)
    defaults_probability: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.gen_types = list(self.gen_types)
        bounds_min = list(self.bounds[0])
        bounds_max = list(self.bounds[1])

        chromo_size = len(self.gen_types)
        if chromo_size == 0:
            raise ValueError("chromosome schema must contain at least one gene")
        if len(bounds_min) != chromo_size or len(bounds_max) != chromo_size:
            raise ValueError("bounds must match chromosome size")
        if any(low > high for low, high in zip(bounds_min, bounds_max)):
            raise ValueError("each lower bound must be <= upper bound")

        if not self.comments:
            self.comments = [""] * chromo_size
        if not self.groups:
            self.groups = [0] * chromo_size
        if not self.defaults:
            self.defaults = [nan] * chromo_size
        if not self.defaults_probability:
            self.defaults_probability = [0.0] * chromo_size

        if len(self.comments) != chromo_size:
            raise ValueError("comments must match chromosome size")
        if len(self.groups) != chromo_size:
            raise ValueError("groups must match chromosome size")
        if len(self.defaults) != chromo_size:
            raise ValueError("defaults must match chromosome size")
        if len(self.defaults_probability) != chromo_size:
            raise ValueError("defaults_probability must match chromosome size")
        if any(probability < 0 or probability > 1 for probability in self.defaults_probability):
            raise ValueError("defaults_probability must contain values between 0 and 1")

        self.bounds = (bounds_min, bounds_max)
        self.comments = list(self.comments)
        self.groups = [int(group) for group in self.groups]
        self.defaults = [float(value) for value in self.defaults]
        self.defaults_probability = [float(value) for value in self.defaults_probability]

    @property
    def chromo_size(self) -> int:
        return len(self.gen_types)