from __future__ import annotations

import math
from typing import Any

import pytest

from tardigradas import ChromosomeSchema, GenType


def test_schema_fills_optional_fields_with_defaults() -> None:
    schema = ChromosomeSchema(
        gen_types=[GenType.bit, GenType.float],
        bounds=([0, -1.0], [1, 1.0]),
    )

    assert schema.chromo_size == 2
    assert schema.comments == ["", ""]
    assert schema.groups == [0, 0]
    assert all(math.isnan(value) for value in schema.defaults)
    assert schema.defaults_probability == [0.0, 0.0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"comments": ["a"]}, "comments must match chromosome size"),
        ({"groups": [1]}, "groups must match chromosome size"),
        ({"defaults": [0.0]}, "defaults must match chromosome size"),
        ({"defaults_probability": [0.5]}, "defaults_probability must match chromosome size"),
    ],
)
def test_schema_rejects_size_mismatches(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ChromosomeSchema(
            gen_types=[GenType.bit, GenType.int],
            bounds=([0, 0], [1, 5]),
            **kwargs,
        )


def test_schema_rejects_bounds_size_mismatch() -> None:
    with pytest.raises(ValueError, match="bounds must match chromosome size"):
        ChromosomeSchema(
            gen_types=[GenType.bit, GenType.int],
            bounds=([0], [1, 5]),
        )


def test_schema_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="each lower bound must be <= upper bound"):
        ChromosomeSchema(
            gen_types=[GenType.float],
            bounds=([2.0], [1.0]),
        )


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_schema_rejects_invalid_default_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="defaults_probability must contain values between 0 and 1"):
        ChromosomeSchema(
            gen_types=[GenType.int],
            bounds=([0], [3]),
            defaults_probability=[probability],
        )