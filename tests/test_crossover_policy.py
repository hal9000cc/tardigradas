from __future__ import annotations

import pytest

from tardigradas import CrossoverBitType, CrossoverFloatType, CrossoverPolicy


def test_explicit_crossover_policy_preserves_selected_operators() -> None:
    policy = CrossoverPolicy.explicit(
        bit=CrossoverBitType.two_point,
        float=CrossoverFloatType.BLX,
    )

    assert policy.is_explicit is True
    assert policy.bit == CrossoverBitType.two_point
    assert policy.float == CrossoverFloatType.BLX


def test_adaptive_crossover_policy_defaults_to_all_candidates() -> None:
    policy = CrossoverPolicy.adaptive()

    assert policy.is_adaptive is True
    assert policy.bit_candidates == tuple(CrossoverBitType)
    assert policy.float_candidates == tuple(CrossoverFloatType)


def test_adaptive_crossover_policy_validates_reward() -> None:
    with pytest.raises(ValueError, match="unsupported adaptive crossover reward"):
        CrossoverPolicy.adaptive(reward="not-supported")