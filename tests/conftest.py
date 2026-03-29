from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from tests.helpers import DefaultsProblem, DummyProblem, create_engine


@pytest.fixture(autouse=True)
def fixed_seed() -> None:
    np.random.seed(12345)


@pytest.fixture
def engine():
    return create_engine(problem=DummyProblem)


@pytest.fixture
def defaults_engine():
    return create_engine(problem=DefaultsProblem)