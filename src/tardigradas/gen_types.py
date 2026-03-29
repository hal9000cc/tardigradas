from enum import Enum


class GenType(Enum):
    bit = 1
    int = 2
    float = 3


class CrossoverBitType(Enum):
    uniform = 0
    one_point = 1
    two_point = 2


class CrossoverFloatType(Enum):
    uniform = 0
    arithmetic = 1
    BLX = 2