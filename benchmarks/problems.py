from __future__ import annotations

import numpy as np

from ._paths import ensure_project_paths


ensure_project_paths()


from tardigradas import ChromosomeSchema, GenType, Individual, Problem, Tardigradas


def build_bit_benchmark_schema(
    chromo_size: int,
    *,
    groups: list[int] | None = None,
) -> ChromosomeSchema:
    return ChromosomeSchema(
        gen_types=[GenType.bit] * chromo_size,
        bounds=([0] * chromo_size, [1] * chromo_size),
        comments=[f"bit-{index}" for index in range(chromo_size)],
        groups=[0] * chromo_size if groups is None else groups,
    )


def build_float_benchmark_schema(
    chromo_size: int,
    *,
    lower_bound: float,
    upper_bound: float,
) -> ChromosomeSchema:
    return ChromosomeSchema(
        gen_types=[GenType.float] * chromo_size,
        bounds=([lower_bound] * chromo_size, [upper_bound] * chromo_size),
        comments=[f"x{index}" for index in range(chromo_size)],
    )


class BitBenchmarkProblem(Problem):
    n_bits = 24

    @staticmethod
    def init_environment(tardigradas: Tardigradas) -> None:
        return None

    @classmethod
    def gen_info(cls, tardigradas: Tardigradas) -> ChromosomeSchema:
        return build_bit_benchmark_schema(cls.n_bits)


class FloatBenchmarkProblem(Problem):
    chromo_size = 4
    lower_bound = -5.0
    upper_bound = 5.0

    @staticmethod
    def init_environment(tardigradas: Tardigradas) -> None:
        return None

    @classmethod
    def gen_info(cls, tardigradas: Tardigradas) -> ChromosomeSchema:
        return build_float_benchmark_schema(
            cls.chromo_size,
            lower_bound=cls.lower_bound,
            upper_bound=cls.upper_bound,
        )


class OneMaxProblem(BitBenchmarkProblem):
    @staticmethod
    def fitness(individual: Individual) -> float:
        return float(np.sum(individual.chromo))


class SphereProblem(FloatBenchmarkProblem):
    chromo_size = 4
    lower_bound = -5.12
    upper_bound = 5.12

    @staticmethod
    def fitness(individual: Individual) -> float:
        chromo = individual.chromo
        return float(-np.sum(chromo * chromo))


class RastriginProblem(FloatBenchmarkProblem):
    chromo_size = 4
    lower_bound = -5.12
    upper_bound = 5.12

    @staticmethod
    def fitness(individual: Individual) -> float:
        chromo = individual.chromo
        chromo_size = len(chromo)
        objective = 10.0 * chromo_size + np.sum(chromo * chromo - 10.0 * np.cos(2.0 * np.pi * chromo))
        return float(-objective)


class RoyalRoadProblem(BitBenchmarkProblem):
    block_size = 4
    n_blocks = 6
    n_bits = block_size * n_blocks

    @classmethod
    def gen_info(cls, tardigradas: Tardigradas) -> ChromosomeSchema:
        groups: list[int] = []
        for group in range(1, cls.n_blocks + 1):
            groups.extend([group] * cls.block_size)
        return build_bit_benchmark_schema(cls.n_bits, groups=groups)

    @classmethod
    def fitness(cls, individual: Individual) -> float:
        total = 0.0
        chromo = individual.chromo.astype(int)
        for start in range(0, len(chromo), cls.block_size):
            block = chromo[start : start + cls.block_size]
            if np.all(block == 1):
                total += cls.block_size
        return total


class RosenbrockProblem(FloatBenchmarkProblem):
    chromo_size = 3
    lower_bound = -2.048
    upper_bound = 2.048

    @staticmethod
    def fitness(individual: Individual) -> float:
        chromo = individual.chromo
        objective = np.sum(100.0 * (chromo[1:] - chromo[:-1] ** 2) ** 2 + (1.0 - chromo[:-1]) ** 2)
        return float(-objective)


class AckleyProblem(FloatBenchmarkProblem):
    chromo_size = 4
    lower_bound = -5.0
    upper_bound = 5.0

    @staticmethod
    def fitness(individual: Individual) -> float:
        chromo = individual.chromo
        chromo_size = len(chromo)
        square_mean = float(np.sum(chromo * chromo) / chromo_size)
        cosine_mean = float(np.sum(np.cos(2.0 * np.pi * chromo)) / chromo_size)
        objective = -20.0 * np.exp(-0.2 * np.sqrt(square_mean)) - np.exp(cosine_mean) + 20.0 + np.e
        return float(-objective)


__all__ = [
    "AckleyProblem",
    "BitBenchmarkProblem",
    "FloatBenchmarkProblem",
    "OneMaxProblem",
    "RastriginProblem",
    "RosenbrockProblem",
    "RoyalRoadProblem",
    "SphereProblem",
    "build_bit_benchmark_schema",
    "build_float_benchmark_schema",
]