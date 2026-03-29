from __future__ import annotations

import numpy as np

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


def build_dummy_schema(
    *,
    defaults: list[float] | None = None,
    defaults_probability: list[float] | None = None,
    comments: list[str] | None = None,
    groups: list[int] | None = None,
) -> ChromosomeSchema:
    return ChromosomeSchema(
        gen_types=[GenType.bit, GenType.int, GenType.float],
        bounds=([0, 0, -1.0], [1, 5, 1.0]),
        comments=["bit", "int", "float"] if comments is None else comments,
        groups=[0, 0, 1] if groups is None else groups,
        defaults=[1.0, 2.0, 0.25] if defaults is None else defaults,
        defaults_probability=[0.0, 0.0, 0.0] if defaults_probability is None else defaults_probability,
    )


def build_fixed_schema() -> ChromosomeSchema:
    return ChromosomeSchema(
        gen_types=[GenType.bit, GenType.int, GenType.float],
        bounds=([1, 2, 0.5], [1, 2, 0.5]),
        comments=["fixed-bit", "fixed-int", "fixed-float"],
        groups=[0, 0, 1],
        defaults=[1.0, 2.0, 0.5],
        defaults_probability=[1.0, 1.0, 1.0],
    )


def build_population(tardigradas: Tardigradas, chromosomes: list[list[float]]) -> list[Individual]:
    return [tardigradas.create_individual(chromo=chromo) for chromo in chromosomes]


class DummyProblem(Problem):
    @staticmethod
    def init_environment(tardigradas: Tardigradas) -> None:
        return None

    @staticmethod
    def gen_info(tardigradas: Tardigradas) -> ChromosomeSchema:
        return build_dummy_schema()

    @staticmethod
    def fitness(individual: Individual) -> float:
        return float(individual[0] + individual[1] + individual[2])


class DefaultsProblem(DummyProblem):
    @staticmethod
    def gen_info(tardigradas: Tardigradas) -> ChromosomeSchema:
        return build_dummy_schema(defaults_probability=[1.0, 1.0, 1.0])


class FixedGenesProblem(DummyProblem):
    @staticmethod
    def gen_info(tardigradas: Tardigradas) -> ChromosomeSchema:
        return build_fixed_schema()


class ConstantFitnessProblem(DummyProblem):
    @staticmethod
    def fitness(individual: Individual) -> float:
        return 1.0


class NonNegativeFloatProblem(DummyProblem):
    @staticmethod
    def chromo_valid(individual: Individual) -> bool:
        return float(individual[2]) >= 0.0


class RejectAllProblem(DummyProblem):
    @staticmethod
    def chromo_valid(individual: Individual) -> bool:
        return False


class VectorFitnessProblem(DummyProblem):
    @staticmethod
    def fitness(individual: Individual) -> list[float]:
        return [float(individual[0] + individual[1]), float(individual[2])]


class EmptyFitnessProblem(DummyProblem):
    @staticmethod
    def fitness(individual: Individual) -> list[float]:
        return []


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


class TaggedIndividual(Individual):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tag = "custom"


class TaggedProblem(DummyProblem):
    individual_class = TaggedIndividual


def create_engine(
    *,
    problem: type[Problem] = DummyProblem,
    population_size: int = 6,
    crossover_fraction: float = 0.5,
    fresh_blood_fraction: float = 0.0,
    gen_mutation_fraction: float = 0.25,
    n_elits: int = 1,
) -> Tardigradas:
    return Tardigradas(
        problem=problem,
        population_size=population_size,
        crossover_fraction=crossover_fraction,
        fresh_blood_fraction=fresh_blood_fraction,
        gen_mutation_fraction=gen_mutation_fraction,
        n_elits=n_elits,
    )