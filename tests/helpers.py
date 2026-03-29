from __future__ import annotations

from benchmarks.problems import (
    AckleyProblem,
    OneMaxProblem,
    RastriginProblem,
    RosenbrockProblem,
    RoyalRoadProblem,
    SphereProblem,
)
from tardigradas import ChromosomeSchema, CrossoverPolicy, GenType, Individual, Problem, Tardigradas


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


class ValidateScoreProblem(DummyProblem):
    @staticmethod
    def validate_score(individual: Individual) -> float:
        return float(individual[1] - individual[2])


class VectorValidateScoreProblem(VectorFitnessProblem):
    @staticmethod
    def validate_score(individual: Individual) -> float:
        return float(individual[1] - individual[2])


class EmptyFitnessProblem(DummyProblem):
    @staticmethod
    def fitness(individual: Individual) -> list[float]:
        return []


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
    crossover_policy: CrossoverPolicy | None = None,
) -> Tardigradas:
    return Tardigradas(
        problem=problem,
        population_size=population_size,
        crossover_fraction=crossover_fraction,
        fresh_blood_fraction=fresh_blood_fraction,
        gen_mutation_fraction=gen_mutation_fraction,
        n_elits=n_elits,
        crossover_policy=crossover_policy,
    )