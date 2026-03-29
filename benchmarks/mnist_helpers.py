from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ._paths import PROJECT_ROOT, ensure_project_paths


ensure_project_paths()


from tardigradas import ChromosomeSchema, CrossoverPolicy, GenType, Individual, Problem, Tardigradas


DEFAULT_MNIST_ROOT = PROJECT_ROOT / ".data" / "mnist"
DEFAULT_BATCH_SIZE = 4096
DEFAULT_WEIGHT_BOUND = 0.25


class TinyMnistConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(8, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        return self.classifier(outputs)


@dataclass(frozen=True)
class ParameterSlice:
    name: str
    shape: tuple[int, ...]
    start: int
    stop: int
    group: int
    bound: float


@dataclass(frozen=True)
class MnistEvaluation:
    loss: float
    accuracy: float
    n_examples: int


@dataclass
class MnistBenchmarkEnvironment:
    device: torch.device
    model: nn.Module
    train_loader: DataLoader
    test_loader: DataLoader
    parameter_slices: list[ParameterSlice]
    bounds_min: list[float]
    bounds_max: list[float]

    @property
    def chromo_size(self) -> int:
        if not self.parameter_slices:
            return 0
        return self.parameter_slices[-1].stop


def resolve_mnist_root(data_root: str | Path | None = None) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    root_from_env = os.getenv("TARDIGRADAS_MNIST_ROOT")
    if root_from_env:
        return Path(root_from_env).expanduser().resolve()
    return DEFAULT_MNIST_ROOT.resolve()


def build_mnist_cnn() -> TinyMnistConvNet:
    return TinyMnistConvNet()


def parameter_bound(parameter: torch.Tensor, fallback: float = DEFAULT_WEIGHT_BOUND) -> float:
    max_abs = float(parameter.detach().abs().max().item())
    return max(max_abs, fallback)


def build_parameter_slices(model: nn.Module) -> tuple[list[ParameterSlice], list[float], list[float]]:
    slices: list[ParameterSlice] = []
    bounds_min: list[float] = []
    bounds_max: list[float] = []
    offset = 0

    for group, (name, parameter) in enumerate(model.named_parameters(), start=1):
        size = int(parameter.numel())
        bound = parameter_bound(parameter)
        slices.append(
            ParameterSlice(
                name=name,
                shape=tuple(parameter.shape),
                start=offset,
                stop=offset + size,
                group=group,
                bound=bound,
            )
        )
        bounds_min.extend([-bound] * size)
        bounds_max.extend([bound] * size)
        offset += size

    return slices, bounds_min, bounds_max


def build_group_vector(parameter_slices: list[ParameterSlice]) -> list[int]:
    groups: list[int] = []
    for parameter_slice in parameter_slices:
        groups.extend([parameter_slice.group] * (parameter_slice.stop - parameter_slice.start))
    return groups


def benchmark_environment(tardigradas: Tardigradas) -> MnistBenchmarkEnvironment:
    environment = tardigradas.environment
    if not isinstance(environment, MnistBenchmarkEnvironment):
        raise TypeError("MNIST benchmark requires MnistBenchmarkEnvironment")
    return environment


def load_chromosome_into_model(
    model: nn.Module,
    chromo: np.ndarray,
    parameter_slices: list[ParameterSlice],
) -> None:
    expected_size = parameter_slices[-1].stop if parameter_slices else 0
    if len(chromo) != expected_size:
        raise ValueError(f"expected chromosome of size {expected_size}, got {len(chromo)}")

    parameters = dict(model.named_parameters())
    first_parameter = next(model.parameters())
    source = torch.as_tensor(chromo, device=first_parameter.device, dtype=first_parameter.dtype)

    with torch.no_grad():
        for parameter_slice in parameter_slices:
            parameter = parameters[parameter_slice.name]
            tensor = source[parameter_slice.start : parameter_slice.stop]
            tensor = tensor.to(dtype=parameter.dtype).view(parameter_slice.shape)
            parameter.copy_(tensor)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> MnistEvaluation:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="sum")

        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_examples += int(targets.numel())

    return MnistEvaluation(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        n_examples=total_examples,
    )


def create_mnist_environment(
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    data_root: str | Path | None = None,
    require_cuda: bool = True,
) -> MnistBenchmarkEnvironment:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("MNIST benchmark requires a CUDA-capable PyTorch device")

    root = resolve_mnist_root(data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(root=str(root), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=str(root), train=False, download=True, transform=transform)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = build_mnist_cnn().to(device)
    parameter_slices, bounds_min, bounds_max = build_parameter_slices(model)

    return MnistBenchmarkEnvironment(
        device=device,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        parameter_slices=parameter_slices,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


class MnistFullTrainConvProblem(Problem):
    batch_size = DEFAULT_BATCH_SIZE
    data_root: str | Path | None = None
    require_cuda = True

    @classmethod
    def init_environment(cls, tardigradas: Tardigradas) -> None:
        if tardigradas.environment is None:
            tardigradas.environment = create_mnist_environment(
                batch_size=cls.batch_size,
                data_root=cls.data_root,
                require_cuda=cls.require_cuda,
            )
            return

        environment = benchmark_environment(tardigradas)
        if cls.require_cuda and environment.device.type != "cuda":
            raise RuntimeError("MNIST benchmark expects a CUDA-backed environment")

    @classmethod
    def gen_info(cls, tardigradas: Tardigradas) -> ChromosomeSchema:
        environment = benchmark_environment(tardigradas)
        chromo_size = environment.chromo_size
        return ChromosomeSchema(
            gen_types=[GenType.float] * chromo_size,
            bounds=(list(environment.bounds_min), list(environment.bounds_max)),
            groups=build_group_vector(environment.parameter_slices),
        )

    @staticmethod
    def fitness(individual: Individual) -> list[float]:
        environment = benchmark_environment(individual.tardigradas)
        load_chromosome_into_model(environment.model, individual.chromo, environment.parameter_slices)
        metrics = evaluate_model(environment.model, environment.train_loader, environment.device)
        return [-metrics.loss, metrics.accuracy]

    @staticmethod
    def chromo_valid(individual: Individual) -> bool:
        return bool(np.isfinite(individual.chromo).all())


def evaluate_individual_on_test_split(individual: Individual) -> MnistEvaluation:
    environment = benchmark_environment(individual.tardigradas)
    load_chromosome_into_model(environment.model, individual.chromo, environment.parameter_slices)
    return evaluate_model(environment.model, environment.test_loader, environment.device)


def evaluate_best_individual_on_test_split(engine: Tardigradas) -> MnistEvaluation:
    if engine.best_individual is None:
        raise ValueError("best_individual is not available")
    return evaluate_individual_on_test_split(engine.best_individual)


def create_mnist_benchmark_engine(
    *,
    problem: type[Problem] = MnistFullTrainConvProblem,
    population_size: int = 12,
    crossover_fraction: float = 0.6,
    fresh_blood_fraction: float = 0.2,
    gen_mutation_fraction: float = 0.02,
    n_elits: int = 2,
    fitness_environment: MnistBenchmarkEnvironment | None = None,
    crossover_policy: CrossoverPolicy | None = None,
) -> Tardigradas:
    return Tardigradas(
        problem=problem,
        population_size=population_size,
        crossover_fraction=crossover_fraction,
        fresh_blood_fraction=fresh_blood_fraction,
        gen_mutation_fraction=gen_mutation_fraction,
        fitness_environment=fitness_environment,
        n_elits=n_elits,
        crossover_policy=crossover_policy,
    )