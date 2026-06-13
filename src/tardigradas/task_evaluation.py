from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskEvaluationContext:
    generation: int
    individual_index: int
    population_size: int
    evaluation_number: int = 1
    attempt: int = 1
    task_id: str | None = None
    task_number: int | None = None


@dataclass(frozen=True)
class EvaluationTaskSpec:
    task_id: str
    individual_index: int
    generation: int
    task_number: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationTaskResult:
    task_id: str
    individual_index: int
    ok: bool
    score: list[float] | None = None
    payload: dict[str, Any] | None = None
    retryable: bool = False
    failure_kind: str | None = None
    error_message: str | None = None


@dataclass
class IndividualTaskState:
    individual_index: int
    payload: dict[str, Any] = field(default_factory=dict)
    scheduled_count: int = 0
    completed_count: int = 0
    stopped: bool = False
    results: list[EvaluationTaskResult] = field(default_factory=list)


@dataclass
class TaskSchedulingDecision:
    new_tasks: list[EvaluationTaskSpec] = field(default_factory=list)
    stop_individual: bool = False
    ready_to_aggregate: bool = False
    individual_failed: bool = False
    failure_message: str | None = None