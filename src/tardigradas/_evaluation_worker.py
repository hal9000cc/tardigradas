from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path
from typing import Any

from .engine import Tardigradas
from .evaluation import EvaluationContext, normalize_fitness_score
from .exceptions import EvaluationFailure


def _resolve_qualified_name(module_name: str, qualified_name: str) -> Any:
    value: Any = importlib.import_module(module_name)
    for part in qualified_name.split("."):
        value = getattr(value, part)
    return value


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        return 2

    request_path = Path(args[0])
    with request_path.open("rb") as file:
        request = pickle.load(file)

    response_path = Path(request["response_path"])
    try:
        problem = _resolve_qualified_name(
            request["problem_module"],
            request["problem_qualified_name"],
        )
        engine = Tardigradas(
            problem=problem,
            population_size=int(request["population_size"]),
            crossover_fraction=float(request["crossover_fraction"]),
            fresh_blood_fraction=float(request["fresh_blood_fraction"]),
            gen_mutation_fraction=float(request["gen_mutation_fraction"]),
            n_elits=int(request["n_elits"]),
        )
        individual = engine.create_individual(chromo=request["chromo"])
        individual.evaluation_context = EvaluationContext(
            generation=int(request["generation"]),
            individual_index=int(request["individual_index"]),
            attempt=int(request["attempt"]),
        )
        score = normalize_fitness_score(individual.fitness()).tolist()
        response = {"ok": True, "score": score}
        with response_path.open("wb") as file:
            pickle.dump(response, file)
        return 0
    except EvaluationFailure as exc:
        response = {
            "ok": False,
            "failure_mode": "transient" if exc.retryable else "permanent",
            "retryable": bool(exc.retryable),
            "failure_kind": exc.failure_kind,
            "error_type": type(exc).__name__,
            "error_message": exc.error_message if exc.error_message is not None else str(exc),
            "error_repr": repr(exc),
            "details": exc.details,
        }
        with response_path.open("wb") as file:
            pickle.dump(response, file)
        return 1
    except Exception as exc:
        response = {
            "ok": False,
            "failure_mode": "exception",
            "retryable": False,
            "failure_kind": "exception",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_repr": repr(exc),
        }
        with response_path.open("wb") as file:
            pickle.dump(response, file)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
