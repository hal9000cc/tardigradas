from __future__ import annotations

from typing import Optional


class TardigradasException(Exception):
    def __init__(self, *args: object) -> None:
        self.message = args[0] if args else None

    def __str__(self) -> str:
        if self.message:
            return f"{self.__class__.__name__}: {self.message}"
        return self.__class__.__name__


class IncompleteEpochError(TardigradasException):
    def __init__(self, missing_indices: list[int]) -> None:
        self.missing_indices = [int(index) for index in missing_indices]
        super().__init__(f"incomplete population fitness evaluation: missing indices {self.missing_indices}")


class EvaluationFailure(TardigradasException):
    retryable = False

    def __init__(
        self,
        failure_kind: str,
        message: Optional[str] = None,
        details: Optional[dict[str, object]] = None,
    ) -> None:
        self.failure_kind = str(failure_kind)
        self.details = {} if details is None else dict(details)
        self.error_message = message
        super().__init__(message if message is not None else self.failure_kind)


class TransientEvaluationError(EvaluationFailure):
    retryable = True


class PermanentEvaluationError(EvaluationFailure):
    retryable = False


TradigradasException = TardigradasException