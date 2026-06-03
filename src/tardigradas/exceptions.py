from __future__ import annotations


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


TradigradasException = TardigradasException