from __future__ import annotations


class TardigradasException(Exception):
    def __init__(self, *args: object) -> None:
        self.message = args[0] if args else None

    def __str__(self) -> str:
        if self.message:
            return f"{self.__class__.__name__}: {self.message}"
        return self.__class__.__name__


TradigradasException = TardigradasException