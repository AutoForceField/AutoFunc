# +
from __future__ import annotations

import abc
from typing import Sequence

__all__ = ["Tensor", "Generators", "Segments", "Elemental"]


class Tensor(abc.ABC):
    @abc.abstractmethod
    def __add__(self, other: "Tensor" | float | int) -> "Tensor":
        ...

    @abc.abstractmethod
    def __sub__(self, other: "Tensor" | float | int) -> "Tensor":
        ...

    @abc.abstractmethod
    def __mul__(self, other: "Tensor" | float | int) -> "Tensor":
        ...

    @abc.abstractmethod
    def __div__(self, other: "Tensor" | float | int) -> "Tensor":
        ...

    @abc.abstractmethod
    def __pow__(self, other: "Tensor" | float | int) -> "Tensor":
        ...

    @abc.abstractmethod
    def __radd__(self, other: "Tensor" | float | int) -> "Tensor":
        ...

    @abc.abstractmethod
    def __rsub__(self, other: "Tensor" | float | int) -> "Tensor":
        ...

    @abc.abstractmethod
    def __rmul__(self, other: "Tensor" | float | int) -> "Tensor":
        ...

    @abc.abstractmethod
    def __rdiv__(self, other: "Tensor" | float | int) -> "Tensor":
        ...

    @abc.abstractmethod
    def __rpow__(self, other: "Tensor" | float | int) -> "Tensor":
        ...


class Generators(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def constant(object) -> Tensor:
        ...

    @staticmethod
    @abc.abstractmethod
    def zeros(shape: tuple[int]) -> Tensor:
        ...

    @staticmethod
    @abc.abstractmethod
    def ones(shape: tuple[int]) -> Tensor:
        ...

    @staticmethod
    @abc.abstractmethod
    def empty(shape: tuple[int]) -> Tensor:
        ...

    @staticmethod
    @abc.abstractmethod
    def full(shape: tuple[int], value: Tensor | float) -> Tensor:
        ...


class Segments(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def cat(t: Sequence[Tensor], dim: int = 0) -> Tensor:
        ...


class Elemental(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def sin(t: Tensor) -> Tensor:
        ...
