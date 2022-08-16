# +
from __future__ import annotations

import abc
from typing import ClassVar, Iterator, Sequence

__all__ = ["Tens", "Alloc", "Seg", "Elem", "Reduc", "Abs"]


class Tens(abc.ABC):
    """
    Tens is short for "Tensor".

    Abstract representation of tensors which are
    implemented in popular packages such as:
    numpy (ndarray), pytorch (Tensor), etc.

    Although, the Tens class defined here reflects
    only a small subset of methods which are identical
    across popular platforms.

    This is mainly defined for static type checking.

    """

    @abc.abstractmethod
    def __add__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __sub__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __mul__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __div__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __truediv__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __pow__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __matmul__(self, other: "Tens") -> "Tens":
        ...

    @abc.abstractmethod
    def __radd__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __rsub__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __rmul__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __rdiv__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __rtruediv__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __rpow__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __rmatmul__(self, other: "Tens") -> "Tens":
        ...

    @abc.abstractmethod
    def __lt__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __gt__(self, other: "Tens" | float | int) -> "Tens":
        ...

    @abc.abstractmethod
    def __neg__(self) -> "Tens":
        ...

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        ...

    @abc.abstractmethod
    def reshape(self, shape: int | tuple[int, ...]) -> "Tens":
        ...

    @abc.abstractmethod
    def __getitem__(self, *args) -> "Tens":
        ...

    @abc.abstractmethod
    def __setitem__(self, *args) -> None:
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator["Tens"]:
        ...


class Alloc(abc.ABC):
    """
    Alloc is short for "Allocations".

    Concrete implementations of this class will contain
    tensor creation methods and automatically handle
    their dtype.

    Since dtype info is only accessible within this class,
    dtype dependent constants (finfo, etc.) will be available
    as its properties.

    """

    eps: ClassVar[Tens]

    @staticmethod
    @abc.abstractmethod
    def tens(object) -> Tens:
        ...

    @staticmethod
    @abc.abstractmethod
    def zeros(shape: int | tuple[int, ...]) -> Tens:
        ...

    @staticmethod
    @abc.abstractmethod
    def ones(shape: int | tuple[int, ...]) -> Tens:
        ...

    @staticmethod
    @abc.abstractmethod
    def empty(shape: int | tuple[int, ...]) -> Tens:
        ...

    @staticmethod
    @abc.abstractmethod
    def full(shape: int | tuple[int, ...], value: Tens | float) -> Tens:
        ...

    @staticmethod
    @abc.abstractmethod
    def rand(shape: int | tuple[int, ...]) -> Tens:
        ...


class Seg(abc.ABC):
    """
    Seg is short for "Segments".

    Concrete implementations of this class will contain
    methods for concatenation, spliting, etc of tensors.

    """

    @staticmethod
    @abc.abstractmethod
    def cat(x: Sequence[Tens], dim: int = 0) -> Tens:
        ...

    @staticmethod
    @abc.abstractmethod
    def stack(x: Sequence[Tens], dim: int = 0) -> Tens:
        ...

    @staticmethod
    @abc.abstractmethod
    def t(x: Tens) -> Tens:
        # TODO: ambigous
        ...


class Elem(abc.ABC):
    """
    Elem is short for "Elemental".

    Concrete implementations of this class will contain
    functions that are element-wise. Thus they can act
    on tensors with any shape and the output always
    has the same shape as the input tensor.

    """

    @staticmethod
    @abc.abstractmethod
    def sqrt(x: Tens) -> Tens:
        ...

    @staticmethod
    @abc.abstractmethod
    def sin(x: Tens) -> Tens:
        ...

    @staticmethod
    @abc.abstractmethod
    def cos(x: Tens) -> Tens:
        ...

    @staticmethod
    @abc.abstractmethod
    def atan2(x: Tens, y: Tens) -> Tens:
        ...

    @staticmethod
    @abc.abstractmethod
    def where(condition: Tens, x: Tens, y: Tens) -> Tens:
        ...

    @staticmethod
    @abc.abstractmethod
    def allclose(x: Tens, y: Tens) -> Tens:
        # TODO: incomplete
        ...


class Reduc(abc.ABC):
    """
    Reduc is short for "Reductions".

    Concrete implementations of this class will contain
    functions that perform reduction operations e.g. sum,
    norm, etc.

    """

    @staticmethod
    @abc.abstractmethod
    def norm(x: Tens, dim: int | None = None, keepdim: bool = False) -> Tens:
        # TODO: incomplete
        ...


class Abs(Reduc, Elem, Seg, Alloc):
    """
    Abs is short for "Abstract".

    This class contains all of the other abstract classes
    and by subclassing this class all of the abstract methods
    become available as attributes of "self".

    """

    pass
