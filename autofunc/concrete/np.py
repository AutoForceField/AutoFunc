# +
from functools import partial

import numpy as np

import autofunc.abstract as ab
from autofunc.concrete.concrete import concrete

__all__ = ["np32", "np64"]


class Generators_np32(ab.Generators):
    @staticmethod
    def constant(object) -> np.ndarray:
        return np.asarray(object, dtype=np.float32)

    @staticmethod
    def zeros(shape: tuple[int]) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def ones(shape: tuple[int]) -> np.ndarray:
        return np.ones(shape, dtype=np.float32)

    @staticmethod
    def empty(shape: tuple[int]) -> np.ndarray:
        return np.empty(shape, dtype=np.float32)

    @staticmethod
    def full(shape: tuple[int], value: np.ndarray | float) -> np.ndarray:
        return np.full(shape, value, dtype=np.float32)


class Generators_np64(ab.Generators):
    @staticmethod
    def constant(object) -> np.ndarray:
        return np.asarray(object, dtype=np.float64)

    @staticmethod
    def zeros(shape: tuple[int]) -> np.ndarray:
        return np.zeros(shape, dtype=np.float64)

    @staticmethod
    def ones(shape: tuple[int]) -> np.ndarray:
        return np.ones(shape, dtype=np.float64)

    @staticmethod
    def empty(shape: tuple[int]) -> np.ndarray:
        return np.empty(shape, dtype=np.float64)

    @staticmethod
    def full(shape: tuple[int], value: np.ndarray | float) -> np.ndarray:
        return np.full(shape, value, dtype=np.float64)


class Elemental_np(ab.Elemental):
    @staticmethod
    def sin(x: np.ndarray) -> np.ndarray:
        return np.sin(x)


np32 = partial(
    concrete, cfg={ab.Generators: Generators_np32, ab.Elemental: Elemental_np}
)
np64 = partial(
    concrete, cfg={ab.Generators: Generators_np64, ab.Elemental: Elemental_np}
)
