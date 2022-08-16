# +
from functools import partial
from typing import Sequence

import numpy as np

import autofunc.abstract as ab
from autofunc.concrete.concrete import concrete

__all__ = ["np32", "np64"]


class Alloc_np32(ab.Alloc):

    eps = np.finfo(np.float32).eps

    @staticmethod
    def tens(object) -> np.ndarray:
        return np.asarray(object, dtype=np.float32)

    @staticmethod
    def zeros(shape: int | tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def ones(shape: int | tuple[int, ...]) -> np.ndarray:
        return np.ones(shape, dtype=np.float32)

    @staticmethod
    def empty(shape: int | tuple[int, ...]) -> np.ndarray:
        return np.empty(shape, dtype=np.float32)

    @staticmethod
    def full(shape: int | tuple[int, ...], value: np.ndarray | float) -> np.ndarray:
        return np.full(shape, value, dtype=np.float32)

    @staticmethod
    def rand(shape: int | tuple[int, ...]) -> np.ndarray:
        return np.asarray(np.random.uniform(size=shape), dtype=np.float64)


class Alloc_np64(ab.Alloc):

    eps = np.finfo(np.float64).eps

    @staticmethod
    def tens(object) -> np.ndarray:
        return np.asarray(object, dtype=np.float64)

    @staticmethod
    def zeros(shape: int | tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape, dtype=np.float64)

    @staticmethod
    def ones(shape: int | tuple[int, ...]) -> np.ndarray:
        return np.ones(shape, dtype=np.float64)

    @staticmethod
    def empty(shape: int | tuple[int, ...]) -> np.ndarray:
        return np.empty(shape, dtype=np.float64)

    @staticmethod
    def full(shape: int | tuple[int, ...], value: np.ndarray | float) -> np.ndarray:
        return np.full(shape, value, dtype=np.float64)

    @staticmethod
    def rand(shape: int | tuple[int, ...]) -> np.ndarray:
        return np.asarray(np.random.uniform(size=shape), dtype=np.float64)


class Seg_np(ab.Seg):
    @staticmethod
    def cat(x: Sequence[np.ndarray], dim: int = 0) -> np.ndarray:
        return np.concatenate(x, axis=dim)

    @staticmethod
    def stack(x: Sequence[np.ndarray], dim: int = 0) -> np.ndarray:
        return np.stack(x, axis=dim)

    @staticmethod
    def t(x: np.ndarray) -> np.ndarray:
        return x.T


class Elem_np(ab.Elem):
    @staticmethod
    def sqrt(x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

    @staticmethod
    def sin(x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    @staticmethod
    def cos(x: np.ndarray) -> np.ndarray:
        return np.cos(x)

    @staticmethod
    def atan2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.arctan2(x, y)

    @staticmethod
    def where(condition: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(condition, x, y)

    @staticmethod
    def allclose(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.allclose(x, y)


class Reduc_np(ab.Reduc):
    @staticmethod
    def norm(
        x: np.ndarray, dim: int | None = None, keepdim: bool = False
    ) -> np.ndarray:
        return np.linalg.norm(x, axis=dim, keepdims=keepdim)


_cfg = {ab.Reduc: Reduc_np, ab.Elem: Elem_np, ab.Seg: Seg_np}
np32 = partial(concrete, cfg={ab.Alloc: Alloc_np32, **_cfg})
np64 = partial(concrete, cfg={ab.Alloc: Alloc_np64, **_cfg})
