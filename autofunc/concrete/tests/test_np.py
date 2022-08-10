# +
import numpy as np

from autofunc.concrete.np import np32, np64
from autofunc.concrete.tests.example import Example


def test_np32() -> bool:
    x = np.random.uniform(size=10)
    f = np32(Example)()
    assert f.a.dtype == np.float32
    f(x)
    return True


def test_np64() -> bool:
    x = np.random.uniform(size=10)
    f = np64(Example)()
    assert f.a.dtype == np.float64
    f(x)
    return True


if __name__ == "__main__":
    test_np32()
    test_np64()
