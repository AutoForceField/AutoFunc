# +
import numpy as np

from autofunc.concrete.np import np32, np64
from autofunc.concrete.tests.example import Example


def test_np32() -> bool:
    f = np32(Example)()
    x = f.tensor([1.0, 2.0, 3.0])
    assert f(x).dtype == np.float32
    return True


def test_np64() -> bool:
    f = np64(Example)()
    x = f.tensor([1.0, 2.0, 3.0])
    assert f(x).dtype == np.float64
    return True


if __name__ == "__main__":
    test_np32()
    test_np64()
