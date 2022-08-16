# +
from math import pi

from autofunc.concrete.numpy import np64
from autofunc.coo3d import Cart2Sph, Rotation, Sph2Cart


def test_transformations() -> bool:
    f = np64(Cart2Sph)()
    g = np64(Sph2Cart)()
    x = 2 * f.rand((1000, 3)) - 1.0
    assert f.allclose(g(f(x)), x)
    return True


def test_Rotation() -> bool:
    axis = [0.0, 0.0, 1.0]
    angle = pi / 2
    rot = np64(Rotation)(axis, angle)
    xyz_in = rot.tens(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    xyz_out = rot.tens(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    assert rot.allclose(rot(xyz_in), xyz_out)
    return True


if __name__ == "__main__":
    test_transformations()
    test_Rotation()
