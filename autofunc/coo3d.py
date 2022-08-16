# +
from __future__ import annotations

import autofunc.abstract as ab


class Cart2Sph(ab.Abs):
    """
    Converst 3d Cartesian coordinates to Spherical:
        (x, y, z) -> (r, theta, phi)
    where:
        r:       radial distance
        theta:   polar angle
        phi:     azimuthal angle

    """

    def __call__(self, xyz: ab.Tens) -> ab.Tens:
        x, y, z = self.t(xyz)
        rxy2 = x * x + y * y
        rxy = self.sqrt(rxy2)
        r = self.sqrt(rxy2 + z * z)
        theta = self.atan2(rxy, z)
        phi = self.atan2(y, x)
        rtp = self.stack([r, theta, phi], dim=1)
        return rtp


class Sph2Cart(ab.Abs):
    """
    Converts 3d Spherical coordinates to Cartesian:
        (r, theta, phi) -> (x, y, z)
    where:
        r:       radial distance
        theta:   polar angle
        phi:     azimuthal angle

    """

    def __call__(self, rtp: ab.Tens) -> ab.Tens:
        r, theta, phi = self.t(rtp)
        x = r * self.sin(theta) * self.cos(phi)
        y = r * self.sin(theta) * self.sin(phi)
        z = r * self.cos(theta)
        xyz = self.stack([x, y, z], dim=1)
        return xyz


class Rotation(ab.Abs):
    """
    Rotations in 3d Cartesian coordinates defined by
    a rotation "axis" and an angle.

    """

    def __init__(self, axis: tuple[float, float, float], angle: float) -> None:
        n = self.tens(axis)
        n = n / self.norm(n)
        a = self.cos(self.tens(angle / 2))
        b, c, d = -n * self.sin(self.tens(angle / 2))
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        self._rot = self.t(
            self.tens(
                [
                    [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
                ]
            )
        )

    def __call__(self, xyz: ab.Tens) -> ab.Tens:
        return xyz @ self._rot
