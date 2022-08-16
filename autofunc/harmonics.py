# +
from __future__ import annotations

from math import pi, sqrt

import autofunc.abstract as ab


class SolSphHarm(ab.Abs):
    """
    SolSphHarm is short for Solid Spherical Harmonics.

    * It maps 3D Cartesian vectors as follows:

        r -> |r|^l Ylm(theta, phi)

      where Ylm is the spherical harmonics function.

    * If |r| = 1, this reduces to spherical harmonics.

    * If the results are multiplied by sqrt(4*pi/(2*l+1)),
      then regular solid harmonics are obtained:

        https://en.wikipedia.org/wiki/Solid_harmonics

    * The results are stored in a matrix Y (per-input)
      such that the full harmonic with (l, m) can be
      retrieved from:

        m > 0 -> Y[l,l-m] + 1.0j*Y[l-m,l]
        m = 0 -> Y[l,l]

      where:
        l = 0, ..., lmax
        m = 0, ..., l

    * With lmax = 3 this arrangement looks like:

            0 1 2 3       0 1 2 3        r i i i
        l = 1 1 2 3   m = 1 0 1 2    Y = r r i i
            2 2 2 3       2 1 0 1        r r r i
            3 3 3 3       3 2 1 0        r r r r

      where r, i indicate real and imaginary components.

    """

    def __init__(self, lmax: int) -> None:
        super().__init__()
        self.lmax = lmax
        self.Yoo = sqrt(1 / (4 * pi))
        self._init_alp()
        self._init_lms()
        self._init_coef()

    def _init_alp(self) -> None:
        lmax = self.lmax
        nan = self.tens(float("nan"))
        a = [nan, nan]
        b = [nan, nan]
        c = [self.tens(1.0), self.tens(sqrt(3))]
        d = [nan, self.tens(-sqrt(1.5))]
        for l in range(2, lmax + 1):
            _aa = []
            _bb = []
            for m in range(l - 1):
                _aa.append(sqrt((4 * l**2 - 1) / (l**2 - m**2)))
                _bb.append(-sqrt(((l - 1) ** 2 - m**2) / (4 * (l - 1) ** 2 - 1)))
            _a = self.tens(_aa).reshape((-1, 1))
            _b = self.tens(_bb).reshape((-1, 1))
            _c = self.tens(sqrt(2 * l + 1))
            _d = self.tens(-sqrt((1 + 1 / (2 * l))))
            a.append(_a)
            b.append(_b)
            c.append(_c)
            d.append(_d)
        self._al = a
        self._bl = b
        self._cl = c
        self._dl = d

    def _init_lms(self) -> None:
        lmax = self.lmax
        shape = (lmax + 1, lmax + 1, 1)
        l = self.empty(shape)
        m = self.empty(shape)
        s = self.empty(shape)
        for i in range(lmax + 1):
            l[:i, i] = i
            l[i, : i + 1] = i
            for j in range(lmax - i + 1):
                m[i, i + j] = j
                m[i + j, i] = j
            s[i, :i] = -1
            s[i, i:] = 1
        self.l = l
        self.m = m
        self.sign = s

    def _init_coef(self) -> None:
        c = (self.l**2 - self.m**2) * (2 * self.l + 1) / (2 * self.l - 1)
        self.coef = self.sqrt(c[1:, 1:])

    def __call__(self, xyz: ab.Tens) -> ab.Tens:

        # 1.
        n = xyz.shape[0]
        x, y, z = self.t(xyz)
        rxy2 = x * x + y * y
        r2 = rxy2 + z * z
        rxy = self.sqrt(rxy2 + self.eps)

        # 2. Associated Legendre Polynomials
        alp = [[self.full((n,), self.Yoo)]]
        for l in range(1, self.lmax + 1):
            alp.append(
                [
                    *(
                        self._al[l][m]
                        * (z * alp[l - 1][m] + r2 * self._bl[l][m] * alp[l - 2][m])
                        for m in range(l - 1)
                    ),
                    self._cl[l] * z * alp[l - 1][l - 1],
                    self._dl[l] * rxy * alp[l - 1][l - 1],
                ]
            )

        # 3. Sin & Cos of m*phi
        pole = rxy < self.eps
        sin_phi = self.where(pole, y, y / rxy)
        cos_phi = self.where(pole, x, x / rxy)
        sin = [self.zeros(n), sin_phi]
        cos = [self.ones(n), cos_phi]
        for m in range(2, self.lmax + 1):
            s = sin_phi * cos[-1] + cos_phi * sin[-1]
            c = cos_phi * cos[-1] - sin_phi * sin[-1]
            sin.append(s)
            cos.append(c)

        # 4. Spherical Harmonics
        Y = self.zeros((self.lmax + 1, self.lmax + 1, n))
        for l in range(self.lmax + 1):
            Y[l, l] = alp[l][0]
            for m in range(l, 0, -1):
                Y[l, l - m] = alp[l][m] * cos[m]
                Y[l - m, l] = alp[l][m] * sin[m]

        return Y
