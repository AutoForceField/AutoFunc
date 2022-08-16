# +
from scipy.special import sph_harm

from autofunc.concrete.numpy import np64
from autofunc.coo3d import Cart2Sph
from autofunc.harmonics import SolSphHarm


def _scipy_SolSphHarm(xyz, lmax):
    rtp = np64(Cart2Sph)()
    r, theta, phi = rtp.t(rtp(xyz))
    rlm = rtp.empty((lmax + 1, lmax + 1, xyz.shape[0]))
    for l in range(0, lmax + 1):
        for m in range(0, l + 1):
            val = r**l * sph_harm(m, l, phi, theta)
            rlm[l, l - m] = val.real
            if m > 0:
                rlm[l - m, l] = val.imag
    return rlm


def test_SolSphHarm(lmax: int = 10) -> bool:
    rlm = np64(SolSphHarm)(lmax)
    x = rlm.tens([[1.0, 0.0, 0.0]])
    y = rlm.tens([[0.0, 1.0, 0.0]])
    z = rlm.tens([[0.0, 0.0, 1.0]])
    error = []
    for r in [x, y, z]:
        a = _scipy_SolSphHarm(r, lmax)
        b = rlm(r)
        error.append(abs(a - b).max())
    assert float(max(error)) < 10 * rlm.eps
    return True


if __name__ == "__main__":
    test_SolSphHarm()
