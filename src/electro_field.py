import numpy as np
from src import const


def electro_field(
    l0s: np.ndarray,
    m0s: np.ndarray,
    SDIV: np.ndarray,
    nsx: int,
    nsy: int,
    ncut: int,
    sx0: float,
    sy0: float,
    linput: np.ndarray,
    minput: np.ndarray,
    ampxx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ex0m = np.zeros((SDIV[nsx, nsy], ncut), dtype=complex)
    Ey0m = np.zeros_like(Ex0m)
    Ez0m = np.zeros_like(Ex0m)

    for isd in range(SDIV[nsx, nsy]):
        kx = sx0 + 2.0 * const.pi / const.dx * l0s[nsx][nsy][isd]
        ky = sy0 + 2.0 * const.pi / const.dy * m0s[nsx][nsy][isd]
        ls = l0s[nsx][nsy][isd] + const.lsmaxX
        ms = m0s[nsx][nsy][isd] + const.lsmaxY
        for i in range(ncut):
            kxplus = kx + 2 * const.pi * linput[i] / const.dx
            kyplus = ky + 2 * const.pi * minput[i] / const.dy
            kxy2 = kxplus**2 + kyplus**2
            klm = np.sqrt(const.k * const.k - kxy2)
            ip = linput[i] + const.lpmaxX
            jp = minput[i] + const.lpmaxY

            Ax_val = ampxx[ls, ms, ip, jp] / np.sqrt(const.k * const.k - kx * kx)
            Ay_val = 0

            EAx = const.i_complex * const.k * Ax_val - const.i_complex / const.k * (
                kxplus**2 * Ax_val + kxplus * kyplus * Ay_val
            )
            EAy = const.i_complex * const.k * Ay_val - const.i_complex / const.k * (
                kxplus * kyplus * Ax_val + kyplus**2 * Ay_val
            )
            EAz = const.i_complex * klm / const.k * (kxplus * Ax_val + kyplus * Ay_val)
            Ex0m[isd, i] = EAx
            Ey0m[isd, i] = EAy
            Ez0m[isd, i] = EAz

    return Ex0m, Ey0m, Ez0m
