import numpy as np
from src import const


def source() -> tuple[list, list, np.ndarray]:
    dkxang = 2.0 * const.pi / const.dx
    dkyang = 2.0 * const.pi / const.dy
    skangx = const.k * const.NA / const.MX * const.sigma1
    skangy = const.k * const.NA / const.MY * const.sigma1
    l0max = int(skangx / dkxang) + 1
    m0max = int(skangy / dkyang) + 1

    l0s = [[[] for _ in range(const.ndivs)] for _ in range(const.ndivs)]
    m0s = [[[] for _ in range(const.ndivs)] for _ in range(const.ndivs)]
    SDIV = np.zeros((const.ndivs, const.ndivs), dtype=int)

    for nsx in range(const.ndivs):
        for nsy in range(const.ndivs):
            for l in range(-l0max, l0max + 1):
                for m in range(-m0max, m0max + 1):
                    skx = l * dkxang + 2.0 * const.pi / const.dx * nsx / const.ndivs
                    sky = m * dkyang + 2.0 * const.pi / const.dy * nsy / const.ndivs
                    skxo = skx * const.MX
                    skyo = sky * const.MY

                    condition = False
                    if const.optical_type == 0:  # circular
                        condition = (skxo**2 + skyo**2) <= (
                            const.k * const.NA * const.sigma1
                        ) ** 2
                    elif const.optical_type == 1:  # annular
                        r = np.sqrt(skxo**2 + skyo**2)
                        condition = (
                            const.k * const.NA * const.sigma2
                            <= r
                            <= const.k * const.NA * const.sigma1
                        )
                    elif const.optical_type == 2:  # dipole
                        r = np.sqrt(skxo**2 + skyo**2)
                        angle_condition = abs(skyo) <= abs(skxo) * np.tan(
                            const.pi * const.openangle / 180.0 / 2.0
                        )
                        condition = (
                            const.k * const.NA * const.sigma2
                            <= r
                            <= const.k * const.NA * const.sigma1
                        ) and angle_condition

                    if condition:
                        l0s[nsx][nsy].append(l)
                        m0s[nsx][nsy].append(m)
                        SDIV[nsx][nsy] += 1

    return l0s, m0s, SDIV
