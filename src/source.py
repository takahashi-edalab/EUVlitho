import numpy as np
from src import const


def source(ndivs: int) -> tuple[list, list, np.ndarray]:
    """Calculate source distribution

    Args:
        NA: Numerical aperture
        type: Source type (0: circular, 1: annular, 2: dipole)
        sigma1: Outer sigma
        sigma2: Inner sigma
        openangle: Opening angle for dipole
        k: Wave number
        dx, dy: Pitch
        ndivs: Number of divisions
        MX, MY: Magnification

    Returns:
        l0s, m0s: Source coordinates
        SDIV: Source division array
    """
    dkxang = 2.0 * const.pi / const.dx
    dkyang = 2.0 * const.pi / const.dy
    skangx = const.k * const.NA / const.MX * const.sigma1
    skangy = const.k * const.NA / const.MY * const.sigma1
    l0max = int(skangx / dkxang) + 1
    m0max = int(skangy / dkyang) + 1

    l0s = [[[] for _ in range(ndivs)] for _ in range(ndivs)]
    m0s = [[[] for _ in range(ndivs)] for _ in range(ndivs)]
    SDIV = np.zeros((ndivs, ndivs), dtype=int)

    for nsx in range(ndivs):
        for nsy in range(ndivs):
            for l in range(-l0max, l0max + 1):
                for m in range(-m0max, m0max + 1):
                    skx = l * dkxang + 2.0 * const.pi / const.dx * nsx / ndivs
                    sky = m * dkyang + 2.0 * const.pi / const.dy * nsy / ndivs
                    skxo = skx * const.MX
                    skyo = sky * const.MY

                    condition = False
                    if type == 0:  # circular
                        condition = (skxo**2 + skyo**2) <= (
                            const.k * const.NA * const.sigma1
                        ) ** 2
                    elif type == 1:  # annular
                        r = np.sqrt(skxo**2 + skyo**2)
                        condition = (
                            const.k * const.NA * const.sigma2
                            <= r
                            <= const.k * const.NA * const.sigma1
                        )
                    elif type == 2:  # dipole
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
