import numpy as np
from elitho import const


def abbe_source() -> tuple[list, list, np.ndarray]:
    # TODO: abbe_division_samplingにあとで変更
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
                    if const.illumination_type == const.IlluminationType.CIRCULAR:
                        condition = (skxo**2 + skyo**2) <= (
                            const.k * const.NA * const.sigma1
                        ) ** 2
                    elif const.illumination_type == const.IlluminationType.ANNULAR:
                        r = np.sqrt(skxo**2 + skyo**2)
                        condition = (
                            const.k * const.NA * const.sigma2
                            <= r
                            <= const.k * const.NA * const.sigma1
                        )
                    elif const.illumination_type == const.IlluminationType.DIPOLE:
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


def uniform_k_source():
    kmesh = const.k * const.mesh * (const.pi / 180.0)
    skangx = const.k * const.NA / const.MX * const.sigma1
    skangy = const.k * const.NA / const.MY * const.sigma1
    l0max = int(skangx / kmesh + 1)
    m0max = int(skangy / kmesh + 1)

    # 元コードと同じ範囲
    l = np.arange(-l0max, l0max + 1)
    m = np.arange(-m0max, m0max + 1)

    L, M = np.meshgrid(l, m, indexing="ij")
    skx = L * kmesh
    sky = M * kmesh
    skxo = skx * const.MX
    skyo = sky * const.MY

    if const.illumination_type == const.IlluminationType.CIRCULAR:
        mask = skxo**2 + skyo**2 <= (const.k * const.NA * const.sigma1) ** 2

    elif const.illumination_type == const.IlluminationType.ANNULAR:
        r = np.sqrt(skxo**2 + skyo**2)
        mask = (const.k * const.NA * const.sigma2 <= r) & (
            r <= const.k * const.NA * const.sigma1
        )

    elif const.illumination_type == const.IlluminationType.DIPOLE:
        r = np.sqrt(skxo**2 + skyo**2)
        angle_cond = np.abs(skyo) <= np.abs(skxo) * np.tan(
            const.pi * const.openangle / 180.0 / 2.0
        )
        mask = (
            (const.k * const.NA * const.sigma2 <= r)
            & (r <= const.k * const.NA * const.sigma1)
            & angle_cond
        )
    else:
        raise ValueError("Invalid illumination type")

    dkx = skx[mask].tolist()
    dky = sky[mask].tolist()

    return dkx, dky, len(dkx)


class UniformKSource:
    def __init__(self):
        self.dkx, self.dky, self.SDIV = uniform_k_source()

    def __iter__(self):
        return zip(self.dkx, self.dky)
