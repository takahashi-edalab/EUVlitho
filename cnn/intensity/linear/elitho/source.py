import numpy as np
from elitho import const
from collections import defaultdict


def illumination_condition(
    illumination_type: "const.IlluminationType", skx, sky
) -> bool:
    if illumination_type == const.IlluminationType.CIRCULAR:
        condition = (skx**2 + sky**2) <= (const.k * const.NA * const.sigma1) ** 2
    elif illumination_type == const.IlluminationType.ANNULAR:
        r = np.sqrt(skx**2 + sky**2)
        condition = (
            const.k * const.NA * const.sigma2 <= r <= const.k * const.NA * const.sigma1
        )
    elif illumination_type == const.IlluminationType.DIPOLE_X:
        r = np.sqrt(skx**2 + sky**2)
        angle_condition = abs(sky) <= abs(skx) * np.tan(
            const.pi * const.openangle / 180.0 / 2.0
        )
        condition = (
            const.k * const.NA * const.sigma2 <= r <= const.k * const.NA * const.sigma1
        ) and angle_condition
    elif illumination_type == const.IlluminationType.DIPOLE_Y:
        r = np.sqrt(skx**2 + sky**2)
        angle_condition = abs(skx) <= abs(sky) * np.tan(
            const.pi * const.openangle / 180.0 / 2.0
        )
        condition = (
            const.k * const.NA * const.sigma2 <= r <= const.k * const.NA * const.sigma1
        ) and angle_condition
    else:
        raise ValueError("Invalid illumination type")
    return condition


def get_valid_diffraction_orders(dkX, dkY):
    lmax = int(const.kX / dkX) + 1
    mmax = int(const.kY / dkY) + 1

    ls, ms = [], []
    for l in range(-lmax, lmax + 1):
        for m in range(-mmax, mmax + 1):
            skx = dkX * l * const.MX
            sky = dkY * m * const.MY
            if (skx**2 + sky**2) <= (const.k * const.NA) ** 2:
                ls.append(l)
                ms.append(m)
    return ls, ms


def get_valid_source_points(illumination_type: "const.IlluminationType", dkX, dkY):
    dkdivX = dkX / const.ndivX
    dkdivY = dkY / const.ndivY
    kXs = const.k * const.NA / const.MX * const.sigma1
    kYs = const.k * const.NA / const.MY * const.sigma1
    ldivmax = int(kXs / dkdivX) + 1
    mdivmax = int(kYs / dkdivY) + 1

    ldiv, mdiv, pdiv = [], [], []
    for l in range(-ldivmax, ldivmax + 1):
        for m in range(-mdivmax, mdivmax + 1):
            skx = dkdivX * l * const.MX
            sky = dkdivY * m * const.MY
            if illumination_condition(illumination_type, skx, sky):
                ldiv.append(l)
                mdiv.append(m)
                pdiv.append(1)
    return ldiv, mdiv, pdiv


def abbe_division_sampling(
    illumination_type: "const.IlluminationType",
) -> tuple[dict, dict, dict]:
    dkX = 2.0 * const.pi / const.dx
    dkY = 2.0 * const.pi / const.dy

    ls, ms = get_valid_diffraction_orders(dkX, dkY)
    ldiv, mdiv, pdiv = get_valid_source_points(illumination_type, dkX, dkY)

    l0s = defaultdict(list)
    m0s = defaultdict(list)
    SDIV = defaultdict(int)

    for nsx in range(-const.ndivX + 1, const.ndivX):
        for nsy in range(-const.ndivY + 1, const.ndivY):
            for l, m in zip(ls, ms):
                shifted_l = l * const.ndivX + nsx
                shifted_m = m * const.ndivY + nsy
                for i, (ld, md) in enumerate(zip(ldiv, mdiv)):
                    if all(
                        [
                            pdiv[i] == 1,
                            ld == shifted_l,
                            md == shifted_m,
                        ]
                    ):
                        l0s[(nsx, nsy)].append(l)
                        m0s[(nsx, nsy)].append(m)
                        SDIV[(nsx, nsy)] += 1
                        pdiv[i] = 0
    return l0s, m0s, SDIV


# def old_abbe_division_sampling() -> tuple[list, list, np.ndarray]:
#     dkxang = 2.0 * const.pi / const.dx
#     dkyang = 2.0 * const.pi / const.dy
#     skangx = const.k * const.NA / const.MX * const.sigma1
#     skangy = const.k * const.NA / const.MY * const.sigma1
#     l0max = int(skangx / dkxang) + 1
#     m0max = int(skangy / dkyang) + 1

#     l0s = [[[] for _ in range(const.ndivs)] for _ in range(const.ndivs)]
#     m0s = [[[] for _ in range(const.ndivs)] for _ in range(const.ndivs)]
#     SDIV = np.zeros((const.ndivs, const.ndivs), dtype=int)

#     for nsx in range(const.ndivs):
#         for nsy in range(const.ndivs):
#             for l in range(-l0max, l0max + 1):
#                 for m in range(-m0max, m0max + 1):
#                     skx = l * dkxang + 2.0 * const.pi / const.dx * nsx / const.ndivs
#                     sky = m * dkyang + 2.0 * const.pi / const.dy * nsy / const.ndivs
#                     skxo = skx * const.MX
#                     skyo = sky * const.MY

#                     condition = False
#                     if const.illumination_type == const.IlluminationType.CIRCULAR:
#                         condition = (skxo**2 + skyo**2) <= (
#                             const.k * const.NA * const.sigma1
#                         ) ** 2
#                     elif const.illumination_type == const.IlluminationType.ANNULAR:
#                         r = np.sqrt(skxo**2 + skyo**2)
#                         condition = (
#                             const.k * const.NA * const.sigma2
#                             <= r
#                             <= const.k * const.NA * const.sigma1
#                         )
#                     elif const.illumination_type == const.IlluminationType.DIPOLE:
#                         r = np.sqrt(skxo**2 + skyo**2)
#                         angle_condition = abs(skyo) <= abs(skxo) * np.tan(
#                             const.pi * const.openangle / 180.0 / 2.0
#                         )
#                         condition = (
#                             const.k * const.NA * const.sigma2
#                             <= r
#                             <= const.k * const.NA * const.sigma1
#                         ) and angle_condition

#                     if condition:
#                         l0s[nsx][nsy].append(l)
#                         m0s[nsx][nsy].append(m)
#                         SDIV[nsx][nsy] += 1

#     return l0s, m0s, SDIV


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
