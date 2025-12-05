import numpy as np
from elitho import config
from collections import defaultdict


def illumination_condition(
    sc: config.SimulationConfig,
    skx: float | np.ndarray,
    sky: float | np.ndarray,
) -> bool | np.ndarray:
    if sc.illumination.type == config.IlluminationType.CIRCULAR:
        condition = (skx**2 + sky**2) <= (
            sc.k * sc.NA * sc.illumination.outer_sigma
        ) ** 2
    elif sc.illumination.type == config.IlluminationType.ANNULAR:
        r = np.sqrt(skx**2 + sky**2)
        condition = (sc.k * sc.NA * sc.illumination.inner_sigma <= r) & (
            r <= sc.k * sc.NA * sc.illumination.outer_sigma
        )
    elif sc.illumination.type == config.IlluminationType.DIPOLE_X:
        r = np.sqrt(skx**2 + sky**2)
        angle_condition = abs(sky) <= abs(skx) * np.tan(
            np.pi * sc.illumination.open_angle / 180.0 / 2.0
        )
        condition = (
            (sc.k * sc.NA * sc.illumination.inner_sigma <= r)
            & (r <= sc.k * sc.NA * sc.illumination.outer_sigma)
        ) & angle_condition
    elif sc.illumination.type == config.IlluminationType.DIPOLE_Y:
        r = np.sqrt(skx**2 + sky**2)
        angle_condition = abs(skx) <= abs(sky) * np.tan(
            np.pi * sc.illumination.open_angle / 180.0 / 2.0
        )
        condition = (
            (sc.k * sc.NA * sc.illumination.inner_sigma <= r)
            & (r <= sc.k * sc.NA * sc.illumination.outer_sigma)
        ) & angle_condition
    else:
        raise ValueError("Invalid illumination type")
    return condition


def get_valid_diffraction_orders(sc: config.SimulationConfig):
    lmax = int(sc.kX / sc.dkx) + 1
    mmax = int(sc.kY / sc.dky) + 1

    ls, ms = [], []
    for l in range(-lmax, lmax + 1):
        for m in range(-mmax, mmax + 1):
            skx = sc.dkx * l * sc.magnification_x
            sky = sc.dky * m * sc.magnification_y
            if (skx**2 + sky**2) <= (sc.k * sc.NA) ** 2:
                ls.append(l)
                ms.append(m)
    return ls, ms


def abbe_division_sampling(sc: config.SimulationConfig) -> tuple[dict, dict, dict]:

    ls, ms = get_valid_diffraction_orders(sc)
    _, _, ldiv, mdiv, pdiv = get_valid_source_points(sc, sc.illumination)

    l0s = defaultdict(list)
    m0s = defaultdict(list)
    SDIV = defaultdict(int)

    for nsx in range(-config.ndivX + 1, config.ndivX):
        for nsy in range(-config.ndivY + 1, config.ndivY):
            for l, m in zip(ls, ms):
                shifted_l = l * config.ndivX + nsx
                shifted_m = m * config.ndivY + nsy
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


def get_valid_source_points(sc: config.SimulationConfig):
    dkdivX = sc.dkx / sc.ndivX
    dkdivY = sc.dky / sc.ndivY
    kXs = sc.k * sc.NA / sc.magnification_x * sc.illumination.outer_sigma
    kYs = sc.k * sc.NA / sc.magnification_y * sc.illumination.outer_sigma
    ldivmax = int(kXs / dkdivX) + 1
    mdivmax = int(kYs / dkdivY) + 1

    ldiv, mdiv, pdiv = [], [], []
    dkx, dky = [], []
    for l in range(-ldivmax, ldivmax + 1):
        for m in range(-mdivmax, mdivmax + 1):
            skx = dkdivX * l * sc.magnification_x
            sky = dkdivY * m * sc.magnification_y
            if illumination_condition(sc, skx, sky):
                dkx.append(skx)
                dky.append(sky)
                ldiv.append(l)
                mdiv.append(m)
                pdiv.append(1)
    return np.array(dkx), np.array(dky), ldiv, mdiv, pdiv


def uniform_k_source(sc: config.SimulationConfig) -> tuple[np.ndarray, np.ndarray, int]:
    kmesh = sc.k * sc.mesh * (np.pi / 180.0)
    skangx = sc.k * sc.NA / sc.magnification_x * sc.illumination.outer_sigma
    skangy = sc.k * sc.NA / sc.magnification_y * sc.illumination.outer_sigma
    l0max = int(skangx / kmesh + 1)
    m0max = int(skangy / kmesh + 1)
    l = np.arange(-l0max, l0max + 1)
    m = np.arange(-m0max, m0max + 1)

    L, M = np.meshgrid(l, m, indexing="ij")
    skx = L * kmesh
    sky = M * kmesh
    skxo = skx * sc.magnification_x
    skyo = sky * sc.magnification_y

    mask = illumination_condition(sc, skxo, skyo)
    dkx = skx[mask]
    dky = sky[mask]
    return dkx, dky, len(dkx)


class UniformKSource:
    def __init__(self):
        self.dkx, self.dky, self.SDIV = uniform_k_source()

    def __iter__(self):
        return zip(self.dkx, self.dky)
