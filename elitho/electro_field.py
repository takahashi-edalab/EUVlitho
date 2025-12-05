import numpy as np
from elitho import config


def polarization_rotation(k, MX, MY, px, py, sx0, sy0) -> np.ndarray:
    s0 = np.array([sx0, sy0, -np.sqrt(k**2 - sx0**2 - sy0**2)])
    p = np.array([px, py, 0.0])
    pp = np.array([MX * px, MY * py, -np.sqrt(k**2 - (MX * px) ** 2 - (MY * py) ** 2)])
    ez = np.array([0.0, 0.0, 1.0])
    ezp = np.array([0.0, 0.0, 1.0])
    eps = 1e-5

    if (px**2 + py**2) > eps:
        es = np.cross(p, s0)
        es = es / np.linalg.norm(es)
        esp = np.cross(pp, ezp)
        esp = esp / np.linalg.norm(esp)

        ps0 = p + s0
        ps0 = ps0 / np.linalg.norm(ps0)

        em = np.cross(es, ps0)
        emp = np.cross(esp, pp) / k

    else:
        es = np.cross(ez, s0)
        es = es / np.linalg.norm(es)
        esp = -es
        em = np.cross(es, s0) / k
        emp = np.cross(esp, -ezp)

    R = np.zeros((3, 2))
    scale = np.sqrt(k / abs(pp[2]))

    for i in range(3):
        for j in range(2):
            R[i, j] = scale * (esp[i] * es[j] + emp[i] * em[j])

    return R


def high_na_electro_field(nsx, nsy, Ax_val, Ay_val, linput, minput, l0s, m0s):
    kxn = (
        (2 * np.pi / config.dx) * (nsx / config.ndivX)
        + (2 * np.pi / config.dx) * l0s
        + (2 * np.pi * linput / config.dx)
    )
    kyn = (
        (2 * np.pi / config.dy) * (nsy / config.ndivY)
        + (2 * np.pi / config.dy) * m0s
        + (2 * np.pi * minput / config.dy)
    )
    EAx = 0.0
    EAy = 0.0
    EAz = 0.0
    p2 = config.MX**2 * kxn**2 + config.MY**2 * kyn**2
    if all(
        [
            (config.NA * config.k * config.co) ** 2 <= p2,
            p2 <= (config.NA * config.k) ** 2,
        ]
    ):
        R = polarization_rotation(
            config.k, config.MX, config.MY, kxn, kyn, config.kx0, config.ky0
        )
        EAx = config.i_complex * config.k * (R[0, 0] * Ax_val + R[0, 1] * Ay_val)
        EAy = config.i_complex * config.k * (R[1, 0] * Ax_val + R[1, 1] * Ay_val)
        EAz = config.i_complex * config.k * (R[2, 0] * Ax_val + R[2, 1] * Ay_val)
    return EAx, EAy, EAz


def standard_na_electro_field(kxplus, kyplus, Ax_val, Ay_val):
    kxy2 = kxplus**2 + kyplus**2
    klm = np.sqrt(config.k**2 - kxy2)
    EAx = config.i_complex * config.k * Ax_val - config.i_complex / config.k * (
        kxplus**2 * Ax_val + kxplus * kyplus * Ay_val
    )
    EAy = config.i_complex * config.k * Ay_val - config.i_complex / config.k * (
        kxplus * kyplus * Ax_val + kyplus**2 * Ay_val
    )
    EAz = config.i_complex * klm / config.k * (kxplus * Ax_val + kyplus * Ay_val)
    return EAx, EAy, EAz


def electro_field(
    polar: config.PolarizationDirection,
    is_high_na: bool,
    nsx: int,
    nsy: int,
    SDIV: np.ndarray,
    l0s: np.ndarray,
    m0s: np.ndarray,
    sx0: float,
    sy0: float,
    pupil_coords: "pupil.PupilCoordinates",
    ampxx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ex0m = np.zeros((SDIV, pupil_coords.n_coordinates), dtype=complex)
    Ey0m = np.zeros_like(Ex0m)
    Ez0m = np.zeros_like(Ex0m)

    for isd in range(SDIV):
        kx = sx0 + 2.0 * np.pi / config.dx * l0s[isd]
        ky = sy0 + 2.0 * np.pi / config.dy * m0s[isd]
        ls = l0s[isd] + config.lsmaxX
        ms = m0s[isd] + config.lsmaxY
        for i in range(pupil_coords.n_coordinates):
            ip = pupil_coords.linput[i] + config.lpmaxX
            jp = pupil_coords.minput[i] + config.lpmaxY

            if polar == config.PolarizationDirection.X:
                Ax_val = ampxx[ls, ms, ip, jp] / np.sqrt(config.k**2 - kx**2)
                Ay_val = 0
            elif polar == config.PolarizationDirection.Y:
                Ax_val = 0
                Ay_val = ampxx[ls, ms, ip, jp] / np.sqrt(config.k**2 - ky**2)
            else:
                raise ValueError("polar must be 'X' or 'Y'")

            if is_high_na:
                EAx, EAy, EAz = high_na_electro_field(
                    nsx,
                    nsy,
                    Ax_val,
                    Ay_val,
                    pupil_coords.linput[i],
                    pupil_coords.minput[i],
                    l0s[isd],
                    m0s[isd],
                )
            else:
                kxplus = kx + 2 * np.pi * pupil_coords.linput[i] / config.dx
                kyplus = ky + 2 * np.pi * pupil_coords.minput[i] / config.dy
                EAx, EAy, EAz = standard_na_electro_field(
                    kxplus, kyplus, Ax_val, Ay_val
                )

            Ex0m[isd, i] = EAx
            Ey0m[isd, i] = EAy
            Ez0m[isd, i] = EAz

    return Ex0m, Ey0m, Ez0m
