import numpy as np
from elitho import const


def abbe_source() -> tuple[list, list, np.ndarray]:
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


def linear_source() -> tuple[list, list, int]:
    kmesh = const.k * const.mesh * (const.pi / 180.0)
    skangx = const.k * const.NA / const.MX * const.sigma1
    skangy = const.k * const.NA / const.MY * const.sigma1
    l0max = int(skangx / kmesh + 1)
    m0max = int(skangy / kmesh + 1)

    dkx = []
    dky = []
    for l in range(-l0max, l0max + 1):
        for m in range(-m0max, m0max + 1):
            skx = l * kmesh
            sky = m * kmesh
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
                dkx.append(skx)
                dky.append(sky)

    return dkx, dky, len(dkx)


def index() -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(const.Lrange) - const.LMAX
    j = np.arange(const.Mrange) - const.MMAX
    ii, jj = np.meshgrid(i, j, indexing="ij")

    mask = (ii / (const.LMAX + 0.01)) ** 2 + (jj / (const.MMAX + 0.01)) ** 2 <= 1.0

    lindexp = ii[mask]
    mindexp = jj[mask]
    return lindexp, mindexp, len(lindexp)


def linear_nrange(Lrange: int, Mrange: int, LMAX: int, MMAX: int) -> int:
    i = np.arange(Lrange) - LMAX
    j = np.arange(Mrange) - MMAX
    base = (np.abs(i)[:, None] / (LMAX + 0.01) + 1.0) * (
        np.abs(j)[None, :] / (MMAX + 0.01) + 1.0
    )
    nrange = np.count_nonzero(base <= 2.0)
    return int(nrange)


def linear_find_valid_output_points(nrange: int) -> tuple[np.ndarray, np.ndarray]:
    linput = np.zeros(nrange, dtype=int)
    minput = np.zeros(nrange, dtype=int)
    xinput = np.zeros(nrange, dtype=int)

    is_arr = np.arange(const.nsourceX) - const.lsmaxX
    js_arr = np.arange(const.nsourceY) - const.lsmaxY
    cond1 = (
        (is_arr[:, None] * const.MX / const.dx) ** 2
        + (js_arr[None, :] * const.MY / const.dy) ** 2
    ) <= (const.NA / const.wavelength) ** 2

    ninput = 0
    for ip in range(const.noutX):
        dx_ip = (ip - const.lpmaxX + is_arr) * const.MX / const.dx
        for jp in range(const.noutY):
            dy_jp = (jp - const.lpmaxY + js_arr) * const.MY / const.dy

            cond2 = (dx_ip[:, None] ** 2 + dy_jp[None, :] ** 2) <= (
                const.NA / const.wavelength
            ) ** 2
            valid_mask = cond1 & cond2
            snum = np.count_nonzero(valid_mask)

            if snum > 0:
                linput[ninput] = ip - const.lpmaxX
                minput[ninput] = jp - const.lpmaxY
                xinput[ninput] = 1 if snum >= 8 else 0
                ninput += 1

    return linput, minput, xinput
