import numpy as np
from elitho import const


def tcc_matrices(linput, minput, n_input):
    TCCXS0 = np.zeros((n_input, n_input), dtype=np.complex128)
    TCCXSX = np.zeros((n_input, n_input), dtype=np.complex128)
    TCCXSY = np.zeros((n_input, n_input), dtype=np.complex128)
    TCCYS0 = np.zeros((n_input, n_input), dtype=np.complex128)
    TCCYSX = np.zeros((n_input, n_input), dtype=np.complex128)
    TCCYSY = np.zeros((n_input, n_input), dtype=np.complex128)

    pmax = (const.k * const.NA) ** 2

    for i in range(n_input):
        kx = 2 * np.pi / const.dx * linput[i]
        ky = 2 * np.pi / const.dy * minput[i]

        for j in range(i + 1):  # j <= i
            kxp = 2 * np.pi / const.dx * linput[j]
            kyp = 2 * np.pi / const.dy * minput[j]

            sumx_s0 = 0 + 0j
            sumy_s0 = 0 + 0j
            sumx_sx = 0 + 0j
            sumy_sx = 0 + 0j
            sumx_sy = 0 + 0j
            sumy_sy = 0 + 0j

            for is_ in range(SDIV):
                sx = dkx[is_]
                sy = dky[is_]

                ksx = kx + sx
                ksy = ky + sy
                ksxp = kxp + sx
                ksyp = kyp + sy

                if (const.MX**2 * ksx**2 + const.MY**2 * ksy**2) <= pmax and (
                    const.MX**2 * ksxp**2 + const.MY**2 * ksyp**2
                ) <= pmax:

                    phase = np.exp(
                        const.i_complex
                        * ((ksx + const.kx0) ** 2 + (ksy + const.ky0) ** 2)
                        / (2 * const.k)
                        * const.z0
                    )
                    phasep = np.exp(
                        const.i_complex
                        * ((ksxp + const.kx0) ** 2 + (ksyp + const.ky0) ** 2)
                        / (2 * const.k)
                        * const.z0
                    )

                    denom_x = const.k**2 - (const.kx0 + sx) ** 2
                    denom_y = const.k**2 - (const.ky0 + sy) ** 2

                    sumx_s0 += phase * phasep.conjugate() / denom_x
                    sumy_s0 += phase * phasep.conjugate() / denom_y
                    sumx_sx += sx * phase * phasep.conjugate() / denom_x
                    sumy_sx += sx * phase * phasep.conjugate() / denom_y
                    sumx_sy += sy * phase * phasep.conjugate() / denom_x
                    sumy_sy += sy * phase * phasep.conjugate() / denom_y

            TCCXS0[i, j] = sumx_s0 / SDIV
            TCCXS0[j, i] = TCCXS0[i, j].conjugate()
            TCCXSX[i, j] = sumx_sx / SDIV
            TCCXSX[j, i] = TCCXSX[i, j].conjugate()
            TCCXSY[i, j] = sumx_sy / SDIV
            TCCXSY[j, i] = TCCXSY[i, j].conjugate()
            TCCYS0[i, j] = sumy_s0 / SDIV
            TCCYS0[j, i] = TCCYS0[i, j].conjugate()
            TCCYSX[i, j] = sumy_sx / SDIV
            TCCYSX[j, i] = TCCYSX[i, j].conjugate()
            TCCYSY[i, j] = sumy_sy / SDIV
            TCCYSY[j, i] = TCCYSY[i, j].conjugate()
    return TCCXS0, TCCXSX, TCCXSY, TCCYS0, TCCYSX, TCCYSY
