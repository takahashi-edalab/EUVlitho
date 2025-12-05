import numpy as np
from elitho import config


def electric_field_distribution(
    U1U: np.ndarray, U1B: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    for ip in range(2):
        if ip == 0:
            delta = 90.0  # TE
        else:
            delta = 0.0  # TM
        Ex0p = np.cos(np.pi / 180.0 * theta) * cosphi * np.cos(
            np.pi / 180.0 * delta
        ) - sinphi * np.sin(np.pi / 180.0 * delta)
        Ey0p = np.cos(np.pi / 180.0 * theta) * sinphi * np.cos(
            np.pi / 180.0 * delta
        ) + cosphi * np.sin(np.pi / 180.0 * delta)
        Ez0p = -np.sin(np.pi / 180.0 * theta) * np.cos(np.pi / 180.0 * delta)
        ES = np.zeros((config.Nrange2), dtype=complex)
        for i in range(config.Nrange):
            i2 = i + config.Nrange
            if (lindex[i] == l0) and (mindex[i] == m0):
                ES[i] = 2.0 * kz * Ex0p
                ES[i2] = 2.0 * kz * Ey0p
        AS = solve(U0, ES)
        A1 = U1U @ AS
        A1p = U1B @ AS

        for i in range(config.Nrange):
            i2 = i + config.Nrange
            sumx = sumy = sumz = 0.0
            if (lindex[i] == l0) and (mindex[i] == m0):
                sumx += -Ex0p
                sumy += -Ey0p
                sumz += -Ez0p
            for n in range(config.Nrange2):
                sumx += (A1[n] + A1p[n]) * B1[i, n]
                sumy += (A1[n] + A1p[n]) * B1[i2, n]
                sumz += (
                    -1j
                    / k
                    * al1[n]
                    * (A1[n] - A1p[n])
                    * (kxplus[i] * br1[i, n] + kyplus[i] * br1[i2, n])
                )
            if ip == 0:
                ExS[IS, i] = sumx
                EyS[IS, i] = sumy
                EzS[IS, i] = sumz
            else:
                ExP[IS, i] = sumx
                EyP[IS, i] = sumy
                EzP[IS, i] = sumz

    ES = [ExS, EyS, EzS]
    EP = [ExP, EyP, EzP]
    return ES, EP
