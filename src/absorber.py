import numpy as np
from src import const


def absorber(
    polar: str,
    kxplus: np.ndarray,
    kyplus: np.ndarray,
    kxy2: np.ndarray,
    eps: np.ndarray,
    eta: np.ndarray,
    zeta: np.ndarray,
    sigma: np.ndarray,
    dabs: float,
    al2: np.ndarray,
    br2: np.ndarray,
    B2: np.ndarray,
    U2U: np.ndarray,
    U2B: np.ndarray,
):
    l = const.lindex[:, None] - const.lindex[None, :] + 2 * const.LMAX
    m = const.mindex[:, None] - const.mindex[None, :] + 2 * const.MMAX
    if polar == "X":
        D = eps[l, m] * const.k**2 - const.i_complex * eta[l, m] * kxplus[None, :]
    elif polar == "Y":
        D = eps[l, m] * const.k**2 - const.i_complex * zeta[l, m] * kyplus[None, :]
    D[np.arange(const.Nrange), np.arange(const.Nrange)] -= kxy2

    # eigenvalues and eigenvectors
    w, br1 = np.linalg.eig(D)
    al1 = np.sqrt(w)
    # calc Cjp
    Cjp = np.linalg.inv(br1) @ br2

    new_sigma = np.zeros((const.Nrange, const.Nrange), dtype=complex)
    for i in range(const.Nrange):
        l = const.lindex[i]
        m = const.mindex[i]
        for ip in range(const.Nrange):
            llp = l - const.lindex[ip] + 2 * const.LMAX
            mmp = m - const.mindex[ip] + 2 * const.MMAX
            if polar == "X":
                new_sigma[i, ip] = sigma[llp, mmp] * kxplus[ip]
            else:
                new_sigma[i, ip] = sigma[llp, mmp] * kyplus[ip]

    B1 = const.i_complex * (
        const.k * br1
        - np.outer(kxplus if polar == "X" else kyplus, np.ones(const.Nrange))
        / const.k
        * new_sigma
        @ br1
    )
    # Cj の計算
    Cj = np.linalg.inv(B1) @ B2

    # T1 行列と U1U, U1B の計算
    gamma = np.exp(const.i_complex * al1 * dabs)
    T1UL = 0.5 * (Cj + np.outer(al2, 1 / al1) * Cjp) / gamma[:, None]
    T1UR = 0.5 * (Cj - np.outer(al2, 1 / al1) * Cjp) / gamma[:, None]
    T1BL = 0.5 * (Cj - np.outer(al2, 1 / al1) * Cjp) * gamma[:, None]
    T1BR = 0.5 * (Cj + np.outer(al2, 1 / al1) * Cjp) * gamma[:, None]

    U1U = T1UL @ U2U + T1UR @ U2B
    U1B = T1BL @ U2U + T1BR @ U2B
    return U1U, U1B, B1, al1, br1
