import cupy as cp
from elitho import const


def calc_sigma(
    polar: str, kxplus: "xp.ndarray", kyplus: "xp.ndarray", sigma: "xp.ndarray"
) -> "xp.ndarray":
    xp = cp.get_array_module(kxplus)
    new_sigma = xp.zeros((const.Nrange, const.Nrange), dtype=complex)
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
    return new_sigma


def absorber(
    polar: str,
    kxplus: "xp.ndarray",
    kyplus: "xp.ndarray",
    kxy2: float,
    eps: "xp.ndarray",
    eta: "xp.ndarray",
    zeta: "xp.ndarray",
    sigma: "xp.ndarray",
    dabs: float,
    al2: "xp.ndarray",
    br2: "xp.ndarray",
    B2: "xp.ndarray",
    U2U: "xp.ndarray",
    U2B: "xp.ndarray",
):
    xp = cp.get_array_module(kxplus)
    print(xp.__name__)
    l = const.lindex[:, None] - const.lindex[None, :] + 2 * const.LMAX
    m = const.mindex[:, None] - const.mindex[None, :] + 2 * const.MMAX
    if polar == "X":
        D = eps[l, m] * const.k**2 - const.i_complex * eta[l, m] * kxplus[None, :]
    elif polar == "Y":
        D = eps[l, m] * const.k**2 - const.i_complex * zeta[l, m] * kyplus[None, :]
    D[xp.arange(const.Nrange), xp.arange(const.Nrange)] -= kxy2

    # eigenvalues and eigenvectors
    w, br1 = xp.linalg.eig(D)
    al1 = xp.sqrt(w)
    Cjp = xp.linalg.solve(br1, br2)  # Cjp = np.linalg.inv(br1) @ br2
    new_sigma = calc_sigma(polar, kxplus, kyplus, sigma)

    B1 = const.i_complex * (
        const.k * br1
        - xp.outer(kxplus if polar == "X" else kyplus, xp.ones(const.Nrange))
        / const.k
        * new_sigma
        @ br1
    )

    Cj = xp.linalg.solve(B1, B2)  # Cj = np.linalg.inv(B1) @ B2
    gamma = xp.exp(const.i_complex * al1 * dabs)
    T1UL = 0.5 * (Cj + xp.outer(1 / al1, al2) * Cjp) / gamma[:, None]
    T1UR = 0.5 * (Cj - xp.outer(1 / al1, al2) * Cjp) / gamma[:, None]
    T1BL = 0.5 * (Cj - xp.outer(1 / al1, al2) * Cjp) * gamma[:, None]
    T1BR = 0.5 * (Cj + xp.outer(1 / al1, al2) * Cjp) * gamma[:, None]

    U1U = T1UL @ U2U + T1UR @ U2B
    U1B = T1BL @ U2U + T1BR @ U2B
    return U1U, U1B, B1, al1, br1
