import cupy as cp
from elitho import const
from elitho.utils.mat_utils import linalg_eig


def calc_sigma(
    polar: str,
    dod,
    doc: "diffraction_order.DiffractionOrderCoordinate",
    kxplus: "xp.ndarray",
    kyplus: "xp.ndarray",
    sigma: "xp.ndarray",
) -> "xp.ndarray":
    lx = doc.valid_x_coords
    ly = doc.valid_y_coords
    Mx = dod.max_diffraction_order_x
    My = dod.max_diffraction_order_y
    llp = lx[:, None] - lx[None, :] + 2 * Mx
    mmp = ly[:, None] - ly[None, :] + 2 * My
    if polar == "X":
        coef = kxplus
    else:
        coef = kyplus
    new_sigma = sigma[llp, mmp] * coef[None, :]
    return new_sigma


def absorber(
    polar: str,
    dod: "descriptors.DiffractionOrderDescriptor",
    doc: "descriptors.DiffractionOrderDescriptor",
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
    l = (
        doc.valid_x_coords[:, None]
        - doc.valid_x_coords[None, :]
        + 2 * dod.max_diffraction_order_x
    )
    m = (
        doc.valid_y_coords[:, None]
        - doc.valid_y_coords[None, :]
        + 2 * dod.max_diffraction_order_y
    )
    if polar == "X":
        D = eps[l, m] * const.k**2 - const.i_complex * eta[l, m] * kxplus[None, :]
    elif polar == "Y":
        D = eps[l, m] * const.k**2 - const.i_complex * zeta[l, m] * kyplus[None, :]
    D[
        xp.arange(doc.num_valid_diffraction_orders),
        xp.arange(doc.num_valid_diffraction_orders),
    ] -= kxy2

    # eigenvalues and eigenvectors
    # w, br1 = xp.linalg.eig(D) # cupy is not compatible with linalg_eig
    w, br1 = linalg_eig(D)
    # np.save("w.npy", w)
    # np.save("br1.npy", br1)
    al1 = xp.sqrt(w)
    Cjp = xp.linalg.solve(br1, br2)  # Cjp = np.linalg.inv(br1) @ br2
    # np.save("Cjp.npy", Cjp)  #
    new_sigma = calc_sigma(polar, dod, doc, kxplus, kyplus, sigma)

    B1 = const.i_complex * (
        const.k * br1
        - xp.outer(
            kxplus if polar == "X" else kyplus,
            xp.ones(doc.num_valid_diffraction_orders),
        )
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
