import cupy as cp
from elitho import config
from elitho.utils.mat_utils import linalg_eig


def calc_sigma(
    polar: config.PolarizationDirection,
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
    if polar == config.PolarizationDirection.X:
        coef = kxplus
    elif polar == config.PolarizationDirection.Y:
        coef = kyplus
    else:
        raise ValueError("Invalid polarization direction")
    new_sigma = sigma[llp, mmp] * coef[None, :]
    return new_sigma


def absorber(
    polar: config.PolarizationDirection,
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
    if polar == config.PolarizationDirection.X:
        D = eps[l, m] * config.k**2 - config.i_complex * eta[l, m] * kxplus[None, :]
    elif polar == config.PolarizationDirection.Y:
        D = eps[l, m] * config.k**2 - config.i_complex * zeta[l, m] * kyplus[None, :]
    else:
        raise ValueError("Invalid polarization direction")
    D[
        xp.arange(doc.num_valid_diffraction_orders),
        xp.arange(doc.num_valid_diffraction_orders),
    ] -= kxy2

    # eigenvalues and eigenvectors
    # w, br1 = xp.linalg.eig(D) # cupy is not compatible with linalg_eig
    w, br1 = linalg_eig(D)
    al1 = xp.sqrt(w)
    Cjp = xp.linalg.solve(br1, br2)  # Cjp = np.linalg.inv(br1) @ br2
    new_sigma = calc_sigma(polar, dod, doc, kxplus, kyplus, sigma)

    B1 = config.i_complex * (
        config.k * br1
        - xp.outer(
            kxplus if polar == config.PolarizationDirection.X else kyplus,
            xp.ones(doc.num_valid_diffraction_orders),
        )
        / config.k
        * new_sigma
        @ br1
    )

    Cj = xp.linalg.solve(B1, B2)  # Cj = np.linalg.inv(B1) @ B2
    gamma = xp.exp(config.i_complex * al1 * dabs)
    T1UL = 0.5 * (Cj + xp.outer(1 / al1, al2) * Cjp) / gamma[:, None]
    T1UR = 0.5 * (Cj - xp.outer(1 / al1, al2) * Cjp) / gamma[:, None]
    T1BL = 0.5 * (Cj - xp.outer(1 / al1, al2) * Cjp) * gamma[:, None]
    T1BR = 0.5 * (Cj + xp.outer(1 / al1, al2) * Cjp) * gamma[:, None]

    U1U = T1UL @ U2U + T1UR @ U2B
    U1B = T1BL @ U2U + T1BR @ U2B
    return U1U, U1B, B1, al1, br1
