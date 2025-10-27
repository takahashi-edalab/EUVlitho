import cupy as cp
from elitho import const, fourier, multilayer, descriptors
from elitho.absorber import absorber


def calc_Ax(FG: "xp.ndarray", kx0: float, ky0: float) -> "xp.ndarray":
    xp = cp.get_array_module(FG)
    Ax = xp.zeros((const.nsourceX, const.nsourceY, const.Nrange), dtype=xp.complex128)
    for ls in range(-const.lsmaxX, const.lsmaxX + 1):
        for ms in range(-const.lsmaxY, const.lsmaxY + 1):
            if (ls * const.MX / const.dx) ** 2 + (ms * const.MY / const.dy) ** 2 <= (
                const.NA / const.wavelength
            ) ** 2:
                kx = kx0 + ls * 2 * const.pi / const.dx
                ky = ky0 + ms * 2 * const.pi / const.dy
                kz = xp.sqrt(const.k**2 - kx**2 - ky**2)
                Ax0p = 1.0
                AS = xp.zeros(const.Nrange, dtype=xp.complex128)
                for i in range(const.Nrange):
                    if const.lindex[i] == ls and const.mindex[i] == ms:
                        AS[i] = 2 * kz * Ax0p
                FGA = FG @ AS
                Ax[ls + const.lsmaxX][ms + const.lsmaxY] = -FGA
                for i in range(const.Nrange):
                    if const.lindex[i] == ls and const.mindex[i] == ms:
                        Ax[ls + const.lsmaxX][ms + const.lsmaxY][i] += Ax0p

    return Ax


def diffraction_amplitude(
    polar: str,
    mask2d: "xp.ndarray",
    kx0: float,
    ky0: float,
    dod: descriptors.DiffractionOrderDescriptor,
) -> "xp.ndarray":
    xp = cp.get_array_module(mask2d)
    # --- 1. calc fourier coefficients for each layer ---
    epsN, etaN, zetaN, sigmaN = fourier.coefficients(mask2d)

    # --- 2. kxplus, kyplus, kxy2, klm
    # kxplus = kx0 + 2 * const.pi * xp.array(const.lindex) / const.dx
    # kyplus = ky0 + 2 * const.pi * xp.array(const.mindex) / const.dy
    kxplus = kx0 + 2 * const.pi * xp.array(dod.valid_x_coords) / const.dx
    kyplus = ky0 + 2 * const.pi * xp.array(dod.valid_y_coords) / const.dy
    kxy2 = kxplus**2 + kyplus**2

    # --- 3.calc absorber sequencially from the most above layer ---
    U1U, U1B = multilayer.multilayer_transfer_matrix(
        polar, const.Nrange, kxplus, kyplus, kxy2
    )

    # --- 4. calc initial B matrix ---
    if polar == "X":
        Bru = xp.diag(
            const.i_complex * const.k
            - const.i_complex / const.k / const.epsilon_ru * kxplus**2
        )
    else:
        Bru = xp.diag(
            const.i_complex * const.k
            - const.i_complex / const.k / const.epsilon_ru * kyplus**2
        )

    B = Bru
    al = xp.sqrt(const.k**2 * const.epsilon_ru - kxy2)
    br = xp.eye(const.Nrange, dtype=complex)
    for n in reversed(range(const.NABS)):
        eps, eta, zeta, sigma = epsN[n], etaN[n], zetaN[n], sigmaN[n]
        dabs = const.dabs[n]
        U1U, U1B, B, al, br = absorber(
            polar,
            kxplus,
            kyplus,
            kxy2,
            eps,
            eta,
            zeta,
            sigma,
            dabs,
            al,
            br,
            B,
            U1U,
            U1B,
        )

    # --- 5. calc Ax ---
    klm = xp.sqrt(const.k**2 - kxy2)
    al_B = al * br
    klm_B = klm[:, xp.newaxis] * br
    T0L = klm_B + al_B
    T0R = klm_B - al_B
    U0 = xp.matmul(T0L, U1U) + xp.matmul(T0R, U1B)
    # TODO: fix me ---
    U0_inv = xp.linalg.inv(U0)
    new_U1U = xp.matmul(U1U - U1B, U0_inv)
    # ----
    FG = al_B / klm[:, xp.newaxis]
    FG = xp.matmul(FG, new_U1U)
    #
    Ax = calc_Ax(FG, kx0, ky0)

    return Ax
