import numpy as np
from src import const, fourier, multilayer
from src.absorber import absorber


def diffraction_amplitude(
    polar: str, mask2d: np.ndarray, kx0: float, ky0: float
) -> np.ndarray:
    # --- 1. calc fourier coefficients for each layer ---
    epsN, etaN, zetaN, sigmaN = fourier.coefficients(mask2d)

    # --- 2. kxplus, kyplus, kxy2, klm
    kxplus = kx0 + 2 * const.pi * np.array(const.lindex) / const.dx
    kyplus = ky0 + 2 * const.pi * np.array(const.mindex) / const.dy
    kxy2 = kxplus**2 + kyplus**2
    klm = np.sqrt(const.k**2 - kxy2)

    # --- 3. calc initial B matrix ---
    if polar == "X":
        Bru = np.diag(
            const.i_complex * const.k
            - const.i_complex / const.k / const.epsilon_ru * kxplus**2
        )
    else:
        Bru = np.diag(
            const.i_complex * const.k
            - const.i_complex / const.k / const.epsilon_ru * kyplus**2
        )

    # --- 4.calc absorber sequencially from the most above layer ---
    U1U, U1B = multilayer.multilayer_transfer_matrix(
        polar, const.Nrange, kxplus, kyplus, kxy2
    )
    B = Bru
    al = np.sqrt(const.k**2 * const.epsilon_ru - kxy2)
    br = np.eye(const.Nrange, dtype=complex)
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
    tmp = klm / (const.i_complex * const.k - const.i_complex / const.k * kxplus**2)
    tmp_B = tmp[:, np.newaxis] * B
    al_br = br * al[np.newaxis, :]

    # T0L, T0R を計算
    T0L = tmp_B + al_br
    T0R = tmp_B - al_br

    # calc U0
    U0 = np.matmul(T0L, U1U) + np.matmul(T0R, U1B)
    U0I = np.linalg.inv(U0)
    U1U = np.matmul(U1U - U1B, U0I)
    FG = al_br / klm[:, np.newaxis]
    FG = np.matmul(FG, U1U)

    Ax = np.zeros((const.nsourceX, const.nsourceY, const.Nrange), dtype=np.complex128)
    for ls in range(-const.lsmaxX, const.lsmaxX + 1):
        for ms in range(-const.lsmaxY, const.lsmaxY + 1):
            if (ls * const.MX / const.dx) ** 2 + (ms * const.MY / const.dy) ** 2 <= (
                const.NA / const.wavelength
            ) ** 2:
                kx = kx0 + ls * 2 * np.pi / const.dx
                ky = ky0 + ms * 2 * np.pi / const.dy
                kz = np.sqrt(const.k**2 - kx**2 - ky**2)
                Ax0p = 1.0
                AS = np.zeros(const.Nrange, dtype=complex)
                for i in range(const.Nrange):
                    if const.lindex[i] == ls and const.mindex[i] == ms:
                        AS[i] = 2 * kz * Ax0p
                FGA = FG @ AS
                Ax[ls + const.lsmaxX][ms + const.lsmaxY] = -FGA
                for i in range(const.Nrange):
                    if const.lindex[i] == ls and const.mindex[i] == ms:
                        Ax[ls + const.lsmaxX][ms + const.lsmaxY][i] += Ax0p

    return Ax
