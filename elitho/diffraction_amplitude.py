import cupy as cp
import numpy as np
from elitho import const, fourier, multilayer, descriptors, diffraction_order
from elitho.absorber import absorber


def calc_Ax(
    FG: "xp.ndarray",
    kx0: float,
    ky0: float,
    doc: diffraction_order.DiffractionOrderCoordinate,
) -> "xp.ndarray":
    xp = cp.get_array_module(FG)
    Ax = xp.zeros(
        (const.nsourceX, const.nsourceY, doc.num_valid_diffraction_orders),
        dtype=xp.complex128,
    )
    for ls in range(-const.lsmaxX, const.lsmaxX + 1):
        for ms in range(-const.lsmaxY, const.lsmaxY + 1):
            if (ls * const.MX / const.dx) ** 2 + (ms * const.MY / const.dy) ** 2 <= (
                const.NA / const.wavelength
            ) ** 2:
                kx = kx0 + ls * 2 * const.pi / const.dx
                ky = ky0 + ms * 2 * const.pi / const.dy
                kz = xp.sqrt(const.k**2 - kx**2 - ky**2)
                Ax0p = 1.0
                AS = xp.zeros(doc.num_valid_diffraction_orders, dtype=xp.complex128)
                for i in range(doc.num_valid_diffraction_orders):
                    if doc.valid_x_coords[i] == ls and doc.valid_y_coords[i] == ms:
                        AS[i] = 2 * kz * Ax0p
                FGA = FG @ AS
                Ax[ls + const.lsmaxX][ms + const.lsmaxY] = -FGA
                for i in range(doc.num_valid_diffraction_orders):
                    if doc.valid_x_coords[i] == ls and doc.valid_y_coords[i] == ms:
                        Ax[ls + const.lsmaxX][ms + const.lsmaxY][i] += Ax0p

    return Ax


def diffraction_amplitude(
    polar: const.PolarizationDirection,
    mask2d: "xp.ndarray",
    kx0: float,
    ky0: float,
    dod: descriptors.DiffractionOrderDescriptor,
    doc: diffraction_order.DiffractionOrderCoordinate,
) -> "xp.ndarray":
    xp = cp.get_array_module(mask2d)
    # --- 1. calc fourier coefficients for each layer ---
    epsN, etaN, zetaN, sigmaN = fourier.coefficients(
        mask2d, const.absorption_amplitudes, dod
    )

    # --- 2. kxplus, kyplus, kxy2, klm
    kxplus = kx0 + 2 * const.pi * xp.array(doc.valid_x_coords) / const.dx
    kyplus = ky0 + 2 * const.pi * xp.array(doc.valid_y_coords) / const.dy
    kxy2 = kxplus**2 + kyplus**2

    # --- 3.calc absorber sequencially from the most above layer ---
    U1U, U1B = multilayer.multilayer_transfer_matrix(
        polar, doc.num_valid_diffraction_orders, kxplus, kyplus, kxy2
    )

    # --- 4. calc initial B matrix ---
    if polar == const.PolarizationDirection.X:
        Bru = xp.diag(
            const.i_complex * const.k
            - const.i_complex / const.k / const.epsilon_ru * kxplus**2
        )
    elif polar == const.PolarizationDirection.Y:
        Bru = xp.diag(
            const.i_complex * const.k
            - const.i_complex / const.k / const.epsilon_ru * kyplus**2
        )
    else:
        raise ValueError("Invalid polarization direction")

    B = Bru
    al = xp.sqrt(const.k**2 * const.epsilon_ru - kxy2)
    br = xp.eye(doc.num_valid_diffraction_orders, dtype=complex)
    # for n in reversed(range(const.NABS)):
    for eps, eta, zeta, sigma, dab in reversed(
        list(zip(epsN, etaN, zetaN, sigmaN, const.absorber_layer_thicknesses))
    ):
        U1U, U1B, B, al, br = absorber(
            polar,
            dod,
            doc,
            kxplus,
            kyplus,
            kxy2,
            eps,
            eta,
            zeta,
            sigma,
            dab,
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
    Ax = calc_Ax(FG, kx0, ky0, doc)

    return Ax


def absorber_and_vacuum_amplitudes(
    polar: const.PolarizationDirection, dod, doc
) -> tuple["xp.ndarray", "xp.ndarray"]:
    # mask with vacuum only
    mask_vacuum = np.zeros((const.NDIVX, const.NDIVY), dtype=np.float32)
    Ax_vacuum = diffraction_amplitude(
        polar, mask_vacuum, const.kx0, const.ky0, dod, doc
    )
    # mask with vacuum only
    mask_absorber = np.ones((const.NDIVX, const.NDIVY), dtype=np.float32)
    Ax_absorber = diffraction_amplitude(
        polar, mask_absorber, const.kx0, const.ky0, dod, doc
    )
    vcxx = np.zeros((const.nsourceX, const.nsourceY), dtype=np.complex128)
    abxx = np.zeros_like(vcxx)
    for ls in range(-const.lsmaxX, const.lsmaxX + 1):
        for ms in range(-const.lsmaxY, const.lsmaxY + 1):
            if ((ls * const.MX / const.dx) ** 2 + (ms * const.MY / const.dy) ** 2) <= (
                const.NA / const.wavelength
            ) ** 2:
                for i in range(doc.num_valid_diffraction_orders):
                    if doc.valid_x_coords[i] == ls and doc.valid_y_coords[i] == ms:
                        vcxx[ls + const.lsmaxX, ms + const.lsmaxY] = Ax_vacuum[
                            ls + const.lsmaxX
                        ][ms + const.lsmaxY][i]
                        abxx[ls + const.lsmaxX, ms + const.lsmaxY] = Ax_absorber[
                            ls + const.lsmaxX
                        ][ms + const.lsmaxY][i]
    return abxx, vcxx


def zero_order_amplitude(
    polar: str,
    dod,  # dod_wide
    doc,  # doc_narrow
) -> tuple[complex, complex, complex]:
    abxx, vcxx = absorber_and_vacuum_amplitudes(polar, dod, doc)
    phasexx = np.zeros((const.nsourceX, const.nsourceY), dtype=np.complex128)
    for x in range(const.nsourceX):
        for y in range(const.nsourceY):
            phasexx[x, y] = vcxx[x, y] / np.abs(vcxx[x, y])

    amp_absorber = abxx[const.lsmaxX, const.lsmaxY]
    amp_vacuum = vcxx[const.lsmaxX, const.lsmaxY]
    return amp_absorber, amp_vacuum, phasexx[const.lsmaxX, const.lsmaxY]


def mask_amplitude(fmask, abxx, vcxx):
    fampxx = np.zeros(
        (const.nsourceX, const.nsourceY, const.noutX, const.noutY), dtype=np.complex128
    )
    for is_ in range(const.nsourceX):
        for js in range(const.nsourceY):
            for ip in range(const.noutX):
                for jp in range(const.noutY):
                    kxp = 2.0 * np.pi * (ip - const.lpmaxX) / const.dx
                    kyp = 2.0 * np.pi * (jp - const.lpmaxY) / const.dy
                    phasesp = np.exp(
                        -const.i_complex
                        * (const.kx0 * kxp + kxp**2 / 2 + const.ky0 * kyp + kyp**2 / 2)
                        / (const.k * const.z0)
                    )
                    fampxx[is_, js, ip, jp] = (
                        fmask[ip, jp]
                        * phasesp
                        * (
                            abxx[const.lsmaxX, const.lsmaxY]
                            - vcxx[const.lsmaxX, const.lsmaxY]
                        )
                    )
                    if ip == const.lpmaxX and jp == const.lpmaxY:
                        fampxx[is_, js, ip, jp] += vcxx[is_, js]
    return fampxx


def extract_valid_amplitudes(doc_wide, Ax):
    ampxx = np.zeros(
        (const.nsourceX, const.nsourceY, const.noutX, const.noutY), dtype=np.complex128
    )
    for x in range(const.nsourceX):
        for y in range(const.nsourceY):
            # 照明源がNA内かチェック
            cond = (
                ((x - const.lsmaxX) * const.MX / const.dx) ** 2
                + ((y - const.lsmaxY) * const.MY / const.dy) ** 2
            ) <= (const.NA / const.wavelength) ** 2

            if not cond:
                continue

            for n in range(doc_wide.num_valid_diffraction_orders):
                ip = doc_wide.valid_x_coords[n] - (x - const.lsmaxX) + const.lpmaxX
                jp = doc_wide.valid_y_coords[n] - (y - const.lsmaxY) + const.lpmaxY
                # pupil plane 内かチェック
                if (doc_wide.valid_x_coords[n] * const.MX / const.dx) ** 2 + (
                    doc_wide.valid_y_coords[n] * const.MY / const.dy
                ) ** 2 <= (const.NA / const.wavelength) ** 2:
                    ampxx[x, y, ip, jp] = Ax[x, y, n]
    return ampxx


def extract_source_phase(fampxx):
    phasexx = np.zeros((const.nsourceX, const.nsourceY), dtype=np.complex128)
    # 複素数を正規化して位相だけに
    for x in range(const.nsourceX):
        for y in range(const.nsourceY):
            phasexx[x, y] = fampxx[x, y, const.lpmaxX, const.lpmaxY] / np.abs(
                fampxx[x, y, const.lpmaxX, const.lpmaxY]
            )
    return phasexx


def normalize_amplitude_by_source_phase(phasexx, fampxx, ampxx):
    for is_ in range(const.nsourceX):
        for js in range(const.nsourceY):
            # 現在の source の位相
            phase = phasexx[is_, js]
            for ip in range(const.noutX):
                for jp in range(const.noutY):
                    # fampxx の振幅を位相で正規化
                    fampxx[is_, js, ip, jp] /= phase
                    # 照明源が NA 内かつ pupil plane が NA 内かチェック
                    cond_source = ((is_ - const.lsmaxX) * const.MX / const.dx) ** 2 + (
                        (js - const.lsmaxY) * const.MY / const.dy
                    ) ** 2 <= (const.NA / const.wavelength) ** 2
                    cond_pupil = (
                        (ip - const.lpmaxX + is_ - const.lsmaxX) * const.MX / const.dx
                    ) ** 2 + (
                        (jp - const.lpmaxY + js - const.lsmaxY) * const.MY / const.dy
                    ) ** 2 <= (
                        const.NA / const.wavelength
                    ) ** 2

                    if cond_source and cond_pupil:
                        ampxx[is_, js, ip, jp] /= phase
    return ampxx


def diffraction_amplitude_difference(ampxx, fampxx):
    dampxx = np.zeros(
        (const.nsourceX, const.nsourceY, const.noutX, const.noutY), dtype=np.complex128
    )
    for is_ in range(const.nsourceX):
        for js in range(const.nsourceY):
            # 照明源が NA 内か確認
            cond_source = ((is_ - const.lsmaxX) * const.MX / const.dx) ** 2 + (
                (js - const.lsmaxY) * const.MY / const.dy
            ) ** 2 <= (const.NA / const.wavelength) ** 2
            if not cond_source:
                continue

            for ip in range(const.noutX):
                for jp in range(const.noutY):
                    # pupil plane が NA 内か確認
                    cond_pupil = (
                        (ip - const.lpmaxX + is_ - const.lsmaxX) * const.MX / const.dx
                    ) ** 2 + (
                        (jp - const.lpmaxY + js - const.lsmaxY) * const.MY / const.dy
                    ) ** 2 <= (
                        const.NA / const.wavelength
                    ) ** 2
                    if cond_pupil:
                        dampxx[is_, js, ip, jp] = (
                            ampxx[is_, js, ip, jp]
                            - fampxx[const.lsmaxX, const.lsmaxY, ip, jp]
                        )
    return dampxx


def compute_diffraction_difference(
    polar, mask: np.ndarray, abxx, vcxx, dod, doc
) -> np.ndarray:
    fmask = fourier.mask(mask)
    fampxx = mask_amplitude(fmask, abxx, vcxx)
    Ax = diffraction_amplitude(polar, mask, const.kx0, const.ky0, dod, doc)
    ampxx = extract_valid_amplitudes(doc, Ax)
    phasexx = extract_source_phase(fampxx)
    norm_ampxx = normalize_amplitude_by_source_phase(phasexx, fampxx, ampxx)
    dampxx = diffraction_amplitude_difference(norm_ampxx, fampxx)
    return dampxx
