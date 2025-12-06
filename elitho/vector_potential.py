import cupy as cp
import numpy as np
from elitho import config, fourier, multilayer, descriptors, diffraction_order
from elitho.absorber import absorber


def calc_Ax(
    sc: config.SimulationConfig,
    FG: "xp.ndarray",
    kx0: float,
    ky0: float,
    doc: diffraction_order.DiffractionOrderCoordinate,
) -> "xp.ndarray":
    xp = cp.get_array_module(FG)
    Ax = xp.zeros(
        (sc.nsourceX, sc.nsourceY, doc.num_valid_diffraction_orders),
        dtype=xp.complex128,
    )
    for ls in range(-sc.lsmaxX, sc.lsmaxX + 1):
        for ms in range(-sc.lsmaxY, sc.lsmaxY + 1):
            if (ls * sc.magnification_x / sc.mask_width) ** 2 + (
                ms * sc.magnification_y / sc.mask_height
            ) ** 2 <= (sc.NA / sc.wavelength) ** 2:
                kx = kx0 + ls * sc.dkx
                ky = ky0 + ms * sc.dky
                kz = xp.sqrt(sc.k**2 - kx**2 - ky**2)
                Ax0p = 1.0
                AS = xp.zeros(doc.num_valid_diffraction_orders, dtype=xp.complex128)
                for i in range(doc.num_valid_diffraction_orders):
                    if doc.valid_x_coords[i] == ls and doc.valid_y_coords[i] == ms:
                        AS[i] = 2 * kz * Ax0p

                FGA = FG @ AS
                Ax[ls + sc.lsmaxX][ms + sc.lsmaxY] = -FGA
                for i in range(doc.num_valid_diffraction_orders):
                    if doc.valid_x_coords[i] == ls and doc.valid_y_coords[i] == ms:
                        Ax[ls + sc.lsmaxX][ms + sc.lsmaxY][i] += Ax0p

    return Ax


def vector_potential(
    sc: config.SimulationConfig,
    polar: config.PolarizationDirection,
    mask2d: "xp.ndarray",
    kx0: float,
    ky0: float,
    dod: descriptors.DiffractionOrderDescriptor,
    doc: diffraction_order.DiffractionOrderCoordinate,
) -> "xp.ndarray":
    xp = cp.get_array_module(mask2d)
    # --- 1. calc fourier coefficients for each layer ---
    epsN, etaN, zetaN, sigmaN = fourier.coefficients(
        mask2d, config.absorption_amplitudes, dod
    )

    # --- 2. kxplus, kyplus, kxy2, klm
    kxplus = kx0 + sc.dkx * xp.array(doc.valid_x_coords)
    kyplus = ky0 + sc.dky * xp.array(doc.valid_y_coords)
    kxy2 = kxplus**2 + kyplus**2

    # --- 3.calc absorber sequencially from the most above layer ---
    U1U, U1B = multilayer.multilayer_transfer_matrix(
        sc.k, polar, doc.num_valid_diffraction_orders, kxplus, kyplus, kxy2
    )

    # --- 4. calc initial B matrix ---
    if polar == config.PolarizationDirection.X:
        Bru = xp.diag(1j * sc.k - 1j / sc.k / config.epsilon_ru * kxplus**2)
    elif polar == config.PolarizationDirection.Y:
        Bru = xp.diag(1j * sc.k - 1j / sc.k / config.epsilon_ru * kyplus**2)
    else:
        raise ValueError("Invalid polarization direction")

    B = Bru
    al = xp.sqrt(sc.k**2 * config.epsilon_ru - kxy2)
    br = xp.eye(doc.num_valid_diffraction_orders, dtype=complex)
    for eps, eta, zeta, sigma, dab in reversed(
        list(zip(epsN, etaN, zetaN, sigmaN, config.absorber_layer_thicknesses))
    ):
        U1U, U1B, B, al, br = absorber(
            sc.k,
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
    klm = xp.sqrt(sc.k**2 - kxy2)
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
    Ax = calc_Ax(sc, FG, kx0, ky0, doc)

    return Ax


def absorber_and_vacuum_amplitudes(
    polar: config.PolarizationDirection, dod, doc
) -> tuple["xp.ndarray", "xp.ndarray"]:
    # mask with vacuum only
    mask_vacuum = np.zeros((config.NDIVX, config.NDIVY), dtype=np.float32)
    Ax_vacuum = diffraction_amplitude(
        polar, mask_vacuum, config.kx0, config.ky0, dod, doc
    )
    # mask with vacuum only
    mask_absorber = np.ones((config.NDIVX, config.NDIVY), dtype=np.float32)
    Ax_absorber = diffraction_amplitude(
        polar, mask_absorber, config.kx0, config.ky0, dod, doc
    )
    vcxx = np.zeros((config.nsourceX, config.nsourceY), dtype=np.complex128)
    abxx = np.zeros_like(vcxx)
    for ls in range(-config.lsmaxX, config.lsmaxX + 1):
        for ms in range(-config.lsmaxY, config.lsmaxY + 1):
            if (
                (ls * config.MX / config.dx) ** 2 + (ms * config.MY / config.dy) ** 2
            ) <= (config.NA / config.wavelength) ** 2:
                for i in range(doc.num_valid_diffraction_orders):
                    if doc.valid_x_coords[i] == ls and doc.valid_y_coords[i] == ms:
                        vcxx[ls + config.lsmaxX, ms + config.lsmaxY] = Ax_vacuum[
                            ls + config.lsmaxX
                        ][ms + config.lsmaxY][i]
                        abxx[ls + config.lsmaxX, ms + config.lsmaxY] = Ax_absorber[
                            ls + config.lsmaxX
                        ][ms + config.lsmaxY][i]
    return abxx, vcxx


def zero_order_amplitude(
    polar: str,
    dod,  # dod_wide
    doc,  # doc_narrow
) -> tuple[complex, complex, complex]:
    abxx, vcxx = absorber_and_vacuum_amplitudes(polar, dod, doc)
    phasexx = np.zeros((config.nsourceX, config.nsourceY), dtype=np.complex128)
    for x in range(config.nsourceX):
        for y in range(config.nsourceY):
            phasexx[x, y] = vcxx[x, y] / np.abs(vcxx[x, y])

    amp_absorber = abxx[config.lsmaxX, config.lsmaxY]
    amp_vacuum = vcxx[config.lsmaxX, config.lsmaxY]
    return amp_absorber, amp_vacuum, phasexx[config.lsmaxX, config.lsmaxY]


def mask_amplitude(fmask, abxx, vcxx):
    fampxx = np.zeros(
        (config.nsourceX, config.nsourceY, config.noutX, config.noutY),
        dtype=np.complex128,
    )
    for is_ in range(config.nsourceX):
        for js in range(config.nsourceY):
            for ip in range(config.noutX):
                for jp in range(config.noutY):
                    kxp = 2.0 * np.pi * (ip - config.lpmaxX) / config.dx
                    kyp = 2.0 * np.pi * (jp - config.lpmaxY) / config.dy
                    phasesp = np.exp(
                        -1j
                        * (
                            config.kx0 * kxp
                            + kxp**2 / 2
                            + config.ky0 * kyp
                            + kyp**2 / 2
                        )
                        / (config.k * config.z0)
                    )
                    fampxx[is_, js, ip, jp] = (
                        fmask[ip, jp]
                        * phasesp
                        * (
                            abxx[config.lsmaxX, config.lsmaxY]
                            - vcxx[config.lsmaxX, config.lsmaxY]
                        )
                    )
                    if ip == config.lpmaxX and jp == config.lpmaxY:
                        fampxx[is_, js, ip, jp] += vcxx[is_, js]
    return fampxx


def extract_valid_amplitudes(doc_wide, Ax):
    ampxx = np.zeros(
        (config.nsourceX, config.nsourceY, config.noutX, config.noutY),
        dtype=np.complex128,
    )
    for x in range(config.nsourceX):
        for y in range(config.nsourceY):
            # 照明源がNA内かチェック
            cond = (
                ((x - config.lsmaxX) * config.MX / config.dx) ** 2
                + ((y - config.lsmaxY) * config.MY / config.dy) ** 2
            ) <= (config.NA / config.wavelength) ** 2

            if not cond:
                continue

            for n in range(doc_wide.num_valid_diffraction_orders):
                ip = doc_wide.valid_x_coords[n] - (x - config.lsmaxX) + config.lpmaxX
                jp = doc_wide.valid_y_coords[n] - (y - config.lsmaxY) + config.lpmaxY
                # pupil plane 内かチェック
                if (doc_wide.valid_x_coords[n] * config.MX / config.dx) ** 2 + (
                    doc_wide.valid_y_coords[n] * config.MY / config.dy
                ) ** 2 <= (config.NA / config.wavelength) ** 2:
                    ampxx[x, y, ip, jp] = Ax[x, y, n]
    return ampxx


def extract_source_phase(fampxx):
    phasexx = np.zeros((config.nsourceX, config.nsourceY), dtype=np.complex128)
    # 複素数を正規化して位相だけに
    for x in range(config.nsourceX):
        for y in range(config.nsourceY):
            phasexx[x, y] = fampxx[x, y, config.lpmaxX, config.lpmaxY] / np.abs(
                fampxx[x, y, config.lpmaxX, config.lpmaxY]
            )
    return phasexx


def normalize_amplitude_by_source_phase(phasexx, fampxx, ampxx):
    for is_ in range(config.nsourceX):
        for js in range(config.nsourceY):
            # 現在の source の位相
            phase = phasexx[is_, js]
            for ip in range(config.noutX):
                for jp in range(config.noutY):
                    # fampxx の振幅を位相で正規化
                    fampxx[is_, js, ip, jp] /= phase
                    # 照明源が NA 内かつ pupil plane が NA 内かチェック
                    cond_source = (
                        (is_ - config.lsmaxX) * config.MX / config.dx
                    ) ** 2 + ((js - config.lsmaxY) * config.MY / config.dy) ** 2 <= (
                        config.NA / config.wavelength
                    ) ** 2
                    cond_pupil = (
                        (ip - config.lpmaxX + is_ - config.lsmaxX)
                        * config.MX
                        / config.dx
                    ) ** 2 + (
                        (jp - config.lpmaxY + js - config.lsmaxY)
                        * config.MY
                        / config.dy
                    ) ** 2 <= (
                        config.NA / config.wavelength
                    ) ** 2

                    if cond_source and cond_pupil:
                        ampxx[is_, js, ip, jp] /= phase
    return ampxx


def diffraction_amplitude_difference(ampxx, fampxx):
    dampxx = np.zeros(
        (config.nsourceX, config.nsourceY, config.noutX, config.noutY),
        dtype=np.complex128,
    )
    for is_ in range(config.nsourceX):
        for js in range(config.nsourceY):
            # 照明源が NA 内か確認
            cond_source = ((is_ - config.lsmaxX) * config.MX / config.dx) ** 2 + (
                (js - config.lsmaxY) * config.MY / config.dy
            ) ** 2 <= (config.NA / config.wavelength) ** 2
            if not cond_source:
                continue

            for ip in range(config.noutX):
                for jp in range(config.noutY):
                    # pupil plane が NA 内か確認
                    cond_pupil = (
                        (ip - config.lpmaxX + is_ - config.lsmaxX)
                        * config.MX
                        / config.dx
                    ) ** 2 + (
                        (jp - config.lpmaxY + js - config.lsmaxY)
                        * config.MY
                        / config.dy
                    ) ** 2 <= (
                        config.NA / config.wavelength
                    ) ** 2
                    if cond_pupil:
                        dampxx[is_, js, ip, jp] = (
                            ampxx[is_, js, ip, jp]
                            - fampxx[config.lsmaxX, config.lsmaxY, ip, jp]
                        )
    return dampxx


def compute_diffraction_difference(
    polar, mask: np.ndarray, abxx, vcxx, dod, doc
) -> np.ndarray:
    fmask = fourier.mask(mask)
    fampxx = mask_amplitude(fmask, abxx, vcxx)
    Ax = diffraction_amplitude(polar, mask, config.kx0, config.ky0, dod, doc)
    ampxx = extract_valid_amplitudes(doc, Ax)
    phasexx = extract_source_phase(fampxx)
    norm_ampxx = normalize_amplitude_by_source_phase(phasexx, fampxx, ampxx)
    dampxx = diffraction_amplitude_difference(norm_ampxx, fampxx)
    return dampxx
