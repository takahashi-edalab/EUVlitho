import numpy as np
from elitho import (
    diffraction_amplitude,
    const,
    descriptors,
    diffraction_order,
    source,
    pupil,
)


def tcc_matrices(
    pupil_coords: pupil.PupilCoordinates,
    uniform_k_source,
) -> dict[str, np.ndarray]:
    TCC = dict(
        XS0=np.zeros(
            (pupil_coords.n_coordinates, pupil_coords.n_coordinates),
            dtype=np.complex128,
        ),
        XSX=np.zeros(
            (pupil_coords.n_coordinates, pupil_coords.n_coordinates),
            dtype=np.complex128,
        ),
        XSY=np.zeros(
            (pupil_coords.n_coordinates, pupil_coords.n_coordinates),
            dtype=np.complex128,
        ),
        YS0=np.zeros(
            (pupil_coords.n_coordinates, pupil_coords.n_coordinates),
            dtype=np.complex128,
        ),
        YSX=np.zeros(
            (pupil_coords.n_coordinates, pupil_coords.n_coordinates),
            dtype=np.complex128,
        ),
        YSY=np.zeros(
            (pupil_coords.n_coordinates, pupil_coords.n_coordinates),
            dtype=np.complex128,
        ),
    )
    pmax = (const.k * const.NA) ** 2
    for i in range(pupil_coords.n_coordinates):
        kx = 2 * np.pi / const.dx * pupil_coords.linput[i]
        ky = 2 * np.pi / const.dy * pupil_coords.minput[i]

        for j in range(i + 1):  # j <= i
            kxp = 2 * np.pi / const.dx * pupil_coords.linput[j]
            kyp = 2 * np.pi / const.dy * pupil_coords.minput[j]

            sumx_s0 = 0 + 0j
            sumy_s0 = 0 + 0j
            sumx_sx = 0 + 0j
            sumy_sx = 0 + 0j
            sumx_sy = 0 + 0j
            sumy_sy = 0 + 0j

            for sx, sy in uniform_k_source:
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

            TCC["XS0"][i, j] = sumx_s0 / uniform_k_source.SDIV
            TCC["XS0"][j, i] = TCC["XS0"][i, j].conjugate()
            TCC["XSX"][i, j] = sumx_sx / uniform_k_source.SDIV
            TCC["XSX"][j, i] = TCC["XSX"][i, j].conjugate()
            TCC["XSY"][i, j] = sumx_sy / uniform_k_source.SDIV
            TCC["XSY"][j, i] = TCC["XSY"][i, j].conjugate()
            TCC["YS0"][i, j] = sumy_s0 / uniform_k_source.SDIV
            TCC["YS0"][j, i] = TCC["YS0"][i, j].conjugate()
            TCC["YSX"][i, j] = sumy_sx / uniform_k_source.SDIV
            TCC["YSX"][j, i] = TCC["YSX"][i, j].conjugate()
            TCC["YSY"][i, j] = sumy_sy / uniform_k_source.SDIV
            TCC["YSY"][j, i] = TCC["YSY"][i, j].conjugate()
    return TCC


def shift_center(fft_mask):
    fmask = np.zeros((const.noutX, const.noutY), dtype=np.complex128)
    # NOTE: NDIVX, NDIVYはもともとFDIVX, FDIVYという名前
    for i in range(const.noutX):
        l = (i - const.lpmaxX + const.NDIVX) % const.NDIVX
        for j in range(const.noutY):
            m = (j - const.lpmaxY + const.NDIVY) % const.NDIVY
            fmask[i, j] = fft_mask[l, m]
    return fmask


def propagation(fmask):
    fampxx = np.zeros((const.noutX, const.noutY), dtype=np.complex128)
    for ip in range(const.noutX):
        for jp in range(const.noutY):
            kxp = 2.0 * np.pi * (ip - const.lpmaxX) / const.dx
            kyp = 2.0 * np.pi * (jp - const.lpmaxY) / const.dy
            phasesp = np.exp(
                -const.i_complex
                * (const.kx0 * kxp + kxp**2 / 2 + const.ky0 * kyp + kyp**2 / 2)
                / (const.k * const.z0)
            )
            fampxx[ip, jp] = fmask[ip, jp] * phasesp
    return fampxx


def calc_xpolar_field_terms(Ax, dxAx, dyAx, kxplus, kyplus, kzplus, lp, mp):
    zi = 1j
    Exs0 = zi * const.k * Ax - zi / const.k * kxplus * kxplus * Ax
    Exsx = (
        -2 * zi / const.k * kxplus * Ax
        + zi * const.k * dxAx
        - zi / const.k * kxplus * kxplus * dxAx
    )
    Exsy = zi * const.k * dyAx - zi / const.k * kxplus * kxplus * dyAx
    Exsxy = (
        Exsx / (const.dx / (2 * np.pi)) * lp / 2
        + Exsy / (const.dy / (2 * np.pi)) * mp / 2
    )

    Eys0 = -zi / const.k * kxplus * kyplus * Ax
    Eysx = -zi / const.k * kyplus * Ax - zi / const.k * kxplus * kyplus * dxAx
    Eysy = -zi / const.k * kxplus * Ax - zi / const.k * kxplus * kyplus * dyAx
    Eysxy = (
        Eysx / (const.dx / (2 * np.pi)) * lp / 2
        + Eysy / (const.dy / (2 * np.pi)) * mp / 2
    )

    Ezs0 = -zi / const.k * kxplus * kzplus * Ax
    Ezsx = -zi / const.k * kzplus * Ax - zi / const.k * kxplus * kzplus * dxAx
    Ezsy = -zi / const.k * kxplus * kzplus * dyAx
    Ezsxy = (
        Ezsx / (const.dx / (2 * np.pi)) * lp / 2
        + Ezsy / (const.dy / (2 * np.pi)) * mp / 2
    )
    return Exs0, Exsx, Exsy, Exsxy, Eys0, Eysx, Eysy, Eysxy, Ezs0, Ezsx, Ezsy, Ezsxy


def calc_ypolar_field_terms():
    pass


def electric_field(
    polar: const.PolarizationDirection,
    pupil_coords: pupil.PupilCoordinates,
    fampxx: np.ndarray,
    a0xx: np.ndarray = None,
    axxx: np.ndarray = None,
    ayxx: np.ndarray = None,
) -> dict[str, np.ndarray]:

    # 複素単位
    zi = 1j

    ef = dict(
        Exs0=np.zeros(pupil_coords.n_coordinates, dtype=complex),
        Eys0=np.zeros(pupil_coords.n_coordinates, dtype=complex),
        Ezs0=np.zeros(pupil_coords.n_coordinates, dtype=complex),
        Exsx=np.zeros(pupil_coords.n_coordinates, dtype=complex),
        Eysx=np.zeros(pupil_coords.n_coordinates, dtype=complex),
        Ezsx=np.zeros(pupil_coords.n_coordinates, dtype=complex),
        Exsy=np.zeros(pupil_coords.n_coordinates, dtype=complex),
        Eysy=np.zeros(pupil_coords.n_coordinates, dtype=complex),
        Ezsy=np.zeros(pupil_coords.n_coordinates, dtype=complex),
        Exsxy=np.zeros(pupil_coords.n_coordinates, dtype=complex),
        Eysxy=np.zeros(pupil_coords.n_coordinates, dtype=complex),
        Ezsxy=np.zeros(pupil_coords.n_coordinates, dtype=complex),
    )

    # 各空間周波数成分ごとに電場成分を計算
    for i in range(pupil_coords.n_coordinates):
        # --- 1) 波数ベクトル(kx, ky, kz) 計算 ---
        kxplus = const.kx0 + 2 * np.pi * pupil_coords.linput[i] / const.dx / 2.0
        kyplus = const.ky0 + 2 * np.pi * pupil_coords.minput[i] / const.dy / 2.0

        # kzはEvanescentを含むので負の平方根
        kzplus = -np.sqrt(const.k * const.k - kxplus * kxplus - kyplus * kyplus)

        # 周波数インデックス (wrap)
        ip = pupil_coords.linput[i] + const.lpmaxX
        jp = pupil_coords.minput[i] + const.lpmaxY
        lp = ip - const.lpmaxX
        mp = jp - const.lpmaxY

        # 電場スペクトル Ax or Ay (偏光成分)
        if polar == const.PolarizationDirection.X:  # X偏光
            if a0xx is None or axxx is None or ayxx is None:
                Ax = fampxx[ip, jp]
                dxAx = 0.0
                dyAx = 0.0
            elif a0xx is not None and axxx is not None and ayxx is not None:
                Ax = fampxx[ip, jp] + a0xx[ip, jp]
                dxAx = axxx[ip, jp] * const.dx / (2 * np.pi)
                dyAx = ayxx[ip, jp] * const.dy / (2 * np.pi)
            else:
                raise ValueError("a0xx, axxx, ayxx must be all None or all not None")

            (
                ef["Exs0"][i],
                ef["Exsx"][i],
                ef["Exsy"][i],
                ef["Exsxy"][i],
                ef["Eys0"][i],
                ef["Eysx"][i],
                ef["Eysy"][i],
                ef["Eysxy"][i],
                ef["Ezs0"][i],
                ef["Ezsx"][i],
                ef["Ezsy"][i],
                ef["Ezsxy"][i],
            ) = calc_xpolar_field_terms(Ax, dxAx, dyAx, kxplus, kyplus, kzplus, lp, mp)

        elif polar == const.PolarizationDirection.Y:  # Y偏光
            if type == 1:
                Ay = fampxx[ip, jp]
                dxAy = 0.0
                dyAy = 0.0
            else:
                Ay = fampxx[ip, jp] + a0xx[ip, jp]
                dxAy = axxx[ip, jp] * const.dx / (2 * np.pi)
                dyAy = ayxx[ip, jp] * const.dy / (2 * np.pi)

            ef["Exs0"][i] = -zi / const.k * kxplus * kyplus * Ay
            ef["Exsx"][i] = (
                -zi / const.k * kyplus * Ay - zi / const.k * kxplus * kyplus * dxAy
            )
            ef["Exsy"][i] = (
                -zi / const.k * kxplus * Ay - zi / const.k * kxplus * kyplus * dyAy
            )
            ef["Exsxy"][i] = (
                ef["Exsx"][i] / (const.dx / (2 * np.pi)) * lp / 2
                + ef["Exsy"][i] / (const.dy / (2 * np.pi)) * mp / 2
            )

            ef["Eys0"][i] = zi / const.k * Ay - zi / const.k * kyplus * kyplus * Ay
            ef["Eysx"][i] = zi / const.k * dxAy - zi / const.k * kyplus * kyplus * dxAy
            ef["Eysy"][i] = (
                -2 * zi / const.k * kyplus * Ay
                + zi / const.k * dyAy
                - zi / const.k * kyplus * kyplus * dyAy
            )
            ef["Eysxy"][i] = (
                ef["Eysx"][i] / (const.dx / (2 * np.pi)) * lp / 2
                + ef["Eysy"][i] / (const.dy / (2 * np.pi)) * mp / 2
            )

            ef["Ezs0"][i] = -zi / const.k * kyplus * kzplus * Ay
            ef["Ezsx"][i] = -zi / const.k * kyplus * kzplus * dxAy
            ef["Ezsy"][i] = (
                -zi / const.k * kzplus * Ay - zi / const.k * kyplus * kzplus * dyAy
            )
            ef["Ezsxy"][i] = (
                ef["Ezsx"][i] / (const.dx / (2 * np.pi)) * lp / 2
                + ef["Ezsy"][i] / (const.dy / (2 * np.pi)) * mp / 2
            )
        else:
            raise ValueError("polar must be 'X' or 'Y'")

    return ef


def calc_tccee(
    polar: const.PolarizationDirection,
    pupil_coords: pupil.PupilCoordinates,
    ef: dict[str, np.ndarray],
    TCC: dict[str, np.ndarray],
) -> np.ndarray:
    # TCCEE initialization
    TCCEE = np.zeros((const.XDIV, const.XDIV), dtype=np.complex128)
    if polar == const.PolarizationDirection.X:
        for i in range(pupil_coords.n_coordinates):
            ix = pupil_coords.linput[i]
            iy = pupil_coords.minput[i]
            for j in range(pupil_coords.n_coordinates):
                jx = pupil_coords.linput[j]
                jy = pupil_coords.minput[j]

                px = (ix - jx + const.XDIV) % const.XDIV
                py = (iy - jy + const.XDIV) % const.XDIV

                TCCEE[px, py] += TCC["XS0"][i, j] * (
                    ef["Exs0"][i] * np.conj(ef["Exs0"][j])
                    + ef["Eys0"][i] * np.conj(ef["Eys0"][j])
                    + ef["Ezs0"][i] * np.conj(ef["Ezs0"][j])
                )

                TCCEE[px, py] += (
                    2.0
                    * TCC["XSX"][i, j]
                    * (
                        ef["Exs0"][i] * np.conj(ef["Exsx"][j])
                        + ef["Eys0"][i] * np.conj(ef["Eysx"][j])
                        + ef["Ezs0"][i] * np.conj(ef["Ezsx"][j])
                    )
                )

                TCCEE[px, py] += (
                    2.0
                    * TCC["XSY"][i, j]
                    * (
                        ef["Exs0"][i] * np.conj(ef["Exsy"][j])
                        + ef["Eys0"][i] * np.conj(ef["Eysy"][j])
                        + ef["Ezs0"][i] * np.conj(ef["Ezsy"][j])
                    )
                )

                TCCEE[px, py] += (
                    2.0
                    * TCC["XS0"][i, j]
                    * (
                        ef["Exs0"][i] * np.conj(ef["Exsxy"][j])
                        + ef["Eys0"][i] * np.conj(ef["Eysxy"][j])
                        + ef["Ezs0"][i] * np.conj(ef["Ezsxy"][j])
                    )
                )

    elif polar == const.PolarizationDirection.Y:
        for i in range(pupil_coords.n_coordinates):
            ix = pupil_coords.linput[i]
            iy = pupil_coords.minput[i]
            for j in range(pupil_coords.n_coordinates):
                jx = pupil_coords.linput[j]
                jy = pupil_coords.minput[j]

                px = (ix - jx + const.XDIV) % const.XDIV
                py = (iy - jy + const.XDIV) % const.XDIV

                TCCEE[px, py] += TCC["YS0"][i, j] * (
                    ef["Exs0"][i] * np.conj(ef["Exs0"][j])
                    + ef["Eys0"][i] * np.conj(ef["Eys0"][j])
                    + ef["Ezs0"][i] * np.conj(ef["Ezs0"][j])
                )

                TCCEE[px, py] += (
                    2.0
                    * TCC["YSX"][i, j]
                    * (
                        ef["Exs0"][i] * np.conj(ef["Exsx"][j])
                        + ef["Eys0"][i] * np.conj(ef["Eysx"][j])
                        + ef["Ezs0"][i] * np.conj(ef["Ezsx"][j])
                    )
                )

                TCCEE[px, py] += (
                    2.0
                    * TCC["YSY"][i, j]
                    * (
                        ef["Exs0"][i] * np.conj(ef["Exsy"][j])
                        + ef["Eys0"][i] * np.conj(ef["Eysy"][j])
                        + ef["Ezs0"][i] * np.conj(ef["Ezsy"][j])
                    )
                )

                TCCEE[px, py] += (
                    2.0
                    * TCC["YS0"][i, j]
                    * (
                        ef["Exs0"][i] * np.conj(ef["Exsxy"][j])
                        + ef["Eys0"][i] * np.conj(ef["Eysxy"][j])
                        + ef["Ezs0"][i] * np.conj(ef["Ezsxy"][j])
                    )
                )
    else:
        raise ValueError("polar must be 'X' or 'Y'")

    return TCCEE


def intensity(
    mask: np.ndarray,
    polar: const.PolarizationDirection,
    a0xx: np.ndarray = None,
    axxx: np.ndarray = None,
    ayxx: np.ndarray = None,
) -> np.ndarray:

    dod_narrow = descriptors.DiffractionOrderDescriptor(1.5)
    dod_wide = descriptors.DiffractionOrderDescriptor(6.0)
    doc_narrow = diffraction_order.DiffractionOrderCoordinate(
        dod_narrow.max_diffraction_order_x,
        dod_narrow.max_diffraction_order_y,
        diffraction_order.ellipse,
    )
    doc_wide = diffraction_order.DiffractionOrderCoordinate(
        dod_wide.max_diffraction_order_x,
        dod_wide.max_diffraction_order_y,
        diffraction_order.rounded_diamond,
    )

    pupil_coords = pupil.PupilCoordinates(doc_wide.num_valid_diffraction_orders)
    uniform_k_source = source.UniformKSource()
    amp_absorber, amp_vacuum, phasexx = diffraction_amplitude.zero_order_amplitude(
        const.PolarizationDirection.X, dod_wide, doc_narrow
    )

    hfpattern = mask * (amp_absorber - amp_vacuum) + amp_vacuum
    fft_mask = np.fft.fft2(hfpattern, norm="forward")
    shifted_fft_mask = shift_center(fft_mask)
    fampxx = propagation(shifted_fft_mask)
    fampxx /= phasexx

    ef = electric_field(polar, pupil_coords, fampxx, a0xx, axxx, ayxx)
    TCC = tcc_matrices(pupil_coords, uniform_k_source)
    TCCEE = calc_tccee(polar, pupil_coords, ef, TCC)
    ifft_tccee = np.fft.ifft2(TCCEE, norm="forward")
    return ifft_tccee.real
