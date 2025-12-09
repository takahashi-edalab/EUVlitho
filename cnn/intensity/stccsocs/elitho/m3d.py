import numpy as np
from elitho import const


def m3d_params(pupil_coords, dampxx):
    a0xx = np.zeros((const.noutX, const.noutY), dtype=np.complex128)
    axxx = np.zeros((const.noutX, const.noutY), dtype=np.complex128)
    ayxx = np.zeros((const.noutX, const.noutY), dtype=np.complex128)

    # ninput個の入力点ループ
    for n in range(pupil_coords.n_coordinates):
        lp = pupil_coords.linput[n]
        mp = pupil_coords.minput[n]
        ip = pupil_coords.linput[n] + const.lpmaxX
        jp = pupil_coords.minput[n] + const.lpmaxY

        # 集計変数の初期化
        xxf1 = 0 + 0j
        xxfx = 0 + 0j
        xxfy = 0 + 0j
        c1 = 0 + 0j
        cx = 0 + 0j
        cy = 0 + 0j
        cx2 = 0 + 0j
        cy2 = 0 + 0j
        cxy = 0 + 0j

        # 光源ループ
        for is_ in range(const.nsourceX):
            for js in range(const.nsourceY):
                ls = is_ - const.lsmaxX + lp / 2.0
                ms = js - const.lsmaxY + mp / 2.0

                # 条件: source と pupil plane が NA 内
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
                    f = dampxx[is_, js, ip, jp]
                    xxf1 += f
                    xxfx += ls * f
                    xxfy += ms * f
                    c1 += 1
                    cx += ls
                    cy += ms
                    cx2 += ls**2
                    cy2 += ms**2
                    cxy += ls * ms

        # xinput[n] に応じて出力計算
        if pupil_coords.xinput[n] >= 8:
            cd = (
                c1 * cx2 * cy2
                + 2.0 * cx * cxy * cy
                - c1 * cxy**2
                - cx2 * cy**2
                - cy2 * cx**2
            )
            a0xx[ip, jp] = (
                (cx2 * cy2 - cxy**2) * xxf1
                + (-cx * cy2 + cy * cxy) * xxfx
                + (cx * cxy - cy * cx2) * xxfy
            ) / cd
            axxx[ip, jp] = (
                (-cx * cy2 + cy * cxy) * xxf1
                + (c1 * cy2 - cy**2) * xxfx
                + (-c1 * cxy + cx * cy) * xxfy
            ) / cd
            ayxx[ip, jp] = (
                (cx * cxy - cy * cx2) * xxf1
                + (-c1 * cxy + cx * cy) * xxfx
                + (c1 * cx2 - cx**2) * xxfy
            ) / cd
        else:
            a0xx[ip, jp] = xxf1 / c1
            axxx[ip, jp] = 0.0
            ayxx[ip, jp] = 0.0

    return a0xx, axxx, ayxx


def m3d_from_mask(mask: np.ndarray) -> np.ndarray:
    from elitho import diffraction_amplitude, const, descriptors, diffraction_order
    from elitho.pupil import PupilCoordinates

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
    pupil_coords = PupilCoordinates(doc_wide.num_valid_diffraction_orders)
    abxx, vcxx = diffraction_amplitude.absorber_and_vacuum_amplitudes(
        const.PolarizationDirection.X, dod_wide, doc_narrow
    )
    dampxx = diffraction_amplitude.compute_diffraction_difference(
        const.PolarizationDirection.X, mask, abxx, vcxx, dod_wide, doc_wide
    )
    a0xx, axxx, ayxx = m3d_params(pupil_coords, dampxx)
    return a0xx, axxx, ayxx
