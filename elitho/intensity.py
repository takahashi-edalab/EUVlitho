import numpy as np
from elitho import const, pupil, descriptors, diffraction_order
from elitho.diffraction_amplitude import diffraction_amplitude
from elitho.source import abbe_source
from elitho.electro_field import electro_field


def na_filter_amplitude_map(
    Ax: np.ndarray, doc: diffraction_order.DiffractionOrderCoordinate
) -> np.ndarray:
    ampxx = np.full(
        (const.nsourceXL, const.nsourceYL, const.noutXL, const.noutYL),
        -1000 + 0j,
        dtype=np.complex128,
    )
    for x in range(const.nsourceXL):
        for y in range(const.nsourceYL):
            cond = (
                ((x - const.lsmaxX) * const.MX / const.dx) ** 2
                + ((y - const.lsmaxY) * const.MY / const.dy) ** 2
            ) <= (const.NA / const.wavelength) ** 2

            if cond:
                for n in range(doc.num_valid_diffraction_orders):
                    ip = doc.valid_x_coords[n] - (x - const.lsmaxX) + const.lpmaxX
                    jp = doc.valid_y_coords[n] - (y - const.lsmaxY) + const.lpmaxY
                    if 0 <= ip < const.noutXL and 0 <= jp < const.noutYL:
                        ampxx[x, y, ip, jp] = Ax[x, y, n]
    return ampxx


def intensity_by_abbe_source(
    ncut: int,
    Ex0m: np.ndarray,
    Ey0m: np.ndarray,
    Ez0m: np.ndarray,
    offset_x: float,
    offset_y: float,
    linput: np.ndarray,
    minput: np.ndarray,
) -> np.ndarray:
    # ---- FFT & isum更新 ----
    fnx = np.zeros((const.XDIV, const.XDIV), dtype=complex)
    fny = np.zeros_like(fnx)
    fnz = np.zeros_like(fnx)
    for n in range(ncut):
        kxn = offset_x + 2.0 * const.pi * linput[n] / const.dx
        kyn = offset_y + 2.0 * const.pi * minput[n] / const.dy

        if (const.MX**2 * kxn**2 + const.MY**2 * kyn**2) <= (const.NA * const.k) ** 2:
            # Calculate phase
            phase = np.exp(
                1j
                * ((kxn + const.kx0) ** 2 + (kyn + const.ky0) ** 2)
                / 2.0
                / const.k
                * const.z0
                + 1j
                * (const.MX**2 * kxn**2 + const.MY**2 * kyn**2)
                / 2.0
                / const.k
                * const.z
            )
            # Calculate electric field components
            fx = Ex0m[n] * phase
            fy = Ey0m[n] * phase
            fz = Ez0m[n] * phase
            # Map to FFT grid
            ix = linput[n]
            iy = minput[n]
            px = (ix + const.XDIV) % const.XDIV
            py = (iy + const.YDIV) % const.YDIV
            fnx[px, py] = fx
            fny[px, py] = fy
            fnz[px, py] = fz

    # Inverse FFT without scaling
    fnx_ifft = np.fft.ifft2(fnx, norm="forward")
    fny_ifft = np.fft.ifft2(fny, norm="forward")
    fnz_ifft = np.fft.ifft2(fnz, norm="forward")
    # Calculate intensity
    intensity = np.abs(fnx_ifft) ** 2 + np.abs(fny_ifft) ** 2 + np.abs(fnz_ifft) ** 2
    return intensity


def intensity(
    mask2d: np.ndarray,
    polar: const.PolarizationDirection = const.PolarizationDirection.X,
    is_high_na: bool = False,
) -> np.ndarray:
    l0s, m0s, SDIV = abbe_source()
    # SDIVMAX = np.max(SDIV)
    SDIVSUM = np.sum(SDIV)

    dod = descriptors.DiffractionOrderDescriptor(6.0)
    doc = diffraction_order.DiffractionOrderCoordinate(
        dod.max_diffraction_order_x,
        dod.max_diffraction_order_y,
        diffraction_order.rounded_diamond,
    )

    linput, minput, _, ninput = pupil.find_valid_pupil_points(
        doc.num_valid_diffraction_orders
    )
    intensity_total = np.zeros((const.XDIV, const.XDIV))
    for nsx in range(const.ndivs):
        for nsy in range(const.ndivs):
            sx0 = 2.0 * const.pi / const.dx * nsx / const.ndivs + const.kx0
            sy0 = 2.0 * const.pi / const.dy * nsy / const.ndivs + const.ky0
            Ax = diffraction_amplitude(polar, mask2d, sx0, sy0, dod, doc)
            ampxx = na_filter_amplitude_map(Ax, doc)
            Ex0m, Ey0m, Ez0m = electro_field(
                polar,
                is_high_na,
                nsx,
                nsy,
                SDIV[nsx, nsy],
                l0s[nsx][nsy],
                m0s[nsx][nsy],
                ninput,
                sx0,
                sy0,
                linput,
                minput,
                ampxx,
            )

            for isd in range(SDIV[nsx, nsy]):
                offset_x = (
                    2.0 * const.pi / const.dx * nsx / const.ndivs
                    + 2.0 * const.pi / const.dx * l0s[nsx][nsy][isd]
                )
                offset_y = (
                    2.0 * const.pi / const.dy * nsy / const.ndivs
                    + 2.0 * const.pi / const.dy * m0s[nsx][nsy][isd]
                )

                intensity_by_a_source = intensity_by_abbe_source(
                    ninput,
                    Ex0m[isd],
                    Ey0m[isd],
                    Ez0m[isd],
                    offset_x,
                    offset_y,
                    linput,
                    minput,
                )
                intensity_total += intensity_by_a_source

    intensity_map = intensity_total / SDIVSUM
    return intensity_map


def stcc_intensity(
    mask: np.ndarray, polar: const.PolarizationDirection = const.PolarizationDirection.X
) -> np.ndarray:
    from elitho import (
        diffraction_amplitude,
        descriptors,
        diffraction_order,
        source,
    )
    from elitho.pupil import find_valid_pupil_points

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
    linput, minput, xinput, n_input = find_valid_pupil_points(
        doc_wide.num_valid_diffraction_orders
    )

    dkx, dky, SDIV = source.uniform_k_source()
    amp_absorber, amp_vacuum, phasexx = diffraction_amplitude.zero_order_amplitude(
        polar, dod_wide, doc_narrow
    )

    hfpattern = mask * (amp_absorber - amp_vacuum) + amp_vacuum
    # fft with scaling
    fft_mask = np.fft.fft2(hfpattern, norm="forward")
    # reshaep
    fmask = np.zeros((const.noutX, const.noutY), dtype=np.complex128)
    for i in range(const.noutX):
        l = (i - const.lpmaxX + const.NDIVX) % const.NDIVX
        for j in range(const.noutY):
            m = (j - const.lpmaxY + const.NDIVY) % const.NDIVY
            fmask[i, j] = fft_mask[l, m]

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

    fampxx /= phasexx

    intensity = np.zeros((const.XDIV, const.XDIV))
    return intensity


def main():
    import time
    from elitho import use_backend, get_backend
    from elitho.mask_pattern import LinePattern

    mask = LinePattern(cd=56, gap=80, direction="V", field_type="DF")(
        const.NDIVX, const.NDIVY
    )
    print(mask.shape)

    use_backend("numpy")
    # xp = get_backend()
    # i = intensity(xp.ones((const.NDIVX, const.NDIVY)))
    start = time.time()
    i = intensity(mask)
    print(f"Elapsed time: {time.time() - start} [s]")


if __name__ == "__main__":
    main()
