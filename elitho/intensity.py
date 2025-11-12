import numpy as np
from elitho import const, pupil, descriptors, diffraction_order, source
from elitho.diffraction_amplitude import diffraction_amplitude
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
    Ex0m: np.ndarray,
    Ey0m: np.ndarray,
    Ez0m: np.ndarray,
    offset_x: float,
    offset_y: float,
    pupil_coords: "pupil.PupilCoordinates",
    is_high_na: bool = False,
    defocus: float = 0.0,
) -> np.ndarray:
    # ---- FFT & isum更新 ----
    fnx = np.zeros((const.XDIV, const.XDIV), dtype=complex)
    fny = np.zeros_like(fnx)
    fnz = np.zeros_like(fnx)
    for n in range(pupil_coords.n_coordinates):
        kxn = offset_x + 2.0 * const.pi * pupil_coords.linput[n] / const.dx
        kyn = offset_y + 2.0 * const.pi * pupil_coords.minput[n] / const.dy
        p2 = const.MX**2 * kxn**2 + const.MY**2 * kyn**2
        if all(
            [
                (const.NA * const.k * const.co) ** 2 <= p2 if is_high_na else True,
                p2 <= (const.NA * const.k) ** 2,
            ]
        ):
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
                * defocus
            )
            # Calculate electric field components
            fx = Ex0m[n] * phase
            fy = Ey0m[n] * phase
            fz = Ez0m[n] * phase
            # Map to FFT grid
            ix = pupil_coords.linput[n]
            iy = pupil_coords.minput[n]
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
    polar: const.PolarizationDirection,
    illumination_type: const.IlluminationType,
    defocus: float = 0.0,
    cutoff_factor: float = 6.0,
    is_high_na: bool = False,
) -> np.ndarray:
    l0s, m0s, SDIV = source.abbe_division_sampling(illumination_type)
    SDIVSUM = np.sum(list(SDIV.values()))

    dod = descriptors.DiffractionOrderDescriptor(cutoff_factor)
    doc = diffraction_order.DiffractionOrderCoordinate(
        dod.max_diffraction_order_x,
        dod.max_diffraction_order_y,
        diffraction_order.rounded_diamond,
    )
    # ここまで多分OK
    intensity_total = np.zeros((const.XDIV, const.XDIV))
    for nsx in range(-const.ndivX + 1, const.ndivX):
        for nsy in range(-const.ndivY + 1, const.ndivY):
            if SDIV[(nsx, nsy)] == 0:
                continue

            sx0 = 2.0 * const.pi / const.dx * nsx / const.ndivX + const.kx0
            sy0 = 2.0 * const.pi / const.dy * nsy / const.ndivY + const.ky0
            Ax = diffraction_amplitude(polar, mask2d, sx0, sy0, dod, doc)
            ampxx = na_filter_amplitude_map(Ax, doc)
            pupil_coords = pupil.PupilCoordinates(
                doc.num_valid_diffraction_orders, nsx, nsy
            )

            Ex0m, Ey0m, Ez0m = electro_field(
                polar,
                is_high_na,
                nsx,
                nsy,
                SDIV[(nsx, nsy)],
                l0s[(nsx, nsy)],
                m0s[(nsx, nsy)],
                sx0,
                sy0,
                pupil_coords,
                ampxx,
            )

            for isd in range(SDIV[(nsx, nsy)]):
                offset_x = (
                    2.0 * const.pi / const.dx * nsx / const.ndivX
                    + 2.0 * const.pi / const.dx * l0s[(nsx, nsy)][isd]
                )
                offset_y = (
                    2.0 * const.pi / const.dy * nsy / const.ndivY
                    + 2.0 * const.pi / const.dy * m0s[(nsx, nsy)][isd]
                )

                intensity_by_a_source = intensity_by_abbe_source(
                    Ex0m[isd],
                    Ey0m[isd],
                    Ez0m[isd],
                    offset_x,
                    offset_y,
                    pupil_coords,
                    is_high_na,
                    defocus,
                )
                intensity_total += intensity_by_a_source

    intensity_map = intensity_total / SDIVSUM
    return intensity_map


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
