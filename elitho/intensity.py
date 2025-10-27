import numpy as np
from elitho import const, pupil, descriptors, diffraction_order
from elitho.diffraction_amplitude import diffraction_amplitude
from elitho.source import abbe_source
from elitho.electro_field import electro_field


def na_filter_amplitude_map(
    Ax: np.ndarray, dod: descriptors.DiffractionOrderDescriptor
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
                for n in range(dod.num_valid_diffraction_orders):
                    ip = dod.valid_x_coords[n] - (x - const.lsmaxX) + const.lpmaxX
                    jp = dod.valid_y_coords[n] - (y - const.lsmaxY) + const.lpmaxY
                    if 0 <= ip < const.noutXL and 0 <= jp < const.noutYL:
                        ampxx[x, y, ip, jp] = Ax[x, y, n]

    return ampxx


def intensity(mask2d: np.ndarray) -> np.ndarray:
    l0s, m0s, SDIV = abbe_source()
    SDIVMAX = np.max(SDIV)
    SDIVSUM = np.sum(SDIV)

    dod = descriptors.DiffractionOrderDescriptor(
        6.0, valid_region_fn=diffraction_order.rounded_diamond
    )

    linput, minput, _, ninput = pupil.find_valid_pupil_points(
        dod.num_valid_diffraction_orders
    )
    ncut = ninput

    isum = np.zeros((const.ndivs, const.ndivs, const.XDIV, const.XDIV, SDIVMAX))
    for nsx in range(const.ndivs):
        for nsy in range(const.ndivs):
            sx0 = 2.0 * const.pi / const.dx * nsx / const.ndivs + const.kx0
            sy0 = 2.0 * const.pi / const.dy * nsy / const.ndivs + const.ky0
            Ax = diffraction_amplitude("X", mask2d, sx0, sy0, dod)
            ampxx = na_filter_amplitude_map(Ax, dod)
            Ex0m, Ey0m, Ez0m = electro_field(
                SDIV, l0s, m0s, nsx, nsy, ncut, sx0, sy0, linput, minput, ampxx
            )

            # ---- FFT & isum更新 ----
            for isd in range(SDIV[nsx, nsy]):
                fnx = np.zeros((const.XDIV, const.XDIV), dtype=complex)
                fny = np.zeros_like(fnx)
                fnz = np.zeros_like(fnx)

                for n in range(ncut):
                    kxn = (
                        2.0 * const.pi / const.dx * nsx / const.ndivs
                        + 2.0 * const.pi / const.dx * l0s[nsx][nsy][isd]
                        + 2.0 * const.pi * linput[n] / const.dx
                    )
                    kyn = (
                        2.0 * const.pi / const.dy * nsy / const.ndivs
                        + 2.0 * const.pi / const.dy * m0s[nsx][nsy][isd]
                        + 2.0 * const.pi * minput[n] / const.dy
                    )

                    if (const.MX**2 * kxn**2 + const.MY**2 * kyn**2) <= (
                        const.NA * const.k
                    ) ** 2:
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
                        fx = Ex0m[isd, n] * phase
                        fy = Ey0m[isd, n] * phase
                        fz = Ez0m[isd, n] * phase

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
                intensity = (
                    np.abs(fnx_ifft) ** 2
                    + np.abs(fny_ifft) ** 2
                    + np.abs(fnz_ifft) ** 2
                )

                isum[nsx, nsy, :, :, isd] = intensity

    intensity_map = isum.sum(axis=(0, 1, 4)) / SDIVSUM
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
