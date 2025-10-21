import numpy as np
from src import const
from src.diffraction_amplitude import diffraction_amplitude
from src.source import abbe_source
from src.electro_field import electro_field


def find_valid_output_points(nrange: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Find valid output points based on source and pupil conditions"""
    linput = np.zeros(nrange, dtype=int)
    minput = np.zeros(nrange, dtype=int)
    ninput = 0
    for ip in range(const.noutX):
        for jp in range(const.noutY):
            snum = 0
            for is_src in range(const.nsourceX):
                for js_src in range(const.nsourceY):
                    source_condition = (
                        (is_src - const.lsmaxX) * const.MX / const.dx
                    ) ** 2 + ((js_src - const.lsmaxY) * const.MY / const.dy) ** 2 <= (
                        const.NA / const.wavelength
                    ) ** 2
                    pupil_condition = (
                        (ip - const.lpmaxX + is_src - const.lsmaxX)
                        * const.MX
                        / const.dx
                    ) ** 2 + (
                        (jp - const.lpmaxY + js_src - const.lsmaxY)
                        * const.MY
                        / const.dy
                    ) ** 2 <= (
                        const.NA / const.wavelength
                    ) ** 2
                    if source_condition and pupil_condition:
                        snum += 1

            if snum > 0:
                linput[ninput] = ip - const.lpmaxX
                minput[ninput] = jp - const.lpmaxY
                ninput += 1

    return linput, minput, ninput


def na_filter_amplitude_map(Ax: np.ndarray) -> np.ndarray:
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
                for n in range(const.Nrange):
                    ip = const.lindex[n] - (x - const.lsmaxX) + const.lpmaxX
                    jp = const.mindex[n] - (y - const.lsmaxY) + const.lpmaxY

                    if 0 <= ip < const.noutXL and 0 <= jp < const.noutYL:
                        ampxx[x, y, ip, jp] = Ax[x, y, n]

    return ampxx


def intensity(mask2d: np.ndarray) -> np.ndarray:
    l0s, m0s, SDIV = abbe_source()
    SDIVMAX = np.max(SDIV)
    SDIVSUM = np.sum(SDIV)

    linput, minput, ninput = find_valid_output_points(const.Nrange)
    ncut = ninput

    isum = np.zeros((const.ndivs, const.ndivs, const.XDIV, const.XDIV, SDIVMAX))
    for nsx in range(const.ndivs):
        for nsy in range(const.ndivs):
            sx0 = 2.0 * const.pi / const.dx * nsx / const.ndivs + const.kx0
            sy0 = 2.0 * const.pi / const.dy * nsy / const.ndivs + const.ky0
            Ax = diffraction_amplitude("X", mask2d, sx0, sy0)
            # ここまでdebug中...
            ampxx = na_filter_amplitude_map(Ax)
            Ex0m, Ey0m, Ez0m = electro_field(
                SDIV, l0s, m0s, nsx, nsy, ncut, sx0, sy0, linput, minput, ampxx
            )
            # np.save("Ex0m.npy", Ex0m)
            # np.save("Ey0m.npy", Ey0m)
            # np.save("Ez0m.npy", Ez0m)
            # exit()

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

                # Inverse FFT to get spatial field distribution
                fnx_ifft = np.fft.ifft2(fnx)
                fny_ifft = np.fft.ifft2(fny)
                fnz_ifft = np.fft.ifft2(fnz)

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
    pass


if __name__ == "__main__":
    main()
