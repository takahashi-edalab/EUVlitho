import numpy as np
from src import const
from src.diffraction_amplitude import diffraction_amplitude
from src.source import source


# def exponential(FDIV: int) -> np.ndarray:
#     """Calculate exponential terms for Fourier transform"""
#     cexp = np.zeros(FDIV + 1, dtype=complex)
#     for i in range(FDIV + 1):
#         cexp[i] = np.exp(-2j * np.pi * i / FDIV)
#     return cexp


def find_valid_output_points() -> tuple[np.ndarray, np.ndarray]:
    """Find valid output points based on source and pupil conditions"""
    linput = []
    minput = []
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
                linput.append(ip - const.lpmaxX)
                minput.append(jp - const.lpmaxY)

    linput = np.array(linput)
    minput = np.array(minput)
    return linput, minput


def na_filter_amplitude_map(Ax: np.ndarray) -> np.ndarray:
    ampxx = np.full(
        (const.nsourceXL, const.nsourceYL, const.noutXL, const.noutYL),
        -1000 + 0j,
        dtype=np.complex128,
    )
    is_idx = np.arange(const.nsourceXL)[:, None]  # shape (nsourceXL, 1)
    js_idx = np.arange(const.nsourceYL)[None, :]  # shape (1, nsourceYL)

    condition = (
        ((is_idx - const.lsmaxX) * const.MX / const.dx) ** 2
        + ((js_idx - const.lsmaxY) * const.MY / const.dy) ** 2
    ) <= (const.NA / const.wavelength) ** 2

    mask3d = condition[:, :, None]

    ip_all = (
        const.lindex[None, None, :]
        - (is_idx - const.lsmaxX)[:, None, None]
        + const.lpmaxX
    )  # shape (nsourceXL, 1, Nrange)
    jp_all = (
        const.mindex[None, None, :]
        - (js_idx - const.lsmaxY)[None, :, None]
        + const.lpmaxY
    )  # shape (1, nsourceYL, Nrange)

    valid_ip = (0 <= ip_all) & (ip_all < const.noutXL)
    valid_jp = (0 <= jp_all) & (jp_all < const.noutYL)
    valid_mask = mask3d & valid_ip & valid_jp
    # Get indices where the mask is True
    I, J, N = np.nonzero(valid_mask)
    ip_valid = ip_all[I, J, N]
    jp_valid = jp_all[I, J, N]

    ampxx[I, J, ip_valid, jp_valid] = Ax[I, J, N]
    return ampxx


def intensity(mask2d: np.ndarray) -> np.ndarray:
    l0s, m0s, SDIV = source()
    SDIVMAX = np.max(SDIV)
    SDIVSUM = np.sum(SDIV)

    linput, minput = find_valid_output_points()
    ncut = len(linput)

    isum = np.zeros((const.ndivs, const.ndivs, const.XDIV, const.XDIV, SDIVMAX))
    for nsx in range(const.ndivs):
        for nsy in range(const.ndivs):
            kx0 = (
                const.k
                * np.sin(np.deg2rad(const.theta0))
                * np.cos(np.deg2rad(const.phi0))
            )
            ky0 = (
                const.k
                * np.sin(np.deg2rad(const.theta0))
                * np.sin(np.deg2rad(const.phi0))
            )
            sx0 = 2.0 * const.pi / const.dx * nsx / const.ndivs + kx0
            sy0 = 2.0 * const.pi / const.dy * nsy / const.ndivs + ky0

            Ax = diffraction_amplitude("X", mask2d, sx0, sy0)
            ampxx = na_filter_amplitude_map(Ax)

            # ---- Ex0m / Ey0m / Ez0m ----
            Ex0m = np.zeros((SDIV[nsx, nsy], ncut), dtype=complex)
            Ey0m = np.zeros_like(Ex0m)
            Ez0m = np.zeros_like(Ex0m)

            for isd in range(SDIV[nsx, nsy]):
                kx = sx0 + 2.0 * const.pi / const.dx * l0s[nsx][nsy][isd]
                ky = sy0 + 2.0 * const.pi / const.dy * m0s[nsx][nsy][isd]
                ls = l0s[nsx][nsy][isd] + const.lsmaxX
                ms = m0s[nsx][nsy][isd] + const.lsmaxY
                for i in range(ncut):
                    kxplus = kx + 2 * const.pi * linput[i] / const.dx
                    kyplus = ky + 2 * const.pi * minput[i] / const.dy
                    kxy2 = kxplus**2 + kyplus**2
                    klm = np.sqrt(const.k * const.k - kxy2)
                    ip = linput[i] + const.lpmaxX
                    jp = minput[i] + const.lpmaxY

                    Ax_val = ampxx[ls, ms, ip, jp] / np.sqrt(
                        const.k * const.k - kx * kx
                    )
                    Ay_val = 0

                    EAx = (
                        const.i_complex * const.k * Ax_val
                        - const.i_complex
                        / const.k
                        * (kxplus**2 * Ax_val + kxplus * kyplus * Ay_val)
                    )
                    EAy = (
                        const.i_complex * const.k * Ay_val
                        - const.i_complex
                        / const.k
                        * (kxplus * kyplus * Ax_val + kyplus**2 * Ay_val)
                    )
                    EAz = (
                        const.i_complex
                        * klm
                        / const.k
                        * (kxplus * Ax_val + kyplus * Ay_val)
                    )
                    Ex0m[isd, i] = EAx
                    Ey0m[isd, i] = EAy
                    Ez0m[isd, i] = EAz

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
                            * ((kxn + kx0) ** 2 + (kyn + ky0) ** 2)
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
                        py = (iy + const.XDIV) % const.XDIV

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
