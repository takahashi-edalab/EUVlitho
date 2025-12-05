import numpy as np
from elitho import config, pupil, descriptors, diffraction_order, source
from elitho.diffraction_amplitude import diffraction_amplitude
from elitho.electro_field import electro_field


def na_filter_amplitude_map(
    Ax: np.ndarray, doc: diffraction_order.DiffractionOrderCoordinate
) -> np.ndarray:
    ampxx = np.full(
        (config.nsourceXL, config.nsourceYL, config.noutXL, config.noutYL),
        -1000 + 0j,
        dtype=np.complex128,
    )
    for x in range(config.nsourceXL):
        for y in range(config.nsourceYL):
            cond = (
                ((x - config.lsmaxX) * config.MX / config.dx) ** 2
                + ((y - config.lsmaxY) * config.MY / config.dy) ** 2
            ) <= (config.NA / config.wavelength) ** 2

            if cond:
                for n in range(doc.num_valid_diffraction_orders):
                    ip = doc.valid_x_coords[n] - (x - config.lsmaxX) + config.lpmaxX
                    jp = doc.valid_y_coords[n] - (y - config.lsmaxY) + config.lpmaxY
                    if 0 <= ip < config.noutXL and 0 <= jp < config.noutYL:
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
    fnx = np.zeros((config.XDIV, config.XDIV), dtype=complex)
    fny = np.zeros_like(fnx)
    fnz = np.zeros_like(fnx)
    for n in range(pupil_coords.n_coordinates):
        kxn = offset_x + 2.0 * np.pi * pupil_coords.linput[n] / config.dx
        kyn = offset_y + 2.0 * np.pi * pupil_coords.minput[n] / config.dy
        p2 = config.MX**2 * kxn**2 + config.MY**2 * kyn**2
        if all(
            [
                (config.NA * config.k * config.co) ** 2 <= p2 if is_high_na else True,
                p2 <= (config.NA * config.k) ** 2,
            ]
        ):
            # Calculate phase
            phase = np.exp(
                1j
                * ((kxn + config.kx0) ** 2 + (kyn + config.ky0) ** 2)
                / 2.0
                / config.k
                * config.z0
                + 1j
                * (config.MX**2 * kxn**2 + config.MY**2 * kyn**2)
                / 2.0
                / config.k
                * defocus
            )
            # Calculate electric field components
            fx = Ex0m[n] * phase
            fy = Ey0m[n] * phase
            fz = Ez0m[n] * phase
            # Map to FFT grid
            ix = pupil_coords.linput[n]
            iy = pupil_coords.minput[n]
            px = (ix + config.XDIV) % config.XDIV
            py = (iy + config.YDIV) % config.YDIV
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


# @profile
def intensity(
    sc: config.SimulationConfig,
    mask2d: np.ndarray,
    polar: config.PolarizationDirection,
    defocus: float = 0.0,
) -> np.ndarray:
    l0s, m0s, SDIV = source.abbe_division_sampling(sc)
    SDIVSUM = np.sum(list(SDIV.values()))

    dod = descriptors.DiffractionOrderDescriptor(sc.cutoff_factor)
    doc = diffraction_order.DiffractionOrderCoordinate(
        dod.max_diffraction_order_x,
        dod.max_diffraction_order_y,
        diffraction_order.rounded_diamond,
    )
    intensity_total = np.zeros((sc.exposure_field_width, sc.exposure_field_height))
    for nsx in range(-sc.ndivX + 1, sc.ndivX):
        for nsy in range(-sc.ndivY + 1, sc.ndivY):
            if SDIV[(nsx, nsy)] == 0:
                continue

            sx0 = sc.dkx * nsx / sc.ndivX + sc.kx0
            sy0 = sc.dky * nsy / sc.ndivY + sc.ky0
            Ax = diffraction_amplitude(polar, mask2d, sx0, sy0, dod, doc)
            ampxx = na_filter_amplitude_map(Ax, doc)
            pupil_coords = pupil.PupilCoordinates(
                doc.num_valid_diffraction_orders, nsx, nsy
            )

            Ex0m, Ey0m, Ez0m = electro_field(
                polar,
                sc.is_high_na,
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
                offset_x = sc.dkx * nsx / sc.ndivX + sc.dkx * l0s[(nsx, nsy)][isd]
                offset_y = sc.dky * nsy / sc.ndivY + sc.dky * m0s[(nsx, nsy)][isd]
                intensity_by_a_source = intensity_by_abbe_source(
                    Ex0m[isd],
                    Ey0m[isd],
                    Ez0m[isd],
                    offset_x,
                    offset_y,
                    pupil_coords,
                    sc.is_high_na,
                    defocus,
                )
                intensity_total += intensity_by_a_source

    intensity_map = intensity_total / SDIVSUM
    return intensity_map


def run(sc: config.SimulationConfig):
    intensities = []
    defocus_values = []

    for defocus in sc.defocus_list:
        pass


def main():
    import time
    from elitho import use_backend, get_backend
    from elitho.mask_pattern import LinePattern
    import numpy as np
    from elitho import config

    mask = LinePattern(cd=56, gap=80, direction="V", field_type="DF")(
        config.NDIVX, config.NDIVY
    )

    # with open("assets/masks/high-na-mask.bin", "rb") as f:
    #     packed = np.frombuffer(f.read(), dtype=np.uint8)
    # unpacked = np.unpackbits(packed)
    # mask = unpacked.reshape((1024, 2 * 1024))
    # mask = 1 - mask
    # mask = mask[: const.NDIVX, : const.NDIVY]

    print(config.is_high_na, mask.shape)

    use_backend("numpy")
    # xp = get_backend()
    # i = intensity(xp.ones((const.NDIVX, const.NDIVY)))
    start = time.time()
    intensity_map = intensity(
        mask,
        config.PolarizationDirection.X,
        config.IlluminationType.DIPOLE_Y,
        is_high_na=config.is_high_na,
    )
    print(f"Elapsed time: {time.time() - start} [s]")


if __name__ == "__main__":
    main()
