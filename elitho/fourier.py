import cupy as cp
from elitho import config, descriptors
from elitho.utils import image_processing as ip


def mask(
    mask_pattern: "xp.ndarray",
    ampta: complex = 1.0,
    ampvc: complex = 0.0,
    extraction_size_x: int = None,
    extraction_size_y: int = None,
    refinement_factor_x: int = 1,
    refinement_factor_y: int = 1,
) -> "xp.ndarray":
    xp = cp.get_array_module(mask_pattern)
    # refine mask pattern
    refined_mask = ip.refine(mask_pattern, refinement_factor_x, refinement_factor_y)
    # create amplitude pattern
    pattern = xp.where(refined_mask, ampvc, ampta).astype(xp.complex128)
    # FFT with scaling
    fft_map = xp.fft.fftshift(xp.fft.fft2(pattern, norm="forward"))
    # extract central region
    if extraction_size_x is not None and extraction_size_y is not None:
        fft_map = ip.extract_central_region(
            fft_map,
            extraction_size_x,
            extraction_size_y,
        )
    return fft_map


def coefficients(
    mask_pattern: "xp.ndarray",
    absorption_amplitudes: list[complex],
    dod: descriptors.DiffractionOrderDescriptor,
) -> tuple["xp.ndarray", "xp.ndarray", "xp.ndarray", "xp.ndarray"]:
    xp = cp.get_array_module(mask_pattern)

    num_absorber_layers = len(absorption_amplitudes)
    epses = xp.zeros(
        (
            num_absorber_layers,
            dod.num_diffraction_orders_x_expanded,
            dod.num_diffraction_orders_y_expanded,
        ),
        dtype=xp.complex128,
    )
    etas = xp.zeros_like(epses)
    zetas = xp.zeros_like(epses)
    sigmas = xp.zeros_like(epses)
    for i in range(num_absorber_layers):
        # eps
        eps = mask(
            mask_pattern=mask_pattern,
            ampta=absorption_amplitudes[i],
            ampvc=1.0,
            extraction_size_x=dod.num_diffraction_orders_x_expanded,
            extraction_size_y=dod.num_diffraction_orders_y_expanded,
        )
        # sigma
        sigma = mask(
            mask_pattern=mask_pattern,
            ampta=1 / absorption_amplitudes[i],
            ampvc=1.0,
            extraction_size_x=dod.num_diffraction_orders_x_expanded,
            extraction_size_y=dod.num_diffraction_orders_y_expanded,
        )
        # leps
        leps = mask(
            mask_pattern=mask_pattern,
            ampta=xp.log(absorption_amplitudes[i]),
            ampvc=0.0,
            extraction_size_x=dod.num_diffraction_orders_x_expanded,
            extraction_size_y=dod.num_diffraction_orders_y_expanded,
        )
        i_idx = (
            xp.arange(dod.num_diffraction_orders_x_expanded)
            - 2 * dod.max_diffraction_order_x
        )
        j_idx = (
            xp.arange(dod.num_diffraction_orders_y_expanded)
            - 2 * dod.max_diffraction_order_y
        )

        zetal = config.i_complex * 2 * np.pi * i_idx[:, None] / config.dx
        zetam = config.i_complex * 2 * np.pi * j_idx[None, :] / config.dy

        eta = zetal * leps
        zeta = zetam * leps

        epses[i] = eps
        sigmas[i] = sigma
        etas[i] = eta
        zetas[i] = zeta
    return epses, etas, zetas, sigmas
