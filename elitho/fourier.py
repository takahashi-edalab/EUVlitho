import cupy as cp
from elitho import const


def mask(pattern_mask: "xp.ndarray", ampta: complex, ampvc: complex) -> "xp.ndarray":
    xp = cp.get_array_module(pattern_mask)
    meshX = const.FDIVX // const.NDIVX
    meshY = const.FDIVY // const.NDIVY

    pattern = xp.where(
        xp.kron(pattern_mask, xp.ones((meshX, meshY))), ampta, ampvc
    ).astype(
        xp.complex128
    )  # shape: (FDIVX, FDIVY)

    famp_full = xp.fft.fftshift(xp.fft.fft2(pattern)) / (const.FDIVX * const.FDIVY)

    # Extract the region around the frequency center
    cx = const.FDIVX // 2
    cy = const.FDIVY // 2
    half_L = const.Lrange2 // 2
    half_M = const.Mrange2 // 2

    famp = famp_full[
        cx - half_L : cx + half_L + (const.Lrange2 % 2),
        cy - half_M : cy + half_M + (const.Mrange2 % 2),
    ]
    return famp


def coefficients(
    pattern_mask: "xp.ndarray",
) -> tuple["xp.ndarray", "xp.ndarray", "xp.ndarray", "xp.ndarray"]:
    xp = cp.get_array_module(pattern_mask)
    epses = xp.zeros((const.NABS, const.Lrange2, const.Mrange2), dtype=xp.complex128)
    etas = xp.zeros_like(epses)
    zetas = xp.zeros_like(epses)
    sigmas = xp.zeros_like(epses)
    for n in range(const.NABS):
        # eps
        eps = mask(
            pattern_mask=pattern_mask,
            ampta=const.eabs[n],
            ampvc=1.0,
        )
        # sigma
        sigma = mask(
            pattern_mask=pattern_mask,
            ampta=1 / const.eabs[n],
            ampvc=1.0,
        )
        # leps
        leps = mask(
            pattern_mask=pattern_mask,
            ampta=xp.log(const.eabs[n]),
            ampvc=0.0,
        )

        i_idx = xp.arange(const.Lrange2) - 2 * const.LMAX
        j_idx = xp.arange(const.Mrange2) - 2 * const.MMAX

        zetal = const.i_complex * 2 * const.pi * i_idx[:, None] / const.dx
        zetam = const.i_complex * 2 * const.pi * j_idx[None, :] / const.dy

        eta = zetal * leps
        zeta = zetam * leps

        epses[n] = eps
        sigmas[n] = sigma
        etas[n] = eta
        zetas[n] = zeta
    return epses, etas, zetas, sigmas
