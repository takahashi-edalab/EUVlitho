import numpy as np
from src import const


def mask(pattern_mask: np.ndarray, ampta: complex, ampvc: complex) -> np.ndarray:
    meshX = const.FDIVX // const.NDIVX
    meshY = const.FDIVY // const.NDIVY

    pattern = np.where(
        np.kron(pattern_mask, np.ones((meshX, meshY))), ampta, ampvc
    ).astype(
        np.complex128
    )  # shape: (FDIVX, FDIVY)

    famp_full = np.fft.fftshift(np.fft.fft2(pattern)) / (const.FDIVX * const.FDIVY)

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


def coefficients(pattern_mask: np.ndarray):
    epses = []
    etas = []
    zetas = []
    sigmas = []
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
            ampta=np.log(const.eabs[n]),
            ampvc=0.0,
        )

        i_idx = np.arange(const.Lrange2) - 2 * const.LMAX
        j_idx = np.arange(const.Mrange2) - 2 * const.MMAX

        zetal = const.i_complex * 2 * const.pi * i_idx[:, None] / const.dx
        zetam = const.i_complex * 2 * const.pi * j_idx[None, :] / const.dy

        eta = zetal * leps
        zeta = zetam * leps

        epses.append(eps)
        sigmas.append(sigma)
        etas.append(eta)
        zetas.append(zeta)
    return epses, etas, zetas, sigmas
