import numpy as np
from src import const


def fourier(FDIV, ll, f, cexp):
    sum = 0.0 + 0.0j
    for i in range(FDIV):
        il = (i * ll) % FDIV
        sum += f[i] * cexp[il]
    return sum / FDIV


def mask(pattern_mask: np.ndarray, ampta: complex, ampvc: complex) -> np.ndarray:
    meshX = const.FDIVX // const.NDIVX
    meshY = const.FDIVY // const.NDIVY

    # pattern = np.zeros((const.FDIVX, const.FDIVY), dtype=np.complex128)
    # for i in range(const.FDIVX):
    #     ii = i // meshX
    #     for j in range(const.FDIVY):
    #         jj = j // meshY
    #         if pattern_mask[ii, jj] == 1:  # mask2d が 1次元フラットな場合
    #             pattern[i, j] = ampta
    #         else:
    #             pattern[i, j] = ampvc

    pattern = np.where(
        np.kron(pattern_mask, np.ones((meshX, meshY))), ampta, ampvc
    ).astype(
        np.complex128
    )  # shape: (FDIVX, FDIVY)

    # y-axis first Fourier transform
    ftmp = np.zeros((const.FDIVX, const.FDIVY), dtype=np.complex128)
    # print(const.FDIVX)  # 512
    # print(const.Mrange2)  # 73
    for i in range(const.FDIVX):
        ampy = pattern[i, :]
        # shift = int((const.Mrange2 + 1) / 2)
        # shifted_ampy = np.roll(ampy, shift)
        # fft_result = np.fft.fft(
        #     shifted_ampy[: const.Mrange2], n=const.Mrange2, norm="forward"
        # )
        # ftmp[i, : const.Mrange2 // 2] = fft_result[: const.Mrange2 // 2]
        # ftmp[i, -(const.Mrange2 // 2 + 1) :] = fft_result[const.Mrange2 // 2 :]

        for ij in range(int((const.Mrange2 + 1) / 2)):
            ftmp[i, ij] = fourier(const.FDIVY, ij, ampy, const.cexpY)
        for ij in range(int((const.Mrange2 + 1) / 2), const.Mrange2):
            m = ij - const.Mrange2
            ftmp[i, ij] = fourier(const.FDIVY, m, ampy, const.cexpY)

    # x-axis second Fourier transform
    famp = np.zeros((const.Lrange2, const.Mrange2), dtype=np.complex128)
    for j in range(const.Mrange2):
        ampx = ftmp[:, j]
        for i in range(int((const.Lrange2 + 1) / 2)):
            famp[i, j] = fourier(const.FDIVX, i, ampx, const.cexpX)
        for i in range(int((const.Lrange2 + 1) / 2), const.Lrange2):
            ll = i - const.Lrange2
            famp[i, j] = fourier(const.FDIVX, ll, ampx, const.cexpX)

    # --- y-axis first Fourier transform ---
    # ftmp = np.fft.fftshift(pattern, axes=0)
    # ftmp = np.fft.fft(pattern, axis=0, norm="forward")

    # --- x-axis second Fourier transform ---
    # famp = np.fft.fftshift(ftmp, axes=1)
    # famp = np.fft.fft(ftmp, axis=1, norm="forward")
    #
    # famp = np.fft.fft2(pattern)
    # famp = np.fft.fftshift(famp)
    # famp = np.fft.fftshift(famp)

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
