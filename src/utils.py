from src import const
import numpy as np
from scipy import sparse

# TODO: GPU (CUDA) 用に cupy 版も実装
# import cupy as cp  # GPU (CUDA) 用
# def matinv_gpu(A: cp.ndarray) -> cp.ndarray:
#     """Compute inverse of complex matrix A (GPU version using CuPy)."""
#     return cp.linalg.inv(A)


def matinv(A: np.ndarray) -> np.ndarray:
    """Compute inverse of complex matrix A (numpy version)."""
    return np.linalg.inv(A)


def matproduct(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix product C = A @ B"""
    return A @ B  # または np.dot(A, B)


def maskamp(pattern_mask, ampta, ampvc):
    """
    完全ベクトル化した2Dマスクフーリエ係数計算

    pattern_mask : (NDIVX, NDIVY) の0/1マスク
    ampta, ampvc : 透過/遮蔽の複素振幅
    cexpX        : (Lrange2, FDIVX) のフーリエ指数
    cexpY        : (Mrange2, FDIVY) のフーリエ指数
    """
    NDIVX, NDIVY = pattern_mask.shape
    FDIVX, FDIVY = const.cexpX.shape[1], const.cexpY.shape[1]

    # マスクを高解像度に拡張
    meshX = FDIVX // NDIVX
    meshY = FDIVY // NDIVY
    pattern = np.where(
        np.kron(pattern_mask, np.ones((meshX, meshY))), ampta, ampvc
    ).astype(
        np.complex128
    )  # shape: (FDIVX, FDIVY)

    # ループなしで2Dフーリエ
    # Y方向にフーリエ変換
    ftmp = pattern @ const.cexpY.T  # shape: (FDIVX, Mrange2)

    # X方向にフーリエ変換
    famp = const.cexpX @ ftmp  # shape: (Lrange2, Mrange2)

    return famp


def fourier_coefficients(pattern_mask):
    epsN = []
    etaN = []
    zetaN = []
    sigmaN = []
    for n in range(const.NABS):
        # eps
        eps = maskamp(
            pattern_mask=pattern_mask,
            ampta=const.eabs[n],
            ampvc=1.0,
        )
        # sigma
        sigma = maskamp(
            pattern_mask=pattern_mask,
            ampta=1 / const.eabs[n],
            ampvc=1.0,
        )
        # leps
        leps = maskamp(
            pattern_mask=pattern_mask,
            ampta=np.log(const.eabs[n]),
            ampvc=0.0,
        )

        # eta, zeta
        i_idx = np.arange(const.Lrange2) - 2 * const.LMAX
        j_idx = np.arange(const.Mrange2) - 2 * const.MMAX

        zetal = const.i_complex * 2 * const.pi * i_idx[:, None] / const.dx
        zetam = const.i_complex * 2 * const.pi * j_idx[None, :] / const.dy

        eta = zetal * leps
        zeta = zetam * leps

        epsN.append(eps)
        sigmaN.append(sigma)
        etaN.append(eta)
        zetaN.append(zeta)
    return epsN, etaN, zetaN, sigmaN


def absorber(
    polar: str,
    kxplus: np.ndarray,
    kyplus: np.ndarray,
    kxy2: np.ndarray,
    eps: np.ndarray,
    eta: np.ndarray,
    zeta: np.ndarray,
    sigma: np.ndarray,
    dabs: float,
    al2,
    br2,
    B2,
    U2U,
    U2B,
):

    # 行列Dの構築
    l = const.lindex[:, None] - const.lindex[None, :] + 2 * const.LMAX
    m = const.mindex[:, None] - const.mindex[None, :] + 2 * const.MMAX
    if polar == "X":
        D = eps[l, m] * const.k**2 - const.i_complex * eta[l, m] * kxplus[None, :]
    elif polar == "Y":
        D = eps[l, m] * const.k**2 - const.i_complex * zeta[l, m] * kyplus[None, :]
    # 対角要素の修正
    D[np.arange(const.Nrange), np.arange(const.Nrange)] -= kxy2

    # 固有値と固有ベクトル
    w1, VR = np.linalg.eig(D)
    al1 = np.sqrt(w1)

    # FG, FGinv, Cjp の計算
    BR = VR.copy()
    FG = np.column_stack(br2)
    FGinv = np.linalg.inv(FG)
    Cjp = FGinv @ FG

    # B1 の構築
    new_sigma = np.zeros((Nrange, Nrange), dtype=complex)
    for i in range(Nrange):
        l = lindex[i]
        m = mindex[i]
        for ip in range(Nrange):
            llp = l - lindex[ip] + 2 * LMAX
            mmp = m - mindex[ip] + 2 * MMAX
            if polar == "X":
                new_sigma[i, ip] = sigma[llp, mmp] * kxplus[ip]
            else:
                new_sigma[i, ip] = sigma[llp, mmp] * kyplus[ip]
    B1 = const.i_complex * (
        const.k * BR
        - np.outer(kxplus if polar == "X" else kyplus, np.ones(Nrange))
        / const.k
        * new_sigma
        @ BR
    )

    # Cj の計算
    Cj = np.linalg.inv(B1) @ B2

    # T1 行列と U1U, U1B の計算
    gamma = np.exp(const.i_complex * al1 * dabs)
    T1UL = 0.5 * (Cj + np.outer(al2, 1 / al1) * Cjp) / gamma[:, None]
    T1UR = 0.5 * (Cj - np.outer(al2, 1 / al1) * Cjp) / gamma[:, None]
    T1BL = 0.5 * (Cj - np.outer(al2, 1 / al1) * Cjp) * gamma[:, None]
    T1BR = 0.5 * (Cj + np.outer(al2, 1 / al1) * Cjp) * gamma[:, None]

    U1U = T1UL @ U2U + T1UR @ U2B
    U1B = T1BL @ U2U + T1BR @ U2B
    return U1U, U1B, B1, al1


def ampS(polar, mask2d, kx0, ky0):

    # --- 1. 各層のフーリエ係数を計算 ---
    epsN, etaN, zetaN, sigmaN = fourier_coefficients(mask2d)

    # --- 2. kxplus, kyplus, kxy2, klm の計算 ---
    kxplus = kx0 + 2 * const.pi * np.array(const.lindex) / const.dx
    kyplus = ky0 + 2 * const.pi * np.array(const.mindex) / const.dy
    kxy2 = kxplus**2 + kyplus**2
    klm = np.sqrt(const.k**2 - kxy2)

    # --- 3. multilayerS の呼び出し ---
    URUU, URUB = multilayer_transfer_matrix(polar, const.Nrange, kxplus, kyplus, kxy2)

    # --- 4. alru, brru, Bru の作成 ---
    alru = np.sqrt(const.k**2 * const.epsilon_ru - kxy2)
    brru = np.eye(const.Nrange, dtype=complex)
    if polar == "X":
        Bru = np.diag(
            const.i_complex * const.k
            - const.i_complex / const.k / const.epsilon_ru * kxplus**2
        )
    else:
        Bru = np.diag(
            const.i_complex * const.k
            - const.i_complex / const.k / const.epsilon_ru * kyplus**2
        )

    # --- 5. 最下層吸収体の absorberS0 を計算 ---
    n = const.NABS - 1
    al1 = [0] * Nrange
    br1 = [np.zeros(Nrange, dtype=complex) for _ in range(Nrange)]

    B1, U1U, U1B = absorberS0(
        polar,
        kxplus,
        kyplus,
        kxy2,
        epsN[n],
        etaN[n],
        zetaN[n],
        sigmaN[n],
        const.dabs[n],
        alru,
        brru,
        Bru,
        URUU,
        URUB,
        al1,
        br1,
    )

    # --- 6. 上層の absorberS を順に計算 ---
    for n in reversed(range(const.NABS - 1)):
        dabs1 = dabs[n]
        eps1, eta1, zeta1, sigma1 = epsN[n], etaN[n], zetaN[n], sigmaN[n]
        U2U, U2B, B2 = U1U.copy(), U1B.copy(), B1.copy()
        al2, br2 = al1.copy(), br1.copy()
        U1U, U1B = absorberS(
            polar,
            kxplus,
            kyplus,
            kxy2,
            eps1,
            eta1,
            zeta1,
            sigma1,
            dabs1,
            al2,
            br2,
            B2,
            U2U,
            U2B,
            al1,
            br1,
            B1,
        )

    # --- 7. 最終出力 Ax の計算 ---

    # tmp[i] を i ごとに計算
    tmp = klm / (1j * const.k - 1j / const.k * kxplus**2)  # shape (Nrange,)

    # tmp を列方向にブロードキャストして B1 と掛ける
    tmp_B1 = tmp[:, np.newaxis] * B1  # shape (Nrange, Nrange)

    # al1[j] * br1[j][i] を (i,j) での配列に
    al1_br1 = al1[np.newaxis, :] * br1.T  # br1.T[i,j] = br1[j][i]

    # T0L, T0R を計算
    T0L = tmp_B1 + al1_br1
    T0R = tmp_B1 - al1_br1

    U0 = np.matmul(T0L, U1U) + np.matmul(T0R, U1B)
    U0I = np.linalg.inv(U0)
    U1U = np.matmul(U1U - U1B, U0I)

    # br1 を (Nrange, Nrange) 配列に変換
    br1_array = np.array(
        [b for b in br1]
    )  # shape (Nrange, Nrange), br1[n][i] -> br1_array[n,i]
    br1_array = br1_array.T  # shape (Nrange, Nrange), now br1_array[i,n] = br1[n][i]

    # al1[n] を列方向にブロードキャスト
    al1_br1 = br1_array * al1[np.newaxis, :]  # shape (Nrange, Nrange)

    # klm[i] を行方向にブロードキャストして割り算
    FG = al1_br1 / klm[:, np.newaxis]  # shape (Nrange, Nrange)
    FG = np.matmul(FG, U1U)

    Ax = np.zeros((const.nsourceX, const.nsourceY, Nrange), dtype=np.complex128)
    for ls in range(-const.lsmaxX, const.lsmaxX + 1):
        for ms in range(-const.lsmaxY, const.lsmaxY + 1):
            if (ls * const.MX / const.dx) ** 2 + (ms * const.MY / const.dy) ** 2 <= (
                const.NA / const.wavelength
            ) ** 2:
                kx = kx0 + ls * 2 * np.pi / const.dx
                ky = ky0 + ms * 2 * np.pi / const.dy
                kz = np.sqrt(const.k**2 - kx**2 - ky**2)
                Ax0p = 1.0
                AS = np.zeros(Nrange, dtype=complex)
                for i in range(Nrange):
                    if lindex[i] == ls and mindex[i] == ms:
                        AS[i] = 2 * kz * Ax0p
                FGA = FG @ AS
                Ax[ls + lsmaxX][ms + lsmaxY] = -FGA
                for i in range(Nrange):
                    if lindex[i] == ls and mindex[i] == ms:
                        Ax[ls + lsmaxX][ms + lsmaxY][i] += Ax0p

    return Ax
