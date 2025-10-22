import numpy as np
from scipy import sparse
from elitho import const


def diag_mat(vals: np.ndarray) -> sparse.csr_matrix:
    return sparse.diags(vals, offsets=0, format="csr", dtype=np.complex128)


def transfer_matrix(
    polar: str,
    matrix_size: int,
    kxplus: np.ndarray,
    kyplus: np.ndarray,
    current_alpha: np.ndarray,
    current_epsilon: complex,
    current_thickness: float,
    next_alpha: np.ndarray,
    next_epsilon: complex,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    # identity diag Triplet (Cjp) -> all ones
    k = const.k
    Cjp_vals = np.ones(matrix_size, dtype=np.complex128)

    # Build Cj depending on polarization (diagonal entries)
    if polar == "X":
        Cj_vals = (k - (kxplus**2) / k / next_epsilon) / (
            k - (kxplus**2) / k / current_epsilon
        )

    else:  # 'Y'
        Cj_vals = (k - (kyplus**2) / k / next_epsilon) / (
            k - (kyplus**2) / k / current_epsilon
        )

    gamma = np.exp(const.i_complex * current_alpha * current_thickness)
    # element-wise arrays for each diag
    ul_vals = 0.5 * (Cj_vals + (next_alpha / current_alpha) * Cjp_vals) / gamma
    ur_vals = 0.5 * (Cj_vals - (next_alpha / current_alpha) * Cjp_vals) / gamma
    bl_vals = 0.5 * (Cj_vals - (next_alpha / current_alpha) * Cjp_vals) * gamma
    br_vals = 0.5 * (Cj_vals + (next_alpha / current_alpha) * Cjp_vals) * gamma
    TMOUL = diag_mat(ul_vals)
    TMOUR = diag_mat(ur_vals)
    TMOBL = diag_mat(bl_vals)
    TMOBR = diag_mat(br_vals)
    return TMOUL, TMOUR, TMOBL, TMOBR


def multilayer_transfer_matrix(
    polar: str,
    matrix_size: int,
    kxplus: np.ndarray,
    kyplus: np.ndarray,
    kxy2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    # compute per-mode propagation constants (complex)
    k = const.k
    alpha_sio2 = np.sqrt(k * k * const.epsilon_si_o2 - kxy2)
    alpha_mo = np.sqrt(k * k * const.epsilon_mo - kxy2)
    alpha_si = np.sqrt(k * k * const.epsilon_si - kxy2)
    alpha_mo_si2 = np.sqrt(k * k * const.epsilon_mo_si2 - kxy2)
    alpha_ru = np.sqrt(k * k * const.epsilon_ru - kxy2)
    alpha_ru_si = np.sqrt(k * k * const.epsilon_ru_si - kxy2)

    # MO layer -> MO/Si2 layer
    TMOUL, TMOUR, TMOBL, TMOBR = transfer_matrix(
        polar,
        matrix_size,
        kxplus,
        kyplus,
        alpha_mo,
        const.epsilon_mo,
        const.thickness_mo,
        alpha_mo_si2,
        const.epsilon_mo_si2,
    )

    # MOSI layer -> MO layer
    TMOSIUL, TMOSIUR, TMOSIBL, TMOSIBR = transfer_matrix(
        polar,
        matrix_size,
        kxplus,
        kyplus,
        alpha_mo_si2,
        const.epsilon_mo_si2,
        const.thickness_mo_si,
        alpha_mo,
        const.epsilon_mo,
    )

    # SI layer -> MOSI2 layer
    TSIUL, TSIUR, TSIBL, TSIBR = transfer_matrix(
        polar,
        matrix_size,
        kxplus,
        kyplus,
        alpha_si,
        const.epsilon_si,
        const.thickness_si,
        alpha_mo_si2,
        const.epsilon_mo_si2,
    )

    # MOSI2 layer -> SI layer
    TSIMOUL, TSIMOUR, TSIMOBL, TSIMOBR = transfer_matrix(
        polar,
        matrix_size,
        kxplus,
        kyplus,
        alpha_mo_si2,
        const.epsilon_mo_si2,
        const.thickness_si_mo,
        alpha_si,
        const.epsilon_si,
    )

    # RU/Si layer -> SI layer
    TSIRUUL, TSIRUUR, TSIRUBL, TSIRUBR = transfer_matrix(
        polar,
        matrix_size,
        kxplus,
        kyplus,
        alpha_ru_si,
        const.epsilon_ru_si,
        const.thickness_si_ru,
        alpha_si,
        const.epsilon_si,
    )

    # RU layer -> RU/Si layer
    TRUUL, TRUUR, TRUBL, TRUBR = transfer_matrix(
        polar,
        matrix_size,
        kxplus,
        kyplus,
        alpha_ru,
        const.epsilon_ru,
        const.thickness_ru,
        alpha_ru_si,
        const.epsilon_ru_si,
    )

    # MO layer -> SIO2 layer
    TNU, _, TNB, _ = transfer_matrix(
        polar,
        matrix_size,
        kxplus,
        kyplus,
        alpha_mo,
        const.epsilon_mo,
        const.thickness_mo,
        alpha_sio2,
        const.epsilon_si_o2,
    )

    UU = TNU
    UB = TNB
    for i in reversed(range(const.NML)):
        if i < const.NML - 1:
            UU, UB = (TMOUL.dot(UU) + TMOUR.dot(UB), TMOBL.dot(UU) + TMOBR.dot(UB))

        UU, UB = (TMOSIUL.dot(UU) + TMOSIUR.dot(UB), TMOSIBL.dot(UU) + TMOSIBR.dot(UB))

        UU, UB = (TSIUL.dot(UU) + TSIUR.dot(UB), TSIBL.dot(UU) + TSIBR.dot(UB))

        if i > 0:
            UU, UB = (
                TSIMOUL.dot(UU) + TSIMOUR.dot(UB),
                TSIMOBL.dot(UU) + TSIMOBR.dot(UB),
            )
        else:
            UU, UB = (
                TSIRUUL.dot(UU) + TSIRUUR.dot(UB),
                TSIRUBL.dot(UU) + TSIRUBR.dot(UB),
            )

    # final combination with TRU* blocks
    URUU = TRUUL.dot(UU) + TRUUR.dot(UB)
    URUB = TRUBL.dot(UU) + TRUBR.dot(UB)
    return URUU.toarray(), URUB.toarray()
