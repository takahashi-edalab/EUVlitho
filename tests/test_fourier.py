# import numpy as np

# def fourier(l, f, cexp, FDIV):
#     sum = 0.0 + 0.0j
#     for i in range(FDIV):
#         j = 0
#         if l >= 0:
#             j = (i * l) % FDIV
#         else:
#             j = FDIV - ((-i * l) % FDIV)

#         sum += f[i] * cexp[j]
#     return sum / FDIV


# def org_mask(pattern_mask: np.ndarray, ampta: complex, ampvc: complex) -> np.ndarray:
#     meshX = const.FDIVX // const.NDIVX
#     meshY = const.FDIVY // const.NDIVY

#     pattern = np.where(
#         np.kron(pattern_mask, np.ones((meshX, meshY))), ampta, ampvc
#     ).astype(
#         np.complex128
#     )  # shape: (FDIVX, FDIVY)

#     # y-axis first Fourier transform
#     ftmp = np.zeros((const.FDIVX, const.FDIVY), dtype=np.complex128)
#     # print(const.FDIVX)  # 512
#     # print(const.Mrange2)  # 73
#     for i in range(const.FDIVX):
#         ampy = pattern[i, :]
#         for j in range(const.Mrange2):
#             m = j - 2 * const.MMAX
#             ftmp[i, j] = fourier(m, ampy, const.cexpY, const.FDIVY)

#     # x-axis second Fourier transform
#     famp = np.zeros((const.Lrange2, const.Mrange2), dtype=np.complex128)
#     for j in range(const.Mrange2):
#         ampx = ftmp[:, j]

#         for i in range(const.Lrange2):
#             l = i - 2 * const.LMAX
#             famp[i, j] = fourier(l, ampx, const.cexpX, const.FDIVX)

#     return famp


# def calc_sigma():
# for i in range(dod.num_valid_diffraction_orders):
#     l = dod.valid_x_coords[i]
#     m = dod.valid_y_coords[i]
#     for ip in range(dod.num_valid_diffraction_orders):
#         llp = l - dod.valid_x_coords[ip] + 2 * dod.max_diffraction_order_x
#         mmp = m - dod.valid_y_coords[ip] + 2 * dod.max_diffraction_order_y
#         if polar == "X":
#             new_sigma[i, ip] = sigma[llp, mmp] * kxplus[ip]
#         else:
#             new_sigma[i, ip] = sigma[llp, mmp] * kyplus[ip]
