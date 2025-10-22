import numpy as np

# TODO: GPU (CUDA) 用に cupy 版も実装
# import cupy as cp  # GPU (CUDA) 用
# def matinv_gpu(A: cp.ndarray) -> cp.ndarray:
#     """Compute inverse of complex matrix A (GPU version using CuPy)."""
#     return cp.linalg.inv(A)


# def matinv(A: np.ndarray) -> np.ndarray:
#     """Compute inverse of complex matrix A (numpy version)."""
#     return np.linalg.inv(A)


# def matproduct(A: np.ndarray, B: np.ndarray) -> np.ndarray:
#     """Matrix product C = A @ B"""
#     return A @ B  # または np.dot(A, B)


#     def load_mask(self, filename: str) -> np.ndarray:
#         shape = (self.NDIVY, self.NDIVX)
#         data = np.fromfile(filename, dtype=np.uint8)
#         unpacked = np.unpackbits(data)[: np.prod(shape)]
#         # mask_restored = unpacked.reshape(shape)
#         return unpacked
