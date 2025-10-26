import numpy as np


def ellipse(LMAX: int, MMAX: int, mesh_x: np.ndarray, mesh_y: np.ndarray) -> np.ndarray:
    return (mesh_x / (LMAX + 0.01)) ** 2 + (mesh_y / (MMAX + 0.01)) ** 2 <= 1.0


def rounded_diamond(
    LMAX: int, MMAX: int, mesh_x: np.ndarray, mesh_y: np.ndarray
) -> np.ndarray:
    return (abs(mesh_x) / (LMAX + 0.01) + 1.0) * (
        abs(mesh_y) / (MMAX + 0.01) + 1.0
    ) <= 2.0


def valid_coordinates(
    LMAX: int, MMAX: int, valid_region_fn
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate diffraction order limits efficiently using NumPy vectorization"""
    # create 1D index arrays
    lvals = np.arange(-LMAX, LMAX + 1)
    mvals = np.arange(-MMAX, MMAX + 1)
    # create 2D grids
    ll, mm = np.meshgrid(lvals, mvals, indexing="ij")
    # apply the condition
    mask = valid_region_fn(LMAX, MMAX, ll, mm)
    # extract indices that satisfy the condition
    return ll[mask], mm[mask]
