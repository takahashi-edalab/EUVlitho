import numpy as np
from functools import cached_property


def ellipse(
    max_diffraction_order_x: int,
    max_diffraction_order_y: int,
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
) -> np.ndarray:
    return (mesh_x / (max_diffraction_order_x + 0.01)) ** 2 + (
        mesh_y / (max_diffraction_order_y + 0.01)
    ) ** 2 <= 1.0


def rounded_diamond(
    max_diffraction_order_x: int,
    max_diffraction_order_y: int,
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
) -> np.ndarray:
    return (abs(mesh_x) / (max_diffraction_order_x + 0.01) + 1.0) * (
        abs(mesh_y) / (max_diffraction_order_y + 0.01) + 1.0
    ) <= 2.0


def valid_coordinates(
    max_diffraction_order_x: int, max_diffraction_order_y: int, valid_region_fn
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate diffraction order limits efficiently using NumPy vectorization"""
    # create 1D index arrays
    x_range = np.arange(-max_diffraction_order_x, max_diffraction_order_x + 1)
    y_range = np.arange(-max_diffraction_order_y, max_diffraction_order_y + 1)
    # create 2D grids
    meshgrid_x_coords, meshgrid_y_coords = np.meshgrid(x_range, y_range, indexing="ij")
    # apply the condition
    mask = valid_region_fn(
        max_diffraction_order_x,
        max_diffraction_order_y,
        meshgrid_x_coords,
        meshgrid_y_coords,
    )
    # extract indices that satisfy the condition
    return meshgrid_x_coords[mask], meshgrid_y_coords[mask]


class DiffractionOrderCoordinate:
    def __init__(
        self,
        max_diffraction_order_x: int,
        max_diffraction_order_y: int,
        valid_region_fn: "callable",
    ):
        # self._max_diffraction_order_x = max_diffraction_order_x
        # self._max_diffraction_order_y = max_diffraction_order_y
        self._valid_region_fn = valid_region_fn
        self._valid_x_coords, self._valid_y_coords = valid_coordinates(
            max_diffraction_order_x,
            max_diffraction_order_y,
            self._valid_region_fn,
        )

    @cached_property
    def valid_x_coords(self) -> np.ndarray:
        """
        Meshgrid of valid diffraction order x-coordinates.

        Returns
        -------
        np.ndarray
            Array of valid diffraction order x-coordinates.
        """
        return self._valid_x_coords

    @cached_property
    def valid_y_coords(self) -> np.ndarray:
        """
        Meshgrid of valid diffraction order y-coordinates.

        Returns
        -------
        np.ndarray
            Array of valid diffraction order y-coordinates.
        """
        return self._valid_y_coords

    @cached_property
    def num_valid_diffraction_orders(self) -> int:
        """
        Number of valid diffraction orders based on the defined region. (Nrange)

        Returns
        -------
        int
            Count of valid diffraction orders.
        """
        return len(self._valid_x_coords)
