from functools import cached_property
import numpy as np
from elitho import const


class DiffractionOrderDescriptor:
    """
    Descriptor for diffraction orders in an EUV lithography simulation.
    Computes spatial frequency cutoffs and diffraction order ranges
    based on NA, sampling points, pixel size, and wavelength.
    """

    def __init__(self, cutoff_factor: float, valid_region_fn: "callable") -> None:
        """
        Initialize the descriptor.

        Parameters
        ----------
        cutoff_factor : float
            Empirical factor for spatial frequency cutoff. Typical value ~6.0.
        valid_region_fn : callable
            Function defining valid diffraction order region (e.g., ellipse, diamond).
        """
        self._cutoff_factor = cutoff_factor
        self._valid_region_fn = valid_region_fn

        from elitho import diffraction_order

        self._meshgrid_x_coords, self._meshgrid_y_coords = (
            diffraction_order.valid_coordinates(
                self.max_diffraction_order_x,
                self.max_diffraction_order_y,
                self._valid_region_fn,
            )
        )

    @cached_property
    def spatial_freq_cutoff_x(self) -> float:
        """
        Maximum spatial frequency in the x direction (pupil plane) to consider.

        Returns
        -------
        float
            Spatial frequency cutoff in x-direction.
        """
        return const.NA / const.MX * self._cutoff_factor

    @cached_property
    def spatial_freq_cutoff_y(self) -> float:
        """
        Maximum spatial frequency in the y direction (pupil plane) to consider.

        Returns
        -------
        float
            Spatial frequency cutoff in y-direction.
        """
        return const.NA / const.MX * self._cutoff_factor

    @cached_property
    def max_diffraction_order_x(self) -> int:
        """
        Maximum diffraction order in x-direction (±LMAX).

        Returns
        -------
        int
            Maximum diffraction order along x.
        """
        return int(self.spatial_freq_cutoff_x * const.dx / const.wavelength)

    @cached_property
    def max_diffraction_order_y(self) -> int:
        """
        Maximum diffraction order in y-direction (±MMAX).

        Returns
        -------
        int
            Maximum diffraction order along y.
        """
        return int(self.spatial_freq_cutoff_y * const.dy / const.wavelength)

    @cached_property
    def num_diffraction_orders_x(self):
        """
        Total number of diffraction orders in x-direction.

        Returns
        -------
        int
            Total diffraction orders along x: 2*max_diffraction_order_x + 1
        """
        return 2 * self.max_diffraction_order_x + 1

    @cached_property
    def num_diffraction_orders_y(self):
        """
        Total number of diffraction orders in y-direction.

        Returns
        -------
        int
            Total diffraction orders along y: 2*max_diffraction_order_y + 1
        """
        return 2 * self.max_diffraction_order_y + 1

    @cached_property
    def num_diffraction_orders_x_expanded(self):
        """
        Expanded number of diffraction orders in x-direction for FFT or padding.

        Returns
        -------
        int
            Expanded diffraction orders along x: 4*max_diffraction_order_x + 1
        """
        return 4 * self.max_diffraction_order_x + 1

    @cached_property
    def num_diffraction_orders_y_expanded(self):
        """
        Expanded number of diffraction orders in y-direction for FFT or padding.

        Returns
        -------
        int
            Expanded diffraction orders along y: 4*max_diffraction_order_y + 1
        """
        return 4 * self.max_diffraction_order_y + 1

    @cached_property
    def meshgrid_x_coords(self) -> np.ndarray:
        """
        Meshgrid of valid diffraction order x-coordinates.

        Returns
        -------
        np.ndarray
            Array of valid diffraction order x-coordinates.
        """
        return self._meshgrid_x_coords

    @cached_property
    def meshgrid_y_coords(self) -> np.ndarray:
        """
        Meshgrid of valid diffraction order y-coordinates.

        Returns
        -------
        np.ndarray
            Array of valid diffraction order y-coordinates.
        """
        return self._meshgrid_y_coords
