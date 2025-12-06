from functools import cached_property
from elitho import config


class DiffractionOrderDescriptor:
    """
    Descriptor for diffraction orders in an EUV lithography simulation.
    Computes spatial frequency cutoffs and diffraction order ranges
    based on NA, sampling points, pixel size, and wavelength.
    """

    def __init__(self, sc: config.SimulationConfig, cutoff_factor: float) -> None:
        """
        Initialize the descriptor.

        Parameters
        ----------
        cutoff_factor : float
            Empirical factor for spatial frequency cutoff. Typical value ~6.0.
        valid_region_fn : callable
            Function defining valid diffraction order region (e.g., ellipse, rounded_diamond).
        """
        self._sc = sc
        self._cutoff_factor = cutoff_factor

    @cached_property
    def spatial_freq_cutoff_x(self) -> float:
        """
        Maximum spatial frequency in the x direction (pupil plane) to consider.

        Returns
        -------
        float
            Spatial frequency cutoff in x-direction.
        """
        return self._sc.NA / self._sc.magnification_x * self._cutoff_factor

    @cached_property
    def spatial_freq_cutoff_y(self) -> float:
        """
        Maximum spatial frequency in the y direction (pupil plane) to consider.

        Returns
        -------
        float
            Spatial frequency cutoff in y-direction.
        """
        return self._sc.NA / self._sc.magnification_y * self._cutoff_factor

    @cached_property
    def max_diffraction_order_x(self) -> int:
        """
        Maximum diffraction order in x-direction (±LMAX).

        Returns
        -------
        int
            Maximum diffraction order along x.
        """
        return int(
            self.spatial_freq_cutoff_x * self._sc.mask_width / self._sc.wavelength
        )

    @cached_property
    def max_diffraction_order_y(self) -> int:
        """
        Maximum diffraction order in y-direction (±MMAX).

        Returns
        -------
        int
            Maximum diffraction order along y.
        """
        return int(
            self.spatial_freq_cutoff_y * self._sc.mask_height / self._sc.wavelength
        )

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
