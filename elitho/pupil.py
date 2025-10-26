import numpy as np
from elitho import const


def count_overlapping_sources(px: int, py: int) -> int:
    """
    Count the number of illumination source points contributing to a given pupil coordinate.

    For a specified pupil coordinate (ip, jp), this function iterates over all
    illumination source points and checks two conditions:
    1. The source point lies within the illumination aperture.
    2. The shifted pupil coordinate (considering the source offset) lies within
       the projection lens aperture.

    The count of source points satisfying both conditions indicates how many
    illumination directions can contribute to this pupil point.

    Parameters
    ----------
    ip : int
        X-index of the pupil-plane coordinate.
    jp : int
        Y-index of the pupil-plane coordinate.

    Returns
    -------
    int
        Number of valid illumination sources overlapping with the given pupil point.
    """
    n_sources = 0
    for is_src in range(const.nsourceX):
        for js_src in range(const.nsourceY):
            source_condition = ((is_src - const.lsmaxX) * const.MX / const.dx) ** 2 + (
                (js_src - const.lsmaxY) * const.MY / const.dy
            ) ** 2 <= (const.NA / const.wavelength) ** 2
            pupil_condition = (
                (px - const.lpmaxX + is_src - const.lsmaxX) * const.MX / const.dx
            ) ** 2 + (
                (py - const.lpmaxY + js_src - const.lsmaxY) * const.MY / const.dy
            ) ** 2 <= (
                const.NA / const.wavelength
            ) ** 2
            if source_condition and pupil_condition:
                n_sources += 1
    return n_sources


def find_valid_pupil_points(
    nrange: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Identify all valid pupil-plane coordinates contributing to image formation.

    This function scans the entire pupil plane and determines which coordinates
    receive illumination from at least one valid source direction. A pupil point
    is considered valid if `count_overlapping_sources(ip, jp)` returns a positive value.

    The resulting coordinates define the effective support of the pupil function,
    which depends on both the illumination aperture and the projection lens NA.

    Parameters
    ----------
    nrange : int
        Maximum number of candidate pupil points to evaluate (array length).

    Returns
    -------
    linput : np.ndarray
        Array of valid pupil X-coordinates (relative to the pupil center).
    minput : np.ndarray
        Array of valid pupil Y-coordinates (relative to the pupil center).
    xinput : np.ndarray
        Array of valid number of sources.
    n_pupil_points : int
        Number of valid pupil points contributing to image formation.
    """
    linput = np.zeros(nrange, dtype=int)
    minput = np.zeros(nrange, dtype=int)
    xinput = np.zeros(nrange, dtype=int)
    n_pupil_points = 0
    for ip in range(const.noutX):
        for jp in range(const.noutY):
            n_sources = count_overlapping_sources(ip, jp)
            if n_sources > 0:
                linput[n_pupil_points] = ip - const.lpmaxX
                minput[n_pupil_points] = jp - const.lpmaxY
                xinput[n_pupil_points] = n_sources
                n_pupil_points += 1
    return linput, minput, xinput, n_pupil_points
