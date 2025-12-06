import numpy as np
from elitho import config


def count_overlapping_sources(
    sc: config.SimulationConfig, px: int, py: int, offset_x: int = 0, offset_y: int = 0
) -> int:
    n_sources = 0
    for is_src in range(sc.nsourceX):
        for js_src in range(sc.nsourceY):
            sx = is_src - sc.lsmaxX + offset_x / sc.ndivX
            sy = js_src - sc.lsmaxY + offset_y / sc.ndivY

            source_condition = (sx * sc.magnification_x / sc.mask_width) ** 2 + (
                sy * sc.magnification_y / sc.mask_height
            ) ** 2 <= (sc.NA / sc.wavelength) ** 2
            pupil_condition = (
                (px - sc.lpmaxX + sx) * sc.magnification_x / sc.mask_width
            ) ** 2 + (
                (py - sc.lpmaxY + sy) * sc.magnification_y / sc.mask_height
            ) ** 2 <= (
                sc.NA / sc.wavelength
            ) ** 2
            if source_condition and pupil_condition:
                n_sources += 1
    return n_sources


def find_valid_pupil_points(
    sc: config.SimulationConfig, nrange: int, offset_x: int = 0, offset_y: float = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    linput = np.zeros(nrange, dtype=int)
    minput = np.zeros(nrange, dtype=int)
    xinput = np.zeros(nrange, dtype=int)
    n_pupil_points = 0
    for ip in range(sc.noutX):
        for jp in range(sc.noutY):
            n_sources = count_overlapping_sources(sc, ip, jp, offset_x, offset_y)
            if n_sources > 0:
                linput[n_pupil_points] = ip - sc.lpmaxX
                minput[n_pupil_points] = jp - sc.lpmaxY
                xinput[n_pupil_points] = n_sources
                n_pupil_points += 1
    return linput, minput, xinput, n_pupil_points


class PupilCoordinates:
    def __init__(
        self,
        sc: config.SimulationConfig,
        nrange: int,
        offset_x: int = 0,
        offset_y: float = 0,
    ):
        (
            self._linput,
            self._minput,
            self._xinput,
            self._n_coordinates,
        ) = find_valid_pupil_points(sc, nrange, offset_x, offset_y)

    @property
    def linput(self) -> np.ndarray:
        return self._linput

    @property
    def minput(self) -> np.ndarray:
        return self._minput

    @property
    def xinput(self) -> np.ndarray:
        return self._xinput

    @property
    def n_coordinates(self) -> int:
        return self._n_coordinates
