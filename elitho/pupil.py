import numpy as np
from elitho import config


def count_overlapping_sources(
    px: int, py: int, offset_x: int = 0, offset_y: int = 0
) -> int:
    n_sources = 0
    for is_src in range(config.nsourceX):
        for js_src in range(config.nsourceY):
            sx = is_src - config.lsmaxX + offset_x / config.ndivX
            sy = js_src - config.lsmaxY + offset_y / config.ndivY

            source_condition = (sx * config.MX / config.dx) ** 2 + (
                sy * config.MY / config.dy
            ) ** 2 <= (config.NA / config.wavelength) ** 2
            pupil_condition = (
                (px - config.lpmaxX + sx) * config.MX / config.dx
            ) ** 2 + ((py - config.lpmaxY + sy) * config.MY / config.dy) ** 2 <= (
                config.NA / config.wavelength
            ) ** 2
            if source_condition and pupil_condition:
                n_sources += 1
    return n_sources


def find_valid_pupil_points(
    nrange: int, offset_x: int = 0, offset_y: float = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    linput = np.zeros(nrange, dtype=int)
    minput = np.zeros(nrange, dtype=int)
    xinput = np.zeros(nrange, dtype=int)
    n_pupil_points = 0
    for ip in range(config.noutX):
        for jp in range(config.noutY):
            n_sources = count_overlapping_sources(ip, jp, offset_x, offset_y)
            if n_sources > 0:
                linput[n_pupil_points] = ip - config.lpmaxX
                minput[n_pupil_points] = jp - config.lpmaxY
                xinput[n_pupil_points] = n_sources
                n_pupil_points += 1
    return linput, minput, xinput, n_pupil_points


class PupilCoordinates:
    def __init__(self, nrange: int, offset_x: int = 0, offset_y: float = 0):
        (
            self._linput,
            self._minput,
            self._xinput,
            self._n_coordinates,
        ) = find_valid_pupil_points(nrange, offset_x, offset_y)

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
