import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
from typing import ClassVar, Literal
from functools import cached_property

# # Default parameters
# MX = 4  # X magnification
# # Optical parameters
# is_high_na = True
# if is_high_na:
#     NA = 0.55
#     theta0 = -5.3  # incidence angle(degree) for high-NA
#     MY = 8  # Y magnification
#     NDIVX = 512  # X pitch (nm)
#     NDIVY = 2 * NDIVX  # Y pitch (nm)

# else:
#     NA = 0.33
#     theta0 = -6.0  # incidence angle (degree)
#     MY = 4  # Y magnification
#     NDIVX = 1024  # X pitch (nm)
#     NDIVY = 1024  # Y pitch (nm)


# dx = NDIVX
# dy = NDIVY
# XDIV = NDIVX // MX
# YDIV = NDIVY // MY
# MASK_REFINEMENT_FACTOR_X = 1  # Mask refinement factor in X
# MASK_REFINEMENT_FACTOR_Y = 1  # Mask refinement factor in Y

# wavelength = 13.5  # wavelength (nm)
# k = 2.0 * pi / wavelength
# kX = k * NA / MX
# kY = k * NA / MY
# azimuth = 0.0  # azimuthal angle (degree)
# phi0 = 90.0 - azimuth
# kx0 = k * np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
# ky0 = k * np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))

# mesh = 0.5
# co = 0.2  # central obscuration of the pupil for high-NA
# # Calculation parameters
# ndivX = max(1, int(180.0 / pi * wavelength / dx / mesh))
# ndivY = max(1, int(180.0 / pi * wavelength / dy / mesh))
# #
# lsmaxX = int(NA * dx / MX / wavelength + 1)
# lsmaxY = int(NA * dy / MY / wavelength + 1)
# lpmaxX = int(NA * dx / MX * 2 / wavelength + 0.0001)
# lpmaxY = int(NA * dy / MY * 2 / wavelength + 0.0001)
# nsourceX = 2 * lsmaxX + 1
# nsourceY = 2 * lsmaxY + 1
# noutX = 2 * lpmaxX + 1
# noutY = 2 * lpmaxY + 1
# nsourceXL = 2 * lsmaxX + 10
# nsourceYL = 2 * lsmaxY + 10
# noutXL = 2 * lpmaxX + 10
# noutYL = 2 * lpmaxY + 10


class PolarizationDirection(Enum):
    X = auto()
    Y = auto()


class IlluminationType(Enum):
    CIRCULAR = auto()
    ANNULAR = auto()
    DIPOLE_X = auto()
    DIPOLE_Y = auto()


class Illumination(ABC):
    pass
    # @abstractmethod
    # def description(self) -> str: ...


@dataclass
class CircularIllumination(Illumination):
    type: ClassVar[IlluminationType] = IlluminationType.CIRCULAR
    outer_sigma: float = 0.9


@dataclass
class AnnularIllumination(Illumination):
    type: ClassVar[IlluminationType] = IlluminationType.ANNULAR
    outer_sigma: float = 0.9
    inner_sigma: float = 0.55


@dataclass
class DipoleIllumination(Illumination):
    type: Literal[IlluminationType.DIPOLE_X, IlluminationType.DIPOLE_Y]
    outer_sigma: float = 0.9
    inner_sigma: float = 0.55
    open_angle: float = 90.0

    def __post_init__(self):
        if self.type not in (IlluminationType.DIPOLE_X, IlluminationType.DIPOLE_Y):
            raise ValueError("type must be 'DIPOLE_X' or 'DIPOLE_Y'")


@dataclass(frozen=True)
class SimulationConfig:
    wavelength: float = 13.5  # nm
    NA: float = 0.33
    is_high_na: bool = False
    illumination: Illumination = CircularIllumination()
    mask_width: int = 1024  # nm
    mask_height: int = 1024  # nm
    mask_refinement_factor_x: int = 1
    mask_refinement_factor_y: int = 1
    magnification_x: int = 4
    magnification_y: int = 4
    mesh: float = 0.5
    incidence_angle: float = -6.0  # degree
    azimuthal_angle: float = 0.0  # degree
    central_obscuration: float = 0.2
    defocus_min: float = 0.0  # nm
    defocus_max: float = None
    defocus_step: float = None
    # TODO:
    cutoff_factor: float = 6.0

    @property
    def defocus_list(self) -> list:
        if self.defocus_max is None or self.defocus_step is None:
            return [self.defocus_min]

        assert self.defocus_step > 0, "Invalid defocus step."
        assert self.defocus_max > self.defocus_min, "Invalid defocus range."
        return list(
            np.arange(
                self.defocus_min,
                self.defocus_max + self.defocus_step,
                self.defocus_step,
            )
        )

    @property
    def exposure_field_width(self):
        return self.mask_width // self.magnification_x

    @property
    def exposure_field_height(self):
        return self.mask_height // self.magnification_y

    @property
    def k(self):
        return 2.0 * np.pi / self.wavelength

    @property
    def kX(self):
        return self.k * self.NA / self.magnification_x

    @property
    def kY(self):
        return self.k * self.NA / self.magnification_y

    @property
    def phi0(self):
        return 90.0 - self.azimuth

    @property
    def kx0(self):
        return (
            self.k
            * np.sin(np.deg2rad(self.incidence_angle))
            * np.cos(np.deg2rad(self.phi0))
        )

    @property
    def ky0(self):
        return (
            self.k
            * np.sin(np.deg2rad(self.incidence_angle))
            * np.sin(np.deg2rad(self.phi0))
        )

    @property
    def ndivX(self):
        return max(
            1, int(180.0 / np.pi * self.wavelength / self.mask_width / self.mesh)
        )

    @property
    def ndivY(self):
        return max(
            1, int(180.0 / np.pi * self.wavelength / self.mask_height / self.mesh)
        )

    @property
    def lsmaxX(self):
        return int(
            self.NA * self.mask_width / self.magnification_x / self.wavelength + 1
        )

    @property
    def lsmaxY(self):
        return int(
            self.NA * self.mask_height / self.magnification_y / self.wavelength + 1
        )

    @property
    def lpmaxX(self):
        return int(
            self.NA * self.mask_width / self.magnification_x * 2 / self.wavelength
            + 0.0001
        )

    @property
    def lpmaxY(self):
        return int(
            self.NA * self.mask_height / self.magnification_y * 2 / self.wavelength
            + 0.0001
        )

    @property
    def nsourceX(self):
        return 2 * self.lsmaxX + 1

    @property
    def nsourceY(self):
        return 2 * self.lsmaxY + 1

    @property
    def noutX(self):
        return 2 * self.lpmaxX + 1

    @property
    def noutY(self):
        return 2 * self.lpmaxY + 1

    @property
    def nsourceXL(self):
        return 2 * self.lsmaxX + 10

    @property
    def nsourceYL(self):
        return 2 * self.lsmaxY + 10

    @property
    def noutXL(self):
        return 2 * self.lpmaxX + 10

    @property
    def noutYL(self):
        return 2 * self.lpmaxY + 10

    @cached_property
    def dkx(self):
        return 2.0 * np.pi / self.mask_width

    @cached_property
    def dky(self):
        return 2.0 * np.pi / self.mask_height


# Mask properties
NML = 40  # number of the multilayer pairs

# complex refractive index
n_mo = 0.9238 + 0.006435j  # Mo layer
n_si = 0.999 + 0.001826j  # Si layer
n_ru = 0.8863 + 0.01706j  # Ru layer
n_mo_si2 = 0.9693 + 0.004333j  # Mo/Si2 mixing layer
n_ru_si = 0.9099 + 0.01547j  # Ru/Si mixing layer
n_si_o2 = 0.978 + 0.01083j  # SiO2 layer

# thickness [nm]
thickness_mo = 2.052
thickness_si = 2.283
thickness_ru = 2.5
thickness_mo_si = 1.661
thickness_si_mo = 1.045
thickness_si_ru = 0.8


# complex permittivity
epsilon_mo = n_mo**2
epsilon_si = n_si**2
epsilon_ru = n_ru**2
epsilon_mo_si2 = n_mo_si2**2
epsilon_ru_si = n_ru_si**2
epsilon_si_o2 = n_si_o2**2


dabst = 60.0  # absorber thickness (nm)
z0 = dabst + 42.0  # reflection point inside ML from the top of the absorber

# absorber properties
nta = 0.9567 + 0.0343j  # absorber complex refractive index
absorption_amplitudes = [nta**2]
absorber_layer_thicknesses = [dabst]
