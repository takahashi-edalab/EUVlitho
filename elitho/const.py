import numpy as np
from enum import Enum


pi = np.pi
i_complex = 1j  # 複素単位

# Default parameters
MX = 4  # X magnification
MY = 4  # Y magnification
# NDIVX = 2048  # X pitch (nm)
# NDIVY = 2048  # Y pitch (nm)
NDIVX = 512  # X pitch (nm)
NDIVY = 512  # Y pitch (nm)
dx = NDIVX
dy = NDIVY
XDIV = NDIVX // MX
YDIV = NDIVY // MY
MASK_REFINEMENT_FACTOR_X = 1  # Mask refinement factor in X
MASK_REFINEMENT_FACTOR_Y = 1  # Mask refinement factor in Y

# Optical parameters
wavelength = 13.5  # wavelength (nm)
k = 2.0 * pi / wavelength
theta0 = -6.0  # chief ray angle (degree)
azimuth = 0.0  # azimuthal angle (degree)
phi0 = 90.0 - azimuth


# NA = 0.33
NA = 0.55
kx0 = k * np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
ky0 = k * np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))


# optical type
mesh = 0.1


class PolarizationDirection(Enum):
    X = 0
    Y = 1


class IlluminationType(Enum):
    CIRCULAR = 0
    ANNULAR = 1
    DIPOLE = 2


illumination_type = IlluminationType.DIPOLE
sigma1 = 0.9  # outer sigma
sigma2 = 0.55  # inner sigma
openangle = 90.0  # opening angle for dipole illumination


# Material properties
nta = 0.9567 + 0.0343j  # absorber complex refractive index
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


# Calculation parameters
sigmadiv = 2.0  # division angle of the source (degree)
ndivs = max(1, int(180.0 / pi * wavelength / dx / sigmadiv))
lsmaxX = int(NA * dx / MX / wavelength + 1)
lsmaxY = int(NA * dy / MY / wavelength + 1)
lpmaxX = int(NA * dx / MX * 2 / wavelength + 0.0001)
lpmaxY = int(NA * dy / MY * 2 / wavelength + 0.0001)
nsourceX = 2 * lsmaxX + 1
nsourceY = 2 * lsmaxY + 1
noutX = 2 * lpmaxX + 1
noutY = 2 * lpmaxY + 1
nsourceXL = 2 * lsmaxX + 10
nsourceYL = 2 * lsmaxY + 10
noutXL = 2 * lpmaxX + 10
noutYL = 2 * lpmaxY + 10


# ?
z = 0.0  # defocus
dabst = 60.0  # absorber thickness (nm)
z0 = dabst + 42.0  # reflection point inside ML from the top of the absorber


absorption_amplitudes = [nta**2]
absorber_layer_thicknesses = [dabst]
