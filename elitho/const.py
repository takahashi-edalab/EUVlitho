import numpy as np

pi = np.pi
i_complex = 1j  # 複素単位

# Default parameters
MX = 4  # X magnification
MY = 4  # Y magnification
NDIVX = 2048  # X pitch (nm)
NDIVY = 2048  # Y pitch (nm)
dx = NDIVX
dy = NDIVY
XDIV = NDIVX // MX
YDIV = NDIVY // MY


# Optical parameters
wavelength = 13.5  # wavelength (nm)
k = 2.0 * pi / wavelength
theta0 = -6.0  # chief ray angle (degree)
azimuth = 0.0  # azimuthal angle (degree)
phi0 = 90.0 - azimuth
NA = 0.33
kx0 = k * np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
ky0 = k * np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))


# optical type
mesh = 0.1
optical_type = 2  # 0: circular, 1: annular, 2: dipole
sigma1 = 0.9  # outer sigma
sigma2 = 0.55  # inner sigma
openangle = 90.0  # opening angle for dipole illumination


# Material properties
nta = 0.9567 + 0.0343j  # absorber complex refractive index
NML = 40  # number of the multilayer pairs
NABS = 1  # number of the absorber layers
dabst = 60.0  # total absorber thickness (nm)
z = 0.0  # defocus
z0 = dabst + 42.0  # reflection point inside ML from the top of the absorber

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
delta = 1.0
FDIVX = int(dx / delta + 0.000001)
FDIVY = int(dy / delta + 0.000001)
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


# TODO: constじゃないっぽい -> linear lithoは
cutx = NA / MX * 6.0
cuty = NA / MY * 6.0
LMAX = int(cutx * dx / wavelength)
MMAX = int(cuty * dy / wavelength)
Lrange = 2 * LMAX + 1
Mrange = 2 * MMAX + 1
Lrange2 = 4 * LMAX + 1
Mrange2 = 4 * MMAX + 1


def diffraction_order_limits(LMAX: int, MMAX: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate diffraction order limits efficiently using NumPy vectorization"""
    # create 1D index arrays
    lvals = np.arange(-LMAX, LMAX + 1)
    mvals = np.arange(-MMAX, MMAX + 1)
    # create 2D grids
    ll, mm = np.meshgrid(lvals, mvals, indexing="ij")
    # apply the condition
    mask = (abs(ll) / (LMAX + 0.01) + 1.0) * (abs(mm) / (MMAX + 0.01) + 1.0) <= 2.0
    # extract indices that satisfy the condition
    lindex = ll[mask]
    mindex = mm[mask]
    return lindex, mindex


lindex, mindex = diffraction_order_limits(LMAX, MMAX)
Nrange = len(lindex)
Nrange2 = Nrange * 2


eabs = np.zeros(100, dtype=complex)  # 各層の複素誘電率
eabs[0] = nta**2  # 吸収体
dabs = np.zeros(100)
dabst = 60.0  # total absorber thickness (nm)
dabs[0] = dabst
z = 0.0  # defocus
z0 = dabst + 42.0  # reflection point inside ML from the top of the absorber

# TODO: delete this
cexpX = np.exp(-2j * np.pi * np.arange(FDIVX + 1) / FDIVX)
cexpY = np.exp(-2j * np.pi * np.arange(FDIVY + 1) / FDIVY)
