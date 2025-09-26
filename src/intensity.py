import numpy as np
import pandas as pd

try:
    from scipy.fft import fft2, ifft2
except Exception:
    # fall back to numpy.fft if scipy.fft is not available in the environment
    from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def exponential(FDIV: int) -> np.ndarray:
    """Calculate exponential terms for Fourier transform"""
    cexp = np.zeros(FDIV + 1, dtype=complex)
    for i in range(FDIV + 1):
        cexp[i] = np.exp(-2j * np.pi * i / FDIV)
    return cexp


def fourier(l: int, f: np.ndarray, cexp: np.ndarray, FDIV: int) -> complex:
    i = np.arange(FDIV)
    if l >= 0:
        j = (i * l) % FDIV
    else:
        j = FDIV - ((-i * l) % FDIV)
    return np.sum(f * cexp[j]) / FDIV


def find_valid_output_points(
    noutX: int,
    noutY: int,
    nsourceX: int,
    nsourceY: int,
    lsmaxX: int,
    lsmaxY: int,
    lpmaxX: int,
    lpmaxY: int,
    NA: float,
    dx: float,
    dy: float,
    MX: int,
    MY: int,
    lambda_wl: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Find valid output points based on source and pupil conditions"""
    linput = []
    minput = []
    for ip in range(noutX):
        for jp in range(noutY):
            snum = 0
            for is_src in range(nsourceX):
                for js_src in range(nsourceY):
                    source_condition = ((is_src - lsmaxX) * MX / dx) ** 2 + (
                        (js_src - lsmaxY) * MY / dy
                    ) ** 2 <= (NA / lambda_wl) ** 2
                    pupil_condition = (
                        (ip - lpmaxX + is_src - lsmaxX) * MX / dx
                    ) ** 2 + ((jp - lpmaxY + js_src - lsmaxY) * MY / dy) ** 2 <= (
                        NA / lambda_wl
                    ) ** 2
                    if source_condition and pupil_condition:
                        snum += 1

            if snum > 0:
                linput.append(ip - lpmaxX)
                minput.append(jp - lpmaxY)

    linput = np.array(linput)
    minput = np.array(minput)
    return linput, minput


class EUVIntensityCalculator:
    """EUV Lithography Intensity Calculation Class

    This class calculates the aerial image intensity for EUV lithography
    based on mask patterns and illumination conditions.
    """

    def __init__(self):
        pass

    def fourier(self, l: int, f: np.ndarray, cexp: np.ndarray, FDIV: int) -> complex:
        """Calculate Fourier coefficient"""
        result = 0.0 + 0.0j
        for i in range(FDIV):
            result += f[i] * cexp[(l * i) % (FDIV + 1)]
        return result / FDIV

    def load_mask(self, filename: str) -> np.ndarray:
        shape = (self.NDIVY, self.NDIVX)
        data = np.fromfile(filename, dtype=np.uint8)
        unpacked = np.unpackbits(data)[: np.prod(shape)]
        # mask_restored = unpacked.reshape(shape)
        return unpacked

    def calculate_intensity(
        self, mask_file: str, output_file: str = "emint.csv"
    ) -> None:
        """Main intensity calculation function"""
        print("Starting EUV intensity calculation...")

        # Load mask pattern
        mask2d = self.load_mask(mask_file)
        print(f"Loaded mask pattern from {mask_file}")
        print(f"Mask pattern shape: {mask2d.shape}")

        # Calculate source parameters
        ndivs = max(1, int(180.0 / self.pi * self.lambda_wl / self.dx / self.sigmadiv))
        l0s, m0s, SDIV = source(
            self.NA,
            self.type,
            self.sigma1,
            self.sigma2,
            self.openangle,
            self.k,
            self.dx,
            self.dy,
            ndivs,
            self.MX,
            self.MY,
        )

        SDIVMAX = np.max(SDIV)
        SDIVSUM = np.sum(SDIV)
        print(
            f"Source divisions: {ndivs}x{ndivs}, SDIVMAX: {SDIVMAX}, SDIVSUM: {SDIVSUM}"
        )

        # Calculate diffraction limits
        lsmaxX = int(self.NA * self.dx / self.MX / self.lambda_wl) + 1
        lsmaxY = int(self.NA * self.dy / self.MY / self.lambda_wl) + 1
        lpmaxX = int(self.NA * self.dx / self.MX * 2 / self.lambda_wl) + 1
        lpmaxY = int(self.NA * self.dy / self.MY * 2 / self.lambda_wl) + 1

        nsourceX = 2 * lsmaxX + 1
        nsourceY = 2 * lsmaxY + 1
        noutX = 2 * lpmaxX + 1
        noutY = 2 * lpmaxY + 1

        FDIVX = int(self.dx / self.delta + 1e-6)
        FDIVY = int(self.dy / self.delta + 1e-6)
        cexpx = exponential(FDIVX)
        cexpy = exponential(FDIVY)
        print(
            f"Diffraction parameters: lsmax=({lsmaxX},{lsmaxY}), lpmax=({lpmaxX},{lpmaxY})"
        )

        # Setup diffraction order limits
        lindex, mindex = difraction_order_limits()
        Nrange = len(lindex)
        print(f"Number of diffraction orders: {Nrange}")

        # Find valid output points
        linput, minput = find_valid_output_points(
            noutX,
            noutY,
            nsourceX,
            nsourceY,
            lsmaxX,
            lsmaxY,
            lpmaxX,
            lpmaxY,
            self.NA,
            self.dx,
            self.dy,
            self.MX,
            self.MY,
            self.lambda_wl,
        )

        ncut = len(linput)
        linput = np.array(linput)
        minput = np.array(minput)
        print(f"Number of valid output points: {ncut}")

        # Initialize intensity arrays
        isum = np.zeros((ndivs, ndivs, self.XDIV, self.XDIV, SDIVMAX))

        # Main calculation loop
        print("Starting main calculation loop...")
        for nsx in range(ndivs):
            for nsy in range(ndivs):
                print(f"Processing source point ({nsx+1}/{ndivs}, {nsy+1}/{ndivs})")

                # Calculate source angles
                kx0 = (
                    self.k
                    * np.sin(self.pi / 180.0 * self.theta0)
                    * np.cos(self.pi / 180.0 * self.phi0)
                )
                ky0 = (
                    self.k
                    * np.sin(self.pi / 180.0 * self.theta0)
                    * np.sin(self.pi / 180.0 * self.phi0)
                )
                sx0 = 2.0 * self.pi / self.dx * nsx / ndivs + kx0
                sy0 = 2.0 * self.pi / self.dy * nsy / ndivs + ky0

                # Calculate diffraction amplitudes (simplified)
                Ax = self.simplified_ampS("X", mask2d, nsourceX, nsourceY)

                # Process each source point
                for is_src in range(SDIV[nsx, nsy]):
                    if is_src >= len(l0s[nsx][nsy]):
                        continue

                    # Calculate wave vectors
                    kx = sx0 + 2.0 * self.pi / self.dx * l0s[nsx][nsy][is_src]
                    ky = sy0 + 2.0 * self.pi / self.dy * m0s[nsx][nsy][is_src]
                    ls = l0s[nsx][nsy][is_src] + lsmaxX
                    ms = m0s[nsx][nsy][is_src] + lsmaxY

                    # Initialize field arrays
                    fnx = np.zeros((self.XDIV, self.XDIV), dtype=complex)
                    fny = np.zeros((self.XDIV, self.XDIV), dtype=complex)
                    fnz = np.zeros((self.XDIV, self.XDIV), dtype=complex)

                    # Calculate electric field components
                    for n in range(ncut):
                        kxn = kx + 2.0 * self.pi * linput[n] / self.dx
                        kyn = ky + 2.0 * self.pi * minput[n] / self.dy

                        if (self.MX**2 * kxn**2 + self.MY**2 * kyn**2) <= (
                            self.NA * self.k
                        ) ** 2:
                            # Calculate field amplitude
                            if (
                                0 <= ls < nsourceX
                                and 0 <= ms < nsourceY
                                and n < Ax.shape[2]
                            ):
                                Ax_val = Ax[ls, ms, n]

                                # Calculate phase
                                phase = np.exp(
                                    1j
                                    * ((kxn + kx0) ** 2 + (kyn + ky0) ** 2)
                                    / 2.0
                                    / self.k
                                    * self.z0
                                    + 1j
                                    * (self.MX**2 * kxn**2 + self.MY**2 * kyn**2)
                                    / 2.0
                                    / self.k
                                    * self.z
                                )

                                # Calculate electric field components
                                fx = Ax_val / np.sqrt(self.k**2 - kx**2) * phase
                                fy = 0.0  # Simplified: only X polarization
                                fz = 0.0  # Simplified: no Z component

                                # Map to FFT grid
                                ix = linput[n]
                                iy = minput[n]
                                px = (ix + self.XDIV) % self.XDIV
                                py = (iy + self.XDIV) % self.XDIV

                                fnx[px, py] = fx
                                fny[px, py] = fy
                                fnz[px, py] = fz

                    # Inverse FFT to get spatial field distribution
                    fnx_spatial = ifft2(fnx) * self.XDIV**2
                    fny_spatial = ifft2(fny) * self.XDIV**2
                    fnz_spatial = ifft2(fnz) * self.XDIV**2

                    # Calculate intensity
                    intensity = (
                        np.abs(fnx_spatial) ** 2
                        + np.abs(fny_spatial) ** 2
                        + np.abs(fnz_spatial) ** 2
                    )

                    isum[nsx, nsy, :, :, is_src] = intensity

        # Sum intensities from all source points
        print("Summing intensities...")
        final_intensity = np.zeros((self.XDIV, self.XDIV))
        for i in range(self.XDIV):
            for j in range(self.XDIV):
                total_sum = 0.0
                for nsx in range(ndivs):
                    for nsy in range(ndivs):
                        for is_src in range(SDIV[nsx, nsy]):
                            total_sum += isum[nsx, nsy, i, j, is_src]
                final_intensity[i, j] = total_sum / SDIVSUM

        # Save results
        self.save_intensity_csv(final_intensity, output_file)
        print(f"Intensity calculation completed. Results saved to {output_file}")

    def save_intensity_csv(self, intensity: np.ndarray, filename: str) -> None:
        """Save intensity data to CSV file in the same format as the original C++ code"""
        with open(filename, "w") as f:
            f.write("data,1\n")
            f.write("memo1\n")
            f.write("memo2\n")

            # Write X coordinates header
            f.write(",")
            for i in range(self.XDIV):
                x = i * self.dx / self.XDIV / self.MX
                f.write(f"{x}")
                if i < self.XDIV - 1:
                    f.write(",")
            f.write("\n")

            # Write data rows
            for j in range(self.XDIV):
                y = j * self.dy / self.XDIV / self.MY
                f.write(f"{y}")
                for i in range(self.XDIV):
                    f.write(f",{intensity[i, j]}")
                f.write("\n")


def main():
    """Main function to run EUV intensity calculation"""
    calculator = EUVIntensityCalculator()

    # Set paths
    mask_file = "mask.csv"
    output_file = "emint.csv"

    # Check if running from emint directory
    if os.path.exists("../emint/mask.csv"):
        mask_file = "../emint/mask.csv"
        output_file = "../emint/emint.csv"
    elif os.path.exists("emint/mask.csv"):
        mask_file = "emint/mask.csv"
        output_file = "emint/emint.csv"

    # Run calculation
    try:
        calculator.calculate_intensity(mask_file, output_file)
        print("EUV intensity calculation completed successfully!")
    except Exception as e:
        print(f"Error in calculation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
