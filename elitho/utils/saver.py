import numpy as np


def save_pupil_points(
    filename: str,
    linput: np.ndarray,
    minput: np.ndarray,
    xinput: np.ndarray,
    n_pupil_points: int,
) -> None:

    import csv

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for n in range(n_pupil_points):
            writer.writerow([linput[n], minput[n], xinput[n]])


def save_mask(filename: str, mask: np.ndarray) -> None:
    compressed_bytes = np.packbits(mask)
    np.savez(filename, data=compressed_bytes, shape=mask.shape)


def load_mask(fpath: str) -> np.ndarray:
    data = np.load(fpath)
    unpacked = np.unpackbits(data["data"])
    mask = unpacked.reshape(data["shape"])
    return mask


def save_m3d_params(
    file_path: str, a0xx: np.ndarray, axxx: np.ndarray, ayxx: np.ndarray
) -> None:
    np.savez(file_path, a0xx=a0xx, axxx=axxx, ayxx=ayxx)


def load_m3d_params(
    file_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = np.load(file_path)
    a0xx = params["a0xx"]
    axxx = params["axxx"]
    ayxx = params["ayxx"]
    return a0xx, axxx, ayxx
