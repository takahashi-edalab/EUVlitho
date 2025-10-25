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
