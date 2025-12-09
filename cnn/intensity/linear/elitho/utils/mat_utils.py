import numpy as np
import cupy as cp
import torch


def linalg_eig(A: "xp.ndarray") -> tuple["xp.ndarray", "xp.ndarray"]:
    xp = cp.get_array_module(A)
    if xp == cp:
        with torch.no_grad():
            tA = torch.from_dlpack(A)
            tw, tv = torch.linalg.eig(tA)
        w = cp.from_dlpack(tw)
        v = cp.from_dlpack(tv)
    elif xp == np:
        w, v = np.linalg.eig(A)
    else:
        raise ValueError("Unsupported array module")

    return w, v
