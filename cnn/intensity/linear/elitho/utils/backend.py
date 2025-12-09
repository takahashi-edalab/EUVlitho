xp = None


def use_backend(name: str):
    global xp
    if name == "numpy":
        import numpy as np

        xp = np
    elif name == "cupy":
        import cupy as cp

        xp = cp
    else:
        raise ValueError(f"Unknown backend: {name}")


def get_backend():
    global xp
    return xp
