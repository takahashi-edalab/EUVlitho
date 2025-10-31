import cupy as cp


def refine(mask: "xp.ndarray", scale_x: int, scale_y: int) -> "xp.ndarray":
    xp = cp.get_array_module(mask)
    refined_pattern = xp.kron(mask, xp.ones((scale_x, scale_y)))
    return refined_pattern


def center_slice(center: int, size: int) -> slice:
    half = size // 2
    start = center - half
    end = start + size
    return slice(start, end)


def extract_central_region(fmap: "xp.ndarray", ew: int, eh: int) -> "xp.ndarray":
    w, h = fmap.shape
    cx = w // 2
    cy = h // 2
    extracted_famp = fmap[
        center_slice(cx, ew),
        center_slice(cy, eh),
    ]
    return extracted_famp
