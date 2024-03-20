import numpy as np
import numpy.typing as npt

from .coords import x1_y1_x2_y2


def get_bright_rect(binary: npt.NDArray[np.uint8 | np.bool_]) -> x1_y1_x2_y2 | None:
    """
    Returns rect around all bright (non-zero) blobs
    """
    nonzeros = binary.nonzero()
    try:
        y1, x1 = np.minimum.reduce(nonzeros, axis=1)
        y2, x2 = np.maximum.reduce(nonzeros, axis=1)
    except ValueError:
        return None
    return x1_y1_x2_y2(x1, y1, x2 + 1, y2 + 1)
