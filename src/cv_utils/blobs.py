import numpy as np
import numpy.typing as npt

from .coords import x1_y1_x2_y2


def get_bright_rect(binary: npt.NDArray[np.uint8]) -> x1_y1_x2_y2:
    nonzeros = binary.nonzero()
    try:
        y1, x1 = np.minimum.reduce(nonzeros, axis=1)
        y2, x2 = np.maximum.reduce(nonzeros, axis=1)
    except ValueError:
        raise ValueError("Image is completely black")
    return x1_y1_x2_y2(x1, y1, x2, y2)
