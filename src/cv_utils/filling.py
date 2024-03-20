import cv2
import numpy as np
import numpy.typing as npt

from cv_utils import padding


def flood_fill_binary(binary: npt.NDArray[np.uint8], x_y: tuple[int, int]) -> npt.NDArray[np.uint8]:
    x, y = x_y
    value = binary[y, x]
    if value == np.max(binary):
        fill_value = int(np.min(binary))
    elif value == np.min(binary):
        fill_value = int(np.max(binary))
    else:
        raise ValueError("You passed non binary image")

    flags = 4 | (fill_value << 8) | cv2.FLOODFILL_FIXED_RANGE
    binary = cv2.floodFill(
        binary,
        None,
        seedPoint=(x, y),
        newVal=fill_value,
        loDiff=0,
        upDiff=0,
        flags=flags,
    )[1]
    return binary


def darken_areas_near_borders(binary: npt.NDArray[np.uint8 | bool]) -> npt.NDArray[np.uint8]:
    binary = binary.astype(np.uint8)

    binary = padding.equal(binary, size=1, value=int(np.max(binary)))
    binary = flood_fill_binary(binary, (0, 0))
    binary = binary[1:-1, 1:-1]

    return binary


def brighten_areas_near_borders(binary: npt.NDArray[np.uint8 | bool]) -> npt.NDArray[np.uint8]:
    binary = binary.astype(np.uint8)

    binary = padding.equal(binary, size=1, value=int(np.min(binary)))
    binary = flood_fill_binary(binary, (0, 0))
    binary = binary[1:-1, 1:-1]

    return binary
