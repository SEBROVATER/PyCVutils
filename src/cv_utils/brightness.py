import cv2
import numpy as np
import numpy.typing as npt

from . import padding
from .blobs import get_bright_rect
from .filling import darken_areas_near_borders


def has_any_bright_border(binary: npt.NDArray[np.uint8 | np.bool_]) -> bool | None:
    try:
        return bool(
            binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any()
        )
    except IndexError:
        return None


def has_any_bright_corner(binary: npt.NDArray[np.uint8 | np.bool_]) -> bool | None:
    try:
        return bool(binary[0, 0] or binary[0, -1] or binary[-1, 0] or binary[-1, -1])
    except IndexError:
        return None


def crop_bright_area_and_pad(
    bgr_or_gray: npt.NDArray[np.uint8],
    thr: int | float = 125,
    inverse: bool = False,
    pad_size: int = 5,
    darken_borders: bool = False,
) -> npt.NDArray[np.uint8] | None:
    """
    Detects bright area using thresholding,
    crops it and pads it using median color if needed
    """
    if bgr_or_gray.size == 0:
        return bgr_or_gray

    type_ = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    img = bgr_or_gray
    if len(img.shape) == 2:
        median_color = int(np.median(img))
        gray = img
    else:
        median_color = tuple(np.median(img.reshape((-1, 3)), axis=0).astype(np.uint8).tolist())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thr, binary = cv2.threshold(src=gray, thresh=thr, maxval=255, type=type_)
    if darken_borders:
        binary = darken_areas_near_borders(binary)

    rect = get_bright_rect(binary)
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    if x1 == x2 or y1 == y2:
        return None

    img = padding.equal(img[y1:y2, x1:x2], value=median_color, size=pad_size)
    return img
