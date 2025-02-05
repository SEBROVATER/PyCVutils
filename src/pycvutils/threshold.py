from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from cv2.typing import MatLike


def binary(img: npt.NDArray[np.uint8] | MatLike, thr: int) -> MatLike | None:
    """Binarize image by specified 'thr' value like "img > thr".

    Returns:
        None: if 'img' is the empty image array
        npt.NDArray[np.uint8]: for any non-empty image array

    """
    return cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)[1]


def inv_binary(img: npt.NDArray[np.uint8] | MatLike, thr: int) -> MatLike | None:
    """Binarize image by specified 'thr' value like "img <= thr".

    Returns:
        None: if 'img' is the empty image array
        npt.NDArray[np.uint8]: for any non-empty image array

    """
    return cv2.threshold(img, thr, 255, cv2.THRESH_BINARY_INV)[1]


def otsu(img: npt.NDArray[np.uint8] | MatLike) -> MatLike | None:
    """Binarize image by calculating optimal 'thr' value and then "img > thr".

    Returns:
        None: if 'img' is the empty image array
        npt.NDArray[np.uint8]: for any non-empty image array

    """
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def inv_otsu(img: npt.NDArray[np.uint8] | MatLike) -> MatLike | None:
    """Binarize image by calculating optimal 'thr' value and then "img <= thr".

    Returns:
        None: If 'img' is the empty image array
        npt.NDArray[np.uint8]: for any non-empty image array

    """
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
