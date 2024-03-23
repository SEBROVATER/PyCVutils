from typing import Any

import numpy.typing as npt


def split_view(img: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], ...] | None:
    """
    Split image by channels without making copies.
    Can slow down next processing in comparison to 'cv2.split'
    """
    if img.ndim == 2:
        return (img,)
    try:
        return tuple(img[:, :, n] for n in range(img.shape[2]))
    except IndexError:
        return None
