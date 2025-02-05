from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import cv2

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from cv2.typing import MatLike


class ResizeCallable(Protocol):
    def __call__(
        self,
        img: npt.NDArray[np.uint8] | MatLike,
        width: int | None = None,
        height: int | None = None,
    ) -> MatLike | None: ...


def _resize_wrapper(interpolation: int) -> ResizeCallable:
    def _resize(
        img: npt.NDArray[np.uint8] | MatLike,
        width: int | None = None,
        height: int | None = None,
    ) -> MatLike | None:
        if img.size == 0:
            return None

        h, w, *_ = img.shape
        if width is None:
            if height is None:
                return img
            width = int(w * (height / h))

        elif height is None:
            height = int(h * (width / w))

        return cv2.resize(img, (width, height), interpolation=interpolation)

    return _resize


nearest = _resize_wrapper(cv2.INTER_NEAREST)
area = _resize_wrapper(cv2.INTER_AREA)
linear = _resize_wrapper(cv2.INTER_LINEAR)
cubic = _resize_wrapper(cv2.INTER_CUBIC)
lanczos4 = _resize_wrapper(cv2.INTER_LANCZOS4)
