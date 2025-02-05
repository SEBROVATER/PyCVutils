from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import cv2

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from cv2.typing import MatLike


class ResizeByRatioCallable(Protocol):
    def __call__(
        self,
        img: npt.NDArray[np.uint8] | MatLike,
        fx: float | None = None,
        fy: float | None = None,
    ) -> MatLike | None: ...


def _resize_by_ratio_wrapper(
    interpolation: int,
) -> ResizeByRatioCallable:
    def _resize(
        img: npt.NDArray[np.uint8] | MatLike,
        fx: float | None = None,
        fy: float | None = None,
    ) -> MatLike | None:
        if img.size == 0:
            return None

        if fx is None:
            if fy is None:
                return img
            fx = fy

        elif fy is None:
            fy = fx

        return cv2.resize(img, dsize=None, fx=fx, fy=fy, interpolation=interpolation)

    return _resize


nearest = _resize_by_ratio_wrapper(cv2.INTER_NEAREST)
area = _resize_by_ratio_wrapper(cv2.INTER_AREA)
linear = _resize_by_ratio_wrapper(cv2.INTER_LINEAR)
cubic = _resize_by_ratio_wrapper(cv2.INTER_CUBIC)
lanczos4 = _resize_by_ratio_wrapper(cv2.INTER_LANCZOS4)
