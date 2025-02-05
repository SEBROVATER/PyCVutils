from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import cv2

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from cv2.typing import MatLike

from pycvutils import resizing


class MatchTemplateCallable(Protocol):
    def __call__(
        self,
        img: npt.NDArray[np.uint8] | MatLike,
        template: npt.NDArray[np.uint8] | MatLike,
        mask: npt.NDArray[np.uint8] | MatLike | None = None,
    ) -> MatLike: ...


def _match_template_wrapper(method: int) -> MatchTemplateCallable:
    def _match_template(
        img: npt.NDArray[np.uint8] | MatLike,
        template: npt.NDArray[np.uint8] | MatLike,
        mask: npt.NDArray[np.uint8] | MatLike | None = None,
    ) -> MatLike:
        try:
            return cv2.matchTemplate(img, template, mask=mask, method=method)
        except cv2.error as exc:
            raise ValueError(str(exc)) from exc

    return _match_template


ccoeff_norm = _match_template_wrapper(cv2.TM_CCOEFF_NORMED)
ccoeff = _match_template_wrapper(cv2.TM_CCOEFF)
ccorr_norm = _match_template_wrapper(cv2.TM_CCORR_NORMED)
ccorr = _match_template_wrapper(cv2.TM_CCORR)
sqdiff_norm = _match_template_wrapper(cv2.TM_SQDIFF_NORMED)
sqdiff = _match_template_wrapper(cv2.TM_SQDIFF)


def compare_with_crop(
    img: npt.NDArray[np.uint8],
    template: npt.NDArray[np.uint8],
    crop_ratio: float = 0.1,
) -> float | None:
    """Use CCOEFF_NORMED matching after center cropping template by some ratio.

    Returns:
        None: if 'img' or 'template' is empty numpy array
        float: for any not empty numpy image array

    """
    if img.size == 0 or template.size == 0:
        return None
    h, w, *_ = template.shape
    h_gap = int(crop_ratio * h) + 1
    w_gap = int(crop_ratio * w) + 1
    template = template[
        h_gap : h - h_gap,
        w_gap : w - w_gap,
    ]

    result = ccoeff_norm(img, template)
    return float(result.max())


def compare_one_to_one(
    img: npt.NDArray[np.uint8] | MatLike, template: npt.NDArray[np.uint8] | MatLike
) -> float | None:
    """Use CCOEFF_NORMED matching after resizing template to 'img' size.

    Returns:
        None: if 'img' or 'template' is empty numpy array
        float: for any not empty numpy image array

    """
    if img.size == 0 or template.size == 0:
        return None

    if img.shape != template.shape:
        h, w, *_ = img.shape
        resized = resizing.nearest(template, width=w, height=h)
        if resized is None:
            return None
        template = resized

    result = ccoeff_norm(img, template)
    return float(result.max())
