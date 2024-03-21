import cv2
import numpy as np
import numpy.typing as npt
from cv_utils import resizing


def _match_template_wrapper(method: int):
    def _match_template(
        img: npt.NDArray[np.uint8],
        template: npt.NDArray[np.uint8],
        mask: npt.NDArray[np.uint8] | None = None,
    ):
        return cv2.matchTemplate(img, template, mask=mask, method=method)

    return _match_template


ccoeff_norm = _match_template_wrapper(cv2.TM_CCOEFF_NORMED)
ccoeff = _match_template_wrapper(cv2.TM_CCOEFF)
ccorr_norm = _match_template_wrapper(cv2.TM_CCORR_NORMED)
ccorr = _match_template_wrapper(cv2.TM_CCORR)
sqdiff_norm = _match_template_wrapper(cv2.TM_SQDIFF_NORMED)
sqdiff = _match_template_wrapper(cv2.TM_SQDIFF)


def compare_with_crop(
    img: npt.NDArray[np.uint8], template: npt.NDArray[np.uint8], crop_ratio: float = 0.1
) -> float | None:
    """
    Uses CCOEFF_NORMED matching after center cropping template by some ratio
    """
    if img.size == 0 or template.size == 0:
        return None
    h, w, *c = template.shape
    h_gap = int(crop_ratio * h) + 1
    w_gap = int(crop_ratio * w) + 1
    template = template[
        h_gap : h - h_gap,
        w_gap : w - w_gap,
    ]

    result = ccoeff_norm(img, template)
    return float(result.max())


def compare_one_to_one(old_image: npt.NDArray[np.uint8], new_image: npt.NDArray[np.uint8]) -> float:
    if old_image.shape != new_image.shape:
        new_image = resizing.nearest(new_image, old_image.shape[1], old_image.shape[0])

    result = cv2.matchTemplate(old_image, new_image, cv2.TM_CCOEFF_NORMED)
    return result.max()


def compare(old_image: npt.NDArray[np.uint8], new_image: npt.NDArray[np]) -> float:
    result = cv2.matchTemplate(old_image, new_image, cv2.TM_CCOEFF_NORMED)
    return result.max()
