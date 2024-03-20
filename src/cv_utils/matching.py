import cv2
import numpy as np
import numpy.typing as npt

from cv_utils import resizing


def compare_with_crop(
    old_image: npt.NDArray[np.uint8], new_image: npt.NDArray[np.uint8], crop_ratio: float = 0.1
) -> float:
    h, w, *c = new_image.shape
    new_image = new_image[
        int(crop_ratio * h + 1) : int((1 - crop_ratio) * h),
        int(crop_ratio * w + 1) : int((1 - crop_ratio) * w),
    ]
    try:
        result = cv2.matchTemplate(old_image, new_image, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        raise ValueError(
            f"opencv error: old_image shape: {old_image.shape}, new_image shape: {new_image.shape}"
        )
    return result.max()


def compare_one_to_one(old_image: npt.NDArray[np.uint8], new_image: npt.NDArray[np.uint8]) -> float:
    if old_image.shape != new_image.shape:
        new_image = resizing.nearest(new_image, old_image.shape[1], old_image.shape[0])

    result = cv2.matchTemplate(old_image, new_image, cv2.TM_CCOEFF_NORMED)
    return result.max()


def compare(old_image: npt.NDArray[np.uint8], new_image: npt.NDArray[np]) -> float:
    result = cv2.matchTemplate(old_image, new_image, cv2.TM_CCOEFF_NORMED)
    return result.max()
