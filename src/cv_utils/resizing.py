import cv2
import numpy as np
import numpy.typing as npt


def _resize_wrapper(interpolation: int):
    def _resize(
            img: npt.NDArray[np.uint8],
            width: int | None = None,
            height: int | None = None,
    ) -> npt.NDArray[np.uint8]:
        if width is None and height is None:
            raise ValueError("Either width or height must be specified")
        if width is None:
            width = int(img.shape[1] * (height / img.shape[0]))
        if height is None:
            height = int(img.shape[0] * (width / img.shape[1]))

        if img.shape[0] == height and img.shape[1] == width:
            return img

        return cv2.resize(img, (width, height), interpolation=interpolation)

    return _resize


nearest = _resize_wrapper(cv2.INTER_NEAREST)
area = _resize_wrapper(cv2.INTER_AREA)
linear = _resize_wrapper(cv2.INTER_LINEAR)
cubic = _resize_wrapper(cv2.INTER_CUBIC)
lanczos4 = _resize_wrapper(cv2.INTER_LANCZOS4)
