import cv2
import numpy as np
import numpy.typing as npt


def binary(gray: npt.NDArray[np.uint8], thr: int) -> npt.NDArray[np.uint8]:
    return cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)[1]


def inv_binary(gray: npt.NDArray[np.uint8], thr: int) -> npt.NDArray[np.uint8]:
    return cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)[1]


def otsu(gray: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def inv_otsu(gray: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
