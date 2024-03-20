import numpy as np
import numpy.typing as npt


def split_view(img: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.uint8], ...]:
    if img.ndim == 2:
        return (img,)
    return tuple(img[:, :, n] for n in range(img.shape[2]))
