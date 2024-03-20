import numpy as np
from cv_utils.matching import ccoeff_norm


def test_ccoeff_norm():
    img = np.empty((5, 5), dtype=np.uint8)
    tmplt = np.empty((3, 3), dtype=np.uint8)
    res = ccoeff_norm(img, tmplt)
    assert res.shape == (3, 3)
    assert 0 <= res.max() <= 1.0
