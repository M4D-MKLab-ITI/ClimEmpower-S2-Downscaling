import numpy as np

from climempower_downscaling.utils import upsample_interp23


def test_upsample_ratio_one_is_identity():
    image = np.random.rand(8, 8, 3)
    result = upsample_interp23(image, ratio=1)
    assert result is image


def test_upsample_doubles_spatial_dims():
    image = np.random.rand(16, 16, 4)
    result = upsample_interp23(image, ratio=2)
    assert result.shape == (32, 32, 4)


def test_upsample_quadruples_with_ratio_four():
    image = np.random.rand(8, 8, 2)
    result = upsample_interp23(image, ratio=4)
    assert result.shape == (32, 32, 2)
