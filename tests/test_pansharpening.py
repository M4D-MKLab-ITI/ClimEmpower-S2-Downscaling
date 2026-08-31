import numpy as np
import pytest

from climempower_downscaling.pansharpening import brovey, ihs, wavelet

METHODS = [brovey, ihs, wavelet]


def _synthetic_pair(ratio=2, hs_size=16, num_bands=3):
    rng = np.random.default_rng(0)
    pan = rng.random((hs_size * ratio, hs_size * ratio, 1), dtype=np.float64)
    hs = rng.random((hs_size, hs_size, num_bands), dtype=np.float64)
    return pan, hs


@pytest.mark.parametrize("method", METHODS)
def test_output_shape_and_dtype(method):
    pan, hs = _synthetic_pair()
    fused = method(pan, hs)
    assert fused.shape == (pan.shape[0], pan.shape[1], hs.shape[-1])
    assert fused.dtype == np.uint16


@pytest.mark.parametrize("method", METHODS)
def test_output_in_valid_range(method):
    pan, hs = _synthetic_pair()
    fused = method(pan, hs)
    assert fused.min() >= 0
    assert fused.max() <= 2**16 - 1


def test_mismatched_aspect_ratio_raises():
    rng = np.random.default_rng(0)
    pan = rng.random((30, 40, 1))
    hs = rng.random((10, 10, 3))
    with pytest.raises(ValueError):
        brovey(pan, hs)
