"""Pansharpening methods: Brovey, IHS, and wavelet-based fusion.

Each function fuses a high-resolution panchromatic band with a lower-resolution
multispectral stack. Both inputs are expected to be normalized to [0, 1]; outputs
are returned as uint16 in the full [0, 65535] range.

Adapted from the MIT-licensed https://github.com/codegaj/py_pansharpening
(methods/Brovey.py, methods/IHS.py, methods/Wavelet.py).
"""

from __future__ import annotations

import numpy as np
import pywt

from .utils import upsample_interp23


def _sharpening_ratio(pan: np.ndarray, hs: np.ndarray) -> int:
    pan_rows, pan_cols, _ = pan.shape
    hs_rows, hs_cols, _ = hs.shape

    ratio_rows = pan_rows / hs_rows
    ratio_cols = pan_cols / hs_cols
    if not np.isclose(ratio_rows, ratio_cols):
        raise ValueError(
            f"Aspect ratio mismatch between pan and hs: {ratio_rows} vs {ratio_cols}"
        )
    return int(np.round(ratio_rows))


def _to_uint16(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0, 1)
    return np.uint16(image * (2**16 - 1))


def brovey(pan: np.ndarray, hs: np.ndarray) -> np.ndarray:
    """Fuse ``pan`` and ``hs`` with the Brovey transform.

    Parameters
    ----------
    pan : np.ndarray
        Panchromatic band, shape (M, N, 1), normalized to [0, 1].
    hs : np.ndarray
        Multispectral stack, shape (m, n, C), normalized to [0, 1].

    Returns
    -------
    np.ndarray
        Fused image, shape (M, N, C), dtype uint16.
    """
    ratio = _sharpening_ratio(pan, hs)
    u_hs = upsample_interp23(hs, ratio)

    intensity = np.mean(u_hs, axis=-1)
    image_hr = (pan - np.mean(pan)) * (np.std(intensity, ddof=1) / np.std(pan, ddof=1)) + np.mean(
        intensity
    )
    image_hr = np.squeeze(image_hr)

    num_bands = hs.shape[-1]
    fused_bands = [
        np.expand_dims(image_hr * u_hs[:, :, band] / (intensity + 1e-8), axis=-1)
        for band in range(num_bands)
    ]
    fused = np.concatenate(fused_bands, axis=-1)
    return _to_uint16(fused)


def ihs(pan: np.ndarray, hs: np.ndarray) -> np.ndarray:
    """Fuse ``pan`` and ``hs`` with an IHS (intensity substitution) transform.

    Parameters
    ----------
    pan : np.ndarray
        Panchromatic band, shape (M, N, 1), normalized to [0, 1].
    hs : np.ndarray
        Multispectral stack, shape (m, n, C), normalized to [0, 1].

    Returns
    -------
    np.ndarray
        Fused image, shape (M, N, C), dtype uint16.
    """
    ratio = _sharpening_ratio(pan, hs)
    u_hs = upsample_interp23(hs, ratio)

    intensity = np.mean(u_hs, axis=-1, keepdims=True)
    matched_pan = (pan - np.mean(pan)) * np.std(intensity, ddof=1) / np.std(pan, ddof=1) + np.mean(
        intensity
    )

    num_bands = hs.shape[-1]
    fused = u_hs + np.tile(matched_pan - intensity, (1, 1, num_bands))
    return _to_uint16(fused)


def wavelet(pan: np.ndarray, hs: np.ndarray) -> np.ndarray:
    """Fuse ``pan`` and ``hs`` by substituting wavelet approximation coefficients.

    Decomposes ``pan`` with a 2-level Haar wavelet transform and, for each band
    of ``hs``, replaces its own approximation coefficients with those of ``pan``
    before reconstructing.

    Parameters
    ----------
    pan : np.ndarray
        Panchromatic band, shape (M, N, 1), normalized to [0, 1].
    hs : np.ndarray
        Multispectral stack, shape (m, n, C), normalized to [0, 1].

    Returns
    -------
    np.ndarray
        Fused image, shape (M, N, C), dtype uint16.
    """
    ratio = _sharpening_ratio(pan, hs)
    u_hs = upsample_interp23(hs, ratio)

    pan_2d = np.squeeze(pan)
    pan_coeffs = pywt.wavedec2(pan_2d, "haar", level=2)

    num_bands = hs.shape[-1]
    fused_bands = []
    for band in range(num_bands):
        band_coeffs = pywt.wavedec2(u_hs[:, :, band], "haar", level=2)
        pan_coeffs[0] = band_coeffs[0]
        reconstructed = pywt.waverec2(pan_coeffs, "haar")
        fused_bands.append(np.expand_dims(reconstructed, -1))

    fused = np.concatenate(fused_bands, axis=-1)
    return _to_uint16(fused)
