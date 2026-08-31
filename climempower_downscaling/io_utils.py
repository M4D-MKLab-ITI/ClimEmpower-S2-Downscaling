"""GeoTIFF read/write and georeference helpers built on rasterio."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.windows import Window


def get_georeference(dataset: DatasetReader, window: Window) -> tuple[Affine, CRS]:
    """Extract the affine transform and CRS for a window of a reference dataset.

    Parameters
    ----------
    dataset : DatasetReader
        An opened rasterio dataset (e.g. one of the bands used to build an AOI).
    window : rasterio.windows.Window
        The AOI window that was read from ``dataset``.

    Returns
    -------
    tuple[Affine, CRS]
        The transform for the windowed subset, and the dataset's CRS.
    """
    return dataset.window_transform(window), dataset.crs


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    transform: Affine,
    crs: CRS,
    dtype: str = "uint16",
    band_names: Optional[Sequence[str]] = None,
) -> None:
    """Write a (H, W, bands) array to a georeferenced GeoTIFF.

    Parameters
    ----------
    path : str | Path
        Output file path.
    array : np.ndarray
        Array of shape (H, W, bands).
    transform : Affine
        Affine georeferencing transform, e.g. from ``get_georeference``.
    crs : CRS
        Coordinate reference system, e.g. from ``get_georeference``.
    dtype : str
        Output band dtype.
    band_names : Sequence[str], optional
        Per-band description to embed in the file (e.g. "Red", "SWIR 1"),
        in the same order as ``array``'s last axis. Set both as each band's
        own description (GDAL/QGIS band-picker UIs) and as a dataset-level
        ``TIFFTAG_IMAGEDESCRIPTION`` summary (shown in most TIFF viewers'
        metadata/info panel), so the band mapping is visible either way.
        Must have one entry per band if given.
    """
    height, width, count = array.shape
    if band_names is not None and len(band_names) != count:
        raise ValueError(f"Expected {count} band_names, got {len(band_names)}")

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for band in range(count):
            dst.write(array[:, :, band], band + 1)
            if band_names is not None:
                dst.set_band_description(band + 1, band_names[band])

        if band_names is not None:
            summary = " | ".join(f"Band {i + 1}: {name}" for i, name in enumerate(band_names))
            dst.update_tags(TIFFTAG_IMAGEDESCRIPTION=summary)


def read_geotiff(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a GeoTIFF back into a (H, W, bands) array plus its metadata.

    Parameters
    ----------
    path : str | Path
        Input file path.

    Returns
    -------
    tuple[np.ndarray, dict]
        The pixel data as (H, W, bands), and the dataset's ``.meta`` dict
        (includes ``crs``, ``transform``, ``dtype``, ``width``, ``height``, ``count``).
    """
    with rasterio.open(path) as src:
        data = src.read()
        meta = src.meta.copy()
    return np.transpose(data, (1, 2, 0)), meta
