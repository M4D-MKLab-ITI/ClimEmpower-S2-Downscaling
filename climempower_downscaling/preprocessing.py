"""Sentinel-2 SAFE product loading: extraction, band discovery, AOI subsetting,
panchromatic band construction, and normalization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.windows import Window

Resolution = Literal["10m", "20m", "60m"]

# Human-readable labels for Sentinel-2 band codes, for use as GeoTIFF band
# descriptions (see io_utils.write_geotiff's band_names) or plot titles.
SENTINEL2_BAND_LABELS = {
    "B01": "Coastal aerosol",
    "B02": "Blue",
    "B03": "Green",
    "B04": "Red",
    "B05": "Red Edge 1",
    "B06": "Red Edge 2",
    "B07": "Red Edge 3",
    "B08": "NIR",
    "B8A": "Narrow NIR",
    "B09": "Water vapour",
    "B11": "SWIR 1",
    "B12": "SWIR 2",
}


def extract_safe_archive(zip_path: str | Path, output_dir: str | Path) -> Path:
    """Extract a Sentinel-2 ``.SAFE`` product zip and return the extracted ``.SAFE`` dir.

    Parameters
    ----------
    zip_path : str | Path
        Path to the product zip file.
    output_dir : str | Path
        Directory to extract into.

    Returns
    -------
    Path
        Path to the extracted ``*.SAFE`` directory.
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    safe_dirs = list(output_dir.glob("*.SAFE"))
    if len(safe_dirs) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .SAFE directory under {output_dir}, found {len(safe_dirs)}"
        )
    return safe_dirs[0]


def discover_band_paths(safe_dir: str | Path, resolution: Resolution) -> dict[str, Path]:
    """Find band files for a given resolution inside a ``.SAFE`` directory.

    Globs the granule's ``IMG_DATA/R{resolution}`` directory for ``*_B##_{resolution}.jp2``
    files, so it works against any tile ID rather than a hardcoded product name.

    Parameters
    ----------
    safe_dir : str | Path
        Path to a ``*.SAFE`` directory.
    resolution : {"10m", "20m", "60m"}
        Resolution subdirectory to search.

    Returns
    -------
    dict[str, Path]
        Mapping of band code (e.g. ``"B02"``, ``"B8A"``) to file path, sorted
        by band code.
    """
    safe_dir = Path(safe_dir)
    granule_dirs = list(safe_dir.glob("GRANULE/*"))
    if len(granule_dirs) != 1:
        raise FileNotFoundError(
            f"Expected exactly one granule under {safe_dir / 'GRANULE'}, found {len(granule_dirs)}"
        )
    band_dir = granule_dirs[0] / "IMG_DATA" / f"R{resolution}"

    band_paths = {}
    for band_file in sorted(band_dir.glob(f"*_B*_{resolution}.jp2")):
        band_code = band_file.stem.split("_")[-2]
        band_paths[band_code] = band_file

    if not band_paths:
        raise FileNotFoundError(f"No band files found in {band_dir}")
    return band_paths


def open_bands(band_paths: dict[str, Path]) -> dict[str, DatasetReader]:
    """Open each band file as a rasterio dataset.

    Parameters
    ----------
    band_paths : dict[str, Path]
        Mapping of band name to file path, as returned by ``discover_band_paths``.

    Returns
    -------
    dict[str, DatasetReader]
        Mapping of band name to opened rasterio dataset. Callers are
        responsible for closing these (e.g. via ``contextlib.ExitStack``).
    """
    return {name: rasterio.open(path, driver="JP2OpenJPEG") for name, path in band_paths.items()}


def make_window(bbox: tuple[int, int, int, int]) -> Window:
    """Build a rasterio ``Window`` from a pixel bounding box.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        ``(col_off, row_off, col_stop, row_stop)`` in pixel coordinates.

    Returns
    -------
    rasterio.windows.Window
    """
    col_off, row_off, col_stop, row_stop = bbox
    return Window.from_slices((row_off, row_stop), (col_off, col_stop))


def read_band_subset(
    datasets: dict[str, DatasetReader], window: Window
) -> dict[str, np.ndarray]:
    """Read one band index from each dataset within ``window``.

    Parameters
    ----------
    datasets : dict[str, DatasetReader]
        Mapping of band name to opened dataset, as returned by ``open_bands``.
    window : rasterio.windows.Window
        AOI window to read.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of band name to 2D array.
    """
    return {name: dataset.read(1, window=window) for name, dataset in datasets.items()}


def generate_panchromatic_band(
    red: np.ndarray, green: np.ndarray, blue: np.ndarray, nir: np.ndarray
) -> np.ndarray:
    """Build a panchromatic band by averaging the four 10m Sentinel-2 bands.

    Parameters
    ----------
    red, green, blue, nir : np.ndarray
        2D arrays of identical shape.

    Returns
    -------
    np.ndarray
        Panchromatic band, same shape as the inputs, dtype float.
    """
    if not (red.shape == green.shape == blue.shape == nir.shape):
        raise ValueError("Input band shapes must be the same")
    return (red + green + blue + nir) / 4.0


def stack_bands(band_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Stack a dict of 2D bands into a single (H, W, bands) array.

    Parameters
    ----------
    band_dict : dict[str, np.ndarray]
        Mapping of band name to 2D array, all of identical shape. Band order
        in the output follows the dict's iteration order.

    Returns
    -------
    np.ndarray
        Array of shape (H, W, len(band_dict)).
    """
    stacked = np.array(list(band_dict.values()))
    return np.transpose(stacked, (1, 2, 0))


def normalize_to_unit_range(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalize an array to [0, 1] per band, over its spatial (H, W) axes.

    Parameters
    ----------
    array : np.ndarray
        Array of shape (H, W) for a single band, or (H, W, bands).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(normalized, min_values, max_values)``, where ``min_values``/``max_values``
        are the per-band values used, so the normalization can be inverted later.
    """
    max_values = np.max(array, axis=(0, 1))
    min_values = np.min(array, axis=(0, 1))
    normalized = np.float32(array - min_values) / (max_values - min_values)
    return normalized, min_values, max_values
