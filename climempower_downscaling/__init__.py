"""climempower_downscaling: Sentinel-2 pansharpening/downscaling toolkit.

Provides classical (Brovey, IHS, wavelet) and CNN-based (PNN) pansharpening
methods, plus Sentinel-2 SAFE product loading and GeoTIFF I/O helpers.
"""

from .io_utils import get_georeference, read_geotiff, write_geotiff
from .pansharpening import brovey, ihs, wavelet
from .preprocessing import (
    SENTINEL2_BAND_LABELS,
    discover_band_paths,
    extract_safe_archive,
    generate_panchromatic_band,
    make_window,
    normalize_to_unit_range,
    open_bands,
    read_band_subset,
    stack_bands,
)
from .utils import BaseCoeff, upsample_interp23

__version__ = "0.1.0"

__all__ = [
    "brovey",
    "ihs",
    "wavelet",
    "get_georeference",
    "read_geotiff",
    "write_geotiff",
    "SENTINEL2_BAND_LABELS",
    "discover_band_paths",
    "extract_safe_archive",
    "generate_panchromatic_band",
    "make_window",
    "normalize_to_unit_range",
    "open_bands",
    "read_band_subset",
    "stack_bands",
    "BaseCoeff",
    "upsample_interp23",
]
