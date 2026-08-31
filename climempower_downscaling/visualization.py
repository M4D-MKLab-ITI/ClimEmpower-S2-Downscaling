"""Small, reusable plotting helpers for comparing pansharpening results.

These operate purely on in-memory arrays produced by the rest of the package —
no file paths or business logic — so the example notebook can stay a thin
walkthrough instead of redefining plotting code inline.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1] for display purposes only.

    Parameters
    ----------
    image : np.ndarray
        Array of any shape/dtype.

    Returns
    -------
    np.ndarray
        Float array scaled to [0, 1].
    """
    min_val, max_val = image.min(), image.max()
    return (image - min_val) / (max_val - min_val)


def make_rgb_composite(band_r: np.ndarray, band_g: np.ndarray, band_b: np.ndarray) -> np.ndarray:
    """Stack three 2D bands into an (H, W, 3) RGB composite.

    Parameters
    ----------
    band_r, band_g, band_b : np.ndarray
        2D arrays of identical shape.

    Returns
    -------
    np.ndarray
        Array of shape (H, W, 3).
    """
    return np.dstack((band_r, band_g, band_b))


def plot_band_comparison(
    original: np.ndarray,
    results: dict[str, np.ndarray],
    band_name: str,
    normalize: bool = True,
):
    """Plot a single band before/after each pansharpening method, side by side.

    Parameters
    ----------
    original : np.ndarray
        The band before pansharpening.
    results : dict[str, np.ndarray]
        Mapping of method name (e.g. "Brovey") to that method's output band,
        used both for the number of panels and their titles.
    band_name : str
        Label used in each subplot title (e.g. "Band 05").
    normalize : bool
        If True, normalize each panel to [0, 1] for display.

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = {"Original": original, **results}
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, band) in zip(axes, panels.items()):
        display_band = normalize_for_display(band) if normalize else band
        ax.imshow(display_band, cmap="gray")
        ax.set_title(f"{band_name} ({label})")
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_band_histogram(band: np.ndarray, title: Optional[str] = None, ax=None):
    """Plot a histogram of a band's pixel values.

    Parameters
    ----------
    band : np.ndarray
        Array of pixel values (any shape; flattened internally).
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure/axes is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.hist(band.flatten(), bins=1024, color="orange", alpha=0.7)
    if title:
        ax.set_title(title)
    ax.set_xlabel("DN values")
    ax.set_ylabel("Number of observations")
    ax.grid(True)
    return ax
