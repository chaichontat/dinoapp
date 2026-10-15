"""Read and render cell crops for saved barrages and native review."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import contourpy
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from PIL import Image

from dino.pixel_corrections import correct_channels
from dino.zarr_source import CellCrop, CellCropReader

DISPLAY_MIN = 0.0
BARRAGE_CMAP = "magma"
FIGURE_DPI = 200
BASE_CROP_SIZE = 100
BASE_PANEL_WIDTH = 4.0
BASE_FIGURE_HEIGHT = 4.0


def read_crop(reader: CellCropReader, row: pd.Series, *, size: int, pixel_correction=None) -> CellCrop:
    crop = reader.read_cell(
        label=int(row["label"]),
        plane_z=int(row["plane_z"]),
        center_y=int(np.rint(float(row["centroid_y"]))),
        center_x=int(np.rint(float(row["centroid_x"]))),
        size=size,
    )
    if pixel_correction is not None:
        origin = (
            int(row["plane_z"]) + reader.offset_zyx[0],
            int(np.rint(float(row["centroid_y"]))) + reader.offset_zyx[1] - size // 2,
            int(np.rint(float(row["centroid_x"]))) + reader.offset_zyx[2] - size // 2,
        )
        crop = replace(
            crop,
            channels=correct_channels(crop.channels, origin, pixel_correction=pixel_correction),
        )
    return crop


def render_crop(
    crop: CellCrop,
    *,
    cell: str,
    display_max: Mapping[str, float],
    panels: Sequence[tuple[str, str]],
) -> Image.Image:
    """Render one panel per (channel key, title) of `panels`, in order."""
    if not panels:
        raise ValueError("Crop rendering requires at least one panel")
    absent = [key for key, _ in panels if key not in crop.channels]
    if absent:
        raise ValueError(f"Crop has no channel for panels {absent}; channels are {sorted(crop.channels)}")
    height, width = crop.mask.shape
    if height != width:
        raise ValueError(f"Expected square crop, got shape={crop.mask.shape}")

    scale = height / BASE_CROP_SIZE
    output_width = round(BASE_PANEL_WIDTH * len(panels) * scale * FIGURE_DPI)
    output_height = round(BASE_FIGURE_HEIGHT * scale * FIGURE_DPI)
    # Matplotlib truncates floating-point inches; nudge upward to preserve the exact pixel target.
    figure = Figure(
        figsize=(
            np.nextafter(output_width / FIGURE_DPI, np.inf),
            np.nextafter(output_height / FIGURE_DPI, np.inf),
        ),
        dpi=FIGURE_DPI,
    )
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(nrows=1, ncols=len(panels), squeeze=False)[0]
    title_size = 12.0 * scale
    has_mask = bool(crop.mask.any())

    # Trace each segmentation once, then draw the same geometry in all channels.
    neighbors = [
        path
        for label in np.unique(crop.labels[~crop.mask])
        if int(label) != 0
        for path in contourpy.contour_generator(z=(crop.labels == label), name="mpl2014").lines(0.5)[0]
    ]
    target = contourpy.contour_generator(z=crop.mask, name="mpl2014").lines(0.5)[0]

    for axis, (name, title) in zip(axes, panels, strict=True):
        channel = crop.channels[name]
        if channel.shape != crop.mask.shape:
            raise ValueError(f"Channel/mask shape mismatch for {name}: {channel.shape} != {crop.mask.shape}")
        mean = float(np.mean(channel[crop.mask])) if has_mask else float("nan")
        axis.imshow(
            channel,
            cmap=BARRAGE_CMAP,
            vmin=DISPLAY_MIN,
            vmax=display_max[name],
        )
        axis.set_title(f"cell={cell}\nch={title} mean={mean:.1f}", fontsize=title_size)
        axis.axis("off")

        axis.add_collection(
            LineCollection(
                neighbors,
                colors="white",
                linewidths=0.6 * scale,
                linestyles="--",
                alpha=0.35,
            )
        )
        if has_mask:
            axis.add_collection(
                LineCollection(
                    target,
                    colors="cyan",
                    linewidths=0.8 * scale,
                    linestyles="--",
                    alpha=0.7,
                )
            )

    figure.tight_layout()
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    return Image.fromarray(rgb)
