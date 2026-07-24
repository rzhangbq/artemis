#!/usr/bin/env python3
"""Render Artemis diagnostic series side-by-side with a shared color scale."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import yt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm


SERIES = (
    ("FDTD CFL=0.8", Path("run_archive/diags_fdtd_08")),
    # ("ADI CFL=0.8", Path("run_archive/diags_adi_08")),
    ("ADI CFL=1.6", Path("run_archive/diags_adi_16")),
    ("ADI CFL=3.2", Path("run_archive/diags_adi_32")),
    ("ADI CFL=6.4", Path("run_archive/diags_adi_64")),
    ("ADI CFL=12.8", Path("run_archive/diags_adi_128")),
)
PLANES = {
    "xy": (0, 1, 2),
    "xz": (0, 2, 1),
    "yz": (1, 2, 0),
}
AXIS_NAME = "xyz"

yt.set_log_level("ERROR")
logging.getLogger("yt").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variable",
        default="|E|",
        choices=("Ex", "Ey", "Ez", "|E|", "Bx", "By", "Bz", "|B|", "epsilon"),
    )
    parser.add_argument("--plane", default="xy", choices=tuple(PLANES))
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="index along the normal axis (default: center)",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--output",
        type=Path,
        # pace ffmpeg/7.1 is built without libvpx/libx264; mpeg4+mp4 works.
        default=Path("circuit_test_run/E_xy_diag_comparison.mp4"),
    )
    return parser.parse_args()


def sorted_plotfiles(directory: Path) -> list[Path]:
    files = sorted(
        directory.glob("plt*"),
        key=lambda path: int(path.name.removeprefix("plt")),
    )
    if not files:
        raise FileNotFoundError(f"No plotfiles found in {directory.resolve()}")
    return files


def load_field(grid, variable: str) -> np.ndarray:
    if variable in ("Ex", "Ey", "Ez", "Bx", "By", "Bz", "epsilon"):
        return grid[("boxlib", variable)].to_ndarray()

    prefix = variable[1]  # E or B from |E| or |B|
    components = [
        grid[("boxlib", f"{prefix}{axis}")].to_ndarray()
        for axis in "xyz"
    ]
    return np.sqrt(sum(component**2 for component in components))


def load_slice(
    path: Path,
    variable: str,
    plane: str,
    slice_index: int | None,
) -> tuple[float, np.ndarray, list[float], int]:
    ax0, ax1, normal = PLANES[plane]
    ds = yt.load(str(path))
    grid = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions,
    )
    field = load_field(grid, variable)
    dims = field.shape
    fixed = dims[normal] // 2 if slice_index is None else slice_index
    if not 0 <= fixed < dims[normal]:
        raise IndexError(
            f"slice index {fixed} is outside axis {AXIS_NAME[normal]} "
            f"with size {dims[normal]} in {path}"
        )

    index = [slice(None)] * 3
    index[normal] = fixed
    slice2d = np.asarray(field[tuple(index)])

    centers = []
    for axis in (ax0, ax1):
        edges = np.linspace(
            float(ds.domain_left_edge[axis]),
            float(ds.domain_right_edge[axis]),
            dims[axis] + 1,
        )
        centers.append(0.5 * (edges[:-1] + edges[1:]))
    extent = [
        float(centers[0][0] * 1e3),
        float(centers[0][-1] * 1e3),
        float(centers[1][0] * 1e3),
        float(centers[1][-1] * 1e3),
    ]
    return float(ds.current_time), slice2d, extent, fixed


def shared_clim(slices: list[np.ndarray], signed: bool) -> tuple[float, float]:
    """Return limits spanning all datasets for the current frame."""
    if signed:
        vmax = max(float(np.max(np.abs(frame))) for frame in slices)
        vmax = max(vmax, np.finfo(float).eps)
        return -vmax, vmax

    vmin = min(float(np.min(frame)) for frame in slices)
    vmax = max(float(np.max(frame)) for frame in slices)
    if vmin == vmax:
        vmax = vmin + np.finfo(float).eps
    return vmin, vmax


def panel_layout(n_series: int) -> tuple[int, int]:
    """Choose a compact (nrows, ncols) grid for n_series panels."""
    if n_series <= 0:
        raise ValueError("SERIES must contain at least one entry")
    if n_series <= 3:
        return 1, n_series
    if n_series <= 6:
        return 2, (n_series + 1) // 2
    ncols = int(np.ceil(np.sqrt(n_series)))
    nrows = int(np.ceil(n_series / ncols))
    return nrows, ncols


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")

    n_series = len(SERIES)
    all_paths = [sorted_plotfiles(directory)[:: args.stride] for _, directory in SERIES]
    frame_counts = [len(paths) for paths in all_paths]
    if len(set(frame_counts)) != 1:
        raise ValueError(f"Diagnostic series have different frame counts: {frame_counts}")

    ax0, ax1, normal = PLANES[args.plane]
    signed = not args.variable.startswith("|") and args.variable != "epsilon"
    cmap = "RdBu_r" if signed else "viridis"

    def read_frame(frame_index: int):
        loaded = [
            load_slice(paths[frame_index], args.variable, args.plane, args.slice_index)
            for paths in all_paths
        ]
        times = np.asarray([item[0] for item in loaded])
        tolerance = max(1.0e-15, 1.0e-8 * float(np.max(np.abs(times))))
        if not np.allclose(times, times[0], rtol=1.0e-8, atol=tolerance):
            raise ValueError(
                f"Physical times do not match at frame {frame_index}: {times.tolist()}"
            )
        return times, [item[1] for item in loaded], [item[2] for item in loaded], [item[3] for item in loaded]

    times, slices, extents, fixed_indices = read_frame(0)
    shapes = [frame.shape for frame in slices]
    if len(set(shapes)) != 1:
        raise ValueError(f"Slice shapes differ in the first frame: {shapes}")

    vmin, vmax = shared_clim(slices, signed)
    nrows, ncols = panel_layout(n_series)
    fig = plt.figure(figsize=(5.5 * ncols + 1.2, 4.8 * nrows + 0.8))
    # Outer gridspec: plot grid | colorbar column
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=(ncols, 0.045),
        left=0.055,
        right=0.97,
        bottom=0.08,
        top=0.88,
        wspace=0.08,
    )
    inner = outer[0, 0].subgridspec(nrows, ncols, wspace=0.18, hspace=0.28)
    axes = []
    for index in range(n_series):
        row, col = divmod(index, ncols)
        if index == 0:
            axes.append(fig.add_subplot(inner[row, col]))
        else:
            axes.append(
                fig.add_subplot(inner[row, col], sharex=axes[0], sharey=axes[0])
            )
    colorbar_axis = fig.add_subplot(outer[0, 1])
    images = []
    for ax, (label, _), frame, extent, fixed in zip(
        axes, SERIES, slices, extents, fixed_indices, strict=True
    ):
        image = ax.imshow(
            frame.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)
        ax.set_title(label)
        ax.set_xlabel(f"{AXIS_NAME[ax0]} (mm)")
        ax.text(
            0.02,
            0.98,
            f"{AXIS_NAME[normal]}-index {fixed}",
            transform=ax.transAxes,
            va="top",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )
    for row in range(nrows):
        left_index = row * ncols
        if left_index < n_series:
            axes[left_index].set_ylabel(f"{AXIS_NAME[ax1]} (mm)")
    colorbar = fig.colorbar(images[0], cax=colorbar_axis)
    colorbar.set_label(args.variable)
    title = fig.suptitle(
        f"{args.variable} on {args.plane} plane, t = {times[0]:.3e} s"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=args.fps,
        bitrate=7200,
        codec="mpeg4",
        extra_args=(
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        ),
    )
    with writer.saving(fig, args.output, dpi=args.dpi):
        for frame_index in tqdm(range(frame_counts[0]), desc="Rendering"):
            if frame_index:
                times, slices, extents, _ = read_frame(frame_index)
            vmin, vmax = shared_clim(slices, signed)
            for image, frame, extent in zip(images, slices, extents, strict=True):
                image.set_data(frame.T)
                image.set_extent(extent)
                image.set_clim(vmin, vmax)
            title.set_text(
                f"{args.variable} on {args.plane} plane, t = {times[0]:.3e} s"
            )
            writer.grab_frame()

    plt.close(fig)
    print(f"Wrote {args.output}")
    print(f"Series: {n_series}")
    print(f"Frames: {frame_counts[0]}")


if __name__ == "__main__":
    main()
