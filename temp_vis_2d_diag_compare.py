#!/usr/bin/env python3
"""Render Artemis diagnostic series side-by-side with a shared color scale."""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import yt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm


SERIES = (
    ("FDTD CFL=0.8", "diags_fdtd_08"),
    ("ADI CFL=1.6", "diags_adi_16"),
    ("ADI CFL=3.2", "diags_adi_32"),
    ("ADI CFL=6.4", "diags_adi_64"),
    ("ADI CFL=12.8", "diags_adi_128"),
    ("ADI CFL=25.6", "diags_adi_256"),
    # ("ADI CFL=51.2", "diags_adi_512"),
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
        "--prefix",
        type=Path,
        default=Path("run_archive"),
        help="directory containing the diag_* folders",
    )
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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel plotfile readers (1 = serial)",
    )
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
    """Load one 2D slice from an AMReX plotfile (only the requested field)."""
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
    # Copy so worker processes return self-contained arrays.
    slice2d = np.asarray(field[tuple(index)], dtype=np.float64).copy()

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


def _load_slice_job(
    job: tuple[str, str, str, int | None],
) -> tuple[float, np.ndarray, list[float], int]:
    path, variable, plane, slice_index = job
    return load_slice(Path(path), variable, plane, slice_index)


def preload_slices(
    all_paths: list[list[Path]],
    variable: str,
    plane: str,
    slice_index: int | None,
    workers: int,
) -> tuple[np.ndarray, list[list[np.ndarray]], list[list[float]], list[int]]:
    """Load every series/frame slice, optionally in parallel.

    Returns times[n_frames], slices[n_series][n_frames], extents[n_series],
    and fixed_indices[n_series].
    """
    n_series = len(all_paths)
    n_frames = len(all_paths[0])
    # Jobs ordered as (frame0 series0..N), (frame1 series0..N), ...
    jobs = [
        (str(all_paths[series][frame]), variable, plane, slice_index)
        for frame in range(n_frames)
        for series in range(n_series)
    ]

    if workers <= 1:
        loaded = [_load_slice_job(job) for job in tqdm(jobs, desc="Loading")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            loaded = list(
                tqdm(
                    pool.map(_load_slice_job, jobs, chunksize=1),
                    total=len(jobs),
                    desc=f"Loading ({workers} workers)",
                )
            )

    times = np.empty(n_frames, dtype=np.float64)
    slices: list[list[np.ndarray]] = [[None] * n_frames for _ in range(n_series)]  # type: ignore[list-item]
    extents: list[list[float]] = [[]] * n_series
    fixed_indices = [0] * n_series

    for frame in range(n_frames):
        frame_times = []
        for series in range(n_series):
            time, slice2d, extent, fixed = loaded[frame * n_series + series]
            frame_times.append(time)
            slices[series][frame] = slice2d
            extents[series] = extent
            fixed_indices[series] = fixed
        frame_times_arr = np.asarray(frame_times)
        tolerance = max(1.0e-15, 1.0e-8 * float(np.max(np.abs(frame_times_arr))))
        if not np.allclose(
            frame_times_arr, frame_times_arr[0], rtol=1.0e-8, atol=tolerance
        ):
            raise ValueError(
                f"Physical times do not match at frame {frame}: {frame_times}"
            )
        times[frame] = frame_times_arr[0]

    return times, slices, extents, fixed_indices


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
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    series = [(label, args.prefix / name) for label, name in SERIES]
    n_series = len(series)
    all_paths = [sorted_plotfiles(directory)[:: args.stride] for _, directory in series]
    frame_counts = [len(paths) for paths in all_paths]
    if len(set(frame_counts)) != 1:
        raise ValueError(f"Diagnostic series have different frame counts: {frame_counts}")
    n_frames = frame_counts[0]

    ax0, ax1, normal = PLANES[args.plane]
    signed = not args.variable.startswith("|") and args.variable != "epsilon"
    cmap = "RdBu_r" if signed else "viridis"

    times, all_slices, extents, fixed_indices = preload_slices(
        all_paths,
        args.variable,
        args.plane,
        args.slice_index,
        args.workers,
    )
    shapes = [frames[0].shape for frames in all_slices]
    if len(set(shapes)) != 1:
        raise ValueError(f"Slice shapes differ in the first frame: {shapes}")

    frame0 = [frames[0] for frames in all_slices]
    vmin, vmax = shared_clim(frame0, signed)
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
        axes, series, frame0, extents, fixed_indices, strict=True
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
        for frame_index in tqdm(range(n_frames), desc="Rendering"):
            slices = [frames[frame_index] for frames in all_slices]
            vmin, vmax = shared_clim(slices, signed)
            for image, frame, extent in zip(images, slices, extents, strict=True):
                image.set_data(frame.T)
                image.set_extent(extent)
                image.set_clim(vmin, vmax)
            title.set_text(
                f"{args.variable} on {args.plane} plane, t = {times[frame_index]:.3e} s"
            )
            writer.grab_frame()

    plt.close(fig)
    print(f"Wrote {args.output}")
    print(f"Series: {n_series}")
    print(f"Frames: {n_frames}")
    print(f"Workers: {args.workers}")


if __name__ == "__main__":
    main()
