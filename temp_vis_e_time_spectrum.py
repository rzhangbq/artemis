#!/usr/bin/env python3
"""Plot E-field component histories and frequency spectra.

Default: average over a mid-plane slice. With --index I J K: sample one cell.
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mplconfig-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import yt
from tqdm import tqdm


SERIES = (
    ("FDTD CFL=0.8", "diags_fdtd_08"),
    # ("ADI CFL=1.6", "diags_adi_16"),
    ("ADI CFL=3.2", "diags_adi_32"),
    ("ADI CFL=6.4", "diags_adi_64"),
    ("ADI CFL=12.8", "diags_adi_128"),
    ("ADI CFL=25.6", "diags_adi_256"),
    ("ADI CFL=51.2", "diags_adi_512"),
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
        "--components",
        nargs="+",
        default=("Ex", "Ey", "Ez"),
        choices=("Ex", "Ey", "Ez"),
        help="E-field components to extract.",
    )
    parser.add_argument(
        "--index",
        nargs=3,
        type=int,
        metavar=("I", "J", "K"),
        default=None,
        help="sample one cell at (i,j,k) instead of mid-plane average",
    )
    parser.add_argument(
        "--plane",
        default="xy",
        choices=tuple(PLANES),
        help="plane to average over when --index is not set",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="index along the normal axis for plane average (default: center)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth plotfile from each diagnostic series.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel plotfile readers (1 = serial)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("circuit_test_run/E_components_avg_time_spectrum.pdf"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--log-spectrum",
        action="store_true",
        help="Use a logarithmic y-axis for the frequency spectrum.",
    )
    parser.add_argument(
        "--keep-dc",
        action="store_true",
        help="Keep the DC/zero-frequency bin in the spectrum.",
    )
    parser.add_argument(
        "--fft-pad",
        type=float,
        default=5.0,
        help="zero-pad factor before FFT (>=1; 1 = no padding). Default: 5",
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


def load_field_samples(
    path: Path,
    components: tuple[str, ...],
    plane: str,
    slice_index: int | None,
    cell_index: tuple[int, int, int] | None,
) -> tuple[float, dict[str, float], str]:
    """Sample E components at one cell, or average over a mid-plane slice."""
    ds = yt.load(str(path))
    dims = tuple(int(n) for n in ds.domain_dimensions)

    if cell_index is not None:
        # Point selection reads only grids that contain the cell, avoiding a
        # full-domain covering_grid materialization.
        i, j, k = cell_index
        for axis, idx in enumerate((i, j, k)):
            if not 0 <= idx < dims[axis]:
                raise IndexError(
                    f"index {cell_index} is outside domain dims {dims} in {path}"
                )
        dds = ds.domain_width / ds.domain_dimensions
        center = ds.domain_left_edge + dds * (np.array([i, j, k], dtype=np.float64) + 0.5)
        point = ds.point(center)
        values = {
            component: float(np.asarray(point[("boxlib", component)]).ravel()[0])
            for component in components
        }
        return (
            float(ds.current_time),
            values,
            f"cell (i,j,k)=({i},{j},{k})",
        )

    grid = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions,
    )
    _, _, normal = PLANES[plane]
    fixed = dims[normal] // 2 if slice_index is None else int(slice_index)
    if not 0 <= fixed < dims[normal]:
        raise IndexError(
            f"slice index {fixed} is outside axis {AXIS_NAME[normal]} "
            f"with size {dims[normal]} in {path}"
        )
    index = [slice(None)] * 3
    index[normal] = fixed
    values = {
        component: float(
            np.mean(
                np.asarray(grid[("boxlib", component)].to_ndarray(), dtype=np.float64)[
                    tuple(index)
                ]
            )
        )
        for component in components
    }
    return (
        float(ds.current_time),
        values,
        f"{plane} mid-plane ({AXIS_NAME[normal]}-index {fixed})",
    )


def _load_field_samples_job(
    job: tuple[str, tuple[str, ...], str, int | None, tuple[int, int, int] | None],
) -> tuple[float, dict[str, float], str]:
    path, components, plane, slice_index, cell_index = job
    return load_field_samples(Path(path), components, plane, slice_index, cell_index)


def preload_series(
    all_paths: list[list[Path]],
    components: tuple[str, ...],
    plane: str,
    slice_index: int | None,
    cell_index: tuple[int, int, int] | None,
    workers: int,
) -> tuple[list[tuple[np.ndarray, dict[str, np.ndarray]]], str]:
    """Load every series/frame sample, optionally in parallel.

    Returns one (times, values) pair per series (frame order preserved),
    plus a location label from the first loaded frame.
    """
    n_series = len(all_paths)
    frame_counts = [len(paths) for paths in all_paths]
    # Jobs ordered as series0 frames..., series1 frames..., ...
    jobs = [
        (str(path), components, plane, slice_index, cell_index)
        for paths in all_paths
        for path in paths
    ]

    if workers <= 1:
        loaded = [_load_field_samples_job(job) for job in tqdm(jobs, desc="Loading")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            loaded = list(
                tqdm(
                    pool.map(_load_field_samples_job, jobs, chunksize=1),
                    total=len(jobs),
                    desc=f"Loading ({workers} workers)",
                )
            )

    data: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    location = loaded[0][2] if loaded else ""
    offset = 0
    for series in range(n_series):
        n_frames = frame_counts[series]
        series_loaded = loaded[offset : offset + n_frames]
        offset += n_frames

        times = np.empty(n_frames, dtype=np.float64)
        values = {
            component: np.empty(n_frames, dtype=np.float64)
            for component in components
        }
        for frame, (time, field_values, _) in enumerate(series_loaded):
            times[frame] = time
            for component in components:
                values[component][frame] = field_values[component]
        data.append((times, values))

    return data, location


def frequency_spectrum(
    times: np.ndarray,
    values: np.ndarray,
    keep_dc: bool,
    fft_pad: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if times.size < 2:
        return np.asarray([]), np.asarray([])

    order = np.argsort(times)
    times = times[order]
    values = values[order]
    dt_values = np.diff(times)
    dt = float(np.median(dt_values))
    if not np.allclose(dt_values, dt, rtol=1.0e-4, atol=max(1.0e-18, abs(dt) * 1.0e-8)):
        uniform_times = np.linspace(float(times[0]), float(times[-1]), times.size)
        values = np.interp(uniform_times, times, values)
        dt = float(uniform_times[1] - uniform_times[0])

    signal = values if keep_dc else values - np.mean(values)
    n_orig = signal.size
    n_fft = max(n_orig, int(np.ceil(n_orig * fft_pad)))
    frequencies = np.fft.rfftfreq(n_fft, d=dt)
    # Normalize by original length so amplitudes stay comparable across pad factors.
    amplitude = np.abs(np.fft.rfft(signal, n=n_fft)) / n_orig
    if n_orig > 2:
        amplitude[1:-1] *= 2.0

    if keep_dc:
        return frequencies, amplitude
    return frequencies[1:], amplitude[1:]


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.fft_pad < 1.0:
        raise ValueError("--fft-pad must be >= 1")

    components = tuple(dict.fromkeys(args.components))
    cell_index = tuple(args.index) if args.index is not None else None
    if cell_index is not None and any(idx < 0 for idx in cell_index):
        raise ValueError("--index values must be non-negative")

    series = [(label, args.prefix / name) for label, name in SERIES]
    all_paths = [sorted_plotfiles(directory)[:: args.stride] for _, directory in series]
    loaded, location = preload_series(
        all_paths,
        components,
        args.plane,
        args.slice_index,
        cell_index,
        args.workers,
    )
    data = [
        (label, times, values)
        for (label, _), (times, values) in zip(series, loaded, strict=True)
    ]
    # Angle brackets denote a spatial average; point samples omit them.
    def field_label(component: str) -> str:
        return component if cell_index is not None else f"<{component}>"

    fig, axes = plt.subplots(
        len(components),
        2,
        figsize=(13.0, 3.0 * len(components)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, component in enumerate(components):
        ax_time, ax_freq = axes[row]
        for label, times, values in data:
            ax_time.plot(times * 1.0e9, values[component], label=label, linewidth=1.6)
            frequencies, amplitudes = frequency_spectrum(
                times,
                values[component],
                args.keep_dc,
                args.fft_pad,
            )
            if frequencies.size:
                ax_freq.plot(frequencies * 1.0e-9, amplitudes, label=label, linewidth=1.6)

        ylabel = field_label(component)
        ax_time.set_title(f"{ylabel} at {location} in time domain")
        ax_time.set_xlabel("time (ns)")
        ax_time.set_ylabel(ylabel)
        ax_time.grid(True, alpha=0.3)

        ax_freq.set_title(f"{ylabel} at {location} frequency spectrum")
        ax_freq.set_xlabel("frequency (GHz)")
        ax_freq.set_ylabel("amplitude")
        ax_freq.grid(True, alpha=0.3)
        if args.log_spectrum:
            ax_freq.set_yscale("log")
        if row == 0:
            ax_time.legend(loc="best", fontsize="small")
            if ax_freq.lines:
                ax_freq.legend(loc="best", fontsize="small")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote {args.output}")
    print(f"Location: {location}")
    print(f"Workers: {args.workers}")
    for label, times, _ in data:
        print(
            f"{label}: {times.size} samples, "
            f"t=[{times.min() * 1.0e9:.3e}, {times.max() * 1.0e9:.3e}] ns"
        )


if __name__ == "__main__":
    main()
