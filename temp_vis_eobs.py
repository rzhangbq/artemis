#!/usr/bin/env python3
"""Plot Eobs reduced diagnostics: time (col 2) vs Ex (col 3)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mplconfig-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

SERIES = (
    ("FDTD CFL=0.8", "diags_fdtd_08"),
    ("ADI CFL=6.4", "diags_adi_64"),
    ("ADI CFL=12.8", "diags_adi_128"),
    ("ADI CFL=25.6", "diags_adi_256"),
    ("ADI CFL=51.2", "diags_adi_512"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path("run_archive"),
        help="directory containing the diags_* folders",
    )
    parser.add_argument(
        "--name",
        default="Eobs1.txt",
        help="reduced-diag filename under each diags_*/reducedfiles/",
    )
    parser.add_argument(
        "--xcol",
        type=int,
        default=1,
        help="0-based column for x-axis (default: 1 = time)",
    )
    parser.add_argument(
        "--ycol",
        type=int,
        default=2,
        help="0-based column for y-axis (default: 2 = Ex)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("circuit_test_run/Eobs1_time_Ex_e9_res.pdf"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_eobs(path: Path, xcol: int, ycol: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] <= max(xcol, ycol):
        raise ValueError(f"{path}: expected >= {max(xcol, ycol) + 1} columns, got {data.shape}")
    return data[:, xcol], data[:, ycol]


def main() -> None:
    args = parse_args()

    fig, ax = plt.subplots(figsize=(10.0, 4.5), constrained_layout=True)
    for label, dirname in SERIES:
        path = args.prefix / dirname / "reducedfiles" / args.name
        if not path.is_file():
            print(f"skip missing: {path}")
            continue
        x, y = load_eobs(path, args.xcol, args.ycol)
        # time in ns if plotting the default time column
        if args.xcol == 1:
            ax.plot(x * 1.0e9, y, label=label, linewidth=1.4)
        else:
            ax.plot(x, y, label=label, linewidth=1.4)
        print(f"{label}: {path}  N={x.size}")

    if args.xcol == 1:
        ax.set_xlabel("time (ns)")
    else:
        ax.set_xlabel(f"column {args.xcol}")
    ax.set_ylabel(f"column {args.ycol}" if args.ycol != 2 else "Ex")
    ax.set_title(f"{args.prefix.name} / {args.name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
