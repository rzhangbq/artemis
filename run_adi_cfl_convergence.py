#!/usr/bin/env python3
"""Grid convergence of the ADI plane-wave test at several CFL numbers."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/artemis_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROOT = Path(__file__).resolve().parent
gc = load_module(ROOT / "run_adi_grid_convergence.py", "adi_grid_conv")

CFL_STYLE = {
    1.0: ("C0", "o", "CFL = 1"),
    4.0: ("C1", "s", "CFL = 4"),
    16.0: ("C2", "^", "CFL = 16"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Periodic sinusoidal IC grid convergence for ADI at several CFLs."
    )
    parser.add_argument(
        "--exe",
        default="Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex",
        help="WarpX/Artemis executable.",
    )
    parser.add_argument(
        "--launcher",
        nargs="*",
        default=[],
        help="Optional launcher, e.g. --launcher srun -n 1",
    )
    parser.add_argument("--cells", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--cfls", nargs="+", type=float, default=[1.0, 4.0, 16.0])
    parser.add_argument("--length", type=float, default=4.0e-6)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument(
        "--direction",
        nargs="+",
        choices=["x", "y", "z"],
        default=["x"],
        help="Propagation direction(s) to test.",
    )
    parser.add_argument(
        "--time-samples",
        type=int,
        default=4,
        help="Requested aligned outputs per period; reduced if CFL leaves too few steps.",
    )
    parser.add_argument("--workdir", default="adi_cfl_convergence")
    parser.add_argument("--plot", default="cfl_convergence.png")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the workdir and rerun every case.",
    )
    parser.add_argument(
        "--require-second-order",
        action="store_true",
        help="Exit nonzero if the finest L2 order is below 1.9 for any CFL/direction.",
    )
    return parser.parse_args()


def nsteps_for(n: int, cfl: float) -> int:
    return max(1, int(round(n / cfl)))


def aligned_time_samples(cells: list[int], cfl: float, requested: int) -> int:
    """Largest sample count <= requested that divides every nsteps in the grid scan."""
    steps = [nsteps_for(n, cfl) for n in cells]
    g = steps[0]
    for nstep in steps[1:]:
        g = math.gcd(g, nstep)
    for samples in range(min(requested, g), 0, -1):
        if g % samples == 0:
            return samples
    return 1


def grid_args(args: argparse.Namespace, cfl: float, time_samples: int) -> argparse.Namespace:
    cfl_args = argparse.Namespace(**vars(args))
    cfl_args.cfl = cfl
    cfl_args.time_samples = time_samples
    return cfl_args


def plotfiles_complete(case_dir: Path, expected_samples: int) -> bool:
    try:
        files = gc.plotfiles(case_dir)
    except FileNotFoundError:
        return False
    n = len(files)
    if n != expected_samples and n != expected_samples + 1:
        return False
    return all((path / "WarpXHeader").is_file() for path in files)


def write_case_inputs(
    case_dir: Path, n: int, cfl_args: argparse.Namespace, direction: str
) -> tuple[Path, int, int]:
    input_file, nsteps, plot_interval = gc.write_inputs(
        case_dir, n, cfl_args, direction
    )
    e_comp = gc.WAVE_CONFIG[direction][0]
    text = input_file.read_text()
    text = text.replace(
        "plt.fields_to_plot = Ex Ey Ez Bx By Bz",
        f"plt.fields_to_plot = {e_comp}",
    )
    input_file.write_text(text)
    return input_file, nsteps, plot_interval


def run_artemis(
    cfl_args: argparse.Namespace,
    exe: Path,
    case_dir: Path,
    n: int,
    direction: str,
    retries: int = 3,
) -> tuple[int, int]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        input_file, nsteps, plot_interval = write_case_inputs(
            case_dir, n, cfl_args, direction
        )
        cmd = [*cfl_args.launcher, str(exe), str(input_file)]
        retry_note = f" (attempt {attempt}/{retries})" if attempt > 1 else ""
        print(
            f"[{direction} CFL={cfl_args.cfl:g}] running N={n}, "
            f"steps={nsteps}, plot_interval={plot_interval}{retry_note}: "
            f"{' '.join(cmd)}"
        )
        log_path = case_dir / "run.log"
        with log_path.open("w") as log:
            result = subprocess.run(
                cmd,
                cwd=case_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode == 0 and plotfiles_complete(
            case_dir, cfl_args.time_samples
        ):
            return nsteps, plot_interval
        last_error = subprocess.CalledProcessError(result.returncode, cmd)
        print(
            f"[{direction} CFL={cfl_args.cfl:g}] N={n} failed "
            f"(exit {result.returncode}); see {log_path}"
        )
    raise last_error


def run_case(
    cfl_args: argparse.Namespace,
    exe: Path,
    case_dir: Path,
    n: int,
    direction: str,
) -> tuple[int, int, float, float, int, float, tuple[int, int, int]]:
    reuse = (
        not cfl_args.fresh
        and (case_dir / "inputs").exists()
        and plotfiles_complete(case_dir, cfl_args.time_samples)
    )
    reuse = True
    if reuse:
        nsteps = nsteps_for(n, cfl_args.cfl)
        plot_interval = nsteps // cfl_args.time_samples
        print(
            f"[{direction} CFL={cfl_args.cfl:g}] reuse N={n}, "
            f"steps={nsteps}, plot_interval={plot_interval}"
        )
    else:
        nsteps, plot_interval = run_artemis(cfl_args, exe, case_dir, n, direction)

    l2, linf, worst_step, worst_time, worst_index = gc.space_time_error(
        case_dir,
        n,
        cfl_args.length,
        cfl_args.amplitude,
        cfl_args.time_samples,
        direction,
    )
    return nsteps, plot_interval, l2, linf, worst_step, worst_time, worst_index


def run_cfl(
    args: argparse.Namespace, exe: Path, workdir: Path, cfl: float
) -> dict[str, list[tuple[int, int, int, float, float, int, float, tuple[int, int, int]]]]:
    time_samples = aligned_time_samples(args.cells, cfl, args.time_samples)
    if time_samples != args.time_samples:
        print(
            f"CFL={cfl:g}: reducing time samples {args.time_samples} -> {time_samples} "
            f"so every N in {args.cells} has nsteps divisible by the sample count"
        )
    cfl_args = grid_args(args, cfl, time_samples)
    cfl_workdir = workdir / f"cfl_{cfl:g}".replace(".", "p")
    cfl_workdir.mkdir(parents=True, exist_ok=True)

    all_rows = {}
    for direction in args.direction:
        dir_workdir = cfl_workdir / f"prop_{direction}"
        dir_workdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for n in args.cells:
            case_dir = dir_workdir / f"n{n:04d}"
            nsteps, plot_interval, l2, linf, worst_step, worst_time, worst_index = (
                run_case(cfl_args, exe, case_dir, n, direction)
            )
            rows.append(
                (n, nsteps, plot_interval, l2, linf, worst_step, worst_time, worst_index)
            )
        all_rows[direction] = rows
        print(f"\n=== CFL = {cfl:g} ===")
        gc.print_table(direction, rows)
    return all_rows


def plot_cfl_convergence(
    all_results: dict[
        float,
        dict[
            str,
            list[tuple[int, int, int, float, float, int, float, tuple[int, int, int]]],
        ],
    ],
    directions: list[str],
    cfls: list[float],
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        len(directions),
        figsize=(5.2 * len(directions), 8.2),
        dpi=700,
        squeeze=False,
        sharex=True,
    )
    for j, direction in enumerate(directions):
        ax_l2 = axes[0][j]
        ax_inf = axes[1][j]
        href = None
        l2ref = None
        linfref = None
        for cfl in cfls:
            rows = all_results[cfl][direction]
            cells = np.array([row[0] for row in rows], dtype=float)
            h = 1.0 / cells
            l2 = np.array([row[3] for row in rows])
            linf = np.array([row[4] for row in rows])
            color, marker, label = CFL_STYLE.get(
                float(cfl), ("C0", "o", f"CFL = {cfl:g}")
            )
            if href is None:
                href = h
                l2ref = l2
                linfref = linf
            ax_l2.loglog(h, l2, marker + "-", color=color, ms=7, label=label)
            ax_inf.loglog(h, linf, marker + "-", color=color, ms=7, label=label)

        for order, style in [(1, "--"), (2, ":")]:
            ax_l2.loglog(
                href,
                l2ref[0] * (href / href[0]) ** order,
                "k" + style,
                alpha=0.45,
                label=rf"$O(h^{order})$",
            )
            ax_inf.loglog(
                href,
                linfref[0] * (href / href[0]) ** order,
                "k" + style,
                alpha=0.45,
                label=rf"$O(h^{order})$",
            )

        for ax, ylabel in ((ax_l2, "relative $L^2$"), (ax_inf, r"relative $L^\infty$")):
            ax.invert_xaxis()
            ax.set_ylabel(ylabel)
            ax.set_title(f"ADI sine-wave +{direction}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(frameon=False, fontsize=8)
        axes[1][j].set_xlabel("grid spacing $h/L$")

    fig.suptitle("ADI grid convergence vs CFL", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def print_order_summary(
    all_results: dict[
        float,
        dict[
            str,
            list[tuple[int, int, int, float, float, int, float, tuple[int, int, int]]],
        ],
    ],
    require: bool,
) -> None:
    print("\n=== finest L2 observed order ===")
    ok = True
    for cfl, by_dir in all_results.items():
        for direction, rows in by_dir.items():
            orders = gc.observed_orders([row[3] for row in rows])
            finest = orders[-1] if orders else float("nan")
            passed = finest >= 1.9
            ok = ok and passed
            status = "PASS" if passed else "FAIL"
            print(f"  CFL={cfl:g} +{direction}: finest L2 order = {finest:.3f}  [{status}]")
    if require and not ok:
        raise SystemExit("second-order convergence not verified for all CFL/direction pairs")


def main() -> None:
    args = parse_args()
    gc.yt.funcs.mylog.setLevel(50)

    exe = Path(args.exe).resolve()
    if not exe.exists():
        raise FileNotFoundError(f"executable not found: {exe}")

    workdir = Path(args.workdir).resolve()
    if workdir.exists() and args.fresh:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for cfl in args.cfls:
        all_results[cfl] = run_cfl(args, exe, workdir, cfl)

    print_order_summary(all_results, args.require_second_order)

    plot_path = Path(args.plot)
    if not plot_path.is_absolute():
        plot_path = workdir / plot_path
    plot_cfl_convergence(all_results, list(args.direction), list(args.cfls), plot_path)
    print(f"\nwrote {plot_path}")


if __name__ == "__main__":
    main()
