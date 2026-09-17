#!/usr/bin/env python3
"""Run PEC soft-drive A–D at CFL=128 and plot Ey along the cavity centerline."""

from __future__ import annotations

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
import yt

yt.funcs.mylog.setLevel(50)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROOT = Path(__file__).resolve().parent
resonance = load_module(ROOT / "run_adi_pec_resonance.py", "resonance")

CFL = 128.0
SCHEMES = ("a", "b", "c", "d")
WORKDIR = ROOT / "adi_dispersion/artemis_pec_soft_lineout_cfl128_every_step"
OUT = ROOT / "adi_dispersion/pec_soft_ey_centerline_cfl128.png"
OUT_VIDEO = ROOT / "adi_dispersion/pec_soft_ey_centerline_cfl128.webm"
CACHE = WORKDIR / "ey_centerline_cache.npz"
PLOT_PREFIX = "diags/plotfiles/plt"
VIDEO_FPS = 10
STILL_N_TIMES = 5


def nsteps_for_cfl(cfl: float) -> tuple[int, float, float, float, int, int]:
    lz, nz, k, _kh = resonance.mode_grid("pec")
    dz = lz / nz
    dt = cfl * dz / resonance.C0
    f0 = resonance.C0 * k / (2.0 * math.pi)
    t0 = 1.0 / f0
    nsteps = int(math.ceil(resonance.N_PERIODS * t0 / dt))
    nx = ny = resonance.N_TRANS
    return nsteps, dt, f0, t0, nx, nz


def load_all_lineouts(plts: list[Path]) -> list[tuple[float, np.ndarray, np.ndarray]]:
    out = []
    for i, path in enumerate(plts):
        out.append(ey_centerline(path))
        if (i + 1) % 20 == 0 or i + 1 == len(plts):
            print(f"    loaded {i + 1}/{len(plts)} plotfiles")
    return out


def write_and_run(scheme: str, exe: Path) -> Path:
    nsteps, dt, f0, t0, nx, nz = nsteps_for_cfl(CFL)
    lz, _nz, _k, _kh = resonance.mode_grid("pec")
    dz = lz / nz
    lx = nx * dz
    ly = nx * dz
    interval = 1
    case_dir = WORKDIR / resonance.case_name("pec", CFL, scheme)
    dumps = sorted(p for p in case_dir.glob(f"{PLOT_PREFIX}[0-9]*") if p.is_dir())
    if len(dumps) >= nsteps:
        print(f"[reuse] {case_dir} ({len(dumps)} plotfiles)")
        return case_dir

    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    text = resonance.build_inputs(
        "pec",
        nsteps=nsteps,
        diag_interval=max(1, nsteps),
        dt=dt,
        nx=nx,
        ny=nx,
        nz=nz,
        lx=lx,
        ly=ly,
        lz=lz,
        eps0=resonance.EPS0,
        mu0=resonance.MU0,
        c0=resonance.C0,
        freq=f0,
        tp=t0,
        z0=0.25 * lz,
        dz=dz,
        adi_e_excitation=scheme,
    )
    text += (
        "\ndiagnostics.diags_names = plt\n"
        "plt.diag_type = Full\n"
        f"plt.intervals = {interval}\n"
        "plt.fields_to_plot = Ey\n"
        f"plt.file_prefix = {PLOT_PREFIX}\n"
    )
    inputs = case_dir / "inputs"
    inputs.write_text(text)

    print(
        f"[Artemis] scheme={scheme} CFL={CFL:g} nsteps={nsteps} "
        f"plot_interval={interval}"
    )
    log_path = case_dir / "run.log"
    with log_path.open("w") as log:
        subprocess.run(
            [str(exe), str(inputs)],
            cwd=case_dir,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return case_dir


def list_plotfiles(case_dir: Path) -> list[Path]:
    return sorted(p for p in case_dir.glob(f"{PLOT_PREFIX}[0-9]*") if p.is_dir())


def ey_centerline(plt_path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """Ey vs z at the transverse center (x=Lx/2, y=Ly/2)."""
    ds = yt.load(str(plt_path))
    dims = tuple(int(n) for n in ds.domain_dimensions)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    ey = np.asarray(grid[("boxlib", "Ey")]).squeeze()
    lo = np.asarray(ds.domain_left_edge)
    hi = np.asarray(ds.domain_right_edge)
    # covering_grid is cell-centered in each dumped index
    z = np.linspace(float(lo[2]), float(hi[2]), ey.shape[-1], endpoint=False)
    z += 0.5 * (float(hi[2]) - float(lo[2])) / ey.shape[-1]
    ix = dims[0] // 2
    iy = dims[1] // 2
    return float(ds.current_time), z, np.asarray(ey[ix, iy, :], dtype=np.float64)


SCHEME_STYLE = {
    "a": ("C0", "-", "A"),
    "b": ("C1", "--", "B"),
    "c": ("C2", "-.", "C"),
    "d": ("C3", ":", "D"),
}


def plot_lineouts(series: dict[str, list[tuple[float, np.ndarray, np.ndarray]]]) -> None:
    n_all = len(next(iter(series.values())))
    pick = np.unique(np.linspace(0, n_all - 1, STILL_N_TIMES, dtype=int))
    sampled = {s: [series[s][i] for i in pick] for s in SCHEMES}
    n_times = len(pick)
    ncols = 3
    nrows = int(math.ceil(n_times / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.6 * nrows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    f0 = resonance.C0 / resonance.LENGTH_Z

    for i in range(n_times):
        ax = axes_flat[i]
        t = sampled[SCHEMES[0]][i][0]
        for scheme in SCHEMES:
            _t, z, ey = sampled[scheme][i]
            color, ls, label = SCHEME_STYLE[scheme]
            ax.plot(z / resonance.LENGTH_Z, ey, color=color, ls=ls, lw=1.4, label=label)
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_title(rf"$t f_0={t * f0:.2f}$")
        ax.grid(alpha=0.25)
        ax.set_xlim(0.0, 1.0)
        if i == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2)

    for j in range(n_times, len(axes_flat)):
        axes_flat[j].set_visible(False)

    for ax in axes_flat[max(0, n_times - ncols) : n_times]:
        ax.set_xlabel(r"$z/L$")
    for r in range(nrows):
        axes_flat[r * ncols].set_ylabel(r"$E_y$ (centerline)")

    fig.suptitle(
        rf"PEC soft drive, CFL $S={CFL:g}$: $E_y(z)$ at $x=L_x/2$, $y=L_y/2$",
        fontsize=12,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(OUT.resolve())


def save_cache(series: dict[str, list[tuple[float, np.ndarray, np.ndarray]]]) -> None:
    payload = {"schemes": np.array(SCHEMES)}
    z0 = series[SCHEMES[0]][0][1]
    times = np.array([t for t, _z, _ey in series[SCHEMES[0]]])
    payload["z"] = z0
    payload["times"] = times
    for scheme in SCHEMES:
        payload[f"ey_{scheme}"] = np.stack([ey for _t, _z, ey in series[scheme]])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **payload)


def load_cache() -> dict[str, list[tuple[float, np.ndarray, np.ndarray]]] | None:
    if not CACHE.exists():
        return None
    data = np.load(CACHE)
    z = data["z"]
    times = data["times"]
    series = {}
    for scheme in SCHEMES:
        key = f"ey_{scheme}"
        if key not in data:
            return None
        series[scheme] = [(float(t), z, data[key][i]) for i, t in enumerate(times)]
    return series


def write_video(series: dict[str, list[tuple[float, np.ndarray, np.ndarray]]]) -> None:
    from matplotlib.animation import FFMpegWriter

    f0 = resonance.C0 / resonance.LENGTH_Z
    times = np.array([t for t, _z, _ey in series[SCHEMES[0]]])
    z = series[SCHEMES[0]][0][1] / resonance.LENGTH_Z
    ey = {scheme: np.stack([row[2] for row in series[scheme]]) for scheme in SCHEMES}
    ymax = 1.05 * max(np.max(np.abs(arr)) for arr in ey.values())

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    lines = {}
    for scheme in SCHEMES:
        color, ls, label = SCHEME_STYLE[scheme]
        (line,) = ax.plot(z, ey[scheme][0], color=color, ls=ls, lw=1.8, label=label)
        lines[scheme] = line
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel(r"$z/L$")
    ax.set_ylabel(r"$E_y$ (centerline)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    title = ax.set_title(rf"$t f_0={times[0] * f0:.2f}$")
    fig.tight_layout()

    writer = FFMpegWriter(fps=VIDEO_FPS, codec="libvpx-vp9")
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(OUT_VIDEO), dpi=160):
        for i, t in enumerate(times):
            for scheme in SCHEMES:
                lines[scheme].set_ydata(ey[scheme][i])
            title.set_text(rf"$t f_0={t * f0:.2f}$")
            writer.grab_frame()
    plt.close(fig)
    print(OUT_VIDEO.resolve())


def main() -> None:
    exe = resonance.EXE.resolve()
    if not exe.exists():
        raise FileNotFoundError(exe)
    WORKDIR.mkdir(parents=True, exist_ok=True)

    series = load_cache()
    if series is None:
        series = {}
        for scheme in SCHEMES:
            case_dir = write_and_run(scheme, exe)
            plts = list_plotfiles(case_dir)
            if not plts:
                raise FileNotFoundError(f"no plotfiles in {case_dir}")
            series[scheme] = load_all_lineouts(plts)
            print(f"  scheme {scheme}: {len(plts)} dumps")
        save_cache(series)
    else:
        print(f"[reuse] lineout cache {CACHE}")

    plot_lineouts(series)
    write_video(series)


if __name__ == "__main__":
    main()
