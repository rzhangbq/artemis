#!/usr/bin/env python3
"""CFL=64 ADI with in-domain PML along z: probe histories + centerline video.

Grid, spacing, and soft Gaussian drive match run_adi_pec_resonance.py.
PEC walls in z are replaced by CFS-PML (warpx.do_pml_in_domain = 1).
The source is the same standing-wave envelope, masked out of the PML layers.
"""

from __future__ import annotations

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
from matplotlib.animation import FuncAnimation, PillowWriter
import yt

yt.funcs.mylog.setLevel(50)


C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6

EXE = Path("Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex")
WORKDIR = Path("adi_dispersion/artemis_pml_cfl64")
OUTDIR = Path("adi_dispersion")

BLOCKING_FACTOR = 8
NZ = 2 * 1464  # 2928 cells on L = 8 µm
LENGTH_Z = 8.0e-6
N_TRANS = BLOCKING_FACTOR
CFL = 64.0
N_PERIODS = 16.0
PLOT_SAMPLES_PER_PERIOD = 16
PML_NCELL = 128
PML_KAPPA_MAX = 1.0
PML_ALPHA_MAX = 0.0
PML_M = 3.0
PML_R = 1.0e-8
PLOT_PREFIX = "diags/plotfiles/plt"


INPUTS = """\
max_step = {nsteps}

geometry.dims = 3
geometry.prob_lo = 0.0 0.0 0.0
geometry.prob_hi = {lx:.17e} {ly:.17e} {lz:.17e}

amr.n_cell = {nx} {ny} {nz}
amr.max_level = 0
amr.max_grid_size = {nz}
amr.blocking_factor = 8

# PML along the wave (z); periodic transversely. In-domain: outer pml_ncell
# cells of this domain are absorbing, outer z faces are Dirichlet.
boundary.field_lo = periodic periodic pml
boundary.field_hi = periodic periodic pml
warpx.do_pml_in_domain = 1
warpx.pml_ncell = {pml_ncell}
warpx.pml_kappa_max = {pml_kappa_max:.17e}
warpx.pml_alpha_max = {pml_alpha_max:.17e}
warpx.pml_m = {pml_m:.17e}
warpx.pml_R = {pml_R:.17e}

warpx.verbose = 1
warpx.const_dt = {dt:.17e}
warpx.use_filter = 0
warpx.do_particle_cfl_guards = 0

algo.em_solver_medium = macroscopic
algo.time_stepping_scheme = adi
algo.macroscopic_sigma_method = laxwendroff

macroscopic.sigma_function(x,y,z) = "0.0"
macroscopic.epsilon_function(x,y,z) = "{eps0:.17e}"
macroscopic.mu_function(x,y,z) = "{mu0:.17e}"

my_constants.pi = 3.141592653589793
my_constants.c = {c0:.17e}
my_constants.L = {lz:.17e}
my_constants.E0 = 1.0
my_constants.dt = {dt:.17e}
my_constants.TP = {tp:.17e}
my_constants.freq = {freq:.17e}
my_constants.zpml = {zpml:.17e}
my_constants.dz = {dz:.17e}

# Compactly supported soft source in the middle half of the domain.  The
# sin^2 window and its first derivative vanish at z/L=0.25 and z/L=0.75.
warpx.E_excitation_on_grid_style = parse_E_excitation_grid_function
warpx.Ex_excitation_flag_function(x,y,z) = "0.0"
warpx.Ey_excitation_flag_function(x,y,z) = "2.0*(z>0.25*L)*(z<0.75*L)"
warpx.Ez_excitation_flag_function(x,y,z) = "0.0"
warpx.Ex_excitation_grid_function(x,y,z,t) = "0.0"
warpx.Ey_excitation_grid_function(x,y,z,t) = "E0*(dt/TP)*sin(2*pi*(z/L-0.25))**2*exp(-(t-3*TP)**2/(2*TP**2))*sin(2*pi*freq*t)"
warpx.Ez_excitation_grid_function(x,y,z,t) = "0.0"

{probes}

diagnostics.diags_names = plt
plt.diag_type = Full
plt.intervals = {plot_interval}
plt.fields_to_plot = Ey
plt.file_prefix = {plot_prefix}
"""


def probe_plan(lz: float, dz: float, n_pml: int) -> list[tuple[str, float, str]]:
    """Several z stations: inside PML, near the interfaces, and in the cavity."""
    zpml = n_pml * dz
    return [
        ("Eobs0", 0.5 * zpml, "inside lo PML"),
        ("Eobs1", zpml + 4.0 * dz, "just inside (lo)"),
        ("Eobs2", 0.25 * lz, r"$z=L/4$"),
        ("Eobs3", 0.50 * lz, r"$z=L/2$"),
        ("Eobs4", 0.75 * lz, r"$z=3L/4$"),
        ("Eobs5", lz - zpml - 4.0 * dz, "just inside (hi)"),
        ("Eobs6", lz - 0.5 * zpml, "inside hi PML"),
    ]


def probe_block(probes: list[tuple[str, float, str]], dz: float, interval: int) -> str:
    names = " ".join(name for name, _, _ in probes)
    lines = [
        f"warpx.reduced_diags_names = {names}",
    ]
    for name, z0, _label in probes:
        lines += [
            f"{name}.type = RawEFieldReduction",
            f"{name}.reduction_type = integral",
            f"{name}.integration_type = surface",
            f"{name}.surface_normal = Z",
            f"{name}.intervals = {interval}",
            f"{name}.reduced_function(x,y,z) = (z > {z0:.17e} - {dz:.17e}/2) "
            f"* (z < {z0:.17e} + {dz:.17e}/2)",
        ]
    return "\n".join(lines)


def case_complete(case_dir: Path) -> bool:
    probe = case_dir / "diags" / "reducedfiles" / "Eobs3.txt"
    return probe.exists() and any(list_plotfiles(case_dir))


def list_plotfiles(case_dir: Path) -> list[Path]:
    return sorted(
        (p for p in case_dir.glob(f"{PLOT_PREFIX}[0-9]*") if p.is_dir()),
        key=lambda p: int(p.name.removeprefix("plt")),
    )


def write_inputs(case_dir: Path) -> tuple[Path, dict]:
    nx = ny = N_TRANS
    lz = LENGTH_Z
    nz = NZ
    dz = lz / nz
    lx = nx * dz
    ly = ny * dz
    k = 2.0 * math.pi / lz
    f0 = C0 * k / (2.0 * math.pi)
    t0 = 1.0 / f0
    dt = CFL * dz / C0
    nsteps = int(math.ceil(N_PERIODS * t0 / dt))
    steps_per_period = max(1, int(round(t0 / dt)))
    plot_interval = max(1, steps_per_period // PLOT_SAMPLES_PER_PERIOD)
    zpml = PML_NCELL * dz
    probes = probe_plan(lz, dz, PML_NCELL)
    meta = {
        "nsteps": nsteps,
        "dt": dt,
        "dz": dz,
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "f0": f0,
        "t0": t0,
        "zpml": zpml,
        "plot_interval": plot_interval,
        "probes": probes,
        "area": lx * ly,
    }
    text = INPUTS.format(
        nsteps=nsteps,
        lx=lx,
        ly=ly,
        lz=lz,
        nx=nx,
        ny=ny,
        nz=nz,
        dt=dt,
        eps0=EPS0,
        mu0=MU0,
        c0=C0,
        freq=f0,
        tp=t0,
        zpml=zpml,
        dz=dz,
        pml_ncell=PML_NCELL,
        pml_kappa_max=PML_KAPPA_MAX,
        pml_alpha_max=PML_ALPHA_MAX,
        pml_m=PML_M,
        pml_R=PML_R,
        probes=probe_block(probes, dz, 1),
        plot_interval=plot_interval,
        plot_prefix=PLOT_PREFIX,
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "inputs"
    path.write_text(text)
    return path, meta


def run_case(exe: Path, case_dir: Path, reuse: bool) -> dict:
    if reuse and case_complete(case_dir):
        _, meta = write_inputs(case_dir)
        print(f"[Artemis] reuse {case_dir}")
        return meta

    if case_dir.exists():
        shutil.rmtree(case_dir)
    inputs, meta = write_inputs(case_dir)
    print(
        f"[Artemis] PML-z CFL={CFL:g}, N={meta['nx']}x{meta['ny']}x{meta['nz']}, "
        f"L={meta['lz']:g}, pml_ncell={PML_NCELL}, steps={meta['nsteps']}, "
        f"plt.intervals={meta['plot_interval']}"
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
    return meta


def read_probe(case_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    path = case_dir / "diags" / "reducedfiles" / f"{name}.txt"
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 1], data[:, 3]


def extract_centerline(plotfile: Path) -> tuple[float, np.ndarray, np.ndarray]:
    ds = yt.load(str(plotfile))
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    ey = np.asarray(grid[("mesh", "Ey")].to_ndarray())
    nx, ny, nz = ey.shape
    z = np.asarray(grid["z"][0, 0, :].to_ndarray())
    return float(ds.current_time.to_value()), z, ey[nx // 2, ny // 2, :]


def load_lineouts(case_dir: Path, cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache.exists():
        data = np.load(cache)
        return data["times"], data["z"], data["ey"]

    files = list_plotfiles(case_dir)
    if not files:
        raise FileNotFoundError(f"no plotfiles in {case_dir}")
    times = []
    eys = []
    z = None
    for i, pf in enumerate(files):
        t, z_pf, ey = extract_centerline(pf)
        times.append(t)
        eys.append(ey)
        z = z_pf if z is None else z
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"  lineout {i + 1}/{len(files)}")
    times_a = np.asarray(times)
    ey_a = np.vstack(eys)
    np.savez_compressed(cache, times=times_a, z=z, ey=ey_a)
    return times_a, z, ey_a


def plot_probes(
    case_dir: Path,
    meta: dict,
    outpath: Path,
) -> None:
    f0 = meta["f0"]
    area = meta["area"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    cmap = plt.cm.viridis
    n = len(meta["probes"])
    for i, (name, z0, label) in enumerate(meta["probes"]):
        t, ey_int = read_probe(case_dir, name)
        ey = ey_int / area
        ax.plot(
            t * f0,
            ey,
            color=cmap(i / max(1, n - 1)),
            lw=1.3,
            label=fr"{label} ({z0 / meta['lz']:.3f}$L$)",
        )
    ax.axvline(3.0, color="k", ls=":", lw=1.0, alpha=0.7, label=r"pulse peak $t=3T_p$")
    ax.set_xlabel(r"$t\,f_0$")
    ax.set_ylabel(r"$E_y$ (surface mean at probe)")
    ax.set_title(
        rf"ADI CFS-PML along $z$, CFL$={CFL:g}$, $N_{{\mathrm{{pml}}}}={PML_NCELL}$"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(outpath.resolve())


def plot_snapshots(
    z: np.ndarray,
    times: np.ndarray,
    ey: np.ndarray,
    meta: dict,
    outpath: Path,
) -> None:
    f0 = meta["f0"]
    zpml = meta["zpml"]
    lz = meta["lz"]
    targets = np.array([0.0, 1.5, 3.0, 6.0, 9.0, 12.0, 15.0])
    tf = times * f0
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axvspan(0.0, zpml * 1e6, color="0.85", zorder=0, label="PML")
    ax.axvspan((lz - zpml) * 1e6, lz * 1e6, color="0.85", zorder=0)
    cmap = plt.cm.plasma
    for i, tgt in enumerate(targets):
        j = int(np.argmin(np.abs(tf - tgt)))
        if abs(tf[j] - tgt) > 0.4:
            continue
        ax.plot(
            z * 1e6,
            ey[j],
            color=cmap(i / max(1, len(targets) - 1)),
            lw=1.3,
            label=fr"$t f_0={tf[j]:.2f}$",
        )
    ax.set_xlabel(r"$z$ ($\mu$m)")
    ax.set_ylabel(r"$E_y$ (centerline)")
    ax.set_title(r"Centerline $E_y(x_c,y_c,z)$ snapshots")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(outpath.resolve())


def make_video(
    z: np.ndarray,
    times: np.ndarray,
    ey: np.ndarray,
    meta: dict,
    outpath: Path,
) -> None:
    f0 = meta["f0"]
    zpml = meta["zpml"]
    lz = meta["lz"]
    ymax = 1.15 * float(np.max(np.abs(ey)))
    if not np.isfinite(ymax) or ymax <= 0.0:
        ymax = 1.0

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.axvspan(0.0, zpml * 1e6, color="0.85", zorder=0, label="PML")
    ax.axvspan((lz - zpml) * 1e6, lz * 1e6, color="0.85", zorder=0)
    (line,) = ax.plot(z * 1e6, ey[0], color="C0", lw=1.5)
    ax.set_xlim(0.0, lz * 1e6)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel(r"$z$ ($\mu$m)")
    ax.set_ylabel(r"$E_y$ (centerline)")
    title = ax.set_title(rf"$t f_0 = {times[0] * f0:.3f}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()

    def update(i: int):
        line.set_ydata(ey[i])
        title.set_text(rf"$t f_0 = {times[i] * f0:.3f}$")
        return line, title

    frame_dir = outpath.parent / (outpath.stem + "_frames")
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    n = len(times)
    for i in range(n):
        line.set_ydata(ey[i])
        title.set_text(rf"$t f_0 = {times[i] * f0:.3f}$")
        fig.savefig(frame_dir / f"frame_{i:04d}.png", dpi=120)
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  video frame {i + 1}/{n}")
    plt.close(fig)

    fps = 20
    pattern = str(frame_dir / "frame_%04d.png")
    attempts = [
        (
            outpath.with_suffix(".webm"),
            [
                "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
                "-c:v", "libvpx-vp9", "-b:v", "1800k", "-pix_fmt", "yuv420p",
            ],
        ),
        (
            outpath.with_suffix(".gif"),
            [
                "ffmpeg", "-y", "-framerate", "16", "-i", pattern,
                "-vf", "scale=840:-1:flags=lanczos",
            ],
        ),
    ]
    last_err = None
    for dest, cmd in attempts:
        cmd = cmd + [str(dest)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(dest.resolve())
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            last_err = err
            print(f"video encoder failed for {dest.name}: {err}")

    gif = outpath.with_suffix(".gif")
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.axvspan(0.0, zpml * 1e6, color="0.85", zorder=0, label="PML")
    ax.axvspan((lz - zpml) * 1e6, lz * 1e6, color="0.85", zorder=0)
    (line,) = ax.plot(z * 1e6, ey[0], color="C0", lw=1.5)
    ax.set_xlim(0.0, lz * 1e6)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel(r"$z$ ($\mu$m)")
    ax.set_ylabel(r"$E_y$ (centerline)")
    title = ax.set_title(rf"$t f_0 = {times[0] * f0:.3f}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()

    def update_gif(i: int):
        line.set_ydata(ey[i])
        title.set_text(rf"$t f_0 = {times[i] * f0:.3f}$")
        return line, title

    anim = FuncAnimation(fig, update_gif, frames=n, blit=True, interval=40)
    try:
        anim.save(str(gif), writer=PillowWriter(fps=12), dpi=80)
        print(gif.resolve())
        plt.close(fig)
        return
    except Exception as err:
        last_err = err
        plt.close(fig)
    raise RuntimeError(f"could not write video: {last_err}")


def main() -> None:
    exe = EXE.resolve()
    if not exe.exists():
        raise FileNotFoundError(f"executable not found: {exe}")

    case_dir = WORKDIR.resolve()
    outdir = OUTDIR.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    reuse = os.environ.get("ADI_PML_REUSE", "0") == "1"
    meta = run_case(exe, case_dir, reuse=reuse)

    plot_probes(case_dir, meta, outdir / "pml_cfl64_probes.png")

    cache = case_dir / "centerline_ey.npz"
    print("extracting centerline Ey(z) from plotfiles")
    times, z, ey = load_lineouts(case_dir, cache)
    plot_snapshots(z, times, ey, meta, outdir / "pml_cfl64_snapshots.png")
    make_video(z, times, ey, meta, outdir / "pml_cfl64_centerline.mp4")


if __name__ == "__main__":
    main()
