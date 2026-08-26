#!/usr/bin/env python3
"""3D PEC air cavity (original Courant case): explicit FDTD vs ADI at CFL=1."""

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


C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6

EXE = Path("Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex")
WORKDIR = Path("adi_dispersion/artemis_cavity3d_te")

# Original reference parameters (explicit Yee Courant limit).
FREQ = 2.0e9  # Hz
DT_COURANT = 5.8e-12  # s
TP = 30.0 * DT_COURANT  # 0.174 ns; 10x wider bandwidth than the original pulse
NSTEPS = 10000
CFL = 1.0

LX, LY, LZ = 9.0e-2, 6.0e-2, 15.0e-2
DX = DY = DZ = 0.3e-2
NX, NY, NZ = int(round(LX / DX)), int(round(LY / DY)), int(round(LZ / DZ))  # 30, 20, 50

SCHEMES = ("fdtd", "adi")  # macroscopic Yee FDTD vs ADI
TE_PEAKS_GHZ = [1.95, 2.60, 3.42, 4.31]
PLOT_INTERVAL = 100
PROBE_INTERVAL = 1

INPUT_TEMPLATE = """\
max_step = {nsteps}

geometry.dims = 3
geometry.prob_lo = 0.0 0.0 0.0
geometry.prob_hi = {lx:.17e} {ly:.17e} {lz:.17e}

amr.n_cell = {nx} {ny} {nz}
amr.max_level = 0
amr.max_grid_size = {nz}
amr.blocking_factor = 2

boundary.field_lo = pec pec pec
boundary.field_hi = pec pec pec

warpx.verbose = 0
warpx.const_dt = {dt:.17e}
warpx.use_filter = 0
warpx.do_particle_cfl_guards = 0

algo.em_solver_medium = macroscopic
algo.time_stepping_scheme = {scheme}
algo.macroscopic_sigma_method = laxwendroff

macroscopic.sigma_function(x,y,z) = "0.0"
macroscopic.epsilon_function(x,y,z) = "{eps0:.17e}"
macroscopic.mu_function(x,y,z) = "{mu0:.17e}"

my_constants.pi = 3.141592653589793
my_constants.freq = {freq:.17e}
my_constants.TP = {tp:.17e}
my_constants.dt = {dt:.17e}
my_constants.dt0 = {dt0:.17e}
my_constants.E0 = 1.0
my_constants.flag_none = 0
my_constants.flag_soft = 2
my_constants.x0 = {x0:.17e}
my_constants.y0 = {y0:.17e}
my_constants.z0 = {z0:.17e}
my_constants.xp = {xp:.17e}
my_constants.yp = {yp:.17e}
my_constants.zp = {zp:.17e}
my_constants.dx = {dx:.17e}
my_constants.dy = {dy:.17e}
my_constants.dz = {dz:.17e}

# Soft Ey line source along y at cavity center (original: E += pulse*cos).
# (dt/dt0)=1 at CFL=1; kept for consistency with larger-CFL ADI runs.
warpx.E_excitation_on_grid_style = parse_E_excitation_grid_function
warpx.Ex_excitation_flag_function(x,y,z) = "flag_none"
warpx.Ey_excitation_flag_function(x,y,z) = "flag_soft * (abs(x-x0) < dx/2) * (abs(z-z0) < dz/2)"
warpx.Ez_excitation_flag_function(x,y,z) = "flag_none"
warpx.Ex_excitation_grid_function(x,y,z,t) = "0.0"
warpx.Ey_excitation_grid_function(x,y,z,t) = "E0*(dt/dt0)*exp(-(t-3*TP)**2/(2*TP**2))*cos(2*pi*freq*t) * (abs(x-x0) < dx/2) * (abs(z-z0) < dz/2)"
warpx.Ez_excitation_grid_function(x,y,z,t) = "0.0"

warpx.reduced_diags_names = Eobs0
Eobs0.type = RawEFieldReduction
Eobs0.reduction_type = integral
Eobs0.integration_type = volume
Eobs0.intervals = {diag_interval}
Eobs0.reduced_function(x,y,z) = (abs(x-xp) < dx/2) * (abs(y-yp) < dy/2) * (abs(z-zp) < dz/2)

diagnostics.diags_names = plt
plt.diag_type = Full
plt.intervals = {plot_interval}
plt.fields_to_plot = Ex Ey Ez Bx By Bz
plt.file_prefix = plt
"""


def case_name(scheme: str) -> str:
    return f"{scheme}_cfl_{CFL:g}".replace(".", "p")


def te_analytic_ghz(m: int, p: int) -> float:
    return 0.5 * C0 * math.sqrt((m / LX) ** 2 + (p / LZ) ** 2) / 1.0e9


def read_ey_vm(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = case_dir / "diags" / "reducedfiles" / "Eobs0.txt"
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ey = data[:, 3] / (DX * DY * DZ)
    return data[:, 1], ey


def run_case(scheme: str, workdir: Path, exe: Path) -> Path:
    dt = CFL * DT_COURANT
    x0, y0, z0 = 0.5 * LX, 0.5 * LY, 0.5 * LZ
    xp, yp, zp = x0 + 2.0 * DX, y0, z0

    case_dir = workdir / case_name(scheme)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    (case_dir / "inputs").write_text(
        INPUT_TEMPLATE.format(
            nsteps=NSTEPS,
            diag_interval=PROBE_INTERVAL,
            plot_interval=PLOT_INTERVAL,
            scheme=scheme,
            dt=dt,
            dt0=DT_COURANT,
            lx=LX,
            ly=LY,
            lz=LZ,
            nx=NX,
            ny=NY,
            nz=NZ,
            dx=DX,
            dy=DY,
            dz=DZ,
            eps0=EPS0,
            mu0=MU0,
            freq=FREQ,
            tp=TP,
            x0=x0,
            y0=y0,
            z0=z0,
            xp=xp,
            yp=yp,
            zp=zp,
        )
    )

    print(
        f"[Artemis] scheme={scheme}, CFL={CFL:g}, N={NX}x{NY}x{NZ}, "
        f"dt={dt:.3e} s, steps={NSTEPS}, plot={PLOT_INTERVAL}"
    )
    with (case_dir / "run.log").open("w") as log:
        subprocess.run(
            [str(exe), str(case_dir / "inputs")],
            cwd=case_dir,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return case_dir


def spectrum(times: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i0 = len(times) // 2
    t = times[i0:]
    y = signal[i0:] - np.mean(signal[i0:])
    dt = float(np.median(np.diff(t)))
    n = len(y)
    n_fft = max(n, 8 * n)
    amp = np.abs(np.fft.rfft(y * np.hanning(n), n=n_fft)) / n
    freqs = np.fft.rfftfreq(n_fft, d=dt) / 1.0e9
    return freqs, amp


def main() -> None:
    exe = EXE.resolve()
    if not exe.exists():
        raise FileNotFoundError(f"executable not found: {exe}")

    print(
        f"3D PEC cavity {LX}x{LY}x{LZ} m, dx={DX} m -> {NX}x{NY}x{NZ}; "
        f"f0={FREQ/1e9:g} GHz, DT={DT_COURANT:g} s (CFL={CFL:g})"
    )
    print("Analytic TE_m0p [GHz]:")
    for m, p in [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2)]:
        print(f"  TE_{m}0{p}: {te_analytic_ghz(m, p):.3f}")

    workdir = WORKDIR.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scheme in SCHEMES:
        case_dir = run_case(scheme, workdir, exe)
        series[scheme] = read_ey_vm(case_dir)

    styles = {
        "fdtd": dict(color="C0", label="FDTD (Yee)"),
        "adi": dict(color="C1", label="ADI"),
    }

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=False)
    for scheme in SCHEMES:
        t, ey = series[scheme]
        axes[0].plot(
            t * 1.0e9,
            ey,
            color=styles[scheme]["color"],
            lw=1.0,
            label=styles[scheme]["label"],
        )
    axes[0].set_xlabel(r"Time [ns]")
    axes[0].set_ylabel(r"Electric Field $E_y$ [V/m]")
    axes[0].set_title(r"$E_y$ two grids from cavity center (air cavity)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    for scheme in SCHEMES:
        freqs, amp = spectrum(*series[scheme])
        axes[1].plot(
            freqs,
            amp,
            color=styles[scheme]["color"],
            lw=1.5,
            label=styles[scheme]["label"],
        )
    for fpeak in TE_PEAKS_GHZ:
        axes[1].axvline(fpeak, color="0.4", ls=":", lw=1.0, alpha=0.8)
    axes[1].set_xlim(0.0, 5.5)
    axes[1].set_xlabel(r"Frequency [GHz]")
    axes[1].set_ylabel(r"$|E_y|$ spectrum (arb.)")
    axes[1].set_title(r"$E_y$ field two grids away from cavity center, air cavity")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle(
        rf"3-D PEC cavity: FDTD vs ADI at CFL$={CFL:g}$ "
        rf"($f_0={FREQ/1e9:g}$ GHz, $N={NSTEPS}$)",
        fontsize=13,
    )
    fig.tight_layout()

    outdir = Path("adi_dispersion")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "cavity3d_te_fdtd_vs_adi.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(path.resolve())


if __name__ == "__main__":
    main()
