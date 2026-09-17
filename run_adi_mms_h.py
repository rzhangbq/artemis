#!/usr/bin/env python3
"""1D ADI MMS with nonzero magnetic source on the 3D Artemis PEC cavity, schemes A–D."""

from __future__ import annotations

import argparse
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


C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6

EXE = Path("Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex")
WORKDIR = Path("adi_mms_h/artemis_pec_mms")
OUTDIR = Path("adi_mms_h")
PLOT_PREFIX = "diags/plotfiles/plt"

L = 8.0e-6
NZ = 2928
N_TRANS = 8
N_PERIODS = 4.0
H0 = 1.0
CFLS = [32.0, 64.0, 128.0, 256.0, 512.0]
SCHEMES = ("a", "b", "c", "d")
FFT_PAD = 8

# ω = (3/2) c k ≠ c k  ⇒  nonzero MMS source
K = 2.0 * math.pi / L
OMEGA = 1.5 * C0 * K  # 3 π c / L
F0 = OMEGA / (2.0 * math.pi)
S_AMP = H0 * (C0**2 * K**2 - OMEGA**2) / OMEGA
DZ = L / NZ
KH = K * DZ
# cos(k L/2) = -1; Bx lives at z-faces (k+1/2)
Z_PROBE = (NZ / 2 - 0.5) * DZ
COS_PROBE = math.cos(K * Z_PROBE)

STYLES = {
    "a": ("C0", "x", r"A: $\Delta t\,S_H^{n+1/2}$ first half"),
    "b": ("C1", "o", r"B: $\Delta t\,S_H^{n+1/2}$ second half"),
    "c": ("C2", "s", r"C: $\frac{1}{2}\Delta t\,S_H^{n+1/2}$ both"),
    "d": ("C3", "D", r"D: $\frac{1}{2}\Delta t\,S_H^{n+1/4},\,\frac{1}{2}\Delta t\,S_H^{n+3/4}$"),
}

INPUT_TEMPLATE = """\
max_step = {nsteps}

geometry.dims = 3
geometry.prob_lo = 0.0 0.0 0.0
geometry.prob_hi = {lx:.17e} {ly:.17e} {lz:.17e}

amr.n_cell = {nx} {ny} {nz}
amr.max_level = 0
amr.max_grid_size = {nz}
amr.blocking_factor = 8

# PEC walls normal to propagation (z); periodic transversely.
boundary.field_lo = periodic periodic pec
boundary.field_hi = periodic periodic pec

warpx.verbose = 0
warpx.const_dt = {dt:.17e}
warpx.use_filter = 0
warpx.do_particle_cfl_guards = 0

algo.em_solver_medium = macroscopic
algo.time_stepping_scheme = adi
algo.adi_b_excitation = {scheme}
algo.macroscopic_sigma_method = laxwendroff

macroscopic.sigma_function(x,y,z) = "0.0"
macroscopic.epsilon_function(x,y,z) = "{eps0:.17e}"
macroscopic.mu_function(x,y,z) = "{mu0:.17e}"

my_constants.pi = 3.141592653589793
my_constants.c = {c0:.17e}
my_constants.L = {lz:.17e}
my_constants.H0 = {h0:.17e}
my_constants.mu0 = {mu0:.17e}
my_constants.omega = {omega:.17e}
my_constants.Samp = {s_amp:.17e}
my_constants.dt = {dt:.17e}
my_constants.z0 = {z0:.17e}
my_constants.dz = {dz:.17e}
my_constants.flag_none = 0
my_constants.flag_soft = 2

# MMS IC: H(z,0)=H0 cos(kz), E=0.  B=μH.
warpx.E_ext_grid_init_style = parse_E_ext_grid_function
warpx.Ex_external_grid_function(x,y,z) = "0.0"
warpx.Ey_external_grid_function(x,y,z) = "0.0"
warpx.Ez_external_grid_function(x,y,z) = "0.0"

warpx.B_ext_grid_init_style = parse_B_ext_grid_function
warpx.Bx_external_grid_function(x,y,z) = "mu0*H0*cos(2*pi*z/L)"
warpx.By_external_grid_function(x,y,z) = "0.0"
warpx.Bz_external_grid_function(x,y,z) = "0.0"

# Soft B increment μ Δt S_H; ADI A–D splits this on the magnetic half-steps.
# S_H = Samp cos(kz) sin(ωt),  Samp = H0 (c²k²-ω²)/ω.
warpx.B_excitation_on_grid_style = parse_b_excitation_grid_function
warpx.Bx_excitation_flag_function(x,y,z) = "flag_soft"
warpx.By_excitation_flag_function(x,y,z) = "flag_none"
warpx.Bz_excitation_flag_function(x,y,z) = "flag_none"
warpx.Bx_excitation_grid_function(x,y,z,t) = "mu0*Samp*dt*cos(2*pi*z/L)*sin(omega*t)"
warpx.By_excitation_grid_function(x,y,z,t) = "0.0"
warpx.Bz_excitation_grid_function(x,y,z,t) = "0.0"

# Surface probe at z=(Nz/2-1/2) dz ≈ L/2 (H antinode): integral over that face.
warpx.reduced_diags_names = Bobs0
Bobs0.type = RawBFieldReduction
Bobs0.reduction_type = integral
Bobs0.integration_type = surface
Bobs0.surface_normal = Z
Bobs0.intervals = 1
Bobs0.reduced_function(x,y,z) = (z > z0 - dz/2) * (z < z0 + dz/2)

diagnostics.diags_names = plt
plt.diag_type = Full
plt.intervals = {nsteps}
plt.fields_to_plot = Bx
plt.file_prefix = {plot_prefix}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="1D MMS PEC test of ADI H-source schemes A–D on 3D Artemis."
    )
    parser.add_argument("--exe", default=str(EXE))
    parser.add_argument("--workdir", default=str(WORKDIR))
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--cfls", nargs="+", type=float, default=CFLS)
    parser.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def kh_label() -> str:
    return format(KH, ".3e").replace("e-0", "e-")


def case_name(scheme: str, cfl: float) -> str:
    return f"mms_h_cfl_{cfl:g}_exc{scheme}".replace(".", "p")


def nsteps_for(cfl: float) -> tuple[int, float]:
    dt = cfl * DZ / C0
    nsteps = int(math.ceil(N_PERIODS / (F0 * dt)))
    return nsteps, dt


def h_mms(z: np.ndarray | float, t: np.ndarray | float) -> np.ndarray | float:
    return H0 * np.cos(K * z) * np.cos(OMEGA * t)


def write_inputs(case_dir: Path, scheme: str, nsteps: int, dt: float, lx: float, ly: float) -> Path:
    text = INPUT_TEMPLATE.format(
        nsteps=nsteps,
        lx=lx,
        ly=ly,
        lz=L,
        nx=N_TRANS,
        ny=N_TRANS,
        nz=NZ,
        dt=dt,
        scheme=scheme,
        eps0=EPS0,
        mu0=MU0,
        c0=C0,
        h0=H0,
        omega=OMEGA,
        s_amp=S_AMP,
        z0=Z_PROBE,
        dz=DZ,
        plot_prefix=PLOT_PREFIX,
    )
    path = case_dir / "inputs"
    path.write_text(text)
    return path


def probe_path(case_dir: Path) -> Path:
    return case_dir / "diags" / "reducedfiles" / "Bobs0.txt"


def list_plotfiles(case_dir: Path) -> list[Path]:
    return sorted(p for p in case_dir.glob(f"{PLOT_PREFIX}[0-9]*") if p.is_dir())


def case_complete(case_dir: Path) -> bool:
    if not probe_path(case_dir).exists():
        return False
    return any(list_plotfiles(case_dir))


def read_probe(case_dir: Path, area: float) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(probe_path(case_dir), comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # columns: step, time, Bx, By, Bz  →  H = Bx / (area μ0)
    return data[:, 1], data[:, 2] / (area * MU0)


def bx_centerline(plt_path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    ds = yt.load(str(plt_path))
    dims = tuple(int(n) for n in ds.domain_dimensions)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    try:
        bx = np.asarray(grid[("mesh", "Bx")]).squeeze()
    except Exception:
        bx = np.asarray(grid[("boxlib", "Bx")]).squeeze()
    lo = np.asarray(ds.domain_left_edge)
    hi = np.asarray(ds.domain_right_edge)
    z = np.linspace(float(lo[2]), float(hi[2]), bx.shape[-1], endpoint=False)
    z += 0.5 * (float(hi[2]) - float(lo[2])) / bx.shape[-1]
    ix = dims[0] // 2
    iy = dims[1] // 2
    h = np.asarray(bx[ix, iy, :], dtype=np.float64) / MU0
    return float(ds.current_time.to_value()), z, h


def last_half(times: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i0 = len(times) // 2
    return times[i0:], signal[i0:]


def compute_fft(times: np.ndarray, signal: np.ndarray, pad_factor: int = FFT_PAD):
    t, y = last_half(times, signal)
    y = y - np.mean(y)
    dt = float(np.median(np.diff(t)))
    n = len(y)
    n_fft = max(n, int(pad_factor) * n)
    spec = np.fft.rfft(y * np.hanning(n), n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    return freqs, np.abs(spec) / n


def peak_metrics(times: np.ndarray, signal: np.ndarray) -> tuple[float, float]:
    freqs, amp = compute_fft(times, signal)
    i = 1 + int(np.argmax(amp[1:]))
    return float(amp[i]), float(freqs[i] / F0)


def spatial_error(plt_path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    t, z, h = bx_centerline(plt_path)
    exact = np.asarray(h_mms(z, t), dtype=np.float64)
    dz = float(z[1] - z[0]) if z.size > 1 else DZ
    err = h - exact
    l2 = float(np.sqrt(dz * np.sum(err * err)))
    return t, z, h, exact, l2


def run_case(
    exe: Path, workdir: Path, scheme: str, cfl: float, lx: float, ly: float, *, fresh: bool
) -> dict:
    nsteps, dt = nsteps_for(cfl)
    case_dir = workdir / case_name(scheme, cfl)
    area = lx * ly

    if not fresh and case_complete(case_dir):
        print(f"[reuse] {case_dir.name}")
    else:
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        inputs = write_inputs(case_dir, scheme, nsteps, dt, lx, ly)
        print(
            f"[Artemis] scheme={scheme} CFL={cfl:g} nsteps={nsteps} dt={dt:.4e} "
            f"t_end f0={nsteps * dt * F0:.3f}"
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

    times, hp = read_probe(case_dir, area)
    exact_probe = np.asarray(h_mms(Z_PROBE, times))
    probe_err = np.abs(hp - exact_probe)
    plts = list_plotfiles(case_dir)
    if not plts:
        raise FileNotFoundError(f"no plotfiles in {case_dir}")
    t_end, z, h_z, exact_z, l2_final = spatial_error(plts[-1])
    peak, ff0 = peak_metrics(times, hp)
    return {
        "t": times,
        "h_probe": hp,
        "exact_probe": exact_probe,
        "err_probe": probe_err,
        "err_probe_max": float(np.max(probe_err)),
        "err_probe_final": float(probe_err[-1]),
        "z": z,
        "H": h_z,
        "H_mms": exact_z,
        "t_end": t_end,
        "err_final": l2_final,
        "fft_peak": peak,
        "f_over_f0": ff0,
        "dt": dt,
        "nsteps": nsteps,
    }


def omega_adi(cfl: float) -> float:
    dt = cfl * DZ / C0
    return (2.0 / dt) * math.atan(cfl * math.sin(0.5 * KH))


def q_approx(cfl: float) -> float:
    dt = cfl * DZ / C0
    gamma = (C0 * dt / 2.0) ** 2
    return gamma * K**2


def a_mid1(t, cfl: float):
    """Scheme A: F_{H,mid,1} ∝ (1+q) sinc cos(ωt)."""
    dt = cfl * DZ / C0
    w1 = omega_adi(cfl)
    th = OMEGA * dt
    qq = q_approx(cfl)
    sinc = 1.0 if abs(th) < 1e-14 else math.sin(0.5 * th) / (0.5 * th)
    ap = (1.0 + qq) * sinc * H0 * (C0**2 * K**2 - OMEGA**2) / (w1**2 - OMEGA**2)
    return (H0 - ap) * np.cos(w1 * t) + ap * np.cos(OMEGA * t)


def a_mid2(t, cfl: float):
    """Scheme C: F_{H,mid,2} has both cos(ωt) and sin(ωt) pieces."""
    dt = cfl * DZ / C0
    w1 = omega_adi(cfl)
    th = OMEGA * dt
    qq = q_approx(cfl)
    denom = w1**2 - OMEGA**2
    pref = S_AMP / dt
    a = pref * (2.0 + qq) * math.sin(0.5 * th)
    b = pref * qq * math.cos(0.5 * th)
    return (
        (H0 - a / denom) * np.cos(w1 * t)
        - (OMEGA / w1) * (b / denom) * np.sin(w1 * t)
        + (a / denom) * np.cos(OMEGA * t)
        + (b / denom) * np.sin(OMEGA * t)
    )


def a_quarter(t, cfl: float):
    """Scheme D: F_{H,quarter} from draft."""
    dt = cfl * DZ / C0
    w1 = omega_adi(cfl)
    th = OMEGA * dt
    qq = q_approx(cfl)
    denom = w1**2 - OMEGA**2
    pref = S_AMP / dt
    a = pref * (math.sin(0.25 * th) + (1.0 + qq) * math.sin(0.75 * th))
    b = pref * qq * math.cos(0.25 * th)
    return (
        (H0 - a / denom) * np.cos(w1 * t)
        - (OMEGA / w1) * (b / denom) * np.sin(w1 * t)
        + (a / denom) * np.cos(OMEGA * t)
        + (b / denom) * np.sin(OMEGA * t)
    )


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(path.resolve())


def plot_all(results: dict, cfls: list[float], outdir: Path) -> None:
    label = kh_label()
    cfl_show = cfls[min(len(cfls) - 1, max(0, len(cfls) // 2 - 1))]
    x = np.asarray(cfls)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)
    for ax, scheme in zip(axes.ravel(), SCHEMES):
        out = results[scheme][cfl_show]
        color, _marker, slabel = STYLES[scheme]
        ax.plot(out["t"] * F0, out["h_probe"], color=color, lw=1.4, label="Artemis")
        ax.plot(out["t"] * F0, out["exact_probe"], "k--", lw=1.2, alpha=0.8, label="MMS")
        ax.set_title(f"{slabel}  (CFL={cfl_show:g})")
        ax.set_ylabel(r"$H_x(z=L/2)$")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel(r"$t\,f_0$")
    fig.suptitle(
        rf"MMS probe: $H=\cos(kz)\cos(\omega t)$, $\omega=\frac{{3}}{{2}} ck$, "
        rf"$N_z={NZ}$, $k\Delta z={label}$, CFL$={cfl_show:g}$",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, outdir / "mms_h_probe_history.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for scheme, (color, _marker, slabel) in STYLES.items():
        out = results[scheme][cfl_show]
        ax.semilogy(out["t"] * F0, out["err_probe"], color=color, lw=1.5, label=slabel)
    ax.set_xlabel(r"$t\,f_0$")
    ax.set_ylabel(r"$|H_x-H_{\mathrm{MMS}}|$ at $z=L/2$")
    ax.set_title(rf"Probe error vs time (CFL={cfl_show:g}, $N_z={NZ}$)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(fig, outdir / "mms_h_probe_error.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for scheme, (color, marker, slabel) in STYLES.items():
        axes[0].loglog(
            x,
            [results[scheme][c]["err_final"] for c in cfls],
            marker + "-",
            color=color,
            ms=7,
            label=slabel,
        )
        axes[1].loglog(
            x,
            [results[scheme][c]["err_probe_max"] for c in cfls],
            marker + "-",
            color=color,
            ms=7,
            label=slabel,
        )
    s_ref = np.array([cfls[0], cfls[-1]])
    e0 = results["c"][cfls[0]]["err_final"]
    axes[0].loglog(s_ref, e0 * (s_ref / cfls[0]) ** 2, "k--", lw=1.3, label=r"$\propto\mathrm{CFL}^2$")
    e0m = results["c"][cfls[0]]["err_probe_max"]
    axes[1].loglog(s_ref, e0m * (s_ref / cfls[0]) ** 2, "k--", lw=1.3, label=r"$\propto\mathrm{CFL}^2$")
    axes[0].set_xlabel("CFL $S$")
    axes[0].set_ylabel(r"$\|H_x-H_{\mathrm{MMS}}\|_{L^2}$ at $t_{\mathrm{end}}$")
    axes[0].set_title("Final-time centerline L2 error")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel("CFL $S$")
    axes[1].set_ylabel(r"$\max_t|H_x-H_{\mathrm{MMS}}|$ at $z=L/2$")
    axes[1].set_title("Max-in-time probe error")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle(
        rf"MMS H-source schemes A–D ($N_z={NZ}$, $k\Delta z={label}$)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, outdir / "mms_h_error_vs_cfl.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for scheme, (color, marker, slabel) in STYLES.items():
        axes[0].loglog(
            x,
            [results[scheme][c]["fft_peak"] for c in cfls],
            marker + "-",
            color=color,
            ms=7,
            label=slabel,
        )
        axes[1].semilogx(
            x,
            [results[scheme][c]["f_over_f0"] for c in cfls],
            marker + "-",
            color=color,
            ms=7,
            label=slabel,
        )
    axes[0].set_xlabel("CFL $S$")
    axes[0].set_ylabel(r"$|\mathrm{FFT}|/N$ peak")
    axes[0].set_title("Spectral peak magnitude")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel("CFL $S$")
    axes[1].set_ylabel(r"$f_\mathrm{peak}/f_0$")
    axes[1].set_title("Frequency ratio")
    axes[1].axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.6)
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle(
        rf"MMS H-source FFT ($N_z={NZ}$, $k\Delta z={label}$, probe $z=L/2$)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, outdir / "mms_h_fft_vs_cfl.png")

    cfl_prof = cfls[0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(results["c"][cfl_prof]["z"] / L, results["c"][cfl_prof]["H_mms"], "k-", lw=2.0, label="MMS")
    for scheme, (color, marker, slabel) in STYLES.items():
        ax.plot(
            results[scheme][cfl_prof]["z"] / L,
            results[scheme][cfl_prof]["H"],
            marker,
            color=color,
            ms=3,
            markevery=max(1, NZ // 32),
            lw=1.0,
            label=slabel,
        )
    ax.set_xlabel(r"$z/L$")
    ax.set_ylabel(r"$H_x(z,t_{\mathrm{end}})$")
    ax.set_title(rf"Profile at $t_{{\mathrm{{end}}}}$ (CFL={cfl_prof:g})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for scheme, (color, marker, slabel) in STYLES.items():
        ax.semilogx(
            cfls,
            [results[scheme][c]["err_final"] / results["c"][c]["err_final"] for c in cfls],
            marker + "-",
            color=color,
            ms=7,
            label=slabel,
        )
    ax.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.5)
    ax.set_xlabel("CFL $S$")
    ax.set_ylabel(r"$\|e\|_{L^2}/\|e\|_{L^2}^{\mathrm{(C)}}$")
    ax.set_title("Final L2 relative to scheme C")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle(rf"MMS H-source spatial check ($N_z={NZ}$)", fontsize=11)
    fig.tight_layout()
    savefig(fig, outdir / "mms_h_profile_and_ranking.png")

    cfls_show = [c for c in (64.0, 128.0, 256.0, 512.0) if c in cfls]
    if len(cfls_show) == 4:
        t_plot = np.linspace(0.0, N_PERIODS / F0, 2000)
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True, sharey=True)
        for ax, cfl in zip(axes.ravel(), cfls_show):
            ax.plot(
                t_plot * F0,
                H0 * np.cos(OMEGA * t_plot),
                "k--",
                lw=1.2,
                alpha=0.7,
                label=r"MMS $\cos(\omega t)$",
            )
            ax.plot(t_plot * F0, a_mid1(t_plot, cfl), color="C0", lw=1.4, label="mid,1")
            ax.plot(t_plot * F0, a_mid2(t_plot, cfl), color="C2", lw=1.4, label="mid,2")
            ax.plot(t_plot * F0, a_quarter(t_plot, cfl), color="C3", lw=1.4, label="quarter")
            ax.set_title(
                rf"CFL$={cfl:g}$, $q_H\approx{q_approx(cfl):.3f}$, "
                rf"$\omega_{{\mathrm{{ADI}}}}/\omega={omega_adi(cfl)/OMEGA:.4f}$"
            )
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=7, ncol=2)
        for ax in axes[-1]:
            ax.set_xlabel(r"$t\,f_0$")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$a(t)$")
        fig.suptitle(
            rf"Forced-oscillator $a(t)$ for $H=a\cos(kz)$ with $\lambda\approx -k^2$ "
            rf"($N_z={NZ}$, $\omega=\frac{{3}}{{2}} ck$, $k\Delta z={label}$)",
            fontsize=11,
        )
        fig.tight_layout()
        savefig(fig, outdir / "mms_h_theory_a.png")

    cfl_ov = 256.0 if 256.0 in cfls else cfls[-1]
    sim_map = {
        "a": ("mid,1", "C0", a_mid1),
        "c": ("mid,2", "C2", a_mid2),
        "d": ("quarter", "C3", a_quarter),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for ax, (scheme, (pname, color, a_fn)) in zip(axes, sim_map.items()):
        out = results[scheme][cfl_ov]
        t_sim = out["t"]
        t_ov = np.linspace(0.0, float(t_sim[-1]), 2000)
        ax.plot(
            t_sim * F0,
            out["h_probe"],
            "x",
            color=color,
            ms=5,
            mew=1.2,
            label=f"Artemis {scheme.upper()} ({pname})",
        )
        ax.plot(
            t_ov * F0,
            COS_PROBE * a_fn(t_ov, cfl_ov),
            "-",
            color=color,
            lw=1.5,
            alpha=0.85,
            label="theory",
        )
        ax.plot(
            t_ov * F0,
            h_mms(Z_PROBE, t_ov),
            "k--",
            lw=1.1,
            alpha=0.7,
            label="MMS",
        )
        ax.set_title(rf"{pname}, CFL$={cfl_ov:g}$")
        ax.set_xlabel(r"$t\,f_0$")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel(r"$H_x(z=L/2)$")
    fig.suptitle(
        rf"Sim vs forced-oscillator theory ($\cos(kz_p)={COS_PROBE:.3f}$, $N_z={NZ}$)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, outdir / "mms_h_theory_vs_sim.png")


def main() -> None:
    args = parse_args()
    exe = Path(args.exe).resolve()
    if not exe.exists():
        raise FileNotFoundError(exe)

    workdir = Path(args.workdir).resolve()
    outdir = Path(args.outdir).resolve()
    if workdir.exists() and args.fresh:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    lx = ly = N_TRANS * DZ
    print(
        f"Nz={NZ}, k dz={KH:.4e}, f0={F0/1e12:.4f} THz, "
        f"omega/(c k)={OMEGA/(C0*K):.3f}, S_amp={S_AMP:.6e}"
    )
    print(f"H probe z/L={Z_PROBE/L:.6f}, cos(kz)={COS_PROBE:.6f}")
    print("CFL:", args.cfls)

    results: dict[str, dict[float, dict]] = {s: {} for s in args.schemes}
    print("=== MMS PEC magnetic source (nonzero) ===")
    print(f"{'scheme':>6} {'CFL':>8} {'L2 final':>12} {'probe max':>12} {'FFT peak':>12} {'f/f0':>10}")
    for scheme in args.schemes:
        for cfl in args.cfls:
            out = run_case(exe, workdir, scheme, cfl, lx, ly, fresh=args.fresh)
            results[scheme][cfl] = out
            print(
                f"{scheme:>6} {cfl:8g} {out['err_final']:12.4e} "
                f"{out['err_probe_max']:12.4e} {out['fft_peak']:12.4e} {out['f_over_f0']:10.6f}"
            )

    print("\n=== CFL sensitivity (max/min final L2) ===")
    for scheme in args.schemes:
        errs = [results[scheme][c]["err_final"] for c in args.cfls]
        print(f"scheme {scheme}  L2 ratio={max(errs)/min(errs):.3g}")

    print("\n=== CFL sensitivity (max/min FFT peak) ===")
    for scheme in args.schemes:
        peaks = [results[scheme][c]["fft_peak"] for c in args.cfls]
        freqs = [results[scheme][c]["f_over_f0"] for c in args.cfls]
        print(
            f"mms {scheme}  FFT ratio={max(peaks)/min(peaks):.3g}  "
            f"f/f0 spread={max(freqs)-min(freqs):.4f}"
        )

    plot_all(results, list(args.cfls), outdir)


if __name__ == "__main__":
    main()
