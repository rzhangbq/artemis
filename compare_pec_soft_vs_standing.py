#!/usr/bin/env python3
"""Compare PEC soft-drive vs standing-wave IC: magnitude and CFL sensitivity."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
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
resonance = load_module(ROOT / "run_adi_pec_resonance.py", "resonance")
standing = load_module(ROOT / "run_adi_pec_standing.py", "standing")

CASES = {
    "soft_drive": dict(mod=resonance, workdir=ROOT / "adi_dispersion/artemis_pec_soft_kh2p15e-3"),
    "standing_ic": dict(mod=standing, workdir=ROOT / "adi_dispersion/artemis_pec_standing_kh2p15e-3"),
}
CFLS = resonance.VERIFY_CFLS
OUT = ROOT / "adi_dispersion/pec_soft_vs_standing_cfl.png"


def metrics(times: np.ndarray, ey: np.ndarray) -> dict[str, float]:
    i0 = len(times) // 2
    t, y = times[i0:], ey[i0:]
    y0 = y - np.mean(y)
    freqs, amp = resonance.compute_fft(times, ey)
    peak_i = 1 + int(np.argmax(amp[1:]))
    f0 = resonance.C0 * (math.pi / resonance.LENGTH_Z) / (2.0 * math.pi)
    return {
        "peak_abs": float(np.max(np.abs(y0))),
        "rms": float(np.sqrt(np.mean(y0**2))),
        "fft_peak": float(amp[peak_i]),
        "f_over_f0": float(freqs[peak_i] / f0),
    }


def run_case(label: str, mod, workdir: Path, exe: Path) -> dict[float, dict[str, float]]:
    workdir.mkdir(parents=True, exist_ok=True)
    out: dict[float, dict[str, float]] = {}
    print(f"\n=== {label} ({workdir.name}) ===")
    print(f"{'CFL':>8} {'peak|E|':>12} {'rms':>12} {'FFT peak':>12} {'f/f0':>10}")
    for cfl in CFLS:
        case_dir = mod.run_artemis_case("pec", cfl, workdir, exe, reuse=False)
        times, ey = mod.read_probe(case_dir)
        m = metrics(times, ey)
        out[cfl] = m
        print(
            f"{cfl:8g} {m['peak_abs']:12.4e} {m['rms']:12.4e} "
            f"{m['fft_peak']:12.4e} {m['f_over_f0']:10.6f}"
        )
    return out


def plot(results: dict[str, dict[float, dict[str, float]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    styles = {"soft_drive": ("C0", "o", "Soft drive"), "standing_ic": ("C1", "s", "Standing IC")}
    x = np.asarray(CFLS)

    for key, (color, marker, label) in styles.items():
        r = results[key]
        axes[0].loglog(x, [r[c]["peak_abs"] for c in CFLS], marker + "-", color=color, ms=7, label=label)
        axes[1].semilogx(x, [r[c]["f_over_f0"] for c in CFLS], marker + "-", color=color, ms=7, label=label)
        axes[2].loglog(x, [r[c]["fft_peak"] for c in CFLS], marker + "-", color=color, ms=7, label=label)

    kh = resonance.mode_grid("pec")[3]
    s = np.logspace(np.log10(min(CFLS)), np.log10(max(CFLS)), 200)
    f_adi = 2 * np.arctan(s * np.sin(kh / 2)) / (s * kh)
    axes[1].plot(s, f_adi, "k--", lw=1.5, label="ADI analytic")
    axes[1].axhline(1.0, color="0.5", ls=":", lw=1)

    axes[0].set_xlabel("CFL $S$")
    axes[0].set_ylabel(r"peak $|\int E_y|$ (last half)")
    axes[0].set_title("Time-domain amplitude")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("CFL $S$")
    axes[1].set_ylabel(r"$f_\mathrm{peak}/f_0$")
    axes[1].set_title("Frequency (dispersion)")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=9)

    axes[2].set_xlabel("CFL $S$")
    axes[2].set_ylabel(r"$|\mathrm{FFT}|/N$ peak")
    axes[2].set_title("Spectral peak magnitude")
    axes[2].grid(alpha=0.25, which="both")
    axes[2].legend(frameon=False)

    fig.suptitle(r"PEC: soft external drive vs standing-wave IC ($k\Delta z\approx2.15\times10^{-3}$)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(OUT.resolve())


def cfl_spread(results: dict[str, dict[float, dict[str, float]]]) -> None:
    print("\n=== CFL sensitivity (max/min ratio across CFLs) ===")
    for key in results:
        peaks = [results[key][c]["peak_abs"] for c in CFLS]
        fft_peaks = [results[key][c]["fft_peak"] for c in CFLS]
        freqs = [results[key][c]["f_over_f0"] for c in CFLS]
        print(
            f"{key:12s}  |E| ratio={max(peaks)/min(peaks):.3g}  "
            f"FFT ratio={max(fft_peaks)/min(fft_peaks):.3g}  "
            f"f/f0 spread={max(freqs)-min(freqs):.4f}"
        )


def main() -> None:
    exe = resonance.EXE.resolve()
    if not exe.exists():
        raise FileNotFoundError(exe)

    results = {}
    for label, cfg in CASES.items():
        results[label] = run_case(label, cfg["mod"], cfg["workdir"], exe)

    cfl_spread(results)
    plot(results)


if __name__ == "__main__":
    main()
