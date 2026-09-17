#!/usr/bin/env python3
"""PEC soft-drive: spectral peak and dispersion vs CFL for ADI E-source schemes A–D."""

from __future__ import annotations

import importlib.util
import math
import os
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

WORKDIR = ROOT / "adi_dispersion/artemis_pec_soft_exc_kh2p15e-3"
CFLS = resonance.VERIFY_CFLS
SCHEMES = ("a", "b", "c", "d")
OUT = ROOT / "adi_dispersion/pec_soft_drive_cfl_ref.png"
STYLES = {
    "a": ("C0", "x", r"A: $S^{n+1/2}$ first half"),
    "b": ("C1", "o", r"B: $S^{n+1/2}$ second half"),
    "c": ("C2", "s", r"C: $\frac{1}{2} S^{n+1/2}$ both"),
    "d": ("C3", "D", r"D: $\frac{1}{2} S^{n+1/4},\,\frac{1}{2} S^{n+3/4}$"),
}


def metrics(times: np.ndarray, ey: np.ndarray) -> dict[str, float]:
    freqs, amp = resonance.compute_fft(times, ey)
    peak_i = 1 + int(np.argmax(amp[1:]))
    _lz, _nz, k, _kh = resonance.mode_grid("pec")
    f0 = resonance.C0 * k / (2.0 * math.pi)
    return {
        "fft_peak": float(amp[peak_i]),
        "f_over_f0": float(freqs[peak_i] / f0),
    }


def run_soft_drive(exe: Path, scheme: str) -> dict[float, dict[str, float]]:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    out: dict[float, dict[str, float]] = {}
    print(f"\n=== soft drive {scheme} ({WORKDIR.name}) ===")
    print(f"{'CFL':>8} {'FFT peak':>12} {'f/f0':>10}")
    for cfl in CFLS:
        case_dir = resonance.run_artemis_case(
            "pec", cfl, WORKDIR, exe, reuse=True, adi_e_excitation=scheme
        )
        times, ey = resonance.read_probe(case_dir)
        m = metrics(times, ey)
        out[cfl] = m
        print(f"{cfl:8g} {m['fft_peak']:12.4e} {m['f_over_f0']:10.6f}")
    return out


def detuning_amplitude(
    cfl: float, kh: float, f0: float, *, freq_ratio: float = 1.0, exp_prefactor: float = 0.5
) -> float:
    """A(S) ∝ exp[-exp_prefactor T_P² (ω_ADI(S) - ω_d)²] with T_P = 1/f₀."""
    tp = 1.0 / f0
    omega_0 = 2.0 * math.pi * f0
    omega_adi = omega_0 * resonance.analytical_f_over_f0(cfl, kh)
    omega_d = freq_ratio * omega_0
    return math.exp(-exp_prefactor * tp**2 * (omega_adi - omega_d) ** 2)


def modal_q(cfl: float, kh: float, dz: float, c0: float) -> float:
    """q = -γ λ for soft drive ∝ sin(kx), λ = -(4/Δx²) sin²(kΔx/2)."""
    dt = cfl * dz / c0
    gamma = (c0 * dt / 2.0) ** 2
    lam_e2 = -(4.0 / dz**2) * math.sin(0.5 * kh) ** 2
    return -gamma * lam_e2


def theta_adi(cfl: float, kh: float) -> float:
    """ω_ADI Δt."""
    return 2.0 * math.atan(cfl * math.sin(0.5 * kh))


def Fmag_over_S_dt(scheme: str, cfl: float, kh: float, dz: float, c0: float, q: float | None = None) -> float:
    """|F̂| / (|Ŝ|/Δt) from draft E-source FT at θ = ω_ADI Δt."""
    th = theta_adi(cfl, kh)
    qq = modal_q(cfl, kh, dz, c0) if q is None else q
    s2, c2 = math.sin(th / 2.0), math.cos(th / 2.0)
    if scheme in ("a", "b"):
        return 2.0 * math.sqrt(s2**2 + qq**2 * c2**2)
    if scheme == "c":
        return 2.0 * abs(s2)
    if scheme == "d":
        return abs((1.0 - qq) * math.sin(th / 4.0) + (1.0 + qq) * math.sin(3.0 * th / 4.0))
    raise ValueError(scheme)


def Fmag(scheme: str, cfl: float, kh: float, dz: float, c0: float, q: float | None = None) -> float:
    """Absolute |F̂| ∝ (|Ŝ|/Δt) × draft dimensionless factor."""
    dt = cfl * dz / c0
    return Fmag_over_S_dt(scheme, cfl, kh, dz, c0, q=q) / dt


def plot(results: dict[str, dict[float, dict[str, float]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.asarray(CFLS)

    for scheme, (color, marker, label) in STYLES.items():
        r = results[scheme]
        axes[0].loglog(
            x, [r[c]["fft_peak"] for c in CFLS], marker + "-", color=color, ms=7, label=label
        )
        axes[1].semilogx(
            x, [r[c]["f_over_f0"] for c in CFLS], marker + "-", color=color, ms=7, label=label
        )

    _lz, nz, k, kh = resonance.mode_grid("pec")
    f0 = resonance.C0 * k / (2.0 * math.pi)
    dz = resonance.LENGTH_Z / nz
    c0 = resonance.C0
    ref_cfl = CFLS[0]
    scale_from = results["c"] if "c" in results else next(iter(results.values()))
    norm0 = detuning_amplitude(ref_cfl, kh, f0) * Fmag("c", ref_cfl, kh, dz, c0)
    ref_peak = scale_from[ref_cfl]["fft_peak"]

    def amp_theory(scheme: str, cfl: float) -> float:
        return ref_peak * detuning_amplitude(cfl, kh, f0) * Fmag(scheme, cfl, kh, dz, c0) / norm0

    s_amp = np.logspace(np.log10(min(CFLS)), np.log10(max(CFLS)), 200)
    for scheme, color, ls, lab in [
        (
            "c",
            "C2",
            "--",
            r"$e^{-(T_P^2/2)(\omega_{\mathrm{ADI}}(S)-\omega_d)^2}"
            r"\left|\frac{2i\widehat{S}}{\Delta t}\sin(\theta/2)\right|$",
        ),
        (
            "a",
            "C0",
            "--",
            r"$e^{-(T_P^2/2)(\omega_{\mathrm{ADI}}(S)-\omega_d)^2}"
            r"\left|\frac{2\widehat{S}}{\Delta t}"
            r"\left[i\sin(\theta/2)-q\cos(\theta/2)\right]\right|$",
        ),
        (
            "d",
            "C3",
            "-.",
            r"$e^{-(T_P^2/2)(\omega_{\mathrm{ADI}}(S)-\omega_d)^2}"
            r"\left|\frac{i\widehat{S}}{\Delta t}\left[(1-q)\sin(\theta/4)"
            r"+(1+q)\sin(3\theta/4)\right]\right|$",
        ),
    ]:
        axes[0].loglog(
            s_amp,
            [amp_theory(scheme, c) for c in s_amp],
            color=color,
            ls=ls,
            lw=1.5,
            label=lab,
        )

    s = np.logspace(np.log10(min(CFLS)), np.log10(max(CFLS)), 200)
    f_adi = 2 * np.arctan(s * np.sin(kh / 2)) / (s * kh)
    axes[1].plot(s, f_adi, "k--", lw=1.5, label="ADI analytic")
    axes[1].axhline(1.0, color="0.5", ls=":", lw=1)

    axes[0].set_xlabel("CFL $S$")
    axes[0].set_ylabel(r"$|\mathrm{FFT}|/N$ peak")
    axes[0].set_title(
        r"Peaks $\propto$ detuning $\times\left|\widehat{F}\right|$"
        r" ($\theta=\omega_{\mathrm{ADI}}\Delta t$, $q$ modal)"
    )
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=6.5)

    axes[1].set_xlabel("CFL $S$")
    axes[1].set_ylabel(r"$f_\mathrm{peak}/f_0$")
    axes[1].set_title("Frequency (dispersion)")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle(
        rf"PEC soft drive, ADI E source A–D ($k\Delta z={resonance.KH_LABEL}$)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(OUT.resolve())


def cfl_spread(results: dict[str, dict[float, dict[str, float]]]) -> None:
    print("\n=== CFL sensitivity (max/min ratio across CFLs) ===")
    for scheme in SCHEMES:
        fft_peaks = [results[scheme][c]["fft_peak"] for c in CFLS]
        freqs = [results[scheme][c]["f_over_f0"] for c in CFLS]
        print(
            f"soft {scheme}      FFT ratio={max(fft_peaks)/min(fft_peaks):.3g}  "
            f"f/f0 spread={max(freqs)-min(freqs):.4f}"
        )


def main() -> None:
    exe = resonance.EXE.resolve()
    if not exe.exists():
        raise FileNotFoundError(exe)

    results = {scheme: run_soft_drive(exe, scheme) for scheme in SCHEMES}
    cfl_spread(results)
    plot(results)


if __name__ == "__main__":
    main()
