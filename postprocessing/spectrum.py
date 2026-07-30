"""Frequency-spectrum helpers shared by Eobs and E-field time scripts."""

from __future__ import annotations

import argparse

import numpy as np


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


def add_spectrum_args(parser: argparse.ArgumentParser) -> None:
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


def validate_fft_pad(fft_pad: float) -> None:
    if fft_pad < 1.0:
        raise ValueError("--fft-pad must be >= 1")
