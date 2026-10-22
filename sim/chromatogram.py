"""Detector trace from a retention table: the chromatogram an operator would see.

``Simulator.run`` reports one ``(t_R, sigma)`` row per analyte. A detector does
not: it records a single signal versus time in which each analyte contributes a
Gaussian band, and bands closer than about two standard deviations fuse into
one visible peak. This module builds that signal,

    signal(t) = sum_i A_i / (sigma_i sqrt(2 pi)) * exp(-(t - t_R,i)^2 / (2 sigma_i^2)),

samples it at the detector rate from ``instrument.detector`` and optionally adds
the two artefacts every real trace carries: Gaussian white noise and a linear
baseline drift. With the shipped configuration and no arguments beyond the
table the output is deterministic and noise-free.

Areas default to 1.0 for every row because the table carries no identities or
amounts. Peak *height* is then set by width alone, A / (sigma sqrt(2 pi)), so a
late, broad peak is visibly shorter than an early, sharp one, as on an
instrument. The unit of the signal is "area 1 per peak"; ``noise_sd`` and
``drift`` are expressed in that unit.

Nothing here feeds back into the simulator or the CRF: the retention table is
the ground truth, and :func:`apparent_peaks` is deliberately the operator's view
of it, which can show fewer peaks than the table has rows.

Command line (``python -m sim.chromatogram``): simulate one method on an
analyte set, print the apparent peak count and apex list, and save a PNG with
``--out``. Plotting is the only part of ``sim`` that needs matplotlib, which is
imported inside :func:`plot` so the rest of the package never loads it.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .config import InstrumentConfig, Result

# Windowing conventions, not physical constants: the default trace runs this
# many standard deviations past the last apex and never ends before t0 plus
# this margin, so even a table of unretained peaks shows some baseline.
TAIL_SIGMAS = 10.0
MIN_WINDOW_AFTER_T0_MIN = 1.0

Table = Sequence[tuple[float, float]]


# --------------------------------------------------------------------------- #
# Time axis
# --------------------------------------------------------------------------- #
def sample_spacing(instrument: InstrumentConfig) -> float:
    """Time between detector samples (min): 1 / (60 * sampling_rate_hz).

    At the default 10 Hz that is 0.1 s = 1/600 min, about twelve samples per
    0.02 min standard deviation, enough that the trapezoid integral of a peak
    reproduces its area to well under 1e-3.
    """
    return 1.0 / (60.0 * instrument.detector.sampling_rate_hz)


def time_axis(t_end: float, instrument: InstrumentConfig) -> np.ndarray:
    """Sample times 0, dt, 2 dt, ... up to and including the last sample <= t_end."""
    if not t_end > 0.0:
        raise ValueError(f"t_end must be positive, got {t_end}")
    per_min = 60.0 * instrument.detector.sampling_rate_hz
    n = int(math.floor(t_end * per_min + 1e-9)) + 1
    if n < 2:
        raise ValueError(f"t_end = {t_end} min is shorter than one detector sample ({1.0 / per_min:.4g} min)")
    return np.arange(n, dtype=float) / per_min


def _unpack(retention_table: Table) -> tuple[np.ndarray, np.ndarray]:
    """(t_R, sigma) arrays from a list of (t_R, sigma) rows; an empty table is allowed."""
    arr = np.asarray(retention_table, dtype=float)
    if arr.size == 0:
        return np.empty(0), np.empty(0)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"retention_table must be rows of (t_R, sigma), got shape {arr.shape}")
    t_r, sigma = arr[:, 0], arr[:, 1]
    if not np.all(np.isfinite(t_r)) or not np.all(np.isfinite(sigma)):
        raise ValueError("retention_table contains non-finite values")
    if np.any(sigma <= 0.0):
        raise ValueError("every sigma in retention_table must be positive")
    return t_r, sigma


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #
def trace(retention_table: Table, instrument: InstrumentConfig, areas: Sequence[float] | None = None,
          t_end: float | None = None, noise_sd: float | None = None, drift: float | None = None,
          seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Detector trace ``(t, signal)`` for a retention table, both float arrays of equal length.

    Parameters
    ----------
    retention_table
        ``[(t_R [min], sigma [min]), ...]`` as in :class:`sim.config.Result`. Order
        does not matter. An empty table gives a flat baseline (plus drift/noise).
    instrument
        Supplies the sampling rate and the default noise and drift from its
        ``detector`` block, and ``t0`` for the minimum window.
    areas
        One area per row, default 1.0 each. Height = area / (sigma sqrt(2 pi)).
    t_end
        End of the time axis (min). Default: the last apex plus ``TAIL_SIGMAS``
        of its own sigma, but never earlier than ``t0 + MIN_WINDOW_AFTER_T0_MIN``.
    noise_sd
        Standard deviation of Gaussian white noise added to every sample, in
        signal units. ``None`` takes ``instrument.detector.noise_sd`` (0.0 in
        the shipped config). 0 adds nothing and touches no random generator.
    drift
        Total linear baseline rise from the first sample to the last, in signal
        units (negative for a falling baseline). ``None`` takes
        ``instrument.detector.drift`` (0.0 in the shipped config).
    seed
        Seed for the noise generator. The same seed gives a bit-identical trace;
        ``None`` draws fresh entropy, so a noisy trace is reproducible only with
        an explicit seed. Ignored when ``noise_sd`` is 0.
    """
    t_r, sigma = _unpack(retention_table)
    n = t_r.size
    if areas is None:
        amp_area = np.ones(n)
    else:
        amp_area = np.asarray(areas, dtype=float).reshape(-1)
        if amp_area.shape != (n,):
            raise ValueError(f"areas must have one entry per row ({n}), got {amp_area.shape[0]}")
        if np.any(amp_area < 0.0) or not np.all(np.isfinite(amp_area)):
            raise ValueError("areas must be finite and non-negative")

    det = instrument.detector
    noise_sd = det.noise_sd if noise_sd is None else float(noise_sd)
    drift = det.drift if drift is None else float(drift)
    if noise_sd < 0.0:
        raise ValueError(f"noise_sd must be non-negative, got {noise_sd}")

    if t_end is None:
        t_end = instrument.t0 + MIN_WINDOW_AFTER_T0_MIN
        if n:
            last = int(np.argmax(t_r))
            t_end = max(t_end, float(t_r[last] + TAIL_SIGMAS * sigma[last]))
    t = time_axis(float(t_end), instrument)

    # One vectorised Gaussian per row keeps memory at O(len(t)) whatever the
    # sampling rate; n is the analyte count, at most a few tens.
    signal = np.zeros_like(t)
    for a, mu, s in zip(amp_area, t_r, sigma):
        z = (t - mu) / s
        signal += a / (s * math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * z * z)

    if drift != 0.0:
        signal += drift * (t / t[-1])
    if noise_sd > 0.0:
        rng = np.random.default_rng(seed)
        signal += rng.normal(0.0, noise_sd, size=t.shape)
    return t, signal


def from_result(result: Result, instrument: InstrumentConfig, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """:func:`trace` of ``result.retention_table``; keyword arguments pass through.

    Raises ``ValueError`` for an infeasible result: no run happened, so there is
    no trace to show, and an empty array would be too easy to mistake for a
    flat chromatogram downstream.
    """
    if not result.feasible or not result.retention_table:
        raise ValueError(f"no chromatogram for an infeasible method: {result.reason}")
    return trace(result.retention_table, instrument, **kwargs)


# --------------------------------------------------------------------------- #
# The operator's view
# --------------------------------------------------------------------------- #
def apparent_peaks(t: np.ndarray, signal: np.ndarray, min_height_frac: float = 0.05) -> list[tuple[float, float]]:
    """Visible peaks ``[(t_apex, height), ...]`` in time order, as an operator would count them.

    A peak is a local maximum of the signal whose height is at least
    ``min_height_frac`` of the tallest sample and which also stands at least
    that far above the valleys on either side (scipy's *prominence*, so noise
    ripples on a peak top are not counted twice). Height is the raw signal at
    the apex, baseline included.

    This is intentionally not the retention table's fixed row count: two
    Gaussians of equal width and area are unimodal until they are more than
    about 2 sigma apart, so co-eluting analytes show as one peak and the count
    can only be less than or equal to the number of rows. Apexes at the very
    first or last sample are not peaks. For a flat or empty trace the list is
    empty.

    On a noisy trace the floor must sit above the noise band: prominence stops
    a peak top from being counted twice, but an isolated noise spike whose
    neighbours dip low enough is a peak like any other. Keep
    ``min_height_frac * max(signal)`` a few noise standard deviations above the
    drifted baseline, or smooth the trace first.
    """
    from scipy.signal import find_peaks

    t = np.asarray(t, dtype=float)
    s = np.asarray(signal, dtype=float)
    if t.shape != s.shape:
        raise ValueError(f"t and signal must have the same shape, got {t.shape} and {s.shape}")
    if s.size < 3:
        return []
    top = float(s.max())
    if not top > 0.0:
        return []
    floor = min_height_frac * top
    idx, _ = find_peaks(s, height=floor, prominence=floor)
    return [(float(t[i]), float(s[i])) for i in idx]


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot(t: np.ndarray, signal: np.ndarray, path=None, title: str | None = None,
         peaks: Sequence[tuple[float, float]] | None = None):
    """Draw the trace with matplotlib and return the figure.

    matplotlib is imported here, not at module level, so nothing else in
    ``sim`` depends on it (install with the ``plot`` extra). With ``path`` the
    figure is saved as a PNG and closed; without it the open figure is returned
    for the caller to show or embed. ``peaks`` (from :func:`apparent_peaks`)
    are marked with a small triangle above each apex.
    """
    import matplotlib.pyplot as plt

    t = np.asarray(t, dtype=float)
    s = np.asarray(signal, dtype=float)
    ink, muted, grid, trace_color, surface = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6", "#fcfcfb"

    fig, ax = plt.subplots(figsize=(9.0, 3.2), dpi=150)
    ax.set_facecolor(surface)
    ax.plot(t, s, color=trace_color, linewidth=1.2)
    top = float(s.max()) if s.size else 1.0
    bottom = min(0.0, float(s.min())) if s.size else 0.0
    if peaks:
        tp = np.array([p[0] for p in peaks])
        hp = np.array([p[1] for p in peaks])
        ax.plot(tp, hp + 0.04 * (top - bottom), marker="v", linestyle="none", markersize=5,
                color="#52514e", clip_on=False)
    ax.set_xlim(0.0, float(t[-1]) if t.size else 1.0)
    ax.set_ylim(bottom - 0.02 * (top - bottom), top + 0.12 * (top - bottom))
    ax.set_xlabel("time (min)", color=ink)
    ax.set_ylabel("detector response (area units)", color=ink)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelsize=8)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=ink)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _main(argv=None) -> None:
    import argparse
    import sys

    from .config import Method
    from .simulator import Simulator

    p = argparse.ArgumentParser(
        prog="python -m sim.chromatogram",
        description="Simulate one gradient method and build the detector trace an operator would see.")
    p.add_argument("analyte_set", help="name under configs/analytes or a path to a .yaml")
    p.add_argument("--phi0", type=float, default=0.05)
    p.add_argument("--phi1", type=float, default=0.95)
    p.add_argument("--tG", type=float, default=10.0)
    p.add_argument("--T", type=float, default=40.0)
    p.add_argument("--out", default=None, metavar="PNG", help="save the chromatogram here (needs matplotlib)")
    p.add_argument("--noise", type=float, default=None, metavar="SD",
                   help="white-noise standard deviation, signal units (default: detector.noise_sd in instrument.yaml, 0.0)")
    p.add_argument("--drift", type=float, default=None,
                   help="total baseline rise over the run, signal units (default: detector.drift in instrument.yaml, 0.0)")
    p.add_argument("--seed", type=int, default=0, help="noise seed; fixed at 0 so repeated runs match (default: 0)")
    a = p.parse_args(argv)

    sim = Simulator.from_configs(a.analyte_set)
    method = Method(phi0=a.phi0, phi1=a.phi1, tG=a.tG, T=a.T)
    res = sim.run(method)
    if not res.feasible:
        print(f"infeasible: {res.reason}", file=sys.stderr)
        sys.exit(1)
    t, signal = from_result(res, sim.instrument, noise_sd=a.noise, drift=a.drift, seed=a.seed)
    peaks = apparent_peaks(t, signal)
    print(f"{len(peaks)} apparent peaks from {len(res.retention_table)} analytes (CRF {res.crf:.3f})")
    for t_apex, height in peaks:
        print(f"  t = {t_apex:7.3f} min   height = {height:8.3f}")
    if a.out:
        title = (f"{sim.analyte_set_name}: phi {a.phi0:.2f} -> {a.phi1:.2f}, "
                 f"tG {a.tG:g} min, T {a.T:g} C   (CRF {res.crf:.2f})")
        plot(t, signal, path=a.out, title=title, peaks=peaks)
        print(f"saved {a.out}")


if __name__ == "__main__":
    _main()
