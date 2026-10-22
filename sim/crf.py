"""Chromatographic response function: pure functions of the retention table.

    R_j   = (t_R,j+1 - t_R,j) / (2 (sigma_j + sigma_j+1))           (4-sigma base widths)
    CRF   = 1/(n-1) sum_j min(R_j / R_target, 1) - w_t max(0, (t_R,n - t_target) / t_target)

Tables must be sorted by t_R along the last axis. All functions broadcast over
leading axes, so a batch of (M, n) tables gives (M,) scores.

Truncated tables. The run ends at t_end; peaks eluting later are not observed.
``crf_truncated`` scores what was seen: resolution credit over adjacent observed
pairs, zero credit for one pair per missing analyte (the denominator stays
n - 1, so a lost peak scores like a fully co-eluting one), and the time term at
min(t_R of the last observed peak, t_end), i.e. at t_end whenever anything is
missing. For a full table it equals ``crf``.
"""
from __future__ import annotations

import numpy as np

from .config import CRFConfig


def resolution(t_r, sigma) -> np.ndarray:
    """Resolution of each adjacent pair; shape (..., n-1)."""
    t_r = np.asarray(t_r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return (t_r[..., 1:] - t_r[..., :-1]) / (2.0 * (sigma[..., 1:] + sigma[..., :-1]))


def resolution_term(t_r, sigma, cfg: CRFConfig) -> np.ndarray:
    """Normalised, clipped mean resolution in [0, 1]; 1.0 for a single analyte."""
    t_r = np.asarray(t_r, dtype=float)
    if t_r.shape[-1] < 2:
        return np.ones(t_r.shape[:-1])
    R = resolution(t_r, sigma)
    return np.minimum(R / cfg.R_target, 1.0).mean(axis=-1)


def time_term(t_r, cfg: CRFConfig) -> np.ndarray:
    """Penalty for the last peak eluting after t_target (>= 0)."""
    t_last = np.asarray(t_r, dtype=float)[..., -1]
    return cfg.w_t * np.maximum(0.0, (t_last - cfg.t_target) / cfg.t_target)


def crf(t_r, sigma, cfg: CRFConfig) -> np.ndarray:
    """CRF score, in (-inf, 1]."""
    return resolution_term(t_r, sigma, cfg) - time_term(t_r, cfg)


def sort_table(t_r, sigma) -> tuple[np.ndarray, np.ndarray]:
    """Sort (t_r, sigma) by t_r along the last axis."""
    t_r = np.asarray(t_r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    order = np.argsort(t_r, axis=-1, kind="stable")
    return np.take_along_axis(t_r, order, axis=-1), np.take_along_axis(sigma, order, axis=-1)


# --------------------------------------------------------------------------- #
# Truncated tables
# --------------------------------------------------------------------------- #
def _observed_first(t_r, sigma, observed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast the inputs and order each table so its observed rows come first, by t_R.

    Truncation only ever removes the latest peaks, so for a sorted table this is
    the identity; it also makes the functions below correct for any mask.
    """
    t_r = np.asarray(t_r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    observed = np.broadcast_to(np.asarray(observed, dtype=bool), t_r.shape)
    if t_r.shape[-1] == 0 or np.all(observed[..., :-1] >= observed[..., 1:]):
        return t_r, sigma, observed          # already observed-first (the simulator's case): no re-sort
    key = np.where(observed, t_r, np.inf)
    order = np.argsort(key, axis=-1, kind="stable")
    take = lambda a: np.take_along_axis(a, order, axis=-1)  # noqa: E731
    return take(t_r), take(sigma), take(observed)


def resolution_term_truncated(t_r, sigma, observed, n_analytes: int, cfg: CRFConfig) -> np.ndarray:
    """Sum over adjacent observed pairs of min(R / R_target, 1), divided by n_analytes - 1.

    Each missing analyte is one pair with zero credit, so the term is in
    [0, 1] and equals ``resolution_term`` when everything is observed.
    """
    t_r, sigma, observed = _observed_first(t_r, sigma, observed)
    if n_analytes < 2:
        return observed.any(axis=-1).astype(float)
    if t_r.shape[-1] < 2:
        return np.zeros(t_r.shape[:-1])
    pair_ok = observed[..., 1:] & observed[..., :-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        credit = np.minimum(resolution(t_r, sigma) / cfg.R_target, 1.0)
    return np.where(pair_ok, credit, 0.0).sum(axis=-1) / (n_analytes - 1)


def time_term_truncated(t_r, observed, n_analytes: int, t_end: float, cfg: CRFConfig) -> np.ndarray:
    """Time penalty at min(t_R of the last observed peak, t_end); at t_end if any analyte is missing."""
    t_r = np.asarray(t_r, dtype=float)
    observed = np.broadcast_to(np.asarray(observed, dtype=bool), t_r.shape)
    missing = n_analytes - observed.sum(axis=-1) > 0
    if t_r.shape[-1] == 0:
        t_last = np.full(t_r.shape[:-1], float(t_end))
    else:
        t_last_seen = np.max(np.where(observed, t_r, -np.inf), axis=-1)
        t_last = np.where(missing, t_end, np.minimum(t_last_seen, t_end))
    return cfg.w_t * np.maximum(0.0, (t_last - cfg.t_target) / cfg.t_target)


def crf_truncated(t_r, sigma, observed, n_analytes: int, t_end: float, cfg: CRFConfig) -> np.ndarray:
    """CRF of a truncated table, in [-w_t (t_end - t_target) / t_target, 1].

    ``observed`` marks the rows of (t_r, sigma) that were seen (t_R <= t_end);
    ``n_analytes`` is the full count, so the arrays may be full width with a
    mask or already compacted to the observed rows.
    """
    return (resolution_term_truncated(t_r, sigma, observed, n_analytes, cfg)
            - time_term_truncated(t_r, observed, n_analytes, t_end, cfg))
