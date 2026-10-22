"""Gradient elution: retention time under a linear gradient with dwell volume.

Two implementations of the same physical model:

* ``t_r_integrated`` — numerical migration integral (RK4). Always correct;
  reference for tests and for cross-checking the closed form.
* ``t_r_closed`` — the linear-solvent-strength (LSS) closed form, fully
  vectorised over methods and analytes; the hot path.

Physical model. The mobile-phase program at the column inlet is delayed by the
dwell time tD. Solvent moves through the column at the dead-time velocity, so an
analyte at fractional position x at time t sits in solvent that entered the
inlet at tau = t - x*t0. Its migration is

    dx/dt = 1 / ( t0 * [1 + k(phi_in(t - x*t0), T)] ).

Changing variable to the inlet time tau gives dx/dtau = 1/(t0 k(phi_in(tau))),
i.e. the analyte elutes when  integral_0^tau_R dtau / k(phi_in(tau)) = t0  and
t_R = tau_R + t0. Evaluating that integral piecewise (isocratic hold during the
dwell, exponential k decay during the gradient, isocratic hold at phi1 after it)
gives the three closed-form regimes below. The regime-2 expression is the
textbook LSS equation with dwell correction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import InstrumentConfig
from .retention import LN10, log_k

REGIME_BEFORE_GRADIENT = 1   # elutes before the gradient front reaches it
REGIME_IN_GRADIENT = 2       # elutes during the gradient (normal case)
REGIME_AFTER_GRADIENT = 3    # still on the column when the gradient ends; finishes at phi1

_B_SMALL = 1e-12


# --------------------------------------------------------------------------- #
# Mobile-phase program
# --------------------------------------------------------------------------- #
def phi_in(t, phi0, phi1, tG, tD):
    """Composition at the column inlet at time t (min). Broadcasts."""
    t = np.asarray(t, dtype=float)
    frac = np.clip((t - tD) / tG, 0.0, 1.0)
    return phi0 + (phi1 - phi0) * frac


# --------------------------------------------------------------------------- #
# Closed form (LSS), vectorised
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClosedForm:
    """Per-(method, analyte) outputs of the closed form, each of shape (M, n)."""

    t_r: np.ndarray       # retention time (min)
    k_e: np.ndarray       # retention factor at the moment of elution
    b: np.ndarray         # gradient steepness S * dphi * t0 / tG
    regime: np.ndarray    # 1, 2 or 3 (int8)
    k0: np.ndarray        # retention factor at phi0


def _col(x) -> np.ndarray:
    """Method-side quantity -> shape (M, 1)."""
    return np.atleast_1d(np.asarray(x, dtype=float))[:, None]


def t_r_closed(phi0, phi1, tG, T_C, log_kw, S, dH, instrument: InstrumentConfig) -> ClosedForm:
    """LSS closed-form retention times.

    ``phi0, phi1, tG, T_C`` are scalars or arrays of shape (M,); ``log_kw, S, dH``
    have shape (n,). Returns arrays of shape (M, n). Accepts phi1 == phi0
    (isocratic limit).
    """
    t0, tD = instrument.t0, instrument.tD
    phi0, phi1, tG, T_C = _col(phi0), _col(phi1), _col(tG), _col(T_C)
    log_kw, S, dH = (np.asarray(a, dtype=float)[None, :] for a in (log_kw, S, dH))
    dphi = phi1 - phi0

    k0 = np.power(10.0, log_k(phi0, T_C, log_kw, S, dH, instrument.T_ref_C))
    k1 = np.power(10.0, log_k(phi1, T_C, log_kw, S, dH, instrument.T_ref_C))
    b = S * dphi * t0 / tG
    small_b = np.abs(b) < _B_SMALL
    b_safe = np.where(small_b, 1.0, b)

    # Regime 1: the analyte leaves before the gradient front catches it.
    t_iso = t0 * (1.0 + k0)
    reg1 = t0 * k0 <= tD

    # Regime 2: standard LSS with dwell correction  (A = k0 - tD/t0 > 0 outside regime 1).
    A = np.maximum(k0 - tD / t0, 0.0)
    arg = LN10 * b_safe * A
    mig = np.where(small_b, t0 * A, t0 * np.log1p(np.where(small_b, 0.0, arg)) / (LN10 * b_safe))
    t_r2 = mig + t0 + tD
    k_e2 = k0 / (1.0 + np.where(small_b, 0.0, arg))

    # Regime 3: gradient ends (inlet time tD + tG) before the analyte has left.
    # Migration by then (inlet-time frame), from the same integral as regime 2:
    #   x_G = [ tD/k0 + t0 * (10**(b tG / t0) - 1) / (ln10 b k0) ] / t0
    gradient_end = tD + tG + t0
    reg3 = (~reg1) & (t_r2 > gradient_end)
    expo = np.where(small_b, 0.0, LN10 * b_safe * tG / t0)
    with np.errstate(over="ignore", invalid="ignore"):
        grad_term = np.where(small_b, tG, t0 * np.expm1(expo) / (LN10 * b_safe))
        x_G = (tD + grad_term) / (t0 * k0)
    x_G = np.clip(np.nan_to_num(x_G, nan=1.0, posinf=1.0), 0.0, 1.0)
    t_r3 = gradient_end + (1.0 - x_G) * t0 * k1

    t_r = np.where(reg1, t_iso, np.where(reg3, t_r3, t_r2))
    k_e = np.where(reg1, k0, np.where(reg3, k1, k_e2))
    regime = np.where(reg1, REGIME_BEFORE_GRADIENT,
                      np.where(reg3, REGIME_AFTER_GRADIENT, REGIME_IN_GRADIENT)).astype(np.int8)
    return ClosedForm(t_r=t_r, k_e=k_e, b=np.broadcast_to(b, t_r.shape), regime=regime,
                      k0=np.broadcast_to(k0, t_r.shape))


# --------------------------------------------------------------------------- #
# Numerical reference: RK4 migration integral
# --------------------------------------------------------------------------- #
def t_r_integrated(phi0, phi1, tG, T_C, log_kw, S, dH, instrument: InstrumentConfig,
                   steps_per_t0: int = 200, t_stop: float | None = None) -> np.ndarray:
    """Retention times (shape (n,)) by fixed-step RK4 on dx/dt = 1/(t0 [1 + k(phi_in(t - x t0), T)]).

    Integration stops per analyte on x >= 1 with linear interpolation of the
    crossing. Analytes still on the column at ``t_stop`` (default 2 * t_max) get nan.
    """
    t0, tD = instrument.t0, instrument.tD
    log_kw, S, dH = (np.asarray(a, dtype=float) for a in (log_kw, S, dH))
    n = log_kw.shape[0]
    T_ref = instrument.T_ref_C
    if t_stop is None:
        t_stop = 2.0 * instrument.t_max_min
    dt = t0 / steps_per_t0

    def rate(t, x):
        phi = phi_in(t - x * t0, phi0, phi1, tG, tD)
        kk = np.power(10.0, log_k(phi, T_C, log_kw, S, dH, T_ref))
        return 1.0 / (t0 * (1.0 + kk))

    x = np.zeros(n)
    t = 0.0
    t_r = np.full(n, np.nan)
    active = np.ones(n, dtype=bool)
    while active.any() and t < t_stop:
        k1 = rate(t, x)
        k2 = rate(t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = rate(t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = rate(t + dt, x + dt * k3)
        x_new = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        crossed = active & (x_new >= 1.0)
        if crossed.any():
            frac = (1.0 - x[crossed]) / (x_new[crossed] - x[crossed])
            t_r[crossed] = t + frac * dt
            active &= ~crossed
        x = np.where(active, x_new, x)
        t += dt
    return t_r
