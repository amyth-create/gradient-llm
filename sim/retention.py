"""Isocratic retention: the only chemistry in the simulator.

    log10 k_i(phi, T) = log10 k_w,i + dH_i * (1/T_ref - 1/T) - S_i * phi

T in kelvin, dH_i in K*log10 units. dH is the van 't Hoff term
dH = Delta_H / (2.303 R): negative for the exothermic adsorption that governs
reversed-phase retention, so retention falls as T rises (at T > T_ref the
bracket is positive and the product negative). S is temperature-independent.

All functions are pure and broadcast numpy-style: pass ``phi`` / ``T_C`` with a
trailing axis of length 1 (shape ``(M, 1)``) and the analyte arrays with shape
``(n,)`` to get an ``(M, n)`` result.
"""
from __future__ import annotations

import numpy as np

from .config import InstrumentConfig

KELVIN = 273.15
LN10 = np.log(10.0)


def log_k(phi, T_C, log_kw, S, dH, T_ref_C: float = 30.0):
    """log10 of the retention factor at composition ``phi`` and temperature ``T_C`` (deg C)."""
    T = np.asarray(T_C, dtype=float) + KELVIN
    T_ref = T_ref_C + KELVIN
    return np.asarray(log_kw, dtype=float) + np.asarray(dH, dtype=float) * (1.0 / T_ref - 1.0 / T) \
        - np.asarray(S, dtype=float) * np.asarray(phi, dtype=float)


def k(phi, T_C, log_kw, S, dH, T_ref_C: float = 30.0):
    """Retention factor k = 10**log_k."""
    return np.power(10.0, log_k(phi, T_C, log_kw, S, dH, T_ref_C))


def t0(instrument: InstrumentConfig) -> float:
    """Dead time t0 = V0 / F (min)."""
    return instrument.t0


def crossover_phi(log_kw_a, S_a, log_kw_b, S_b):
    """Composition at which two analytes' log k lines cross (elution order flips)."""
    return (log_kw_a - log_kw_b) / (S_a - S_b)
