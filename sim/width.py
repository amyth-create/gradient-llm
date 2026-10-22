"""Peak width: plate count, retention factor at elution, band compression, extra-column term.

    sigma_i = sqrt( ( t0 (1 + k_e,i) G_i / sqrt(N) )^2 + sigma_ec^2 )

G is the gradient band-compression factor, G = 1 - 0.25 b / (1 + b) for an
analyte eluting inside the gradient and G = 1 for one eluting outside it.
"""
from __future__ import annotations

import numpy as np

from .config import InstrumentConfig
from .gradient import REGIME_IN_GRADIENT


def compression(b, in_gradient) -> np.ndarray:
    """Band-compression factor G."""
    b = np.asarray(b, dtype=float)
    return np.where(in_gradient, 1.0 - 0.25 * b / (1.0 + b), 1.0)


def sigma(k_e, G, instrument: InstrumentConfig) -> np.ndarray:
    """Peak standard deviation in time units (min). Broadcasts over k_e and G."""
    column = instrument.t0 * (1.0 + np.asarray(k_e, dtype=float)) * np.asarray(G, dtype=float) / np.sqrt(instrument.N)
    return np.sqrt(column**2 + instrument.sigma_ec_min**2)


def sigma_from_closed_form(k_e, b, regime, instrument: InstrumentConfig) -> np.ndarray:
    """Convenience: sigma for the outputs of ``gradient.t_r_closed``."""
    G = compression(b, np.asarray(regime) == REGIME_IN_GRADIENT)
    return sigma(k_e, G, instrument)
