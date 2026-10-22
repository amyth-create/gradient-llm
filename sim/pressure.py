"""Column pressure drop: Darcy's law with a Kozeny-Carman permeability.

STANDALONE HELPER. Pressure was removed from the study: ``Simulator`` neither reports it
nor checks a limit, and no table or optimizer sees it. The model is kept here, unit-tested,
so it can be wired back in if a pressure constraint is ever wanted.

    dP(phi, T) = phi_0 * eta(phi, T) * u * L / dp^2,    u = F / (eps * pi * (dc/2)^2)

With flow fixed, pressure varies only through the mobile-phase viscosity, which
depends on composition and temperature. The reported pressure is the maximum
over the gradient, i.e. eta evaluated at the composition in [phi0, phi1] closest
to the viscosity maximum (phi ~ 0.2 for water/acetonitrile).
"""
from __future__ import annotations

import numpy as np

from .config import InstrumentConfig, ViscosityModel

PA_PER_BAR = 1.0e5


def viscosity(phi, T_C, vm: ViscosityModel) -> np.ndarray:
    """Mobile-phase viscosity in mPa.s."""
    phi = np.asarray(phi, dtype=float)
    T_C = np.asarray(T_C, dtype=float)
    poly = vm.eta_water + vm.a_mix * phi * (1.0 - phi) - vm.c_lin * phi
    return poly * np.exp(-vm.T_coef * (T_C - vm.T_ref_C))


def linear_velocity_m_s(instrument: InstrumentConfig) -> float:
    """Interstitial linear velocity u = F / (eps * A) in m/s."""
    F = instrument.F_mL_min * 1.0e-6 / 60.0                      # m^3/s
    A = np.pi * (instrument.dc_mm * 1.0e-3 / 2.0) ** 2            # m^2
    return float(F / (instrument.eps * A))


def delta_p(phi, T_C, instrument: InstrumentConfig) -> np.ndarray:
    """Pressure drop in bar at composition ``phi`` and temperature ``T_C``."""
    eta = viscosity(phi, T_C, instrument.viscosity) * 1.0e-3     # Pa.s
    u = linear_velocity_m_s(instrument)
    L = instrument.L_mm * 1.0e-3
    dp = instrument.dp_um * 1.0e-6
    return instrument.flow_resistance * eta * u * L / dp**2 / PA_PER_BAR


def phi_at_max_pressure(phi0, phi1, instrument: InstrumentConfig) -> np.ndarray:
    """Composition in [phi0, phi1] where viscosity, hence pressure, is largest."""
    lo = np.minimum(phi0, phi1)
    hi = np.maximum(phi0, phi1)
    return np.clip(instrument.viscosity.phi_max, lo, hi)


def max_delta_p(phi0, phi1, T_C, instrument: InstrumentConfig) -> np.ndarray:
    """Worst-point pressure over a gradient from phi0 to phi1 at temperature T_C (bar)."""
    phi0 = np.asarray(phi0, dtype=float)
    phi1 = np.asarray(phi1, dtype=float)
    return delta_p(phi_at_max_pressure(phi0, phi1, instrument), T_C, instrument)
