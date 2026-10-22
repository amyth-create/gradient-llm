import numpy as np
import pytest

from sim.pressure import delta_p, linear_velocity_m_s, max_delta_p, phi_at_max_pressure, viscosity


def test_viscosity_endpoints(inst):
    vm = inst.viscosity
    assert float(viscosity(0.0, 25.0, vm)) == pytest.approx(0.89)      # water
    assert float(viscosity(1.0, 25.0, vm)) == pytest.approx(0.37)      # acetonitrile
    assert vm.phi_max == pytest.approx(0.20, abs=0.01)


def test_pressure_maximum_near_phi_0_2_and_falls_with_temperature(inst):
    phis = np.linspace(0.0, 1.0, 2001)
    dp25 = delta_p(phis, 25.0, inst)
    dp60 = delta_p(phis, 60.0, inst)
    assert 0.15 <= phis[np.argmax(dp25)] <= 0.25
    assert dp60.max() < dp25.max()
    assert np.all(dp60 < dp25)
    # ~2 % per degree C
    assert dp60.max() / dp25.max() == pytest.approx(np.exp(-0.021 * 35.0), rel=1e-9)


def test_pressure_magnitude_is_below_P_max_but_not_by_much(inst):
    """Defaults (1.8 um, 100 mm, 0.5 mL/min): the limit is real but not dominant."""
    worst = float(max_delta_p(0.05, 0.95, 25.0, inst))
    assert 400.0 < worst < 600.0
    assert float(max_delta_p(0.05, 0.95, 60.0, inst)) == pytest.approx(worst * np.exp(-0.021 * 35.0))
    assert linear_velocity_m_s(inst) == pytest.approx(3.7e-3, rel=0.02)


def test_max_over_gradient_is_the_true_maximum(inst):
    phis = np.linspace(0.0, 1.0, 100001)
    for phi0, phi1 in [(0.05, 0.95), (0.30, 0.90), (0.05, 0.15), (0.20, 0.30)]:
        inside = (phis >= phi0) & (phis <= phi1)
        brute = delta_p(phis[inside], 40.0, inst).max()
        assert float(max_delta_p(phi0, phi1, 40.0, inst)) == pytest.approx(brute, rel=1e-6)
    assert float(phi_at_max_pressure(0.3, 0.9, inst)) == pytest.approx(0.3)
    assert float(phi_at_max_pressure(0.05, 0.15, inst)) == pytest.approx(0.15)


def test_pressure_vectorises_over_methods(inst):
    phi0 = np.array([0.05, 0.3, 0.5])
    phi1 = np.array([0.95, 0.9, 0.7])
    T = np.array([25.0, 40.0, 60.0])
    out = max_delta_p(phi0, phi1, T, inst)
    assert out.shape == (3,)
    assert np.all(np.diff(out) < 0)
