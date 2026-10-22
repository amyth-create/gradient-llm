import numpy as np
import pytest

from sim.config import InstrumentConfig
from sim.gradient import t_r_closed, t_r_integrated
from sim.retention import crossover_phi, k, log_k, t0


def test_dead_volume_and_time_match_spec(inst: InstrumentConfig):
    # L = 100 mm, dc = 2.1 mm, eps = 0.65 -> V0 = 0.225 mL, t0 = 0.45 min at 0.5 mL/min
    assert inst.V0_mL == pytest.approx(0.2251, abs=5e-4)
    assert t0(inst) == pytest.approx(0.450, abs=1e-3)
    assert inst.tD == pytest.approx(0.2)


def test_log_k_sanity_numbers():
    # log kw = 3, S = 6 at T_ref: k = 10 at phi = 1/3 and k = 1 at phi = 0.5
    assert k(1.0 / 3.0, 30.0, 3.0, 6.0, -2000.0) == pytest.approx(10.0)
    assert k(0.5, 30.0, 3.0, 6.0, -2000.0) == pytest.approx(1.0)
    assert log_k(0.0, 30.0, 3.0, 6.0, -2000.0) == pytest.approx(3.0)


def test_negative_dH_means_retention_falls_with_temperature():
    lk = [float(log_k(0.3, T, 3.0, 6.0, -2000.0)) for T in (25.0, 30.0, 45.0, 60.0)]
    assert lk == sorted(lk, reverse=True)
    assert lk[1] == pytest.approx(3.0 - 1.8)            # T = T_ref -> no temperature term


def test_log_k_broadcasts_methods_by_analytes():
    phi = np.array([[0.1], [0.5]])                       # (M=2, 1)
    out = log_k(phi, 40.0, np.array([2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0]), np.array([-1000.0] * 3))
    assert out.shape == (2, 3)


@pytest.mark.parametrize("tG", [2.0, 10.0, 40.0])
def test_isocratic_limit_closed_form_and_integrator(inst, three_analytes, tG):
    """phi0 = phi1: t_R = t0 (1 + k(phi0, T)) exactly, both methods."""
    log_kw, S, dH = three_analytes
    phi, T = 0.45, 42.0
    expected = inst.t0 * (1.0 + k(phi, T, log_kw, S, dH, inst.T_ref_C))
    closed = t_r_closed(phi, phi, tG, T, log_kw, S, dH, inst).t_r[0]
    integrated = t_r_integrated(phi, phi, tG, T, log_kw, S, dH, inst)
    np.testing.assert_allclose(closed, expected, rtol=1e-12)
    np.testing.assert_allclose(integrated, expected, rtol=1e-3)


def test_crossover_phi_formula():
    assert crossover_phi(3.0, 6.0, 2.5, 4.0) == pytest.approx(0.25)
    # at the crossover both analytes have the same log k
    assert log_k(0.25, 30.0, 3.0, 6.0, -1000.0) == pytest.approx(log_k(0.25, 30.0, 2.5, 4.0, -1000.0))
