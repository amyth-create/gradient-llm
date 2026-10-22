import numpy as np
import pytest

from sim.gradient import REGIME_IN_GRADIENT, t_r_closed
from sim.width import compression, sigma, sigma_from_closed_form


def test_worked_case_from_spec(inst):
    """k_e = 2 at t0 = 0.45 min, N = 15 000: column term 0.011 min, total 0.023 min with sigma_ec = 0.02."""
    no_ec = inst.model_copy(update={"sigma_ec_min": 0.0})
    assert float(sigma(2.0, 1.0, no_ec)) == pytest.approx(0.45 * 3.0 / np.sqrt(15000.0), rel=2e-3)
    assert float(sigma(2.0, 1.0, inst)) == pytest.approx(0.023, abs=5e-4)


def test_compression_factor():
    b = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(compression(b, True), 1.0 - 0.25 * b / (1.0 + b))
    np.testing.assert_allclose(compression(b, False), 1.0)
    assert np.all(compression(np.array([0.1, 1.0, 1.9]), True) < 1.0)


def test_sigma_increases_with_k_e_and_decreases_with_N(inst):
    k_e = np.array([0.5, 1.0, 2.0, 5.0])
    s = sigma(k_e, 1.0, inst)
    assert np.all(np.diff(s) > 0)
    high_N = inst.model_copy(update={"N": 60000.0})
    assert np.all(sigma(k_e, 1.0, high_N) < s)
    assert np.all(s >= inst.sigma_ec_min)      # extra-column term is a floor


def test_sigma_from_closed_form_uses_compression_only_in_gradient(inst):
    log_kw, S, dH = np.array([0.5, 3.0]), np.array([5.0, 6.0]), np.array([-1500.0, -1500.0])
    cf = t_r_closed(0.3, 0.9, 5.0, 40.0, log_kw, S, dH, inst)
    s = sigma_from_closed_form(cf.k_e, cf.b, cf.regime, inst)
    G = np.where(cf.regime == REGIME_IN_GRADIENT, 1.0 - 0.25 * cf.b / (1.0 + cf.b), 1.0)
    np.testing.assert_allclose(s, sigma(cf.k_e, G, inst))
    assert cf.regime[0, 0] == 1 and cf.regime[0, 1] == REGIME_IN_GRADIENT
