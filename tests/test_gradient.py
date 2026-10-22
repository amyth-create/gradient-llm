import numpy as np
import pytest

from sim.config import InstrumentConfig
from sim.gradient import (REGIME_AFTER_GRADIENT, REGIME_BEFORE_GRADIENT, REGIME_IN_GRADIENT,
                          phi_in, t_r_closed, t_r_integrated)
from sim.retention import crossover_phi, k
from sim.width import sigma_from_closed_form


def test_phi_in_program(inst):
    tD = inst.tD
    t = np.array([0.0, tD - 1e-9, tD, tD + 5.0, tD + 10.0, 30.0])
    out = phi_in(t, 0.1, 0.9, 10.0, tD)
    np.testing.assert_allclose(out, [0.1, 0.1, 0.1, 0.5, 0.9, 0.9])


def test_closed_form_vs_integrator_all_regimes(inst, three_analytes):
    """3 analytes x 5 random methods (+ a few hand-picked ones): rel. error in t_R < 1 %, all regimes seen."""
    log_kw, S, dH = three_analytes
    rng = np.random.default_rng(42)
    methods = [(0.05, 0.95, 10.0, 40.0), (0.30, 0.50, 3.0, 30.0), (0.10, 0.50, 2.0, 30.0)]
    for _ in range(5):
        phi0 = rng.uniform(0.05, 0.4)
        methods.append((phi0, rng.uniform(max(phi0 + 0.3, 0.6), 0.95), rng.uniform(2.0, 20.0), rng.uniform(25.0, 60.0)))
    regimes = set()
    for phi0, phi1, tG, T in methods:
        cf = t_r_closed(phi0, phi1, tG, T, log_kw, S, dH, inst)
        ref = t_r_integrated(phi0, phi1, tG, T, log_kw, S, dH, inst)
        assert np.all(np.isfinite(ref)), (phi0, phi1, tG, T)
        rel = np.abs(cf.t_r[0] - ref) / ref
        assert rel.max() < 0.01, (phi0, phi1, tG, T, rel)
        regimes.update(int(r) for r in cf.regime[0])
    assert regimes == {REGIME_BEFORE_GRADIENT, REGIME_IN_GRADIENT, REGIME_AFTER_GRADIENT}


def test_regime_1_is_isocratic_at_phi0(inst):
    # k0 so small that t0*k0 <= tD: the analyte leaves before the gradient front reaches it.
    cf = t_r_closed(0.3, 0.9, 5.0, 40.0, np.array([0.5]), np.array([5.0]), np.array([-1000.0]), inst)
    k0 = float(k(0.3, 40.0, 0.5, 5.0, -1000.0))
    assert inst.t0 * k0 <= inst.tD
    assert cf.regime[0, 0] == REGIME_BEFORE_GRADIENT
    assert cf.t_r[0, 0] == pytest.approx(inst.t0 * (1.0 + k0))
    assert cf.k_e[0, 0] == pytest.approx(k0)


def test_dwell_correction(inst):
    """tD = 0 vs 0.2 min: a strongly retained analyte (k0 > 50) shifts by tD, a weak one by less."""
    no_dwell = inst.model_copy(update={"VD_mL": 0.0})
    assert no_dwell.tD == 0.0 and inst.tD == pytest.approx(0.2)
    log_kw, S, dH = np.array([3.5, 1.0]), np.array([5.0, 5.0]), np.array([-1500.0, -1500.0])
    phi0, phi1, tG, T = 0.1, 0.9, 9.0, 30.0
    k0 = k(phi0, T, log_kw, S, dH)
    assert k0[0] > 50 and k0[1] < 5
    shift = t_r_closed(phi0, phi1, tG, T, log_kw, S, dH, inst).t_r[0] - \
        t_r_closed(phi0, phi1, tG, T, log_kw, S, dH, no_dwell).t_r[0]
    assert shift[0] == pytest.approx(inst.tD, rel=0.02)
    assert 0.0 < shift[1] < 0.8 * inst.tD
    # the reference integrator shows the same shifts
    shift_ref = t_r_integrated(phi0, phi1, tG, T, log_kw, S, dH, inst) - \
        t_r_integrated(phi0, phi1, tG, T, log_kw, S, dH, no_dwell)
    np.testing.assert_allclose(shift_ref, shift, atol=2e-3)


def test_temperature_direction(inst, demo_set):
    """T = 25 vs 60 C: every t_R decreases and every sigma decreases (moderate gradient)."""
    log_kw, S, dH = demo_set.arrays()
    out = {}
    for T in (25.0, 60.0):
        cf = t_r_closed(0.05, 0.95, 15.0, T, log_kw, S, dH, inst)
        out[T] = (cf.t_r[0], sigma_from_closed_form(cf.k_e, cf.b, cf.regime, inst)[0])
    assert np.all(out[60.0][0] < out[25.0][0])
    assert np.all(out[60.0][1] < out[25.0][1])


def test_steepness_direction(inst):
    """tG = 5 vs 30 min, same phi range: late peaks move later and widen; order unchanged for equal S."""
    log_kw = np.array([1.6, 2.2, 2.8, 3.4, 4.0])
    S = np.full(5, 6.0)
    dH = np.full(5, -1500.0)
    res = {}
    for tG in (5.0, 30.0):
        cf = t_r_closed(0.1, 0.9, tG, 35.0, log_kw, S, dH, inst)
        res[tG] = (cf.t_r[0], sigma_from_closed_form(cf.k_e, cf.b, cf.regime, inst)[0], cf.regime[0])
    in_gradient = (res[5.0][2] != 1) & (res[30.0][2] != 1)
    assert in_gradient.sum() >= 3
    assert np.all(res[30.0][0][in_gradient] > res[5.0][0][in_gradient])
    assert np.all(res[30.0][1][in_gradient] > res[5.0][1][in_gradient])
    assert np.array_equal(np.argsort(res[5.0][0]), np.argsort(res[30.0][0]))
    assert np.array_equal(np.argsort(res[5.0][0]), np.arange(5))      # equal S: order follows log_kw


def test_elution_order_crossover(inst):
    """Two analytes with different S swap order at phi_c = (log_kw,a - log_kw,b) / (S_a - S_b)."""
    log_kw, S, dH = np.array([3.0, 2.5]), np.array([6.0, 4.0]), np.array([-1500.0, -1500.0])
    phi_c = crossover_phi(log_kw[0], S[0], log_kw[1], S[1])
    assert phi_c == pytest.approx(0.25)
    T = inst.T_ref_C

    def order(phi):
        return np.argsort(t_r_closed(phi, phi, 10.0, T, log_kw, S, dH, inst).t_r[0])

    assert order(phi_c - 0.05).tolist() == [1, 0]        # analyte b elutes first below the crossover
    assert order(phi_c + 0.05).tolist() == [0, 1]        # ... and last above it
    at_cross = t_r_closed(phi_c, phi_c, 10.0, T, log_kw, S, dH, inst).t_r[0]
    assert at_cross[0] == pytest.approx(at_cross[1], rel=1e-9)
    # the same flip under a gradient that ends below / above phi_c
    lo = t_r_closed(0.05, phi_c - 0.05, 2.0, T, log_kw, S, dH, inst).t_r[0]
    hi = t_r_closed(0.05, 0.95, 40.0, T, log_kw, S, dH, inst).t_r[0]
    assert lo[1] < lo[0] and hi[0] < hi[1]


def test_regime_3_matches_integrator_exactly(inst):
    """An analyte that is still on the column when the gradient ends finishes isocratically at phi1."""
    log_kw, S, dH = np.array([4.3]), np.array([5.0]), np.array([-2500.0])
    cf = t_r_closed(0.1, 0.5, 2.0, 30.0, log_kw, S, dH, inst)
    assert cf.regime[0, 0] == REGIME_AFTER_GRADIENT
    ref = t_r_integrated(0.1, 0.5, 2.0, 30.0, log_kw, S, dH, inst)
    assert cf.t_r[0, 0] == pytest.approx(ref[0], rel=2e-3)
    assert cf.k_e[0, 0] == pytest.approx(float(k(0.5, 30.0, 4.3, 5.0, -2500.0)))
