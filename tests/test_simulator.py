import math
import time

import numpy as np
import pytest

from sim import InstrumentConfig, Method, Result, Simulator
from sim.config import AnalyteSet
from sim.crf import resolution
from sim.gradient import t_r_closed
from sim.optimum import load_optimum
from sim.simulator import INFEASIBLE_BOUNDS, OK, REASONS

SETS = ["easy", "medium", "hard"]


@pytest.fixture(scope="module")
def sim(inst, demo_set, crf_cfg) -> Simulator:
    return Simulator(inst, demo_set, crf_cfg)


def test_public_interface_contract():
    from sim import CRFConfig, InstrumentConfig, Method, Result, Simulator  # noqa: F401
    assert set(InstrumentConfig.load().bounds()) == {"phi0", "dphi", "tG", "T"}


def test_result_shape_and_sorting(sim, demo_set):
    res = sim.run(Method(phi0=0.1, phi1=0.9, tG=10.0, T=40.0))
    assert isinstance(res, Result)
    assert res.feasible and res.reason is None
    assert len(res.retention_table) == len(demo_set)
    t = [row[0] for row in res.retention_table]
    assert t == sorted(t)
    assert all(len(row) == 2 and row[1] > 0 for row in res.retention_table)
    assert res.run_time == pytest.approx(t[-1] + sim.instrument.t_eq_min)
    assert res.crf <= 1.0
    assert res.n_missing == 0


def test_t_end_is_t_max_minus_t_eq(inst):
    assert inst.t_end == pytest.approx(inst.t_max_min - inst.t_eq_min)
    assert inst.t_end == 42.0
    assert set(REASONS) == {OK, INFEASIBLE_BOUNDS}            # run time is no longer a feasibility reason


def test_dphi_interface_roundtrip():
    m = Method(phi0=0.1, phi1=0.8, tG=10.0, T=40.0)
    assert m.dphi == pytest.approx(0.7)
    assert Method.from_vector(m.to_vector()) == m


@pytest.mark.parametrize("method", [
    Method(phi0=0.3, phi1=0.3, tG=10.0, T=40.0),       # isocratic: dphi below minimum
    Method(phi0=0.3, phi1=0.35, tG=10.0, T=40.0),      # dphi below minimum
    Method(phi0=0.02, phi1=0.9, tG=10.0, T=40.0),      # phi0 below box
    Method(phi0=0.1, phi1=0.97, tG=10.0, T=40.0),      # phi1 above box
    Method(phi0=0.1, phi1=0.9, tG=1.0, T=40.0),        # tG below box
    Method(phi0=0.1, phi1=0.9, tG=10.0, T=70.0),       # T above box
])
def test_feasibility_bounds(sim, method):
    res = sim.run(method)
    assert not res.feasible
    assert math.isnan(res.crf)
    assert res.retention_table == []
    assert res.reason.startswith("bounds")
    assert res.n_missing == len(sim._analytes)                # nothing observed: no run happened


def test_truncation_replaces_run_time_infeasibility(inst, crf_cfg):
    """tG = 40, phi1 = 0.95, late analyte: the second peak elutes after t_end = 42 min and is not observed.

    The method stays feasible; the table is one row short, n_missing = 1, and the CRF is
    finite but poor: the only pair earns nothing and the time term is charged at t_end.
    """
    late = AnalyteSet.from_rows("late", [(2.0, 5.0, -1500.0), (5.6, 4.0, -1500.0)])
    s = Simulator(inst, late, crf_cfg)
    m = Method(phi0=0.05, phi1=0.95, tG=40.0, T=25.0)
    res = s.run(m)
    assert res.feasible and res.reason is None
    assert res.n_missing == 1
    assert len(res.retention_table) == len(late) - 1
    assert res.retention_table[-1][0] <= inst.t_end
    assert math.isfinite(res.crf) and res.crf <= -1.5
    assert res.crf == pytest.approx(-crf_cfg.w_t * (inst.t_end - crf_cfg.t_target) / crf_cfg.t_target)
    assert res.run_time == pytest.approx(inst.t_end + inst.t_eq_min) == pytest.approx(inst.t_max_min)
    br = s.run_batch(m.to_vector())
    assert br.reason[0] == OK and br.feasible[0]
    assert br.observed.tolist() == [[True, False]] and br.n_missing.tolist() == [1]
    assert br.t_r.shape == (1, 2) and br.t_r[0, 1] > inst.t_end      # the lost peak is still in the raw arrays


def test_bounds_are_the_only_infeasibility(inst, crf_cfg):
    """An out-of-bounds method is infeasible whatever its run time; an in-bounds one is always scored."""
    late = AnalyteSet.from_rows("late", [(5.6, 4.0, -1500.0)])
    s = Simulator(inst, late, crf_cfg)
    X = np.array([
        [0.02, 0.9, 40.0, 25.0],   # out of bounds, peak after t_end -> bounds
        [0.05, 0.9, 40.0, 60.0],   # in bounds, peak after t_end     -> feasible, truncated
    ])
    br = s.run_batch(X)
    assert br.reason.tolist() == [INFEASIBLE_BOUNDS, OK]
    assert br.feasible.tolist() == [False, True]
    assert math.isnan(br.crf[0]) and math.isfinite(br.crf[1])
    assert br.n_missing.tolist() == [1, 1]                    # the observed mask is physics, not feasibility
    res = s.run(Method.from_vector(X[1]))
    assert res.feasible and res.retention_table == [] and res.n_missing == 1
    assert res.crf == pytest.approx(-crf_cfg.w_t * (inst.t_end - crf_cfg.t_target) / crf_cfg.t_target)


def test_determinism(sim):
    m = Method(phi0=0.123, phi1=0.876, tG=7.5, T=43.2)
    a, b = sim.run(m), sim.run(m)
    assert a == b
    assert a.model_dump() == b.model_dump()
    np.testing.assert_array_equal(sim.run_batch(m.to_vector()).t_r, sim.run_batch(m.to_vector()).t_r)


def test_run_matches_run_batch(sim):
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.uniform(0.05, 0.6, 50), rng.uniform(0.1, 0.9, 50),
                         rng.uniform(2, 40, 50), rng.uniform(25, 60, 50)])
    br = sim.run_batch(X)
    for i in range(X.shape[0]):
        res = sim.run(Method.from_vector(X[i]))
        assert res.feasible == bool(br.feasible[i])
        if res.feasible:
            assert res.crf == pytest.approx(float(br.crf[i]))
            assert res.n_missing == int(br.n_missing[i])
            np.testing.assert_allclose([r[0] for r in res.retention_table], br.t_r[i][br.observed[i]])


@pytest.mark.parametrize("name", SETS)
def test_crf_bounds_on_every_set(name, crf_cfg):
    s = Simulator.from_configs(name)
    lo, hi = s.instrument.box()
    rng = np.random.default_rng(11)
    X = lo + rng.uniform(size=(2000, 4)) * (hi - lo)
    br = s.run_batch(X)
    assert np.all(np.isnan(br.crf) | (br.crf <= 1.0 + 1e-12))
    assert np.isnan(br.crf[~br.feasible]).all()
    feas = br.feasible                                        # the (phi0, dphi) box overshoots phi1 <= 0.95
    assert feas.any() and (~feas).any()
    assert np.isfinite(br.crf[feas]).all()                    # every in-bounds method is scored
    assert (br.reason[~feas] == INFEASIBLE_BOUNDS).all()
    lower = -crf_cfg.w_t * (s.instrument.t_end - crf_cfg.t_target) / crf_cfg.t_target
    assert np.all(br.crf[feas] >= lower - 1e-12)
    assert br.t_r.shape == br.observed.shape == (2000, s.n_analytes)
    assert np.all(br.observed[:, :-1] >= br.observed[:, 1:])  # observed peaks are a prefix of the sorted table
    assert np.array_equal(br.n_missing, s.n_analytes - br.observed.sum(axis=1))
    assert (br.n_missing > 0).any() and (br.n_missing == 0).any()
    np.testing.assert_allclose(br.run_time, np.fmin(br.t_r[:, -1], s.instrument.t_end) + s.instrument.t_eq_min)


def test_stored_easy_optimum_scores_one_iff_fully_resolved_within_target(crf_cfg):
    s = Simulator.from_configs("easy")
    opt = load_optimum("easy")
    res = s.run(Method(**opt["method"]))
    assert res.feasible
    t = np.array([r[0] for r in res.retention_table])
    sig = np.array([r[1] for r in res.retention_table])
    fully = bool(np.all(resolution(t, sig) >= crf_cfg.R_target) and t[-1] <= crf_cfg.t_target)
    assert (abs(res.crf - 1.0) < 1e-12) == fully
    assert res.crf == pytest.approx(opt["crf"])


def test_speed_closed_form_10k_calls_on_hard(inst):
    """10 000 closed-form calls on hard, vectorised: < 50 ms total."""
    aset = AnalyteSet.load("hard")
    log_kw, S, dH = aset.arrays()
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(0.05, 0.6, 10_000), rng.uniform(0.1, 0.9, 10_000),
                         rng.uniform(2, 40, 10_000), rng.uniform(25, 60, 10_000)])
    args = (X[:, 0], X[:, 0] + X[:, 1], X[:, 2], X[:, 3], log_kw, S, dH, inst)
    t_r_closed(*args)                                       # warm up
    best = min(_timed(lambda: t_r_closed(*args)) for _ in range(3))
    assert best < 0.050, f"{best * 1e3:.1f} ms"


def test_speed_single_run_under_5_ms():
    s = Simulator.from_configs("hard")
    m = Method(phi0=0.1, phi1=0.9, tG=10.0, T=40.0)
    s.run(m)
    best = min(_timed(lambda: s.run(m)) for _ in range(20))
    assert best < 0.005, f"{best * 1e3:.2f} ms"


def _timed(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t
