import numpy as np
import pytest

from sim.config import AnalyteSet, Method
from sim.generate import CRF_FLOOR, RULES, easy, hard, medium
from sim.optimum import find_optimum, grid_points, load_optimum
from sim.simulator import Simulator

SETS = ["easy", "medium", "hard"]


def test_grid_covers_the_box():
    s = Simulator.from_configs("easy")
    X = grid_points(s, (3, 3, 3, 2))
    lo, hi = s.instrument.box()
    assert X.shape == (54, 4)
    np.testing.assert_allclose(X.min(axis=0), lo)
    np.testing.assert_allclose(X.max(axis=0), hi)


@pytest.mark.parametrize("name", SETS)
def test_stored_optimum_is_consistent_and_above_floor(name):
    s = Simulator.from_configs(name)
    opt = load_optimum(name)
    res = s.run(Method(**opt["method"]))
    assert res.feasible
    assert res.crf == pytest.approx(opt["crf"], abs=1e-12)
    assert [tuple(r) for r in opt["retention_table"]] == pytest.approx(res.retention_table)
    assert opt["crf"] >= CRF_FLOOR[name]
    assert len(opt["retention_table"]) == s.n_analytes


@pytest.mark.parametrize("name", SETS)
def test_optimum_regression(name):
    """Re-running the finder lands within 1 % of the stored CRF."""
    s = Simulator.from_configs(name)
    stored = load_optimum(name)
    fresh = find_optimum(s)
    assert fresh["crf"] == pytest.approx(stored["crf"], rel=0.01)
    assert fresh["crf"] <= 1.0 + 1e-12


def test_generator_easy_rule_pairwise_distinct():
    aset = easy(123)
    log_kw, S, _ = aset.arrays()
    assert len(aset) == 6
    for i in range(6):
        for j in range(i + 1, 6):
            assert abs(log_kw[i] - log_kw[j]) > RULES.min_dlog_kw or abs(S[i] - S[j]) > RULES.min_dS
    assert np.all((log_kw >= 1.8) & (log_kw <= 4.0)) and np.all((S >= 4.0) & (S <= 8.0))


def test_generator_near_pairs_have_different_S():
    for aset, n_pairs in ((medium(5), 2), (hard(5), 3)):
        log_kw, S, _ = aset.arrays()
        for p in range(n_pairs):
            i, j = 2 * p, 2 * p + 1
            assert abs(log_kw[i] - log_kw[j]) <= RULES.near_dlog_kw
            assert RULES.near_dS_min <= abs(S[i] - S[j]) <= RULES.near_dS_max
    h = hard(5)
    log_kw, _, _ = h.arrays()
    assert len(h) == 10 and RULES.early_log_kw in log_kw and RULES.late_log_kw in log_kw


def test_generator_is_seeded_and_deterministic():
    assert easy(4) == easy(4)
    assert easy(4) != easy(5)


@pytest.mark.parametrize("name", SETS)
def test_stored_sets_follow_their_rules(name):
    aset = AnalyteSet.load(name)
    n = {"easy": 6, "medium": 8, "hard": 10}[name]
    assert len(aset) == n
    _, _, dH = aset.arrays()
    assert np.all(dH < 0)
