import numpy as np
import pytest

from sim.config import CRFConfig
from sim.crf import (crf, crf_truncated, resolution, resolution_term, resolution_term_truncated, sort_table,
                     time_term, time_term_truncated)

T_END = 42.0


def test_resolution_hand_computed():
    t = np.array([1.0, 1.3, 2.0])
    s = np.array([0.02, 0.03, 0.05])
    R = resolution(t, s)
    assert R[0] == pytest.approx(0.3 / (2 * 0.05))      # 3.0
    assert R[1] == pytest.approx(0.7 / (2 * 0.08))      # 4.375


def test_time_penalty_examples(crf_cfg):
    assert float(time_term(np.array([1.0, 20.0]), crf_cfg)) == pytest.approx(0.5)
    assert float(time_term(np.array([1.0, 45.0]), crf_cfg)) == pytest.approx(1.75)
    assert float(time_term(np.array([1.0, 9.0]), crf_cfg)) == 0.0


def test_crf_bounds_and_equality_condition(crf_cfg):
    rng = np.random.default_rng(1)
    for _ in range(500):
        n = rng.integers(2, 11)
        t = np.sort(rng.uniform(0.5, 40.0, n))
        s = rng.uniform(0.01, 0.2, n)
        score = float(crf(t, s, crf_cfg))
        assert score <= 1.0 + 1e-12
        fully = np.all(resolution(t, s) >= crf_cfg.R_target) and t[-1] <= crf_cfg.t_target
        assert (abs(score - 1.0) < 1e-12) == bool(fully)


def test_crf_equals_one_iff_resolved_within_target(crf_cfg):
    s = np.full(4, 0.02)
    t = np.array([1.0, 2.0, 3.0, 9.0])                 # R = 12.5 everywhere, last peak at 9 < 10
    assert float(crf(t, s, crf_cfg)) == pytest.approx(1.0)
    t_late = np.array([1.0, 2.0, 3.0, 12.0])           # resolved but 20 % late -> 1 - 0.5 * 0.2
    assert float(crf(t_late, s, crf_cfg)) == pytest.approx(0.9)
    t_close = np.array([1.0, 1.06, 3.0, 9.0])          # pair 0: R = 0.75 -> 0.5 credit
    assert float(crf(t_close, s, crf_cfg)) == pytest.approx((0.5 + 1.0 + 1.0) / 3.0)


def test_resolution_term_monotonic_in_gaps(crf_cfg):
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = rng.integers(2, 9)
        t = np.sort(rng.uniform(0.5, 9.0, n))
        s = rng.uniform(0.01, 0.3, n)
        t_wide = t[0] + 1.1 * (t - t[0])
        assert float(resolution_term(t_wide, s, crf_cfg)) >= float(resolution_term(t, s, crf_cfg)) - 1e-12


def test_coelution(crf_cfg):
    """Two identical analytes: R = 0 for that pair; resolution term <= (n - 2)/(n - 1)."""
    t = np.array([1.0, 3.0, 3.0, 5.0, 7.0])
    s = np.full(5, 0.02)
    R = resolution(t, s)
    assert R[1] == 0.0
    n = len(t)
    assert float(resolution_term(t, s, crf_cfg)) == pytest.approx((n - 2) / (n - 1))
    all_coelute = np.full(5, 3.0)
    assert float(resolution_term(all_coelute, s, crf_cfg)) == 0.0


def test_sort_table_and_batching(crf_cfg):
    t = np.array([[3.0, 1.0, 2.0], [0.5, 0.7, 0.6]])
    s = np.array([[0.3, 0.1, 0.2], [0.05, 0.07, 0.06]])
    ts, ss = sort_table(t, s)
    np.testing.assert_array_equal(ts, [[1.0, 2.0, 3.0], [0.5, 0.6, 0.7]])
    np.testing.assert_array_equal(ss, [[0.1, 0.2, 0.3], [0.05, 0.06, 0.07]])
    assert crf(ts, ss, crf_cfg).shape == (2,)


def test_single_analyte_has_full_resolution_term():
    cfg = CRFConfig()
    assert float(crf(np.array([5.0]), np.array([0.02]), cfg)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Truncated tables (peaks eluting after t_end are not observed)
# --------------------------------------------------------------------------- #
def test_truncated_full_table_equals_crf(crf_cfg):
    rng = np.random.default_rng(2)
    for _ in range(300):
        n = int(rng.integers(1, 11))
        t = np.sort(rng.uniform(0.5, T_END, n))
        s = rng.uniform(0.01, 0.2, n)
        assert float(crf_truncated(t, s, np.ones(n, bool), n, T_END, crf_cfg)) == \
            pytest.approx(float(crf(t, s, crf_cfg)), abs=1e-12)
    t = np.sort(rng.uniform(0.5, T_END, (50, 6)), axis=1)
    s = rng.uniform(0.01, 0.2, (50, 6))
    np.testing.assert_allclose(crf_truncated(t, s, np.ones((50, 6), bool), 6, T_END, crf_cfg), crf(t, s, crf_cfg))


def test_truncated_missing_peak_scores_as_coelution_at_t_end(crf_cfg):
    """[1, 2, 3, lost], sigma 0.02: pairs 1-2 and 2-3 earn full credit (R = 12.5), the lost peak's pair
    earns nothing, and the time term is charged at t_end: 2/3 - 0.5 * (42 - 10) / 10 = 2/3 - 1.6."""
    t = np.array([1.0, 2.0, 3.0, 50.0])
    s = np.full(4, 0.02)
    obs = np.array([True, True, True, False])
    score = float(crf_truncated(t, s, obs, 4, T_END, crf_cfg))
    assert score == pytest.approx(2.0 / 3.0 - 1.6)
    # the same as the lost peak co-eluting exactly with its neighbour ...
    coelute = np.array([1.0, 2.0, 3.0, 3.0])
    assert float(resolution_term_truncated(t, s, obs, 4, crf_cfg)) == \
        pytest.approx(float(resolution_term(coelute, s, crf_cfg))) == pytest.approx(2.0 / 3.0)
    # ... with the last observed peak sitting at t_end
    assert float(time_term_truncated(t, obs, 4, T_END, crf_cfg)) == \
        pytest.approx(float(time_term(np.array([1.0, 2.0, T_END]), crf_cfg))) == pytest.approx(1.6)
    # only the observed rows, with the full analyte count, give the same score
    assert float(crf_truncated(t[:3], s[:3], np.ones(3, bool), 4, T_END, crf_cfg)) == pytest.approx(score)
    # whatever sits in the unobserved slot is irrelevant
    t2, s2 = t.copy(), s.copy()
    t2[3], s2[3] = np.inf, np.nan
    assert float(crf_truncated(t2, s2, obs, 4, T_END, crf_cfg)) == pytest.approx(score)
    # two lost peaks: one credited pair out of three
    assert float(crf_truncated(t, s, np.array([True, True, False, False]), 4, T_END, crf_cfg)) == \
        pytest.approx(1.0 / 3.0 - 1.6)


def test_truncated_score_bounded_below(crf_cfg):
    lower = -crf_cfg.w_t * (T_END - crf_cfg.t_target) / crf_cfg.t_target
    assert lower == pytest.approx(-1.6)
    rng = np.random.default_rng(3)
    for _ in range(500):
        n = int(rng.integers(1, 11))
        k = int(rng.integers(0, n + 1))                           # k observed, n - k lost
        t = np.concatenate([np.sort(rng.uniform(0.5, T_END, k)), np.sort(rng.uniform(T_END, 80.0, n - k))])
        s = rng.uniform(0.01, 0.3, n)
        obs = np.arange(n) < k
        score = float(crf_truncated(t, s, obs, n, T_END, crf_cfg))
        assert lower - 1e-12 <= score <= 1.0 + 1e-12
        if k == 0:
            assert score == pytest.approx(lower)                   # nothing observed: the floor exactly
        if k < n:
            assert score <= 1.0 - 1.0 / max(n - 1, 1) - (1.6 - 1e-12) + 1e-12 or n == 1


def test_losing_a_peak_never_beats_observing_it(crf_cfg):
    """Observed k peaks; the next either elutes anywhere in (t_k, t_end] or is lost. Lost is never better."""
    rng = np.random.default_rng(4)
    for _ in range(500):
        n = int(rng.integers(2, 9))
        k = int(rng.integers(1, n))
        t_seen = np.sort(rng.uniform(0.5, T_END - 0.01, k))
        t_next = rng.uniform(t_seen[-1], T_END)
        t = np.append(t_seen, t_next)
        s = rng.uniform(0.01, 0.3, k + 1)
        seen = np.ones(k + 1, bool)
        lost = seen.copy()
        lost[-1] = False
        assert float(crf_truncated(t, s, lost, n, T_END, crf_cfg)) <= \
            float(crf_truncated(t, s, seen, n, T_END, crf_cfg)) + 1e-12
