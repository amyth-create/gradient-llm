import numpy as np
import pytest

from optimizers import TableOracle, bo_search, random_search
from optimizers.bo import GP, expected_improvement, matern52
from optimizers.oracle import format_run
from optimizers.plot import plot_comparison
from optimizers.random_search import random_order
from sim.simulator import Simulator
from sim.table import build_rows, method_grid, write_csv


@pytest.fixture(scope="module")
def oracle() -> TableOracle:
    return TableOracle("easy")


@pytest.fixture(scope="module")
def oracle_with_infeasible(tmp_path_factory) -> TableOracle:
    """A coarse easy table (whose phi1 steps past 0.95) with two more out-of-bounds methods appended;
    the default tables have no infeasible rows at all."""
    sim = Simulator.from_configs("easy")
    grid = method_grid(sim.instrument, phi_step=0.25, tG_step=19.0, T_step=35.0)
    grid = np.vstack([grid, [[0.02, 0.90, 10.0, 40.0], [0.01, 0.50, 5.0, 30.0]]])
    d = tmp_path_factory.mktemp("tables")
    write_csv(build_rows(sim, grid), d / "easy.csv")
    return TableOracle("easy", table_dir=d)


def test_oracle_features_budget_and_regret(oracle):
    assert oracle.n_candidates == 22080
    assert oracle.X_unit.min() == 0.0 and oracle.X_unit.max() == 1.0
    run = oracle.new_run("test", seed=0, budget=2)
    feasible_rows = np.flatnonzero(oracle.feasible)
    o1 = oracle.query(run, int(feasible_rows[0]))
    assert o1.run == 1 and o1.feasible and o1.best == o1.crf
    assert o1.regret == pytest.approx(oracle.optimum - o1.crf)
    with pytest.raises(ValueError):
        oracle.query(run, int(feasible_rows[0]))
    o2 = oracle.query(run, int(feasible_rows[1]))
    assert o2.best == pytest.approx(max(o1.crf, o2.crf))
    with pytest.raises(RuntimeError):
        oracle.query(run, int(feasible_rows[2]))


def test_every_table_row_is_feasible_and_scored(oracle):
    assert oracle.feasible.all() and np.isfinite(oracle.crf).all()
    assert (oracle.n_missing > 0).any() and (oracle.n_missing == 0).any()
    for i in np.flatnonzero(oracle.n_missing > 0)[:50]:
        assert len(oracle.retention_table[i]) == 6 - oracle.n_missing[i]


def test_truncated_rows_count_like_any_other_injection(oracle):
    i = int(np.flatnonzero(oracle.n_missing > 0)[0])
    run = oracle.new_run("test", seed=0, budget=1)
    o = oracle.query(run, i)
    assert o.feasible and o.n_missing == oracle.n_missing[i] > 0
    assert np.isfinite(o.crf) and o.best == o.crf and o.regret == pytest.approx(oracle.optimum - o.crf)
    assert run.n_missing_per_run().tolist() == [o.n_missing]
    assert "missing" in format_run(run)


def test_infeasible_rows_do_not_count_as_best(oracle_with_infeasible):
    o = oracle_with_infeasible
    bad = np.flatnonzero(~o.feasible)
    good = int(np.flatnonzero(o.feasible)[0])
    assert len(bad) >= 2 and np.isnan(o.crf[bad]).all() and set(o.n_missing[bad].tolist()) == {6}
    assert o.n_candidates - 2 in bad and o.n_candidates - 1 in bad        # the appended rows
    run = o.new_run("test", seed=0, budget=3)
    obs = o.query(run, int(bad[0]))
    assert not obs.feasible and np.isnan(obs.crf) and np.isnan(obs.best) and np.isnan(obs.regret)
    assert obs.reason.startswith("bounds")
    obs = o.query(run, good)
    assert obs.best == obs.crf
    obs = o.query(run, int(bad[1]))
    assert obs.best == pytest.approx(o.crf[good])
    assert "NO" in format_run(run)


def test_random_search_is_seeded_and_never_repeats(oracle):
    a = random_search(oracle, budget=25, seed=3)
    b = random_search(oracle, budget=25, seed=3)
    c = random_search(oracle, budget=25, seed=4)
    assert a.indices() == b.indices() and a.indices() != c.indices()
    assert len(set(a.indices())) == 25
    assert np.all(np.diff(a.best_curve()[np.isfinite(a.best_curve())]) >= 0)
    assert "optimum CRF" in format_run(a)


def test_gp_interpolates_training_points():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(15, 4))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 - 0.5 * X[:, 2] * X[:, 3]
    gp = GP.fit(X, y, rng)
    mu, sd = gp.predict(X)
    np.testing.assert_allclose(mu, y, atol=0.05)
    assert np.all(sd < 0.2)
    far_mu, far_sd = gp.predict(np.full((1, 4), 3.0))
    assert far_sd[0] > sd.max()                       # uncertainty grows away from the data
    K = matern52(X, X, np.ones(4), 1.0)
    assert np.allclose(np.diag(K), 1.0) and np.all(K <= 1.0 + 1e-12)


def test_expected_improvement_tradeoff():
    assert expected_improvement(np.array([1.0]), np.array([0.1]), 0.0)[0] > \
        expected_improvement(np.array([0.0]), np.array([0.1]), 0.0)[0]
    assert expected_improvement(np.array([0.0]), np.array([1.0]), 0.0)[0] > \
        expected_improvement(np.array([0.0]), np.array([0.1]), 0.0)[0]
    assert expected_improvement(np.array([-5.0]), np.array([0.01]), 0.0)[0] >= 0.0


def test_bo_shares_initial_design_and_respects_budget(oracle):
    run = bo_search(oracle, budget=8, seed=1, n_init=3)
    assert len(run) == 8 and len(set(run.indices())) == 8
    assert run.indices()[:3] == random_order(oracle, 1)[:3].tolist()
    assert run.method == "bo"


def test_bo_at_least_matches_random_on_easy(oracle):
    seeds = range(4)
    rnd = np.mean([random_search(oracle, budget=15, seed=s).best_curve()[-1] for s in seeds])
    bo = np.mean([bo_search(oracle, budget=15, seed=s).best_curve()[-1] for s in seeds])
    assert bo >= rnd - 0.02


def test_plot_writes_png(oracle, tmp_path):
    runs = {"random": [random_search(oracle, budget=6, seed=s) for s in range(2)],
            "bo": [bo_search(oracle, budget=6, seed=s, n_init=3) for s in range(2)]}
    out = plot_comparison(runs, tmp_path / "easy_search.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_efficiency_plot_and_table(oracle, tmp_path):
    from optimizers.efficiency import checkpoint_table, plot_efficiency
    runs = {"easy": [random_search(oracle, budget=10, seed=s) for s in range(3)]}
    out = plot_efficiency(runs, tmp_path / "eff.png")
    assert out.stat().st_size > 10_000
    table = checkpoint_table(runs, checkpoints=(5, 10))
    assert "easy" in table and "@10" in table


def test_convergence_plot(oracle, tmp_path):
    from optimizers.convergence import found_fraction, mean_and_ci, plot_convergence
    rs = [random_search(oracle, budget=10, seed=s) for s in range(4)]
    mean, lo, hi = mean_and_ci(np.vstack([r.best_curve() for r in rs]))
    assert np.all(lo <= mean) and np.all(mean <= hi) and len(mean) == 10
    frac = found_fraction(rs)
    assert np.all(np.diff(frac) >= 0) and 0 <= frac[-1] <= 1
    out = plot_convergence({"easy": {"random": rs, "bo": [bo_search(oracle, budget=10, seed=s, n_init=4) for s in range(2)]}},
                           tmp_path / "conv.png")
    assert out.stat().st_size > 10_000


def test_plot_crf_per_run_writes_png(oracle, tmp_path):
    from optimizers.per_run import plot_crf_per_run
    runs = {"easy": bo_search(oracle, budget=6, seed=0, n_init=3)}
    out = plot_crf_per_run(runs, tmp_path / "per_run.png", n_init=3)
    assert out.exists() and out.stat().st_size > 10_000


def test_repeats_plot_and_tables(oracle, tmp_path):
    from optimizers.repeats import crf_matrix, infeasible_share, per_injection_table, plot_crf_over_runs, truncated_share
    runs = {"easy": [bo_search(oracle, budget=6, seed=s, n_init=3) for s in range(3)]}
    C = crf_matrix(runs["easy"])
    assert C.shape == (3, 6)
    share = infeasible_share(runs["easy"])
    assert share.shape == (6,) and np.all((share >= 0) & (share <= 1))
    trunc = truncated_share(runs["easy"])
    assert trunc.shape == (6,) and np.all((trunc >= 0) & (trunc <= 1)) and np.all(trunc + share <= 1 + 1e-12)
    table = per_injection_table(runs, checkpoints=(5, 10))      # 10 exceeds the budget and must be dropped
    assert "inj 5" in table and "inj 10" not in table
    out = plot_crf_over_runs(runs, tmp_path / "repeats.png", n_init=3)
    assert out.exists() and out.stat().st_size > 10_000


def test_plot_methods_over_runs_writes_png(oracle, tmp_path):
    from optimizers.repeats import plot_methods_over_runs
    runs = {"easy": {"random": [random_search(oracle, budget=6, seed=s) for s in range(2)],
                     "bo": [bo_search(oracle, budget=6, seed=s, n_init=3) for s in range(2)]}}
    out = plot_methods_over_runs(runs, tmp_path / "both.png", n_init=3)
    assert out.exists() and out.stat().st_size > 10_000
