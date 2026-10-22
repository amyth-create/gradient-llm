import numpy as np
import pytest

from sim.simulator import Simulator
from sim.table import COLUMNS, build_rows, load_table, lookup, method_grid, write_csv


@pytest.fixture(scope="module")
def small_table(tmp_path_factory):
    sim = Simulator.from_configs("easy")
    grid = method_grid(sim.instrument, phi_step=0.25, tG_step=19.0, T_step=35.0)
    rows = build_rows(sim, grid)
    path = write_csv(rows, tmp_path_factory.mktemp("tables") / "easy.csv")
    return sim, grid, rows, load_table(path)


def test_grid_respects_bounds():
    inst = Simulator.from_configs("easy").instrument
    g = method_grid(inst)
    b = inst.method_bounds
    assert g.shape == (22080, 4)
    assert g[:, 0].min() == b.phi0[0] and g[:, 0].max() == b.phi0[1]
    assert g[:, 1].max() == b.phi1_max
    assert np.all(g[:, 1] >= g[:, 0] + b.min_dphi - 1e-9)
    assert g[:, 2].min() == b.tG[0] and g[:, 2].max() == b.tG[1]
    assert g[:, 3].min() == b.T[0] and g[:, 3].max() == b.T[1]
    assert len(np.unique(g, axis=0)) == len(g)


def test_rows_match_simulator_and_roundtrip(small_table):
    sim, grid, rows, table = small_table
    assert list(rows[0]) == COLUMNS
    assert {"n_missing", "n_observed"} <= set(COLUMNS)
    assert len(table["crf"]) == len(grid)
    t_end, t_eq, cfg = sim.instrument.t_end, sim.instrument.t_eq_min, sim.crf_config
    for i in range(len(grid)):
        X = np.array([[grid[i, 0], grid[i, 1] - grid[i, 0], grid[i, 2], grid[i, 3]]])
        br = sim.run_batch(X)
        assert table["feasible"][i] == bool(br.feasible[0])
        if br.feasible[0]:
            assert np.isfinite(table["crf"][i])
            assert table["crf"][i] == pytest.approx(float(br.crf[0]), abs=1e-6)
            rt = table["retention_table"][i]
            assert len(rt) == table["n_observed"][i] == int(br.observed[0].sum())
            assert table["n_missing"][i] == int(br.n_missing[0])
            assert table["n_missing"][i] + table["n_observed"][i] == sim.n_analytes
            assert table["time_term"][i] >= 0 and 0 <= table["resolution_term"][i] <= 1
            assert table["crf"][i] == pytest.approx(table["resolution_term"][i] - table["time_term"][i], abs=2e-6)
            assert table["reason"][i] == ""
            assert all(t <= t_end for t, _ in rt)
            if table["n_missing"][i] > 0:
                assert table["run_time"][i] == pytest.approx(t_end + t_eq)
                assert table["time_term"][i] == pytest.approx(cfg.w_t * (t_end - cfg.t_target) / cfg.t_target, abs=1e-6)
            else:
                assert table["t_last"][i] == pytest.approx(rt[-1][0]) and table["t_first"][i] == pytest.approx(rt[0][0])
                assert table["run_time"][i] == pytest.approx(rt[-1][0] + t_eq)
        else:
            assert np.isnan(table["crf"][i]) and table["retention_table"][i] == []
            assert table["reason"][i] != ""
    # this coarse grid steps phi1 past 0.95, so its only infeasible rows are bounds failures
    bad = ~table["feasible"]
    assert bad.any() and np.all(table["phi1"][bad] > 0.95)
    assert all(r.startswith("bounds") for r in table["reason"][bad])
    assert (table["n_missing"][~bad] > 0).any() and (table["n_missing"][~bad] == 0).any()


def test_out_of_bounds_row_is_infeasible_with_nothing_observed():
    sim = Simulator.from_configs("easy")
    rows = build_rows(sim, np.array([[0.02, 0.90, 10.0, 40.0], [0.05, 0.50, 10.0, 40.0]]))
    bad, good = rows
    assert bad["feasible"] is False and bad["reason"].startswith("bounds")
    assert bad["crf"] == "" and bad["retention_table"] == ""
    assert bad["n_observed"] == 0 and bad["n_missing"] == sim.n_analytes
    assert good["feasible"] is True and good["reason"] == "" and isinstance(good["crf"], float)
    assert good["n_observed"] + good["n_missing"] == sim.n_analytes


def test_lookup_snaps_to_nearest_grid_point(small_table):
    sim, grid, rows, table = small_table
    row = lookup(table, phi0=0.06, phi1=0.52, tG=3.0, T=27.0, instrument=sim.instrument)
    assert (row["phi0"], row["phi1"], row["tG"], row["T"]) == (0.05, 0.55, 2.0, 25.0)
    assert isinstance(row["crf"], float) and isinstance(row["method_id"], int)
