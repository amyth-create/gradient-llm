import numpy as np
import pytest

from optimizers import TableOracle, random_search
from optimizers.landscape import plot_heatmaps, plot_landscape, profile, projected_landscape


@pytest.fixture(scope="module")
def oracle() -> TableOracle:
    return TableOracle("easy")


def test_projection_is_max_over_other_variables(oracle):
    xs, ys, Z, present = projected_landscape(oracle, 0, 1)        # phi0 x phi1
    assert Z.shape == (len(ys), len(xs)) == (17, 12)
    assert np.nanmax(Z) == pytest.approx(np.nanmax(oracle.crf))
    # brute force one cell
    a, b = 2, 9
    rows = (oracle.X[:, 0] == xs[a]) & (oracle.X[:, 1] == ys[b]) & oracle.feasible
    assert Z[b, a] == pytest.approx(oracle.crf[rows].max())
    # phi1 < phi0 + 0.1 cells contain no method at all
    assert not present[0, -1] and present[-1, 0]


def test_profile_matches_brute_force(oracle):
    xs, z = profile(oracle, 3)                                       # temperature
    assert len(xs) == 8
    for x, v in zip(xs, z):
        rows = (oracle.X[:, 3] == x) & oracle.feasible
        assert v == pytest.approx(oracle.crf[rows].max())


def test_plot_landscape_with_and_without_run(oracle, tmp_path):
    p1 = plot_landscape(oracle, tmp_path / "landscape.png")
    p2 = plot_landscape(oracle, tmp_path / "traj.png", random_search(oracle, budget=10, seed=0))
    assert p1.stat().st_size > 20_000 and p2.stat().st_size > 20_000


def test_plot_heatmaps_only(oracle, tmp_path):
    p = plot_heatmaps(oracle, tmp_path / "heat.png")
    assert p.stat().st_size > 20_000
