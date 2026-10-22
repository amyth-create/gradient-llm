import numpy as np

from sim.config import AnalyteSet
from sim.plot_analytes import clusters, plot_analyte_sets


def test_clusters_follow_generator_rule():
    log_kw = np.array([3.00, 3.05, 2.00, 3.70, 3.72, 3.75])
    S = np.array([6.0, 6.2, 6.0, 7.0, 7.3, 7.1])
    groups = sorted(sorted(g) for g in clusters(log_kw, S))
    assert groups == [[0, 1], [3, 4, 5]]                  # a pair and a triple; #2 is far in log k_w
    assert clusters(np.array([1.0, 1.1]), np.array([4.0, 6.0])) == []   # same log k_w but S differs by 2


def test_stored_sets_have_expected_clusters():
    assert clusters(*AnalyteSet.load("easy").arrays()[:2]) == []
    assert len(clusters(*AnalyteSet.load("medium").arrays()[:2])) == 2
    assert len(clusters(*AnalyteSet.load("hard").arrays()[:2])) == 3


def test_plot_renders(tmp_path):
    out = plot_analyte_sets([AnalyteSet.load("medium"), AnalyteSet.load("hard")], tmp_path / "sets.png")
    assert out.stat().st_size > 20_000
