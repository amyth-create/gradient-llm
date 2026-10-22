"""sim.chromatogram: detector trace, apparent peaks, detector config, plot and CLI."""
import math

import numpy as np
import pytest

from sim import InstrumentConfig, Method, Simulator
from sim.chromatogram import TAIL_SIGMAS, _main, apparent_peaks, from_result, plot, sample_spacing, trace
from sim.config import DetectorConfig

HARD_METHOD = Method(phi0=0.05, phi1=0.5, tG=6.0, T=55.0)
# Three well separated peaks (gaps of 25 sigma or more), apexes on the 10 Hz grid.
TABLE = [(1.0, 0.02), (2.0, 0.03), (3.5, 0.04)]


def _trapezoid(y, x) -> float:
    f = getattr(np, "trapezoid", None) or np.trapz
    return float(f(y, x))


def _height(area: float, sigma: float) -> float:
    return area / (sigma * math.sqrt(2.0 * math.pi))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_detector_config_loads_from_yaml(inst):
    assert inst.detector == DetectorConfig(sampling_rate_hz=10.0, noise_sd=0.0, drift=0.0)
    assert InstrumentConfig().detector.sampling_rate_hz == 10.0       # default without a detector block
    with pytest.raises(ValueError):
        DetectorConfig(sampling_rate_hz=0.0)
    with pytest.raises(ValueError):
        DetectorConfig(bogus=1.0)


def test_sample_spacing_is_one_over_sixty_f(inst):
    """(f) 10 Hz -> 0.1 s = 1/600 min between samples (~12 samples per 0.02 min sigma)."""
    dt = sample_spacing(inst)
    assert dt == pytest.approx(1.0 / (60.0 * inst.detector.sampling_rate_hz))
    assert dt == pytest.approx(1.0 / 600.0)
    t, s = trace(TABLE, inst)
    assert t[0] == 0.0 and t.shape == s.shape
    np.testing.assert_allclose(np.diff(t), dt, rtol=0, atol=1e-12)
    fast = inst.model_copy(update={"detector": DetectorConfig(sampling_rate_hz=40.0)})
    assert np.diff(trace(TABLE, fast)[0])[0] == pytest.approx(dt / 4.0)


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #
def test_default_window(inst):
    dt = sample_spacing(inst)
    t, _ = trace(TABLE, inst)
    assert t[-1] == pytest.approx(3.5 + TAIL_SIGMAS * 0.04, abs=dt)
    # the last apex sets the end even if the table is unsorted
    assert trace(TABLE[::-1], inst)[0].shape == t.shape
    # an early single peak never ends the trace before t0 + 1 min
    t_early, _ = trace([(0.6, 0.02)], inst)
    assert t_early[-1] == pytest.approx(inst.t0 + 1.0, abs=dt)
    t_fixed, s_fixed = trace(TABLE, inst, t_end=12.0)
    assert t_fixed[-1] == pytest.approx(12.0, abs=dt) and s_fixed.shape == t_fixed.shape
    # an empty table is a flat baseline, not an error
    t_empty, s_empty = trace([], inst)
    assert t_empty[-1] == pytest.approx(inst.t0 + 1.0, abs=dt)
    assert np.all(s_empty == 0.0)


def test_area_is_conserved(inst):
    """(a) the trapezoid integral of a noise-free trace equals the sum of areas to 1e-3 relative."""
    t, s = trace(TABLE, inst)
    assert _trapezoid(s, t) == pytest.approx(len(TABLE), rel=1e-3)          # default: area 1 each
    areas = [0.5, 2.0, 1.25]
    t, s = trace(TABLE, inst, areas=areas)
    assert _trapezoid(s, t) == pytest.approx(sum(areas), rel=1e-3)
    assert np.all(s >= 0.0)


def test_height_is_area_over_sigma_root_two_pi(inst):
    """Hand case: area 1, sigma 0.02 min -> apex 1 / (0.02 sqrt(2 pi)) = 19.947 at t = 2.0 (on the grid)."""
    t, s = trace([(2.0, 0.02)], inst)
    assert s.max() == pytest.approx(19.947, abs=1e-3)
    assert s.max() == pytest.approx(_height(1.0, 0.02), rel=1e-6)
    assert t[int(np.argmax(s))] == pytest.approx(2.0, abs=1e-12)
    # 1 sigma off the apex the signal is exp(-1/2) of the height
    i = int(np.argmin(np.abs(t - 2.02)))
    assert s[i] == pytest.approx(_height(1.0, 0.02) * math.exp(-0.5), rel=1e-6)


def test_apex_within_one_sample_of_t_r(inst):
    """(b) well separated peaks: every apex lands within one sample of its t_R."""
    t, s = trace(TABLE, inst)
    peaks = apparent_peaks(t, s)
    assert len(peaks) == len(TABLE)
    dt = sample_spacing(inst)
    for (t_apex, height), (t_r, sigma) in zip(peaks, TABLE):
        assert abs(t_apex - t_r) <= dt + 1e-12
        assert height == pytest.approx(_height(1.0, sigma), rel=1e-2)
    # same for an apex that falls between two samples
    t, s = trace([(2.0 + 0.4 / 600.0, 0.02)], inst)
    (t_apex, _), = apparent_peaks(t, s)
    assert abs(t_apex - (2.0 + 0.4 / 600.0)) <= dt


def test_coeluting_rows_make_one_apparent_peak(inst):
    """(c) identical rows fuse into one visible peak; 6 sigma apart they are two."""
    t, s = trace([(3.0, 0.02), (3.0, 0.02)], inst)
    peaks = apparent_peaks(t, s)
    assert len(peaks) == 1
    assert peaks[0][0] == pytest.approx(3.0, abs=1e-12)
    assert peaks[0][1] == pytest.approx(_height(2.0, 0.02), rel=1e-6)
    t, s = trace([(3.0, 0.02), (3.0 + 6 * 0.02, 0.02)], inst)
    assert len(apparent_peaks(t, s)) == 2
    # 1 sigma apart: one broader bump (equal Gaussians are unimodal until > 2 sigma apart)
    t, s = trace([(3.0, 0.02), (3.02, 0.02)], inst)
    assert len(apparent_peaks(t, s)) == 1


def test_min_height_fraction_filters_small_peaks(inst):
    t, s = trace([(2.0, 0.02), (4.0, 0.02)], inst, areas=[1.0, 0.01])
    assert len(apparent_peaks(t, s)) == 1                               # 1 % of the tallest: ignored
    assert len(apparent_peaks(t, s, min_height_frac=0.005)) == 2
    assert apparent_peaks(t, np.zeros_like(t)) == []


def test_noise_is_seeded(inst):
    """(d) same seed -> bit-identical traces; different seeds differ; the noise-free default is exact."""
    kw = dict(noise_sd=0.3, drift=1.0)
    t1, s1 = trace(TABLE, inst, seed=7, **kw)
    t2, s2 = trace(TABLE, inst, seed=7, **kw)
    _, s3 = trace(TABLE, inst, seed=8, **kw)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(s1, s2)
    assert not np.array_equal(s1, s3)
    clean = trace(TABLE, inst)[1]
    np.testing.assert_array_equal(clean, trace(TABLE, inst, seed=123)[1])      # seed irrelevant without noise
    np.testing.assert_array_equal(clean, trace(TABLE, inst, noise_sd=0.0, seed=1)[1])
    residual = s1 - clean - 1.0 * t1 / t1[-1]
    assert residual.std() == pytest.approx(0.3, rel=0.1)
    assert abs(residual.mean()) < 0.05
    # With the 5 % floor well above the noise band (here 10 noise sigma), ripples on a
    # peak top are not counted twice (prominence, not raw local maxima) and the
    # baseline adds nothing: one apparent peak per row.
    t4, s4 = trace(TABLE, inst, noise_sd=0.1, seed=7)
    assert len(apparent_peaks(t4, s4)) == len(TABLE)
    # raw local maxima would over-count the same trace many times over
    inner = s4[1:-1]
    raw = int(np.count_nonzero((inner > s4[:-2]) & (inner >= s4[2:]) & (inner >= 0.05 * s4.max())))
    assert raw > len(TABLE)


def test_drift_is_a_linear_baseline(inst):
    t, s = trace([], inst, drift=2.0)
    assert s[0] == 0.0 and s[-1] == pytest.approx(2.0)
    np.testing.assert_allclose(s, 2.0 * t / t[-1])
    assert apparent_peaks(t, s) == []                                   # a rising baseline has no apex
    # the instrument's detector defaults are used when the call does not set them
    drifting = inst.model_copy(update={"detector": DetectorConfig(drift=2.0)})
    np.testing.assert_array_equal(trace([], drifting)[1], s)


def test_from_result_on_hard():
    """(e) a feasible hard result gives a trace whose apparent peak count is <= n_analytes."""
    sim = Simulator.from_configs("hard")
    res = sim.run(HARD_METHOD)
    assert res.feasible
    t, s = from_result(res, sim.instrument)
    peaks = apparent_peaks(t, s)
    assert 0 < len(peaks) <= sim.n_analytes
    # CRF 1.0 means every pair is >= 6 sigma apart, so here all ten are visible,
    # each apex within a sample of its row, and the last (broadest) is the shortest.
    assert res.crf == pytest.approx(1.0)
    assert len(peaks) == sim.n_analytes
    dt = sample_spacing(sim.instrument)
    for (t_apex, _), (t_r, _) in zip(peaks, res.retention_table):
        assert abs(t_apex - t_r) <= dt
    heights = [h for _, h in peaks]
    assert heights[-1] == min(heights)
    t_last, sig_last = res.retention_table[-1]
    assert t[-1] == pytest.approx(t_last + TAIL_SIGMAS * sig_last, abs=dt)
    assert _trapezoid(s, t) == pytest.approx(sim.n_analytes, rel=1e-3)


def test_from_result_refuses_infeasible():
    sim = Simulator.from_configs("hard")
    res = sim.run(Method(phi0=0.3, phi1=0.3, tG=10.0, T=40.0))           # dphi below the minimum
    assert not res.feasible
    with pytest.raises(ValueError, match="infeasible"):
        from_result(res, sim.instrument)


def test_input_validation(inst):
    with pytest.raises(ValueError):
        trace([(1.0, 0.0)], inst)                                       # sigma must be positive
    with pytest.raises(ValueError):
        trace(TABLE, inst, areas=[1.0, 2.0])                            # one area per row
    with pytest.raises(ValueError):
        trace(TABLE, inst, areas=[1.0, -2.0, 1.0])
    with pytest.raises(ValueError):
        trace(TABLE, inst, noise_sd=-1.0)
    with pytest.raises(ValueError):
        trace(TABLE, inst, t_end=0.0)
    with pytest.raises(ValueError):
        trace([(1.0, 0.02, 3.0)], inst)
    with pytest.raises(ValueError):
        apparent_peaks(np.arange(5.0), np.arange(4.0))


# --------------------------------------------------------------------------- #
# Plot and CLI (need matplotlib; skipped without it)
# --------------------------------------------------------------------------- #
def test_plot_writes_png(inst, tmp_path):
    pytest.importorskip("matplotlib")
    t, s = trace(TABLE, inst)
    out = tmp_path / "chrom.png"
    fig = plot(t, s, path=out, title="demo", peaks=apparent_peaks(t, s))
    assert fig is not None
    assert out.exists() and out.stat().st_size > 1000


def test_cli(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "hard.png"
    _main(["hard", "--phi0", "0.05", "--phi1", "0.5", "--tG", "6", "--T", "55",
           "--out", str(out), "--noise", "0.2", "--seed", "1"])
    printed = capsys.readouterr().out
    assert printed.startswith("10 apparent peaks from 10 analytes")
    assert printed.rstrip().endswith(f"saved {out}")
    assert out.exists()
    with pytest.raises(SystemExit):
        _main(["hard", "--phi0", "0.3", "--phi1", "0.3"])                # infeasible: exit status 1
    assert "infeasible" in capsys.readouterr().err
