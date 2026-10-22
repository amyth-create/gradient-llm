import json

import numpy as np
import pytest

from sim.config import AnalyteSet, InstrumentConfig
from sim.import_hplcsim import (DEFAULT_SOURCE, PHASE_KEYS, LibraryCompound, build_set, fit_compound,
                                import_set, load_library, phase_compounds, select)
from sim.retention import log_k


@pytest.fixture(scope="module")
def library():
    return load_library(DEFAULT_SOURCE)


def test_vendored_library_shape(library):
    assert set(library) == set(PHASE_KEYS.values())
    assert len(phase_compounds(library, "beh")) == 21
    assert len(phase_compounds(library, "sb")) == 22
    for comps in library.values():
        assert len({c.id for c in comps}) == len(comps)
        for c in comps:
            assert len(c.coefficients) == 2 and all(len(row) == 4 for row in c.coefficients)
            m, w, n, v = c.coefficients[0]
            assert m < 0 and v < 0                     # retention falls with T and with phi


def test_js_source_parses_like_json(library, tmp_path):
    data = json.loads(DEFAULT_SOURCE.read_text())
    js = tmp_path / "compounds.js"
    js.write_text("// header\n(function(root){\nconst phases = " + json.dumps(data["phases"]) + ";\n})(this);\n")
    assert load_library(js) == library


def test_fit_is_exact_when_surface_is_in_the_model_family(inst):
    # n = 0 (slope independent of T); the library's linear-in-T term is matched by the van 't Hoff
    # term to within the curvature of 1/T over 25-60 C, which is tiny for small m.
    c = LibraryCompound(phase="x", id=0, name="synthetic", molar_volume=100.0,
                        coefficients=((-0.0001, 2.5, 0.0, -5.0), (0.0, 0.0, 0.0, 0.0)))
    f = fit_compound(c, inst)
    assert f.S == pytest.approx(5.0, abs=1e-3)
    assert f.log_kw == pytest.approx(2.5 - 0.0001 * inst.T_ref_C, abs=2e-3)
    assert f.max_abs_all < 2e-3


def test_fit_reproduces_library_surface_in_window(library, inst):
    phi = np.linspace(0.05, 0.95, 37)[:, None]
    T = np.array([25.0, 40.0, 60.0])[None, :]
    for phase in ("beh", "sb"):
        for c in phase_compounds(library, phase):
            f = fit_compound(c, inst)
            assert f.dH < 500.0 and 1.0 < f.S < 8.0 and -0.5 < f.log_kw < 5.0, c.name
            truth = c.log_k(phi, T)
            ours = log_k(phi, T, f.log_kw, f.S, f.dH, inst.T_ref_C)
            k = 10.0**truth
            window = (k >= 0.5) & (k <= 50.0)
            assert np.abs(ours - truth)[window].max() < 0.3, c.name
            assert f.rmse_window < 0.15, c.name


def test_build_set_keeps_names_and_order(library, inst):
    comps = phase_compounds(library, "sb")[:5]
    s, fits = build_set("demo", comps, inst)
    assert [a.name for a in s.analytes] == [c.name for c in comps] == [f.name for f in fits]
    assert len(s) == 5 and s.seed is None


def test_select_by_id_and_name(library):
    comps = phase_compounds(library, "beh")
    picked = select(comps, ids=[11], names=["Anthracene"])
    assert sorted(c.name for c in picked) == ["anthracene", "caffeine"]
    with pytest.raises(KeyError):
        select(comps, names=["unobtainium"])


def test_import_set_writes_loadable_yaml_and_record(tmp_path):
    s, fits, path = import_set("beh", ids=[3, 11, 20], name="tmp_beh3", directory=tmp_path)
    loaded = AnalyteSet.load(path)
    assert loaded == s and len(loaded) == 3
    meta = json.loads((tmp_path / "tmp_beh3.import.json").read_text())
    assert meta["n"] == 3 and {f["name"] for f in meta["fits"]} == {"benzophenone", "caffeine", "anthracene"}
    assert meta["worst_rmse_window"] == max(f.rmse_window for f in fits)


def test_hplcsim_sets_resolve_to_their_own_root():
    from sim.config import PROJECT_ROOT, analyte_dir, results_dir, set_root, table_dir
    from sim.import_hplcsim import HPLCSIM_ROOT
    from sim.optimum import optimum_path
    for name in ("hplcsim_beh", "hplcsim_sb"):
        assert set_root(name) == HPLCSIM_ROOT
        assert analyte_dir(name) == HPLCSIM_ROOT / "configs" / "analytes"
        assert table_dir(name) == HPLCSIM_ROOT / "tables"
        assert results_dir(name) == HPLCSIM_ROOT / "results"
        assert optimum_path(name).exists()
    assert set_root("hard") == PROJECT_ROOT
    assert set_root("no_such_set") == PROJECT_ROOT


def test_shipped_sets_match_the_importer():
    inst = InstrumentConfig.load()
    library = load_library(DEFAULT_SOURCE)
    for phase, name in (("beh", "hplcsim_beh"), ("sb", "hplcsim_sb")):
        shipped = AnalyteSet.load(name)
        rebuilt, _ = build_set(name, phase_compounds(library, phase), inst)
        assert shipped == rebuilt
