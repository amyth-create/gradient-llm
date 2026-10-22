"""Real-compound analyte sets from the HPLC Simulator library.

    python -m sim.import_hplcsim                              # hplcsim_beh (21) and hplcsim_sb (22), + optima
    python -m sim.import_hplcsim --phase sb --solvent methanol --name hplcsim_sb_meoh
    python -m sim.import_hplcsim --phase beh --ids 3 4 9 11 15 17 20 --name hplcsim_beh7 --table

The library (``hplcsim/configs/external/hplcsim_compounds.json``, see the README
there for provenance and licence) gives each compound a measured retention surface

    log10 k = (w + m T) + (v + n T) phi          T in deg C

per organic modifier. The simulator's model is

    log10 k = log_kw + dH (1/T_ref - 1/T) - S phi

with a temperature-independent slope S and a van 't Hoff temperature term.
The two families do not coincide (the library's slope drifts with T), so each
compound is fitted by weighted linear least squares over the instrument's
method box, weighting the window 0.5 <= k <= 50 where a gradient actually
moves the band and the error matters. The fit is exact whenever the library
surface has n = 0 and m is small. Residuals are written next to the YAML in
``<set>.import.json`` so the approximation is on record.

Only the three parameters cross into the analyte YAML; the optimizers never see
compound names. Everything from this library lives under ``hplcsim/`` (a data
root mirroring the project layout: ``configs/analytes``, ``tables``,
``results``), and the loaders find a set by name in whichever root holds it, so
the sets run through ``sim.optimum``, ``sim.table`` and ``optimizers.run``
exactly like the generated ones.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import PROJECT_ROOT, Analyte, AnalyteSet, InstrumentConfig, analyte_dir
from .retention import KELVIN

# Everything from this library lives in its own data root (see sim.config.EXTRA_ROOTS):
# the vendored library, the fitted sets with their optima, their tables and their results.
HPLCSIM_ROOT = PROJECT_ROOT / "hplcsim"
HPLCSIM_ANALYTE_DIR = HPLCSIM_ROOT / "configs" / "analytes"
DEFAULT_SOURCE = HPLCSIM_ROOT / "configs" / "external" / "hplcsim_compounds.json"
SOLVENTS = ("acetonitrile", "methanol")
PHASE_KEYS = {"beh": "Waters Acquity BEH C18", "sb": "Agilent Zorbax SB-C18"}
DEFAULT_NAMES = {("beh", "acetonitrile"): "hplcsim_beh", ("sb", "acetonitrile"): "hplcsim_sb",
                 ("beh", "methanol"): "hplcsim_beh_meoh", ("sb", "methanol"): "hplcsim_sb_meoh"}
K_WINDOW = (0.5, 50.0)      # retention factors that carry full weight in the fit
OUTSIDE_WEIGHT = 0.1        # weight of grid points outside the window


@dataclass(frozen=True)
class LibraryCompound:
    phase: str
    id: int
    name: str
    coefficients: tuple[tuple[float, float, float, float], ...]   # one (m, w, n, v) row per solvent
    molar_volume: float

    def log_k(self, phi, T_C, solvent: int = 0) -> np.ndarray:
        """The library's own surface, broadcast over ``phi`` and ``T_C``."""
        m, w, n, v = self.coefficients[solvent]
        phi = np.asarray(phi, dtype=float)
        T = np.asarray(T_C, dtype=float)
        return (w + m * T) + (v + n * T) * phi


@dataclass(frozen=True)
class Fit:
    name: str
    library_id: int
    log_kw: float
    S: float
    dH: float
    rmse_window: float      # log10 k, over grid points with k in K_WINDOW
    max_abs_window: float
    max_abs_all: float      # over the whole method box
    n_window: int


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #
def load_library(source: str | Path = DEFAULT_SOURCE) -> dict[str, list[LibraryCompound]]:
    """Phases of the library keyed by their full name. ``source`` is a path or an http(s) URL to the JSON or the JS."""
    text = _read(source)
    data = json.loads(text) if text.lstrip().startswith("{") else {"phases": _phases_from_js(text)}
    out: dict[str, list[LibraryCompound]] = {}
    for phase in data["phases"]:
        out[phase["name"]] = [
            LibraryCompound(phase=phase["name"], id=int(c["id"]), name=str(c["name"]),
                            coefficients=tuple(tuple(float(x) for x in row) for row in c["coefficients"]),
                            molar_volume=float(c["molarVolume"]))
            for c in phase["compounds"]
        ]
    return out


def _read(source: str | Path) -> str:
    s = str(source)
    if s.startswith(("http://", "https://")):
        with urllib.request.urlopen(s, timeout=30) as resp:   # noqa: S310 - explicit user-supplied URL
            return resp.read().decode("utf-8")
    return Path(s).read_text()


def _phases_from_js(text: str) -> list:
    """The upstream ``compounds.js`` wraps a JSON array; cut it out."""
    start, end = text.index("["), text.rindex("];")
    return json.loads(text[start:end + 1])


def phase_compounds(library: dict[str, list[LibraryCompound]], phase: str) -> list[LibraryCompound]:
    key = PHASE_KEYS.get(phase, phase)
    if key not in library:
        raise KeyError(f"phase {phase!r} not in library; have {sorted(library)}")
    return library[key]


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def fit_grid(instrument: InstrumentConfig, phi_step: float = 0.05, T_step: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """(phi, T) grid over the method box: every composition a gradient can visit at every allowed temperature."""
    b = instrument.method_bounds
    phi = np.arange(b.phi0[0], b.phi1_max + 1e-9, phi_step)
    T = np.arange(b.T[0], b.T[1] + 1e-9, T_step)
    P, TT = np.meshgrid(phi, T, indexing="ij")
    return P.ravel(), TT.ravel()


def fit_compound(c: LibraryCompound, instrument: InstrumentConfig, solvent: int = 0,
                 k_window: tuple[float, float] = K_WINDOW, outside_weight: float = OUTSIDE_WEIGHT) -> Fit:
    """Weighted least squares of the simulator model to the library surface."""
    phi, T_C = fit_grid(instrument)
    y = c.log_k(phi, T_C, solvent)
    T = T_C + KELVIN
    T_ref = instrument.T_ref_C + KELVIN
    A = np.column_stack([np.ones_like(phi), 1.0 / T_ref - 1.0 / T, -phi])      # -> (log_kw, dH, S)
    k = np.power(10.0, y)
    inside = (k >= k_window[0]) & (k <= k_window[1])
    wts = np.where(inside, 1.0, outside_weight)
    sw = np.sqrt(wts)
    params, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    resid = A @ params - y
    r_in = resid[inside] if inside.any() else resid
    return Fit(name=c.name, library_id=c.id,
               log_kw=float(params[0]), dH=float(params[1]), S=float(params[2]),
               rmse_window=float(np.sqrt(np.mean(r_in**2))),
               max_abs_window=float(np.max(np.abs(r_in))),
               max_abs_all=float(np.max(np.abs(resid))),
               n_window=int(inside.sum()))


def build_set(name: str, compounds: Iterable[LibraryCompound], instrument: InstrumentConfig,
              solvent: int = 0) -> tuple[AnalyteSet, list[Fit]]:
    fits = [fit_compound(c, instrument, solvent) for c in compounds]
    analytes = [Analyte(log_kw=f.log_kw, S=f.S, dH=f.dH, name=f.name) for f in fits]
    return AnalyteSet(name=name, analytes=analytes, seed=None), fits


def select(compounds: Sequence[LibraryCompound], ids: Sequence[int] | None = None,
           names: Sequence[str] | None = None) -> list[LibraryCompound]:
    if ids is None and names is None:
        return list(compounds)
    wanted_ids = set(ids or [])
    wanted_names = {n.lower() for n in (names or [])}
    picked = [c for c in compounds if c.id in wanted_ids or c.name.lower() in wanted_names]
    missing = (wanted_ids - {c.id for c in picked}) | (wanted_names - {c.name.lower() for c in picked})
    if missing:
        raise KeyError(f"not in this phase: {sorted(map(str, missing))}")
    return picked


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_set(analyte_set: AnalyteSet, fits: list[Fit], source: str | Path, phase: str, solvent: str,
              directory: Path = HPLCSIM_ANALYTE_DIR) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    yaml_path = directory / f"{analyte_set.name}.yaml"
    analyte_set.save(yaml_path)
    meta = {
        "analyte_set": analyte_set.name,
        "source": str(source),
        "phase": phase,
        "solvent": solvent,
        "model_note": "library log10 k = (w + m T) + (v + n T) phi fitted to log_kw + dH (1/T_ref - 1/T) - S phi "
                      f"over the method box; full weight for {K_WINDOW[0]} <= k <= {K_WINDOW[1]}, "
                      f"{OUTSIDE_WEIGHT} elsewhere; residuals in log10 k",
        "n": len(fits),
        "fits": [asdict(f) for f in fits],
        "worst_rmse_window": max(f.rmse_window for f in fits),
        "worst_max_abs_window": max(f.max_abs_window for f in fits),
    }
    meta_path = directory / f"{analyte_set.name}.import.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return yaml_path, meta_path


def import_set(phase: str, solvent: str = "acetonitrile", name: str | None = None,
               source: str | Path = DEFAULT_SOURCE, ids: Sequence[int] | None = None,
               names: Sequence[str] | None = None, instrument: InstrumentConfig | None = None,
               directory: Path = HPLCSIM_ANALYTE_DIR) -> tuple[AnalyteSet, list[Fit], Path]:
    """Fit one phase/solvent (or a subset of its compounds) and write the YAML + import record."""
    instrument = instrument or InstrumentConfig.load()
    library = load_library(source)
    compounds = select(phase_compounds(library, phase), ids, names)
    set_name = name or DEFAULT_NAMES.get((phase, solvent), f"hplcsim_{phase}_{solvent}")
    analyte_set, fits = build_set(set_name, compounds, instrument, SOLVENTS.index(solvent))
    yaml_path, _ = write_set(analyte_set, fits, source, PHASE_KEYS.get(phase, phase), solvent, directory)
    return analyte_set, fits, yaml_path


def _main(argv=None) -> None:
    from .optimum import find_optimum, write_optimum
    from .simulator import Simulator

    p = argparse.ArgumentParser(description="Fit HPLC Simulator compounds to the simulator's retention model.")
    p.add_argument("--phase", choices=("beh", "sb", "all"), default="all")
    p.add_argument("--solvent", choices=SOLVENTS, default="acetonitrile")
    p.add_argument("--name", help="analyte set name (only with a single --phase)")
    p.add_argument("--ids", type=int, nargs="+", help="library ids to keep (default: whole phase)")
    p.add_argument("--names", nargs="+", help="compound names to keep (default: whole phase)")
    p.add_argument("--source", default=str(DEFAULT_SOURCE), help="JSON/JS path or URL")
    p.add_argument("--out", type=Path, default=HPLCSIM_ANALYTE_DIR,
                   help="where the YAML goes; optimum and table only follow when this is a data root's configs/analytes")
    p.add_argument("--skip-optimum", action="store_true", help="do not run sim.optimum on the new set(s)")
    p.add_argument("--table", action="store_true", help="also write tables/<set>.csv in the set's data root")
    a = p.parse_args(argv)
    phases = ("beh", "sb") if a.phase == "all" else (a.phase,)
    if a.name and len(phases) > 1:
        p.error("--name needs a single --phase")
    for phase in phases:
        analyte_set, fits, path = import_set(phase, a.solvent, a.name, a.source, a.ids, a.names, directory=a.out)
        print(f"{analyte_set.name}: {len(fits)} compounds from {PHASE_KEYS[phase]} / {a.solvent} -> {path}")
        for f in fits:
            print(f"  {f.name:<24s} log_kw {f.log_kw:6.3f}  S {f.S:6.3f}  dH {f.dH:8.1f}   "
                  f"rmse(window) {f.rmse_window:.3f}  max|r| window {f.max_abs_window:.3f} / box {f.max_abs_all:.3f}")
        # Only a set that the loaders can find by name gets an optimum and a table next to it.
        resolvable = Path(a.out).resolve() == analyte_dir(analyte_set.name).resolve()
        if not resolvable:
            print(f"  ({a.out} is not a data root's configs/analytes: no optimum or table written)")
            continue
        if not a.skip_optimum:
            sim = Simulator.from_configs(analyte_set.name)
            opt = find_optimum(sim)
            write_optimum(analyte_set.name, opt)
            print(f"  optimum CRF {opt['crf']:.4f} at {opt['method']} (run {opt['run_time']:.2f} min)")
        if a.table:
            from .table import build_table
            build_table(analyte_set.name)


if __name__ == "__main__":
    _main()
