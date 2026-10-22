"""Precomputed method -> CRF tables for benchmarking search strategies.

    python -m sim.table                                   # tables/{easy,medium,hard}.csv
    python -m sim.table --phi-step 0.10 --tG-step 5 --T-step 5 --out tables/coarse

Each row is one method on a round-number grid over the interface variables
(phi0, phi1, tG, T) together with everything the simulator returns for it:
feasibility and reason (bounds only), CRF and its two terms, the worst observed
pair resolution, first and last observed retention time, run time, how many
analytes were observed and how many eluted after the run ended, and the
retention table of the observed peaks as JSON.
Random search samples rows; Bayesian optimisation treats the rows as the
candidate set; an LLM search proposes a method and ``lookup`` snaps it to the
nearest grid point. Analyte identities are never written.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import InstrumentConfig, table_dir
from .crf import resolution, resolution_term_truncated, time_term_truncated
from .optimum import load_optimum
from .simulator import REASONS, Simulator

TABLE_DIR = Path(__file__).resolve().parent.parent / "tables"
SET_NAMES = ("easy", "medium", "hard")

COLUMNS = [
    "method_id", "phi0", "phi1", "tG", "T",
    "feasible", "reason", "crf", "resolution_term", "time_term",
    "min_resolution", "n_unresolved_pairs", "t_first", "t_last", "run_time",
    "n_missing", "n_observed", "retention_table",
]
_INT_COLUMNS = {"method_id", "n_unresolved_pairs", "n_missing", "n_observed"}
_FLOAT_COLUMNS = {"phi0", "phi1", "tG", "T", "crf", "resolution_term", "time_term",
                  "min_resolution", "t_first", "t_last", "run_time"}


def _steps(lo: float, hi: float, step: float) -> np.ndarray:
    n = int(round((hi - lo) / step))
    return np.round(lo + step * np.arange(n + 1), 10)


def method_grid(instrument: InstrumentConfig, phi_step: float = 0.05, tG_step: float = 2.0,
                T_step: float = 5.0) -> np.ndarray:
    """All in-bounds (phi0, phi1, tG, T) grid points, shape (M, 4)."""
    b = instrument.method_bounds
    phi0_vals = _steps(b.phi0[0], b.phi0[1], phi_step)
    phi1_vals = _steps(b.phi0[0], b.phi1_max, phi_step)
    tG_vals = _steps(b.tG[0], b.tG[1], tG_step)
    T_vals = _steps(b.T[0], b.T[1], T_step)
    p0, p1, tg, tt = np.meshgrid(phi0_vals, phi1_vals, tG_vals, T_vals, indexing="ij")
    X = np.stack([p0.ravel(), p1.ravel(), tg.ravel(), tt.ravel()], axis=1)
    keep = X[:, 1] >= X[:, 0] + b.min_dphi - 1e-9
    return X[keep]


def build_rows(sim: Simulator, grid: np.ndarray, ndigits: int = 6) -> list[dict]:
    """Evaluate every grid method and assemble one dict per row (vectorised simulation)."""
    X = np.column_stack([grid[:, 0], grid[:, 1] - grid[:, 0], grid[:, 2], grid[:, 3]])
    br = sim.run_batch(X)
    cfg, n, M = sim.crf_config, sim.n_analytes, len(grid)
    obs = br.observed
    # Diagnostics over the observed peaks only: pairs with both members seen.
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(obs[:, 1:] & obs[:, :-1], resolution(br.t_r, br.sigma), np.nan)
    res_term = resolution_term_truncated(br.t_r, br.sigma, obs, n, cfg)
    t_term = time_term_truncated(br.t_r, obs, n, sim.instrument.t_end, cfg)
    has_pair = np.isfinite(R).any(axis=1)
    min_R = np.full(M, np.nan)
    if has_pair.any():
        min_R[has_pair] = np.nanmin(R[has_pair], axis=1)
    n_unresolved = (R < cfg.R_target).sum(axis=1)            # nan compares False
    n_obs = obs.sum(axis=1)
    t_first = np.where(n_obs > 0, br.t_r[:, 0], np.nan)
    t_last = np.where(n_obs > 0, br.t_r[np.arange(M), n_obs - 1], np.nan)

    def f(x: float) -> float | str:
        return "" if not np.isfinite(x) else round(float(x), ndigits)

    rows = []
    for i in range(M):
        feasible = bool(br.feasible[i])
        table = [[round(float(t), ndigits), round(float(s), ndigits)]
                 for t, s, seen in zip(br.t_r[i], br.sigma[i], obs[i]) if seen] if feasible else []
        rows.append({
            "method_id": i,
            "phi0": f(grid[i, 0]), "phi1": f(grid[i, 1]), "tG": f(grid[i, 2]), "T": f(grid[i, 3]),
            "feasible": feasible,
            "reason": REASONS[int(br.reason[i])] or "",
            "crf": f(br.crf[i]) if feasible else "",
            "resolution_term": f(res_term[i]) if feasible else "",
            "time_term": f(t_term[i]) if feasible else "",
            "min_resolution": f(min_R[i]),
            "n_unresolved_pairs": int(n_unresolved[i]),
            "t_first": f(t_first[i]), "t_last": f(t_last[i]),
            "run_time": f(br.run_time[i]),
            "n_missing": n - len(table), "n_observed": len(table),
            "retention_table": json.dumps(table, separators=(",", ":")) if feasible else "",
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def load_table(path: str | Path) -> dict[str, np.ndarray | list]:
    """Read a table back: float columns as arrays (nan where empty), retention_table as lists."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, np.ndarray | list] = {}
    for col in COLUMNS:
        vals = [r[col] for r in rows]
        if col in _FLOAT_COLUMNS:
            out[col] = np.array([float(v) if v != "" else np.nan for v in vals])
        elif col in _INT_COLUMNS:
            out[col] = np.array([int(v) for v in vals])
        elif col == "feasible":
            out[col] = np.array([v == "True" for v in vals])
        elif col == "retention_table":
            out[col] = [json.loads(v) if v else [] for v in vals]
        else:
            out[col] = np.array(vals, dtype=object)
    return out


def lookup(table: dict, phi0: float, phi1: float, tG: float, T: float,
           instrument: InstrumentConfig | None = None) -> dict:
    """Snap a proposed method to the nearest grid row (distances scaled by the box widths)."""
    instrument = instrument or InstrumentConfig.load()
    b = instrument.method_bounds
    scale = np.array([b.phi0[1] - b.phi0[0], b.phi1_max - b.phi0[0], b.tG[1] - b.tG[0], b.T[1] - b.T[0]])
    pts = np.column_stack([table["phi0"], table["phi1"], table["tG"], table["T"]])
    d = (((pts - np.array([phi0, phi1, tG, T])) / scale) ** 2).sum(axis=1)
    i = int(np.argmin(d))
    row = {}
    for col in COLUMNS:
        v = table[col][i]
        row[col] = v.item() if isinstance(v, np.generic) else v
    return row


def build_table(set_name: str, out_dir: Path | None = None, phi_step: float = 0.05, tG_step: float = 2.0,
                T_step: float = 5.0, verbose: bool = True) -> Path:
    """Write ``<set>.csv`` into the set's own data root (``tables/`` beside its YAML) unless ``out_dir`` is given."""
    sim = Simulator.from_configs(set_name)
    grid = method_grid(sim.instrument, phi_step, tG_step, T_step)
    rows = build_rows(sim, grid)
    out = Path(out_dir) if out_dir is not None else table_dir(set_name)
    out.mkdir(parents=True, exist_ok=True)
    path = write_csv(rows, out / f"{set_name}.csv")
    if verbose:
        crf = np.array([r["crf"] if r["crf"] != "" else np.nan for r in rows], dtype=float)
        feasible = np.isfinite(crf)
        truncated = np.array([r["n_missing"] > 0 for r in rows]) & feasible
        try:
            opt = load_optimum(set_name)["crf"]
        except FileNotFoundError:
            opt = float("nan")
        print(f"{set_name:10s} rows {len(rows):6d}  feasible {feasible.mean() * 100:5.1f}%  "
              f"truncated {truncated.mean() * 100:5.1f}%  "
              f"best grid CRF {np.nanmax(crf):.4f} (stored optimum {opt:.4f})  "
              f"CRF>=0.99: {int((crf >= 0.99).sum()):5d} rows  median {np.nanmedian(crf):.3f}  -> {path}")
    return path


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Write method -> CRF tables for the analyte sets.")
    p.add_argument("--sets", nargs="+", default=list(SET_NAMES))
    p.add_argument("--out", type=Path, default=None, help="default: the tables/ folder of each set's data root")
    p.add_argument("--phi-step", type=float, default=0.05)
    p.add_argument("--tG-step", type=float, default=2.0)
    p.add_argument("--T-step", type=float, default=5.0)
    a = p.parse_args(argv)
    for name in a.sets:
        build_table(name, a.out, a.phi_step, a.tG_step, a.T_step)


if __name__ == "__main__":
    _main()
