"""Known-optimum finder: dense grid over the 4-D box, then Nelder-Mead refinement.

    python -m sim.optimum easy medium hard

writes ``configs/analytes/<set>.optimum.json`` with the best method, its CRF and
retention table. The regret reference for the efficiency metrics is this CRF; a
method within 1 % of it counts as "found the optimum".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .config import VECTOR_NAMES, Method, analyte_dir
from .simulator import Simulator

DEFAULT_GRID: tuple[int, int, int, int] = (30, 30, 30, 8)   # phi0 x dphi x tG x T
INFEASIBLE_PENALTY = 10.0


def grid_points(sim: Simulator, shape: tuple[int, int, int, int] = DEFAULT_GRID) -> np.ndarray:
    """All grid points of the box, shape (prod(shape), 4), in VECTOR_NAMES order."""
    lo, hi = sim.instrument.box()
    axes = [np.linspace(lo[i], hi[i], shape[i]) for i in range(4)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def grid_search(sim: Simulator, shape: tuple[int, int, int, int] = DEFAULT_GRID,
                chunk: int = 27_000) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the whole grid in chunks (memory-bounded). Returns (X, crf) with nan for infeasible."""
    X = grid_points(sim, shape)
    scores = np.empty(X.shape[0])
    for start in range(0, X.shape[0], chunk):
        scores[start:start + chunk] = sim.run_batch(X[start:start + chunk]).crf
    return X, scores


def _objective(sim: Simulator, lo: np.ndarray, hi: np.ndarray):
    span = hi - lo

    def f(u: np.ndarray) -> float:
        # u lives in the unit cube; clip into the box and penalise the excursion.
        u_c = np.clip(u, 0.0, 1.0)
        penalty = float(np.abs(u - u_c).sum()) * INFEASIBLE_PENALTY
        br = sim.run_batch((lo + u_c * span)[None, :])
        if not br.feasible[0]:
            return INFEASIBLE_PENALTY + penalty
        return -float(br.crf[0]) + penalty

    return f


def refine(sim: Simulator, x0: np.ndarray, step: float = 0.05, maxiter: int = 2000) -> tuple[np.ndarray, float]:
    """Nelder-Mead from x0 (box coordinates) on the closed-form simulator. Returns (x_best, crf_best)."""
    lo, hi = sim.instrument.box()
    span = hi - lo
    u0 = (np.asarray(x0, dtype=float) - lo) / span
    simplex = [u0]
    for i in range(4):
        v = u0.copy()
        v[i] = v[i] + step if v[i] + step <= 1.0 else v[i] - step
        simplex.append(v)
    res = minimize(_objective(sim, lo, hi), u0, method="Nelder-Mead",
                   options={"initial_simplex": np.array(simplex), "xatol": 1e-5, "fatol": 1e-7,
                            "maxiter": maxiter, "maxfev": maxiter * 2})
    u_best = np.clip(res.x, 0.0, 1.0)
    x_best = lo + u_best * span
    crf_best = float(sim.run_batch(x_best[None, :]).crf[0])
    # Never return something worse than the starting point.
    crf0 = float(sim.run_batch(np.asarray(x0, dtype=float)[None, :]).crf[0])
    if not np.isfinite(crf_best) or crf_best < crf0:
        return np.asarray(x0, dtype=float), crf0
    return x_best, crf_best


def find_optimum(sim: Simulator, shape: tuple[int, int, int, int] = DEFAULT_GRID, top: int = 20) -> dict[str, Any]:
    """Grid search, refine the ``top`` grid points, keep the best. Deterministic."""
    X, scores = grid_search(sim, shape)
    finite = np.isfinite(scores)
    if not finite.any():
        raise RuntimeError("no feasible method on the grid")
    order = np.argsort(np.where(finite, -scores, np.inf), kind="stable")[:top]
    best_x, best_crf = X[order[0]], float(scores[order[0]])
    for idx in order:
        x, c = refine(sim, X[idx])
        if c > best_crf:
            best_x, best_crf = x, c
    method = Method.from_vector(best_x)
    result = sim.run(method)
    return {
        "analyte_set": sim.analyte_set_name,
        "method": method.model_dump(),
        "vector": dict(zip(VECTOR_NAMES, (float(v) for v in best_x))),
        "crf": result.crf,
        "run_time": result.run_time,
        "retention_table": result.retention_table,
        "grid": list(shape),
        "grid_best_crf": float(scores[order[0]]),
    }


def optimum_path(set_name: str, directory: Path | None = None) -> Path:
    """``<set>.optimum.json`` next to the set's YAML (whichever data root holds it) unless ``directory`` says otherwise."""
    return Path(directory if directory is not None else analyte_dir(set_name)) / f"{set_name}.optimum.json"


def write_optimum(set_name: str, opt: dict[str, Any], directory: Path | None = None) -> Path:
    path = optimum_path(set_name, directory)
    with open(path, "w") as fh:
        json.dump(opt, fh, indent=2)
    return path


def load_optimum(set_name: str, directory: Path | None = None) -> dict[str, Any]:
    with open(optimum_path(set_name, directory)) as fh:
        return json.load(fh)


def _main(argv=None) -> None:
    import argparse
    import time

    p = argparse.ArgumentParser(description="Find and store the known optimum of analyte sets.")
    p.add_argument("sets", nargs="+")
    p.add_argument("--grid", type=int, nargs=4, default=list(DEFAULT_GRID), metavar=("PHI0", "DPHI", "TG", "T"))
    p.add_argument("--top", type=int, default=20)
    a = p.parse_args(argv)
    for name in a.sets:
        sim = Simulator.from_configs(name)
        t = time.perf_counter()
        opt = find_optimum(sim, tuple(a.grid), a.top)
        path = write_optimum(name, opt)
        print(f"{name}: CRF {opt['crf']:.4f} (grid {opt['grid_best_crf']:.4f}) "
              f"method {opt['method']} run {opt['run_time']:.2f} min "
              f"[{time.perf_counter() - t:.1f} s] -> {path.name}")


if __name__ == "__main__":
    _main()
