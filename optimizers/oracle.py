"""Table-backed oracle: one injection per query, a fixed budget, regret against the stored optimum."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sim.optimum import load_optimum
from sim.config import table_dir as set_table_dir
from sim.table import load_table

FEATURES: tuple[str, ...] = ("phi0", "phi1", "tG", "T")


@dataclass(frozen=True)
class Observation:
    run: int            # 1-based injection number
    index: int          # table row (method_id)
    phi0: float
    phi1: float
    tG: float
    T: float
    feasible: bool      # False only for an out-of-bounds method (none in the default tables)
    crf: float          # nan when infeasible
    reason: str
    n_missing: int      # analytes that eluted after the run ended; the CRF scores them as co-eluting
    best: float         # best feasible CRF so far (nan until the first feasible injection)
    regret: float       # optimum - best


@dataclass
class SearchRun:
    method: str
    set_name: str
    seed: int
    budget: int
    optimum: float
    observations: list[Observation] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.observations)

    def indices(self) -> list[int]:
        return [o.index for o in self.observations]

    def crf_per_run(self) -> np.ndarray:
        return np.array([o.crf for o in self.observations], dtype=float)

    def n_missing_per_run(self) -> np.ndarray:
        return np.array([o.n_missing for o in self.observations], dtype=int)

    def best_curve(self) -> np.ndarray:
        return np.array([o.best for o in self.observations], dtype=float)

    def regret_curve(self) -> np.ndarray:
        return np.array([o.regret for o in self.observations], dtype=float)

    def runs_to_optimum(self, tol: float = 0.01) -> int | None:
        """First injection at which the best CRF is within ``tol`` (relative) of the optimum."""
        for o in self.observations:
            if np.isfinite(o.best) and o.best >= (1.0 - tol) * self.optimum:
                return o.run
        return None

    def to_rows(self) -> list[dict]:
        return [{"method": self.method, "set": self.set_name, "seed": self.seed, **o.__dict__}
                for o in self.observations]


class TableOracle:
    """The discrete search space: every in-bounds grid method of one analyte set.

    ``X`` holds the raw features in FEATURES order, ``X_unit`` the same scaled to
    the unit cube (what a surrogate model should see). Querying a row costs one
    injection; a row may be queried once per run.
    """

    def __init__(self, set_name: str, table_dir: str | Path | None = None):
        self.set_name = set_name
        directory = Path(table_dir) if table_dir is not None else set_table_dir(set_name)
        table = load_table(directory / f"{set_name}.csv")
        self.X = np.column_stack([table[f] for f in FEATURES]).astype(float)
        lo, hi = self.X.min(axis=0), self.X.max(axis=0)
        self.X_unit = (self.X - lo) / np.where(hi > lo, hi - lo, 1.0)
        self.crf = table["crf"].astype(float)
        self.feasible = table["feasible"].astype(bool)
        self.reason = table["reason"]
        self.n_missing = table["n_missing"].astype(int)
        self.retention_table = table["retention_table"]
        self.optimum = float(load_optimum(set_name)["crf"])

    @property
    def n_candidates(self) -> int:
        return self.X.shape[0]

    def new_run(self, method: str, seed: int, budget: int) -> SearchRun:
        return SearchRun(method=method, set_name=self.set_name, seed=seed, budget=budget, optimum=self.optimum)

    def query(self, run: SearchRun, index: int) -> Observation:
        """Spend one injection on table row ``index`` and append the observation to ``run``."""
        index = int(index)
        if len(run) >= run.budget:
            raise RuntimeError(f"budget of {run.budget} injections exhausted")
        if index in run.indices():
            raise ValueError(f"row {index} was already queried in this run")
        crf = float(self.crf[index])
        feasible = bool(self.feasible[index])
        prev_best = run.observations[-1].best if run.observations else np.nan
        best = np.nanmax([prev_best, crf]) if (feasible or np.isfinite(prev_best)) else np.nan
        obs = Observation(
            run=len(run) + 1, index=index,
            phi0=float(self.X[index, 0]), phi1=float(self.X[index, 1]),
            tG=float(self.X[index, 2]), T=float(self.X[index, 3]),
            feasible=feasible, crf=crf if feasible else float("nan"),
            reason=str(self.reason[index]), n_missing=int(self.n_missing[index]),
            best=float(best), regret=float(self.optimum - best) if np.isfinite(best) else float("nan"),
        )
        run.observations.append(obs)
        return obs


def format_run(run: SearchRun) -> str:
    """Human-readable injection log."""
    lines = [f"== {run.set_name} · {run.method} · seed {run.seed} · budget {run.budget} · optimum CRF {run.optimum:.3f}",
             f"{'run':>3s}  {'phi0':>5s} {'phi1':>5s} {'tG':>5s} {'T':>4s}  {'feasible':8s} {'missing':>7s} "
             f"{'crf':>6s} {'best':>6s} {'regret':>6s}"]
    for o in run.observations:
        crf = f"{o.crf:6.3f}" if o.feasible else "   nan"
        best = f"{o.best:6.3f}" if np.isfinite(o.best) else "     -"
        regret = f"{o.regret:6.3f}" if np.isfinite(o.regret) else "     -"
        flag = "yes" if o.feasible else "NO"
        missing = f"{o.n_missing:7d}" if o.feasible else "      -"
        lines.append(f"{o.run:3d}  {o.phi0:5.2f} {o.phi1:5.2f} {o.tG:5.1f} {o.T:4.0f}  {flag:8s} {missing} {crf} {best} {regret}")
    hit = run.runs_to_optimum()
    tail = f"within 1 % of the optimum after {hit} injections" if hit else "did not reach 1 % of the optimum"
    lines.append(f"   -> best CRF {run.best_curve()[-1]:.3f}, {tail}")
    return "\n".join(lines)
