"""The only public entry point: ``Simulator.run(method) -> Result``.

A deterministic function from one gradient method to one retention table and
one CRF score. Optimizers query it one method at a time and never see the
analyte parameters behind it. ``run_batch`` evaluates many methods at once
(vectorised) for the optimum search.

The run ends at ``instrument.t_end`` = t_max - t_eq. Peaks eluting later are
not observed: they are dropped from the table, counted in ``n_missing`` and
scored by ``crf_truncated`` as if fully co-eluting, with the time term charged
at t_end. Feasibility is a bounds check only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import VECTOR_NAMES, AnalyteSet, CRFConfig, InstrumentConfig, Method, Result
from .crf import crf_truncated, sort_table
from .gradient import t_r_closed
from .width import sigma_from_closed_form

# Feasibility reason codes.
OK = 0
INFEASIBLE_BOUNDS = 1
REASONS = {
    OK: None,
    INFEASIBLE_BOUNDS: "bounds: method outside the instrument box or phi1 - phi0 below the minimum",
}
_TOL = 1e-9


@dataclass(frozen=True)
class BatchResult:
    """Vectorised outputs for M methods on n analytes.

    ``t_r`` and ``sigma`` are full width (every analyte, sorted by t_r);
    ``observed`` masks the peaks that elute by t_end, a prefix of each row.
    Rows of infeasible (out-of-bounds) methods keep their raw retention data
    for diagnostics but have ``crf = nan`` and ``feasible = False``.
    """

    t_r: np.ndarray        # (M, n) sorted retention times
    sigma: np.ndarray      # (M, n) sorted alongside t_r
    observed: np.ndarray   # (M, n) bool, t_r <= t_end
    n_missing: np.ndarray  # (M,) int, analytes not observed
    crf: np.ndarray        # (M,)
    run_time: np.ndarray   # (M,) min(t_r last, t_end) + t_eq
    feasible: np.ndarray   # (M,) bool
    reason: np.ndarray     # (M,) int code, see REASONS


class Simulator:
    def __init__(self, instrument: InstrumentConfig, analytes: AnalyteSet, crf: CRFConfig):
        self.instrument = instrument
        self.crf_config = crf
        self._analytes = analytes
        self._log_kw, self._S, self._dH = analytes.arrays()
        self.n_analytes = len(analytes)

    @classmethod
    def from_configs(cls, analyte_set: str | Path, instrument: str | Path | None = None,
                     crf: str | Path | None = None) -> "Simulator":
        return cls(InstrumentConfig.load(instrument), AnalyteSet.load(analyte_set), CRFConfig.load(crf))

    @property
    def analyte_set_name(self) -> str:
        return self._analytes.name

    # ------------------------------------------------------------------ #
    def bounds(self) -> dict[str, tuple[float, float]]:
        """Box bounds on (phi0, dphi, tG, T) — what an optimizer is allowed to see."""
        return self.instrument.bounds()

    # ------------------------------------------------------------------ #
    def _check_bounds(self, phi0, dphi, tG, T) -> np.ndarray:
        b = self.instrument.method_bounds
        phi1 = phi0 + dphi
        ok = (phi0 >= b.phi0[0] - _TOL) & (phi0 <= b.phi0[1] + _TOL)
        ok &= dphi >= b.min_dphi - _TOL
        ok &= phi1 <= b.phi1_max + _TOL
        ok &= (tG >= b.tG[0] - _TOL) & (tG <= b.tG[1] + _TOL)
        ok &= (T >= b.T[0] - _TOL) & (T <= b.T[1] + _TOL)
        return ok

    def run_batch(self, X: np.ndarray) -> BatchResult:
        """Evaluate M methods given as an (M, 4) array over VECTOR_NAMES = (phi0, dphi, tG, T)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != len(VECTOR_NAMES):
            raise ValueError(f"expected shape (M, {len(VECTOR_NAMES)}), got {X.shape}")
        phi0, dphi, tG, T = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        phi1 = phi0 + dphi
        instr = self.instrument
        t_end = instr.t_end

        in_bounds = self._check_bounds(phi0, dphi, tG, T)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            cf = t_r_closed(phi0, phi1, tG, T, self._log_kw, self._S, self._dH, instr)
            sig = sigma_from_closed_form(cf.k_e, cf.b, cf.regime, instr)
            t_r, sig = sort_table(cf.t_r, sig)
            # A peak is observed if it elutes by t_end with a finite width; nan/inf never is.
            observed = (t_r <= t_end + _TOL) & np.isfinite(sig)
            n_missing = self.n_analytes - observed.sum(axis=1)
            run_time = np.where(n_missing > 0, t_end, t_r[:, -1]) + instr.t_eq_min
            score = crf_truncated(t_r, sig, observed, self.n_analytes, t_end, self.crf_config)

        reason = np.full(X.shape[0], OK, dtype=np.int8)
        reason[~in_bounds] = INFEASIBLE_BOUNDS
        feasible = reason == OK
        score = np.where(feasible, score, np.nan)
        return BatchResult(t_r=t_r, sigma=sig, observed=observed, n_missing=n_missing, crf=score,
                           run_time=run_time, feasible=feasible, reason=reason)

    def run(self, method: Method) -> Result:
        """Simulate one method. Never raises for a bad method: reports ``feasible = False``."""
        br = self.run_batch(method.to_vector()[None, :])
        feasible = bool(br.feasible[0])
        table = ([(float(t), float(s)) for t, s, seen in zip(br.t_r[0], br.sigma[0], br.observed[0]) if seen]
                 if feasible else [])
        return Result(
            retention_table=table,
            crf=float(br.crf[0]) if feasible else float("nan"),
            run_time=float(br.run_time[0]),
            feasible=feasible,
            reason=REASONS[int(br.reason[0])],
            n_missing=self.n_analytes - len(table),
        )


def _main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run one gradient method on an analyte set.")
    p.add_argument("analyte_set", help="name under configs/analytes or a path to a .yaml")
    p.add_argument("--phi0", type=float, default=0.05)
    p.add_argument("--phi1", type=float, default=0.95)
    p.add_argument("--tG", type=float, default=10.0)
    p.add_argument("--T", type=float, default=40.0)
    a = p.parse_args(argv)
    sim = Simulator.from_configs(a.analyte_set)
    res = sim.run(Method(phi0=a.phi0, phi1=a.phi1, tG=a.tG, T=a.T))
    print(res.to_json())


if __name__ == "__main__":
    _main()
