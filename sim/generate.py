"""Seeded analyte-set generator with the rejection rules from the spec.

    python -m sim.generate            # writes easy, medium, hard (+ optima)
    python -m sim.generate --seed 7   # a fourth set is one command away

Construction rules
------------------
easy (6)        log_kw ~ U[1.8, 4.0], S ~ U[4, 8], dH ~ U[-3000, -1000]; a draw is rejected
                unless it differs from every accepted analyte by > 0.3 in log_kw or > 1.0 in S.
medium (8)      easy rule, then two pairs overwritten so the members are within 0.08 in
                log_kw and within 0.3 in S, with |S_a - S_b| >= 0.15 (equal S would make the
                pair inseparable at every phi).
hard (10)       medium rule with three near pairs, plus one analyte with log_kw = 1.2 and
                one with log_kw = 4.6.

Each generated set is accepted only if its known optimum reaches the CRF floor
(0.9 easy/medium, 0.7 hard); otherwise the seed is bumped and the set redrawn.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ANALYTE_DIR, AnalyteSet, CRFConfig, InstrumentConfig
from .optimum import find_optimum, write_optimum
from .simulator import Simulator


@dataclass(frozen=True)
class GeneratorRules:
    log_kw_range: tuple[float, float] = (1.8, 4.0)
    S_range: tuple[float, float] = (4.0, 8.0)
    dH_range: tuple[float, float] = (-3000.0, -1000.0)
    min_dlog_kw: float = 0.3      # distinct if log_kw differs by more than this ...
    min_dS: float = 1.0           # ... or S differs by more than this
    near_dlog_kw: float = 0.08    # near pair: |d log_kw| <= this
    near_dS_max: float = 0.30     # ... and |dS| <= this
    near_dS_min: float = 0.15     # ... but |dS| >= this
    early_log_kw: float = 1.2
    late_log_kw: float = 4.6
    max_draws: int = 100_000


RULES = GeneratorRules()
CRF_FLOOR = {"easy": 0.9, "medium": 0.9, "hard": 0.7}
SET_NAMES = ("easy", "medium", "hard")


def _distinct(cand: np.ndarray, accepted: list[np.ndarray], rules: GeneratorRules) -> bool:
    return all(abs(cand[0] - a[0]) > rules.min_dlog_kw or abs(cand[1] - a[1]) > rules.min_dS for a in accepted)


def draw_distinct(rng: np.random.Generator, n: int, rules: GeneratorRules = RULES) -> list[np.ndarray]:
    """Rejection sampling: each (log_kw, S, dH) row is redrawn until it is distinct from all accepted rows."""
    rows: list[np.ndarray] = []
    for _ in range(rules.max_draws):
        cand = np.array([rng.uniform(*rules.log_kw_range), rng.uniform(*rules.S_range), rng.uniform(*rules.dH_range)])
        if _distinct(cand, rows, rules):
            rows.append(cand)
            if len(rows) == n:
                return rows
    raise RuntimeError("could not draw a distinct set; loosen the rules")


def make_near_pairs(rng: np.random.Generator, rows: list[np.ndarray], n_pairs: int,
                    rules: GeneratorRules = RULES) -> list[np.ndarray]:
    """Overwrite rows 1, 3, 5, ... so that (0,1), (2,3), ... are near pairs with different S."""
    rows = [r.copy() for r in rows]
    for p in range(n_pairs):
        i, j = 2 * p, 2 * p + 1
        base = rows[i]
        log_kw = base[0] + rng.uniform(-rules.near_dlog_kw, rules.near_dlog_kw)
        dS = rng.uniform(rules.near_dS_min, rules.near_dS_max) * rng.choice([-1.0, 1.0])
        S = base[1] + dS
        if not rules.S_range[0] <= S <= rules.S_range[1]:
            S = base[1] - dS
        rows[j] = np.array([log_kw, S, rng.uniform(*rules.dH_range)])
    return rows


def easy(seed: int, rules: GeneratorRules = RULES) -> AnalyteSet:
    rng = np.random.default_rng(seed)
    return AnalyteSet.from_rows("easy", draw_distinct(rng, 6, rules), seed=seed)


def medium(seed: int, rules: GeneratorRules = RULES) -> AnalyteSet:
    rng = np.random.default_rng(seed)
    rows = make_near_pairs(rng, draw_distinct(rng, 8, rules), 2, rules)
    return AnalyteSet.from_rows("medium", rows, seed=seed)


def hard(seed: int, rules: GeneratorRules = RULES) -> AnalyteSet:
    rng = np.random.default_rng(seed)
    rows = make_near_pairs(rng, draw_distinct(rng, 8, rules), 3, rules)
    rows.append(np.array([rules.early_log_kw, rng.uniform(*rules.S_range), rng.uniform(*rules.dH_range)]))
    rows.append(np.array([rules.late_log_kw, rng.uniform(*rules.S_range), rng.uniform(*rules.dH_range)]))
    return AnalyteSet.from_rows("hard", rows, seed=seed)


def accept(aset: AnalyteSet, instrument: InstrumentConfig, crf_cfg: CRFConfig, floor: float) -> tuple[bool, dict]:
    """Run the optimum finder and compare its CRF with the floor for this difficulty."""
    opt = find_optimum(Simulator(instrument, aset, crf_cfg))
    return opt["crf"] >= floor, opt


def generate_all(seed: int = 0, out_dir: Path = ANALYTE_DIR, instrument: InstrumentConfig | None = None,
                 crf_cfg: CRFConfig | None = None, max_tries: int = 50, verbose: bool = True) -> dict[str, dict]:
    """Generate the three sets, each with its optimum; bump the seed until the CRF floor is met."""
    instrument = instrument or InstrumentConfig.load()
    crf_cfg = crf_cfg or CRFConfig.load()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    def produce(name: str, factory) -> AnalyteSet:
        for attempt in range(max_tries):
            s = seed + 1000 * SET_NAMES.index(name) + attempt
            aset = factory(s)
            ok, opt = accept(aset, instrument, crf_cfg, CRF_FLOOR[name])
            if verbose:
                print(f"  {name}: seed {s} -> optimum CRF {opt['crf']:.3f} {'ok' if ok else 'rejected'}")
            if ok:
                aset.save(out_dir / f"{name}.yaml")
                write_optimum(name, opt, out_dir)
                results[name] = opt
                return aset
        raise RuntimeError(f"no acceptable {name} set in {max_tries} tries")

    produce("easy", easy)
    produce("medium", medium)
    produce("hard", hard)
    return results


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate the analyte sets and their known optima.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ANALYTE_DIR)
    a = p.parse_args(argv)
    generate_all(a.seed, a.out)


if __name__ == "__main__":
    _main()
