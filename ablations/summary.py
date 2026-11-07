"""Summarise the ablation campaigns against the baseline LLM, BO and random runs.

    .venv/bin/python ablations/summary.py            # prints one table, writes ablations/summary.csv
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    ("baseline", ROOT / "results" / "crf_over_runs.csv"),                   # random, bo, llm on easy/medium/hard
    ("no_theory", ROOT / "ablations" / "no_theory" / "results" / "llm_crf_over_runs.csv"),
    ("warm_start", ROOT / "ablations" / "warm_start" / "results" / "llm_crf_over_runs.csv"),
    ("fresh", ROOT / "ablations" / "fresh" / "results" / "crf_over_runs.csv"),  # random, bo, llm on easy7/medium7/hard7
]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["seed"], r["run"], r["n_missing"] = int(r["seed"]), int(r["run"]), int(r["n_missing"])
        r["crf"], r["best"], r["regret"] = float(r["crf"]), float(r["best"]), float(r["regret"])
    return rows


def summarise(label: str, rows: list[dict]) -> list[dict]:
    out = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["set"])].append(r)
    for (m, s), g in sorted(groups.items()):
        seeds = sorted({r["seed"] for r in g})
        opt = g[0]["best"] + g[0]["regret"]
        thr = 0.99 * opt
        budget = max(r["run"] for r in g)
        last = np.array([r["best"] for r in g if r["run"] == budget])
        firsts = []
        for sd in seeds:
            hit = [r["run"] for r in g if r["seed"] == sd and r["best"] >= thr]
            if hit:
                firsts.append(min(hit))
        f = np.array(firsts) if firsts else np.array([np.nan])
        found_at = {k: int(sum(1 for r in g if r["run"] == k and r["best"] >= thr)) for k in (5, 10, 15, 25)}
        out.append({
            "variant": label, "method": m, "set": s, "campaigns": len(seeds),
            "mean_best_25": round(float(last.mean()), 3), "sd_best_25": round(float(last.std(ddof=1)), 3) if len(last) > 1 else 0.0,
            "found": int((last >= thr).sum()),
            "median_inj": float(np.nanmedian(f)), "q1_inj": float(np.nanpercentile(f, 25)), "q3_inj": float(np.nanpercentile(f, 75)),
            "max_inj": float(np.nanmax(f)),
            "found_5": found_at[5], "found_10": found_at[10], "found_15": found_at[15],
            "truncated_share": round(float(np.mean([r["n_missing"] > 0 for r in g])), 3),
            "mean_crf": round(float(np.mean([r["crf"] for r in g])), 3),
            "first_crf": round(float(np.mean([r["crf"] for r in g if r["run"] == 1])), 3),
        })
    return out


def main() -> int:
    table = []
    for label, path in SOURCES:
        table += summarise(label, load(path))
    if not table:
        print("no data"); return 1
    cols = list(table[0].keys())
    with open(ROOT / "ablations" / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(table)
    print(f"{'variant':11s}{'method':8s}{'set':9s}{'n':>4s}{'best@25':>9s}{'found':>7s}{'med':>6s}{'IQR':>10s}{'max':>5s}{'@5/10/15':>11s}{'trunc':>7s}{'crf':>6s}{'crf1':>6s}")
    for r in table:
        iqr = f"{r['q1_inj']:.0f}-{r['q3_inj']:.0f}" if not np.isnan(r["q1_inj"]) else "-"
        print(f"{r['variant']:11s}{r['method']:8s}{r['set']:9s}{r['campaigns']:4d}{r['mean_best_25']:9.3f}{r['found']:7d}"
              f"{r['median_inj']:6.1f}{iqr:>10s}{r['max_inj']:5.0f}{r['found_5']:4d}/{r['found_10']:2d}/{r['found_15']:2d}"
              f"{r['truncated_share']:7.3f}{r['mean_crf']:6.3f}{r['first_crf']:6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
