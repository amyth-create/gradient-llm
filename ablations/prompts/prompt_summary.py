"""Summarise the hard-set prompt study: five prompt variants x two models, 50 campaigns each.

    .venv/bin/python ablations/prompts/prompt_summary.py     # prints one table, writes ablations/prompts/summary.csv

Sources: the main campaigns (results/, deployments/alt-model/results/), the no-theory ablation
(ablations/no_theory/) and ablations/prompts/<variant>/<model>/results/.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "ablations"))
from summary import load, summarise  # noqa: E402

MODELS = ("gpt-6-astra", "gpt-5.4")
VARIANTS = ("full", "no_theory", "no_howto", "no_score", "minimal")


def source(model: str, variant: str) -> Path:
    if variant == "full":
        return ROOT / "results" / "crf_over_runs.csv" if model == "gpt-6-astra" \
            else ROOT / "deployments" / "alt-model" / "results" / "llm_crf_over_runs.csv"
    if variant == "no_theory" and model == "gpt-6-astra":
        return ROOT / "ablations" / "no_theory" / "results" / "llm_crf_over_runs.csv"
    return HERE / variant / model / "results" / "llm_crf_over_runs.csv"


def transcript_stats(path: Path) -> dict:
    """Repeat re-asks, tokens, latency and the opening method over the hard-set transcripts in a results folder."""
    files = sorted((path.parent / "llm_transcripts").glob("hard_seed*.jsonl"))
    rows = [json.loads(l) for f in files for l in open(f)]
    own = [r for r in rows if not r.get("warm_start")]
    if not own:
        return {}
    first = [r for r in own if r["injection"] == 1]
    opening = Counter((r["run_as"]["phi0"], r["run_as"]["phi1"], r["run_as"]["tG"], r["run_as"]["T"]) for r in first)
    (p0, p1, tg, t), n_open = opening.most_common(1)[0]
    moves = [r for r in own if r["injection"] > 1]
    return {
        "repeats_per_campaign": round(sum(r.get("repeat_attempts", 0) for r in own) / len(files), 2),
        "parse_errors": sum(r.get("parse_errors", 0) for r in own),
        "prompt_tokens_per_campaign": int(sum(r["usage"]["prompt_tokens"] for r in own) / len(files)),
        "completion_tokens_per_call": int(np.mean([r["usage"]["completion_tokens"] for r in own])),
        "latency_s": round(float(np.mean([r["latency_s"] for r in own])), 1),
        "opening": f"{p0:.2f}-{p1:.2f}/{tg:.0f}min/{t:.0f}C x{n_open}",
    }


def main() -> int:
    table = []
    for model in MODELS:
        for variant in VARIANTS:
            path = source(model, variant)
            rows = [r for r in load(path) if r["method"] == "llm" and r["set"] == "hard"]
            if not rows:
                continue
            for rec in summarise(f"{model}/{variant}", rows):
                rec["model"], rec["prompt"] = model, variant
                rec.update(transcript_stats(path))
                table.append(rec)
    if not table:
        print("no data"); return 1
    cols = list(table[0].keys())
    with open(HERE / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(table)
    print(f"{'model':12s}{'prompt':10s}{'n':>3s}{'found':>6s}{'med':>5s}{'IQR':>8s}{'max':>4s}{'@5/10/15':>10s}{'trunc':>6s}{'crf1':>6s}{'crf':>6s}"
          f"{'rep':>5s}{'ptok/camp':>10s}{'ctok':>5s}{'lat':>5s}  opening")
    for r in table:
        print(f"{r['model']:12s}{r['prompt']:10s}{r['campaigns']:3d}{r['found']:6d}{r['median_inj']:5.1f}"
              f"{r['q1_inj']:4.0f}-{r['q3_inj']:<3.0f}{r['max_inj']:4.0f}{r['found_5']:4d}/{r['found_10']:2d}/{r['found_15']:2d}"
              f"{r['truncated_share']:6.3f}{r['first_crf']:6.3f}{r['mean_crf']:6.3f}{r.get('repeats_per_campaign', 0):5.2f}"
              f"{r.get('prompt_tokens_per_campaign', 0):10d}{r.get('completion_tokens_per_call', 0):5d}{r.get('latency_s', 0):5.1f}  {r.get('opening', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
