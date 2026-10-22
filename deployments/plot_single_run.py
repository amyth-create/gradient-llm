"""One campaign per deployment, overlaid: CRF of each injection and the best so far.

    .venv/bin/python deployments/plot_single_run.py --set hard --seed 0 \
        --run gpt-6-astra results/llm_crf_over_runs.csv \
        --run gpt-5.4 deployments/alt-model/results/llm_crf_over_runs.csv \
        --out deployments/alt-model/results/hard_seed0_gpt-5.4_vs_gpt-6-astra.png

Reads the ``llm_crf_over_runs.csv`` each deployment's repeats runner writes (the main
model's in ``results/``, a second deployment's in ``deployments/<name>/results/``).
Same marks as ``optimizers.plot``: dots for the CRF of each injection, a step line for
the best so far, the known optimum dashed.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from optimizers.plot import GRID, INFEASIBLE, INK, TRUNCATED_MARKER, _style  # noqa: E402
from sim.optimum import load_optimum  # noqa: E402

# Categorical slots 1 and 2 of the project's palette, in fixed order: first --run gets blue.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


def load_campaign(path: Path, set_name: str, seed: int) -> dict[str, np.ndarray]:
    rows = [r for r in csv.DictReader(open(path)) if r["set"] == set_name and int(r["seed"]) == seed]
    if not rows:
        sys.exit(f"no rows for set {set_name!r}, seed {seed} in {path}")
    rows.sort(key=lambda r: int(r["run"]))
    return {
        "x": np.array([int(r["run"]) for r in rows]),
        "crf": np.array([float(r["crf"]) for r in rows]),
        "best": np.array([float(r["best"]) for r in rows]),
        "feasible": np.array([r["feasible"] == "True" for r in rows]),
        "truncated": np.array([int(r["n_missing"] or 0) > 0 for r in rows]),
    }


def optimum_of(set_name: str) -> float:
    return float(load_optimum(set_name)["crf"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--set", default="hard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "CSV"), required=True,
                   help="a deployment's label and its llm_crf_over_runs.csv; give two or three")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if len(a.run) > len(SERIES):
        sys.exit(f"at most {len(SERIES)} runs")

    optimum = optimum_of(a.set)
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    handles: dict[str, object] = {}
    for (label, path), color in zip(a.run, SERIES):
        c = load_campaign(Path(path), a.set, a.seed)
        x, crf, best, feas, trunc = c["x"], c["crf"], c["best"], c["feasible"], c["truncated"]
        handles[f"{label}: CRF of each injection"] = ax.scatter(
            x[feas & ~trunc], crf[feas & ~trunc], s=34, color=color, edgecolors="white", linewidths=0.8, zorder=3)
        handles[f"{label}: best so far"] = ax.step(x, best, where="post", color=color, linewidth=2, zorder=2)[0]
        if trunc.any():
            handles["truncated run (peaks missing), at its observed CRF"] = ax.scatter(
                x[trunc], crf[trunc], marker=TRUNCATED_MARKER, s=44, color=INFEASIBLE, edgecolors="white", zorder=3)
        if (~feas).any():
            handles["infeasible method"] = ax.scatter(
                x[~feas], np.full((~feas).sum(), np.nanmin(crf) - 0.1), marker="x", s=44, color=INFEASIBLE,
                linewidths=1.6, zorder=3)
    handles["known optimum"] = ax.axhline(optimum, color=INK, linewidth=1, linestyle="--")

    _style(ax, f"{a.set} set, one 25-injection campaign (seed {a.seed}), optimum CRF {optimum:.3f}", "injection", "CRF")
    ax.set_xticks(range(1, 26, 2))
    fig.suptitle(f"LLM search on the {a.set} set: {' vs '.join(l for l, _ in a.run)}, same harness and prompt",
                 x=0.01, ha="left", fontsize=12, color="#0b0b0b")
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
