"""Mean best CRF against injection: the LLM search warm-started on BO's eight random rows, beside BO and random
search from the same rows, 50 campaigns per set.

    .venv/bin/python ablations/warm_start/plot_warm_start.py   -> ablations/warm_start/results/warm_start.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from ablations.summary import load  # noqa: E402
from optimizers.plot import COLORS, GRID, INK, _style  # noqa: E402

SETS = ("easy", "medium", "hard")
SERIES = [("random", ROOT / "results" / "crf_over_runs.csv", "Random search"),
          ("bo", ROOT / "results" / "crf_over_runs.csv", "Bayesian optimisation (GP-EI)"),
          ("llm", HERE / "results" / "llm_crf_over_runs.csv", "LLM search, warm-started on the same 8 rows")]


def curves(rows):
    seeds = sorted({r["seed"] for r in rows})
    best = np.array([[r["best"] for r in sorted((x for x in rows if x["seed"] == s), key=lambda x: x["run"])] for s in seeds])
    opt = rows[0]["best"] + rows[0]["regret"]
    return best, int((best[:, -1] >= 0.99 * opt).sum()), len(seeds)


def main() -> int:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, s in zip(axes, SETS):
        for m, path, label in SERIES:
            rows = [r for r in load(path) if r["method"] == m and r["set"] == s]
            best, found, n = curves(rows)
            x = np.arange(1, best.shape[1] + 1)
            ax.plot(x, best.mean(0), color=COLORS[m], linewidth=2, label=f"{label} · {found}/{n}")
        ax.axhline(1.0, color=INK, linewidth=1, linestyle="--")
        ax.axvline(8.5, color=INK, linewidth=1, linestyle=":")
        ax.set_ylim(0.3, 1.02); ax.set_xticks([1, 5, 8, 10, 15, 20, 25])
        _style(ax, f"{s} set · 50 campaigns per search", "injection", "best CRF so far" if s == "easy" else "")
        ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.suptitle("Injections 1 to 8 are the same seeded random rows for all three searches; the dotted line marks the hand-over",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    out = HERE / "results" / "warm_start.png"
    fig.savefig(out, dpi=160); print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
