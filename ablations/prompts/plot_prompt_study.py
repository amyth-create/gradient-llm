"""Mean best CRF against injection for the five prompt variants on the hard set, one panel per model.

    .venv/bin/python ablations/prompts/plot_prompt_study.py   -> ablations/prompts/prompt_study.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
from prompt_summary import MODELS, VARIANTS, source  # noqa: E402
from ablations.summary import load  # noqa: E402
from optimizers.plot import GRID, INK, _style  # noqa: E402

COLORS = {"full": "#2e9e5b", "no_theory": "#2a78d6", "no_howto": "#eb6834", "no_score": "#8e63d6", "minimal": "#52514e"}
LABELS = {"full": "full prompt", "no_theory": "without retention model", "no_howto": "without procedural rules",
          "no_score": "without the CRF formula", "minimal": "minimal (instrument, bounds, format)"}


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, model in zip(axes, MODELS):
        for v in VARIANTS:
            rows = [r for r in load(source(model, v)) if r["method"] == "llm" and r["set"] == "hard"]
            if not rows:
                continue
            seeds = sorted({r["seed"] for r in rows})
            best = np.array([[r["best"] for r in sorted((x for x in rows if x["seed"] == s), key=lambda x: x["run"])] for s in seeds])
            opt = rows[0]["best"] + rows[0]["regret"]
            found = int((best[:, -1] >= 0.99 * opt).sum())
            x = np.arange(1, best.shape[1] + 1)
            ax.plot(x, best.mean(0), color=COLORS[v], linewidth=2, label=f"{LABELS[v]} · {found}/{len(seeds)}")
        ax.axhline(1.0, color=INK, linewidth=1, linestyle="--")
        ax.set_ylim(0.6, 1.02)
        ax.set_xticks([1, 5, 10, 15, 20, 25])
        _style(ax, f"{model} · hard set · mean best CRF so far, 50 campaigns per prompt", "injection", "best CRF so far")
        ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    out = HERE / "prompt_study.png"
    fig.savefig(out, dpi=160)
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
