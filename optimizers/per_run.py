"""CRF per injection for one search method, one panel per analyte set.

    python -m optimizers.per_run --method bo --seed 0      # results/bo_crf_per_run.png
    python -m optimizers.per_run --method random --seed 0  # results/random_crf_per_run.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .bo import BOConfig, bo_search  # noqa: E402
from .oracle import SearchRun, TableOracle  # noqa: E402
from .plot import COLORS, GRID, INFEASIBLE, INFEASIBLE_LABEL, INK, LABELS, TRUNCATED_LABEL, TRUNCATED_MARKER, _style  # noqa: E402
from .random_search import random_search  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SET_NAMES = ("easy", "medium", "hard")
METHODS = {"random": random_search, "bo": bo_search}


def plot_crf_per_run(runs: dict[str, SearchRun], path: str | Path, n_init: int | None = None) -> Path:
    """One panel per set: CRF of each injection, best so far, truncated and infeasible marks, optimum."""
    method = next(iter(runs.values())).method
    budget = next(iter(runs.values())).budget
    c = COLORS[method]
    x = np.arange(1, budget + 1)
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.8), layout="constrained", sharey=True)
    axes = np.atleast_1d(axes)
    all_crf = np.concatenate([r.crf_per_run() for r in runs.values()])
    y_floor = min(-0.15, float(np.nanmin(all_crf)) - 0.1) if np.isfinite(all_crf).any() else -0.15
    y_mark = y_floor - 0.08
    for ax, (set_name, run) in zip(axes, runs.items()):
        crf = run.crf_per_run()
        feas = np.isfinite(crf)
        trunc = feas & (run.n_missing_per_run() > 0)
        ax.axhline(run.optimum, color=INK, linewidth=1, linestyle="--")
        if n_init:
            ax.axvline(n_init + 0.5, color=INK, linewidth=1, linestyle=":")
        ax.step(x, run.best_curve(), where="post", color=c, linewidth=2, zorder=2)
        ax.scatter(x[feas & ~trunc], crf[feas & ~trunc], s=34, color=c, edgecolors="white", linewidths=0.8, zorder=3)
        if trunc.any():
            ax.scatter(x[trunc], crf[trunc], marker=TRUNCATED_MARKER, s=44, color=INFEASIBLE, edgecolors="white",
                       linewidths=0.8, zorder=3)
        if (~feas).any():
            ax.scatter(x[~feas], np.full((~feas).sum(), y_mark), marker="x", s=42, color=INFEASIBLE, linewidths=1.6, zorder=3)
        best = run.best_curve()[-1]
        hit = run.runs_to_optimum()
        found = f"optimum found at injection {hit}" if hit else f"regret {run.optimum - best:.3f}"
        _style(ax, f"{set_name} · best CRF {best:.3f} · {found}", "injection", "CRF")
        ax.set_xlim(0.5, budget + 0.5)
        ax.set_ylim(y_mark - 0.1, 1.08)
        ax.set_xticks(np.arange(0, budget + 1, 5)[1:])
    handles = [
        Line2D([], [], marker="o", color=c, markeredgecolor="white", markersize=8, linestyle="none", label="CRF of each injection"),
        Line2D([], [], color=c, linewidth=2, label="best so far"),
        Line2D([], [], marker=TRUNCATED_MARKER, color=INFEASIBLE, markeredgecolor="white", markersize=8, linestyle="none",
               label=TRUNCATED_LABEL),
        Line2D([], [], marker="x", color=INFEASIBLE, markersize=8, markeredgewidth=1.6, linestyle="none",
               label=INFEASIBLE_LABEL),
        Line2D([], [], color=INK, linestyle="--", label="known optimum"),
    ]
    if n_init:
        handles.append(Line2D([], [], color=INK, linestyle=":", label=f"first {n_init} injections are random; surrogate takes over after"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    seed = next(iter(runs.values())).seed
    fig.suptitle(f"{LABELS[method]} · CRF per injection · seed {seed} · {budget} injections", x=0.01, ha="left",
                 fontsize=12, color="#0b0b0b")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description="CRF per injection for one method, one panel per set.")
    p.add_argument("--method", default="bo", choices=list(METHODS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=25)
    p.add_argument("--sets", nargs="+", default=list(SET_NAMES))
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    runs = {name: METHODS[a.method](TableOracle(name), budget=a.budget, seed=a.seed) for name in a.sets}
    out = a.out or RESULTS_DIR / f"{a.method}_crf_per_run.png"
    n_init = BOConfig().n_init if a.method == "bo" else None
    print(f"-> {plot_crf_per_run(runs, out, n_init=n_init)}")
    for name, run in runs.items():
        crf = " ".join(f"{v:.2f}" if np.isfinite(v) else " nan" for v in run.crf_per_run())
        print(f"{name:7s} {crf}")


if __name__ == "__main__":
    _main()
