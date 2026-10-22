"""Optimisation efficiency of one search method across the analyte sets: CRF per injection.

    python -m optimizers.efficiency --method random            # results/random_efficiency.png
    python -m optimizers.efficiency --method bo --seeds 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .bo import bo_search  # noqa: E402
from .oracle import SearchRun, TableOracle  # noqa: E402
from .plot import COLORS, GRID, INFEASIBLE, INFEASIBLE_LABEL, INK, LABELS, TRUNCATED_LABEL, TRUNCATED_MARKER, _style  # noqa: E402
from .random_search import random_search  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SET_NAMES = ("easy", "medium", "hard")
METHODS = {"random": random_search, "bo": bo_search}
CHECKPOINTS = (5, 10, 15, 20, 25)


def plot_efficiency(runs_by_set: dict[str, list[SearchRun]], path: str | Path, shown_seed: int = 0) -> Path:
    """One panel per set: CRF of each injection and best-so-far for one seed, mean best-so-far over all seeds."""
    sets = list(runs_by_set)
    method = runs_by_set[sets[0]][0].method
    budget = runs_by_set[sets[0]][0].budget
    n_seeds = len(runs_by_set[sets[0]])
    color = COLORS[method]
    x = np.arange(1, budget + 1)
    handles: dict[str, object] = {}

    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 5.0), layout="constrained", sharey=True)
    axes = np.atleast_1d(axes)
    all_crf = np.concatenate([r.crf_per_run() for runs in runs_by_set.values() for r in runs])
    y_floor = min(-0.15, float(np.nanmin(all_crf)) - 0.1)
    y_mark = y_floor - 0.08
    for ax, name in zip(axes, sets):
        runs = runs_by_set[name]
        run = next((r for r in runs if r.seed == shown_seed), runs[0])
        crf = run.crf_per_run()
        feas = np.isfinite(crf)
        trunc = feas & (run.n_missing_per_run() > 0)
        handles[f"CRF of each injection (seed {shown_seed})"] = ax.scatter(
            x[feas & ~trunc], crf[feas & ~trunc], s=34, color=color, edgecolors="white", linewidths=0.8, zorder=3)
        if trunc.any():
            handles[TRUNCATED_LABEL] = ax.scatter(
                x[trunc], crf[trunc], marker=TRUNCATED_MARKER, s=44, color=INFEASIBLE, edgecolors="white",
                linewidths=0.8, zorder=3)
        if (~feas).any():
            handles[INFEASIBLE_LABEL] = ax.scatter(
                x[~feas], np.full((~feas).sum(), y_mark), marker="x", s=44, color=INFEASIBLE, linewidths=1.6, zorder=3)
        handles[f"best so far (seed {shown_seed})"] = ax.step(
            x, run.best_curve(), where="post", color=color, linewidth=2.2, zorder=2)[0]
        B = np.vstack([r.best_curve() for r in runs])
        handles[f"best so far, mean of {n_seeds} seeds"] = ax.plot(
            x, np.nanmean(B, axis=0), color=color, linewidth=1.6, linestyle="--", zorder=2)[0]
        handles["known optimum"] = ax.axhline(run.optimum, color=INK, linewidth=1, linestyle=":")
        hit = run.runs_to_optimum()
        tail = f"found at injection {hit}" if hit else "optimum not found"
        _style(ax, f"{name} · best {run.best_curve()[-1]:.3f} · {tail}", "injection", "CRF" if ax is axes[0] else "")
        ax.set_xlim(0.5, budget + 0.5)
        ax.set_ylim(y_mark - 0.1, 1.08)
    fig.suptitle(f"{LABELS[method]} · optimisation efficiency as CRF per injection · {budget} injections",
                 x=0.01, ha="left", fontsize=12, color="#0b0b0b")
    fig.legend(list(handles.values()), list(handles.keys()), loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def checkpoint_table(runs_by_set: dict[str, list[SearchRun]], checkpoints=CHECKPOINTS) -> str:
    """Mean best CRF (and share of seeds within 1 % of the optimum) after n injections."""
    lines = [f"{'set':8s}" + "".join(f"{f'@{c}':>14s}" for c in checkpoints)]
    for name, runs in runs_by_set.items():
        B = np.vstack([r.best_curve() for r in runs])
        opt = runs[0].optimum
        cells = []
        for c in checkpoints:
            col = B[:, c - 1]
            found = np.nanmean(col >= 0.99 * opt) * 100
            cells.append(f"{np.nanmean(col):6.3f} ({found:3.0f}%)")
        lines.append(f"{name:8s}" + "".join(f"{cell:>14s}" for cell in cells))
    return "\n".join(lines)


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description="CRF per injection for one search method across the sets.")
    p.add_argument("--method", choices=list(METHODS), default="random")
    p.add_argument("--sets", nargs="+", default=list(SET_NAMES))
    p.add_argument("--budget", type=int, default=25)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--shown-seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    fn = METHODS[a.method]
    runs_by_set = {name: [fn(TableOracle(name), budget=a.budget, seed=s) for s in range(a.seeds)] for name in a.sets}
    print(f"{LABELS[a.method]}: mean best CRF after n injections over {a.seeds} seeds (share of seeds within 1 % of the optimum)")
    print(checkpoint_table(runs_by_set))
    out = a.out or RESULTS_DIR / f"{a.method}_efficiency.png"
    print(f"-> {plot_efficiency(runs_by_set, out, a.shown_seed)}")


if __name__ == "__main__":
    _main()
