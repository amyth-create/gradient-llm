"""Convergence view: mean best CRF after n injections, with a confidence band and the found-fraction.

    python -m optimizers.convergence --seeds 50          # results/convergence.png

Top row: mean best-so-far CRF against injection number, one line per method,
with a 95 % confidence interval of the mean (1.96 x standard error over seeds).
Bottom row: the share of seeds whose best is within 1 % of the known optimum by
injection n, i.e. the probability that a campaign of that length has found it.
One column per analyte set, shared axes, so sets and methods compare directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .bo import bo_search  # noqa: E402
from .efficiency import checkpoint_table  # noqa: E402
from .oracle import SearchRun, TableOracle  # noqa: E402
from .plot import COLORS, INK, LABELS, _style  # noqa: E402
from .random_search import random_search  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SET_NAMES = ("easy", "medium", "hard")
METHODS = {"random": random_search, "bo": bo_search}


def mean_and_ci(curves: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean over seeds and a z-sigma confidence interval of the mean, ignoring nan (no feasible injection yet)."""
    n = np.isfinite(curves).sum(axis=0)
    mean = np.nanmean(curves, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    sem = np.where(n > 1, sem, 0.0)
    return mean, mean - z * sem, mean + z * sem


def found_fraction(runs: list[SearchRun], tol: float = 0.01) -> np.ndarray:
    """Share of seeds within tol of the optimum after each injection."""
    B = np.vstack([r.best_curve() for r in runs])
    return np.nanmean(B >= (1.0 - tol) * runs[0].optimum, axis=0)


def plot_convergence(runs: dict[str, dict[str, list[SearchRun]]], path: str | Path, tol: float = 0.01) -> Path:
    """``runs[set][method]`` -> list of SearchRun. Writes a 2 x n_sets figure."""
    sets = list(runs)
    methods = list(runs[sets[0]])
    budget = runs[sets[0]][methods[0]][0].budget
    n_seeds = len(runs[sets[0]][methods[0]])
    x = np.arange(1, budget + 1)
    handles: dict[str, object] = {}

    fig, axes = plt.subplots(2, len(sets), figsize=(5.2 * len(sets), 8.2), layout="constrained",
                             sharex=True, sharey="row", squeeze=False)
    for col, name in enumerate(sets):
        top, bottom = axes[0, col], axes[1, col]
        optimum = runs[name][methods[0]][0].optimum
        for method in methods:
            rs = runs[name][method]
            B = np.vstack([r.best_curve() for r in rs])
            mean, lo, hi = mean_and_ci(B)
            top.fill_between(x, lo, hi, color=COLORS[method], alpha=0.18, linewidth=0)
            handles[LABELS[method]] = top.plot(x, mean, color=COLORS[method], linewidth=2.2)[0]
            bottom.step(x, found_fraction(rs, tol), where="post", color=COLORS[method], linewidth=2.2)
        handles["known optimum"] = top.axhline(optimum, color=INK, linewidth=1, linestyle=":")
        handles[f"within {tol:.0%} of the optimum"] = top.axhline((1 - tol) * optimum, color=INK, linewidth=1, linestyle="--")
        _style(top, f"{name} · mean best CRF after n injections", "", "best CRF" if col == 0 else "")
        _style(bottom, f"{name} · share of seeds that found the optimum", "injection",
               "fraction of seeds" if col == 0 else "")
        top.set_ylim(0.3, 1.02)
        bottom.set_ylim(0.0, 1.02)
        bottom.set_xlim(0.5, budget + 0.5)
    handles["95 % CI of the mean"] = fig.axes[0].fill_between([], [], [], color=INK, alpha=0.18, linewidth=0)
    fig.suptitle(f"Convergence over {n_seeds} seeds · {budget} injections per campaign", x=0.01, ha="left",
                 fontsize=12, color="#0b0b0b")
    fig.legend(list(handles.values()), list(handles.keys()), loc="outside lower center", ncol=5, fontsize=9, frameon=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Mean best CRF after n injections, with confidence bands and found-fraction.")
    p.add_argument("--sets", nargs="+", default=list(SET_NAMES))
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--budget", type=int, default=25)
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "convergence.png")
    a = p.parse_args(argv)
    runs: dict[str, dict[str, list[SearchRun]]] = {}
    for name in a.sets:
        oracle = TableOracle(name)
        runs[name] = {m: [METHODS[m](oracle, budget=a.budget, seed=s) for s in range(a.seeds)] for m in a.methods}
    for m in a.methods:
        print(f"{LABELS[m]}: mean best CRF after n injections over {a.seeds} seeds (share of seeds within 1 % of the optimum)")
        print(checkpoint_table({name: runs[name][m] for name in a.sets}))
        print()
    print(f"-> {plot_convergence(runs, a.out)}")


if __name__ == "__main__":
    _main()
