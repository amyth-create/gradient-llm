"""Repeated search campaigns: CRF over runs (injections) averaged over many seeds, one panel per analyte set.

    python -m optimizers.repeats --repeats 50                   # results/bo_crf_over_runs.png
    python -m optimizers.repeats --method random --repeats 50   # results/random_crf_over_runs.png
    python -m optimizers.repeats --methods random bo            # results/crf_over_runs.png, both overlaid

Top row: the CRF of each injection. Every repeat is drawn as a faint line, the
mean over repeats as a solid line with a 95 % confidence interval of the mean,
and the mean best-so-far as a dashed line. Bottom row: the share of repeats
whose injection at that run number was truncated (peaks still on the column
when the run ended; scored at the observed CRF, so included in the mean) and
the share that was infeasible (no chromatogram, left out of the mean). Repeat ``s`` is seed ``s``, so the
campaigns are the same ones ``optimizers.run`` and ``optimizers.convergence``
draw for those seeds. With several ``--methods`` the figure overlays the
methods' means and bands per set, drops the individual traces and omits the
truncated/infeasible row.
"""
from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .bo import BOConfig, bo_search  # noqa: E402
from .convergence import mean_and_ci  # noqa: E402
from .efficiency import CHECKPOINTS, checkpoint_table  # noqa: E402
from .oracle import SearchRun, TableOracle  # noqa: E402
from .plot import COLORS, INFEASIBLE, INK, LABELS, _style  # noqa: E402
from .random_search import random_search  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SET_NAMES = ("easy", "medium", "hard")
METHODS = {"random": random_search, "bo": bo_search}


def repeat_search(method: str, set_name: str, repeats: int, budget: int = 25) -> list[SearchRun]:
    """``repeats`` campaigns of one method on one set, seeds 0 .. repeats-1."""
    oracle = TableOracle(set_name)
    return [METHODS[method](oracle, budget=budget, seed=s) for s in range(repeats)]


def crf_matrix(runs: list[SearchRun]) -> np.ndarray:
    """Repeats x injections; nan where the injection was infeasible."""
    return np.vstack([r.crf_per_run() for r in runs])


def infeasible_share(runs: list[SearchRun]) -> np.ndarray:
    return np.mean(~np.isfinite(crf_matrix(runs)), axis=0)


def truncated_share(runs: list[SearchRun]) -> np.ndarray:
    """Share of repeats whose n-th injection was feasible but had peaks elute after the run ended."""
    missing = np.vstack([r.n_missing_per_run() for r in runs])
    return np.mean((missing > 0) & np.isfinite(crf_matrix(runs)), axis=0)


def per_injection_table(runs_by_set: dict[str, list[SearchRun]], checkpoints=CHECKPOINTS) -> str:
    """Mean CRF of the n-th injection itself (feasible only), with the share truncated and the share infeasible."""
    budget = next(iter(runs_by_set.values()))[0].budget
    checkpoints = tuple(c for c in checkpoints if c <= budget)
    lines = [f"{'set':8s}" + "".join(f"{f'inj {c}':>22s}" for c in checkpoints)]
    for name, runs in runs_by_set.items():
        C, tr, inf = crf_matrix(runs), truncated_share(runs), infeasible_share(runs)
        cells = []
        for c in checkpoints:
            col = C[:, c - 1]
            feas = np.isfinite(col)
            mean = f"{col[feas].mean():6.3f}" if feas.any() else "     -"
            cells.append(f"{mean} ({100 * tr[c - 1]:3.0f}% tr {100 * inf[c - 1]:3.0f}% inf)")
        lines.append(f"{name:8s}" + "".join(f"{cell:>22s}" for cell in cells))
    return "\n".join(lines)


def plot_crf_over_runs(runs_by_set: dict[str, list[SearchRun]], path: str | Path, n_init: int | None = None,
                       tol: float = 0.01) -> Path:
    """``runs_by_set[set]`` -> list of SearchRun (one per repeat). Writes a 2 x n_sets figure."""
    sets = list(runs_by_set)
    first = runs_by_set[sets[0]][0]
    method, budget, n_rep = first.method, first.budget, len(runs_by_set[sets[0]])
    c = COLORS[method]
    x = np.arange(1, budget + 1)

    fig, axes = plt.subplots(2, len(sets), figsize=(5.2 * len(sets), 7.4), layout="constrained",
                             sharex=True, sharey="row", height_ratios=[3, 1], squeeze=False)
    all_crf = np.concatenate([crf_matrix(runs).ravel() for runs in runs_by_set.values()])
    y_floor = min(-0.15, float(np.nanmin(all_crf)) - 0.05) if np.isfinite(all_crf).any() else -0.15
    for col, name in enumerate(sets):
        runs = runs_by_set[name]
        top, bottom = axes[0, col], axes[1, col]
        C, B = crf_matrix(runs), np.vstack([r.best_curve() for r in runs])
        optimum = runs[0].optimum

        # every repeat, faintly, so the spread is visible behind the mean
        for row in C:
            top.plot(x, row, color=c, alpha=0.18, linewidth=0.8, zorder=1)
        with warnings.catch_warnings():                 # a column with one feasible value has no sample sd
            warnings.simplefilter("ignore", RuntimeWarning)
            mean, lo, hi = mean_and_ci(C)
        top.fill_between(x, lo, hi, color=c, alpha=0.18, linewidth=0, zorder=2)
        top.plot(x, mean, color=c, linewidth=2.2, marker="o", markersize=4.5, markeredgecolor="white",
                 markeredgewidth=0.8, zorder=4)
        top.plot(x, np.nanmean(B, axis=0), color=c, linewidth=1.8, linestyle="--", zorder=3)
        top.axhline(optimum, color=INK, linewidth=1, linestyle="--")
        if n_init:
            for ax in (top, bottom):
                ax.axvline(n_init + 0.5, color=INK, linewidth=1, linestyle=":")

        final = B[:, -1]
        found = int(np.sum(final >= (1.0 - tol) * optimum))
        _style(top, f"{name} · final best {np.nanmean(final):.3f} ± {np.nanstd(final):.3f} · "
                    f"optimum found {found}/{n_rep}", "", "CRF" if col == 0 else "")
        top.set_ylim(y_floor, 1.08)

        bottom.bar(x - 0.2, truncated_share(runs), width=0.38, color=INFEASIBLE, alpha=0.45, linewidth=0, zorder=2)
        bottom.bar(x + 0.2, infeasible_share(runs), width=0.38, color=INFEASIBLE, linewidth=0, zorder=2)
        _style(bottom, f"{name} · share truncated / infeasible at injection n", "injection",
               "share of repeats" if col == 0 else "")
        bottom.set_ylim(0, 1.0)
        bottom.set_xlim(0.5, budget + 0.5)
        bottom.set_xticks(np.arange(0, budget + 1, 5)[1:])

    handles = [
        Line2D([], [], color=c, alpha=0.45, linewidth=0.8, label=f"CRF per injection, one line per repeat ({n_rep})"),
        Line2D([], [], color=c, linewidth=2.2, marker="o", markersize=5, markeredgecolor="white",
               label="mean CRF of injection n (feasible only), band = 95 % CI"),
        Line2D([], [], color=c, linewidth=1.8, linestyle="--", label="mean best CRF so far"),
        Line2D([], [], color=INK, linestyle="--", label="known optimum"),
        Patch(color=INFEASIBLE, alpha=0.45, label="share of repeats truncated (peaks missing, scored as observed)"),
        Patch(color=INFEASIBLE, label="share of repeats infeasible (no chromatogram)"),
    ]
    if n_init:
        handles.append(Line2D([], [], color=INK, linestyle=":",
                              label=f"surrogate takes over after injection {n_init}"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"{LABELS[method]} · CRF over runs · {n_rep} repeats per set · {budget} injections per campaign",
                 x=0.01, ha="left", fontsize=12, color="#0b0b0b")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_methods_over_runs(runs: dict[str, dict[str, list[SearchRun]]], path: str | Path,
                           n_init: int | None = None, tol: float = 0.01) -> Path:
    """``runs[set][method]`` -> list of SearchRun. All methods on the same axes, one panel per set:
    mean CRF of injection n with its 95 % CI and the mean best-so-far."""
    sets = list(runs)
    methods = list(runs[sets[0]])
    first = runs[sets[0]][methods[0]][0]
    budget, n_rep = first.budget, len(runs[sets[0]][methods[0]])
    x = np.arange(1, budget + 1)

    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 5.6), layout="constrained",
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    y_floor = 1.0
    for col, name in enumerate(sets):
        top = axes[col]
        optimum = runs[name][methods[0]][0].optimum
        short = {"random": "random", "bo": "BO", "llm": "LLM"}
        title = []
        for method in methods:
            rs, c = runs[name][method], COLORS[method]
            C, B = crf_matrix(rs), np.vstack([r.best_curve() for r in rs])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean, lo, hi = mean_and_ci(C)
            y_floor = min(y_floor, float(np.nanmin(lo)))
            top.fill_between(x, lo, hi, color=c, alpha=0.18, linewidth=0, zorder=2)
            top.plot(x, mean, color=c, linewidth=2.2, marker="o", markersize=4.5, markeredgecolor="white",
                     markeredgewidth=0.8, zorder=4)
            top.plot(x, np.nanmean(B, axis=0), color=c, linewidth=1.8, linestyle="--", zorder=3)
            found = int(np.sum(B[:, -1] >= (1.0 - tol) * optimum))
            title.append(f"{short.get(method, method)} {np.nanmean(B[:, -1]):.3f} ({found}/{n_rep})")
        top.axhline(optimum, color=INK, linewidth=1, linestyle="--")
        if n_init:
            top.axvline(n_init + 0.5, color=INK, linewidth=1, linestyle=":")
        _style(top, f"{name} · best @{budget}\n" + " · ".join(title), "injection", "CRF" if col == 0 else "")
        top.title.set_fontsize(10)
        top.set_xlim(0.5, budget + 0.5)
        top.set_xticks(np.arange(0, budget + 1, 5)[1:])
    for ax in axes:
        ax.set_ylim(min(-0.05, y_floor - 0.05), 1.05)

    handles = []
    for method in methods:
        c = COLORS[method]
        handles.append(Line2D([], [], color=c, linewidth=2.2, marker="o", markersize=5, markeredgecolor="white",
                              label=f"{LABELS[method]}: mean CRF of injection n, band = 95 % CI"))
        handles.append(Line2D([], [], color=c, linewidth=1.8, linestyle="--", label=f"{LABELS[method]}: mean best CRF so far"))
    handles.append(Line2D([], [], color=INK, linestyle="--", label="known optimum"))
    if n_init:
        handles.append(Line2D([], [], color=INK, linestyle=":",
                              label=f"shared random start: surrogate takes over after injection {n_init}"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f"CRF over runs · {' vs '.join(LABELS[m] for m in methods)} · {n_rep} repeats per set · "
                 f"{budget} injections per campaign · (n/{n_rep}) = repeats within {tol:.0%} of the optimum",
                 x=0.01, ha="left", fontsize=12, color="#0b0b0b")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_runs_csv(runs: list[SearchRun], path: Path) -> Path:
    rows = [row for r in runs for row in r.to_rows()]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Repeated search campaigns: CRF over injections, one panel per set.")
    p.add_argument("--method", default="bo", choices=list(METHODS))
    p.add_argument("--methods", nargs="+", choices=list(METHODS), default=None,
                   help="several methods overlaid on one figure (overrides --method)")
    p.add_argument("--repeats", type=int, default=50, help="campaigns per set (seeds 0 .. repeats-1)")
    p.add_argument("--budget", type=int, default=25, help="injections per campaign")
    p.add_argument("--sets", nargs="+", default=list(SET_NAMES))
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    methods = a.methods or [a.method]
    runs: dict[str, dict[str, list[SearchRun]]] = {}
    for name in a.sets:
        runs[name] = {}
        for m in methods:
            t = time.perf_counter()
            runs[name][m] = repeat_search(m, name, a.repeats, a.budget)
            print(f"{name:7s} {m:7s} {a.repeats} repeats in {time.perf_counter() - t:.1f} s")
    for m in methods:
        by_set = {name: runs[name][m] for name in a.sets}
        print(f"\n{LABELS[m]}: mean CRF of injection n over {a.repeats} repeats (share truncated, share infeasible)")
        print(per_injection_table(by_set))
        print(f"\n{LABELS[m]}: mean best CRF after n injections (share of repeats within 1 % of the optimum)")
        print(checkpoint_table(by_set, checkpoints=tuple(c for c in CHECKPOINTS if c <= a.budget)))
    n_init = BOConfig().n_init if "bo" in methods else None
    if len(methods) == 1:
        m = methods[0]
        out = a.out or RESULTS_DIR / f"{m}_crf_over_runs.png"
        print(f"\n-> {plot_crf_over_runs({name: runs[name][m] for name in a.sets}, out, n_init=n_init)}")
        csv_path = write_runs_csv([r for name in a.sets for r in runs[name][m]], out.with_name(f"{m}_crf_over_runs.csv"))
    else:
        out = a.out or RESULTS_DIR / "crf_over_runs.png"
        print(f"\n-> {plot_methods_over_runs(runs, out, n_init=n_init)}")
        csv_path = write_runs_csv([r for name in a.sets for m in methods for r in runs[name][m]],
                                  out.with_name("crf_over_runs.csv"))
    print(f"-> {csv_path}")


if __name__ == "__main__":
    _main()
