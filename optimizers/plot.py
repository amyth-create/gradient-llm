"""Efficiency and regret plots for a set of search runs."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .oracle import SearchRun  # noqa: E402

# Categorical slots 1 and 2 of the validated reference palette; status red for truncated and infeasible.
COLORS = {"random": "#2a78d6", "bo": "#eb6834", "llm": "#2e9e5b"}
LABELS = {"random": "Random search", "bo": "Bayesian optimisation (GP-EI)", "llm": "LLM search (Azure OpenAI)"}
INFEASIBLE = "#e34948"
TRUNCATED_MARKER = "v"           # a run whose last peaks eluted after the run ended, drawn at its observed CRF
TRUNCATED_LABEL = "truncated run (peaks missing), at its observed CRF"
INFEASIBLE_LABEL = "infeasible injection (outside bounds, no chromatogram), shown at the bottom edge"
GRID = "#e4e3df"
INK = "#52514e"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK)


def _stack(runs: list[SearchRun], attr: str) -> np.ndarray:
    return np.vstack([getattr(r, attr)() for r in runs])


def plot_comparison(runs_by_method: dict[str, list[SearchRun]], path: str | Path, shown_seed: int = 0,
                    tol: float = 0.01) -> Path:
    """Three panels: CRF per injection for one seed, best-so-far across seeds, regret across seeds."""
    any_runs = next(iter(runs_by_method.values()))
    set_name, optimum, budget = any_runs[0].set_name, any_runs[0].optimum, any_runs[0].budget
    n_seeds = len(any_runs)
    x = np.arange(1, budget + 1)
    handles: dict[str, object] = {}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), layout="constrained")
    fig.suptitle(f"{set_name} set · {budget} injections · optimum CRF {optimum:.3f}", x=0.01, ha="left",
                 fontsize=12, color="#0b0b0b")

    # (a) CRF per injection, one seed; truncated runs get their own marker at their CRF,
    #     infeasible (out-of-bounds) injections are marked along the bottom edge
    ax = axes[0]
    all_crf = np.concatenate([r.crf_per_run() for runs in runs_by_method.values() for r in runs])
    y_floor = min(-0.15, float(np.nanmin(all_crf)) - 0.1) if np.isfinite(all_crf).any() else -0.15
    y_mark = y_floor - 0.08
    truncated_handle = infeasible_handle = None
    for method, runs in runs_by_method.items():
        run = next((r for r in runs if r.seed == shown_seed), runs[0])
        crf = run.crf_per_run()
        feas = np.isfinite(crf)
        trunc = feas & (run.n_missing_per_run() > 0)
        c = COLORS[method]
        handles[f"{LABELS[method]}: CRF of each injection"] = ax.scatter(
            x[feas & ~trunc], crf[feas & ~trunc], s=30, color=c, edgecolors="white", linewidths=0.8, zorder=3)
        handles[f"{LABELS[method]}: best so far"] = ax.step(
            x, run.best_curve(), where="post", color=c, linewidth=2, zorder=2)[0]
        if trunc.any():
            truncated_handle = ax.scatter(
                x[trunc], crf[trunc], marker=TRUNCATED_MARKER, s=40, color=INFEASIBLE, edgecolors="white",
                linewidths=0.8, zorder=3)
        if (~feas).any():
            infeasible_handle = ax.scatter(
                x[~feas], np.full((~feas).sum(), y_mark), marker="x", s=40, color=INFEASIBLE, linewidths=1.6, zorder=3)
    if truncated_handle is not None:
        handles[TRUNCATED_LABEL] = truncated_handle
    if infeasible_handle is not None:
        handles[INFEASIBLE_LABEL] = infeasible_handle
    handles["known optimum"] = ax.axhline(optimum, color=INK, linewidth=1, linestyle="--")
    ax.set_ylim(y_mark - 0.1, 1.08)
    ax.set_xlim(0.5, budget + 0.5)
    _style(ax, f"Optimisation efficiency: CRF per injection (seed {shown_seed})", "injection", "CRF")

    # (b) best so far, mean over seeds with min-max band
    ax = axes[1]
    for method, runs in runs_by_method.items():
        B = _stack(runs, "best_curve")
        ax.fill_between(x, np.nanmin(B, axis=0), np.nanmax(B, axis=0), color=COLORS[method], alpha=0.15, linewidth=0)
        ax.plot(x, np.nanmean(B, axis=0), color=COLORS[method], linewidth=2)
    ax.axhline(optimum, color=INK, linewidth=1, linestyle="--")
    ax.set_xlim(0.5, budget + 0.5)
    _style(ax, f"Best CRF so far (mean of {n_seeds} seeds, band = min to max)", "injection", "best CRF")

    # (c) regret
    ax = axes[2]
    for method, runs in runs_by_method.items():
        R = _stack(runs, "regret_curve")
        ax.fill_between(x, np.nanmin(R, axis=0), np.nanmax(R, axis=0), color=COLORS[method], alpha=0.15, linewidth=0)
        ax.plot(x, np.nanmean(R, axis=0), color=COLORS[method], linewidth=2)
    handles[f"within {tol:.0%} of the optimum = found"] = ax.axhline(tol * optimum, color=INK, linewidth=1, linestyle=":")
    ax.set_xlim(0.5, budget + 0.5)
    ax.set_ylim(bottom=0)
    _style(ax, f"Regret = optimum − best so far (mean of {n_seeds} seeds)", "injection", "regret")

    fig.legend(list(handles.values()), list(handles.keys()), loc="outside lower center", ncol=4,
               fontsize=8.5, frameon=False)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
