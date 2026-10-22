"""The CRF landscape of a method table, and how a search walks through it.

The search space is four-dimensional, so the landscape is drawn as a corner
plot: each lower-triangle panel shows, for one pair of variables, the best CRF
over the other two (a max-projection: "is there any good method with these two
values?"); grey cells have no feasible method at all. The diagonal shows the
best CRF as a function of each variable alone. A search run is overlaid as its
injections coloured by order and connected in sequence.

    python -m optimizers.landscape --set hard --seed 0
    -> results/hard_landscape.png, results/hard_trajectory_{random,bo,llm}.png, results/hard_walks.png
       (the LLM run is replayed from results/llm_transcripts/hard_seed0.jsonl and skipped if absent;
        the walks figure is the three runs side by side on the phi1 x tG projection)
    python -m optimizers.landscape --set all --heatmaps-only
    -> results/<set>_landscape_heatmaps.png (the six pairwise heatmaps alone)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from sim.config import results_dir  # noqa: E402
from sim.optimum import load_optimum  # noqa: E402

from .bo import bo_search  # noqa: E402
from .oracle import FEATURES, SearchRun, TableOracle  # noqa: E402
from .plot import GRID, INFEASIBLE, INK, LABELS, TRUNCATED_MARKER  # noqa: E402
from .random_search import random_search  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SET_NAMES = ("easy", "medium", "hard")
AXIS_LABELS = {"phi0": "phi0, initial organic", "phi1": "phi1, final organic", "tG": "tG (min)", "T": "T (°C)"}

# Sequential blue ramp (reference palette steps 100 -> 700) for CRF magnitude.
CRF_RAMP = LinearSegmentedColormap.from_list("crf", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
# Second sequential context (injection order) takes the next categorical hue, orange, as its own ramp.
ORDER_RAMP = LinearSegmentedColormap.from_list("order", ["#fbd3bf", "#eb6834", "#7a2a0e"])
NO_FEASIBLE = "#e4e3df"
PATH_COLOR = "#6b6a66"


def projected_landscape(oracle: TableOracle, i: int, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best CRF over all other variables on the (feature i, feature j) grid.

    Returns (x_values, y_values, Z, present): Z has shape (len(y), len(x)), nan where
    no feasible method exists; ``present`` marks cells that contain any method at all.
    """
    xi, inv_i = np.unique(oracle.X[:, i], return_inverse=True)
    xj, inv_j = np.unique(oracle.X[:, j], return_inverse=True)
    Z = np.full((len(xj), len(xi)), -np.inf)
    f = oracle.feasible
    np.maximum.at(Z, (inv_j[f], inv_i[f]), oracle.crf[f])
    present = np.zeros(Z.shape, dtype=bool)
    present[inv_j, inv_i] = True
    return xi, xj, np.where(np.isfinite(Z), Z, np.nan), present


def profile(oracle: TableOracle, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Best CRF as a function of feature i alone."""
    xi, inv = np.unique(oracle.X[:, i], return_inverse=True)
    z = np.full(len(xi), -np.inf)
    f = oracle.feasible
    np.maximum.at(z, inv[f], oracle.crf[f])
    return xi, np.where(np.isfinite(z), z, np.nan)


def _edges(vals: np.ndarray) -> np.ndarray:
    if len(vals) == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5])
    mids = 0.5 * (vals[1:] + vals[:-1])
    return np.concatenate([[vals[0] - (mids[0] - vals[0])], mids, [vals[-1] + (vals[-1] - mids[-1])]])


def _optimum_point(oracle: TableOracle) -> np.ndarray:
    v = load_optimum(oracle.set_name)["vector"]
    return np.array([v["phi0"], v["phi0"] + v["dphi"], v["tG"], v["T"]])


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)


def plot_landscape(oracle: TableOracle, path: str | Path, run: SearchRun | None = None,
                   label_every: int = 5) -> Path:
    """Corner plot of the landscape, optionally with one search run overlaid."""
    d = len(FEATURES)
    opt = _optimum_point(oracle)
    crf_norm = Normalize(0.0, 1.0)
    fig, axes = plt.subplots(d, d, figsize=(13.5, 12.5), layout="constrained")
    if run is not None:
        idx = np.array(run.indices())
        order = np.arange(1, len(idx) + 1)
        feas = np.array([o.feasible for o in run.observations])
        trunc = feas & (run.n_missing_per_run() > 0)
        best_i = int(np.nanargmax(np.where(feas, run.crf_per_run(), -np.inf)))
        order_norm = Normalize(1, run.budget)
        labelled = [k for k in range(len(idx)) if (k + 1) % label_every == 0 or k == 0]

    for r in range(d):
        for c in range(d):
            ax = axes[r, c]
            if c > r:
                ax.axis("off")
                continue
            if r == c:                                     # 1-D profile
                xs, z = profile(oracle, r)
                ax.plot(xs, z, color="#2a78d6", linewidth=2, marker="o", markersize=3)
                ax.axvline(opt[r], color=INK, linewidth=1, linestyle="--")
                lo = min(0.0, float(np.nanmin(z)) - 0.05)
                ax.set_ylim(lo - 0.12, 1.08)
                ax.set_xlim(_edges(xs)[0], _edges(xs)[-1])
                ax.set_ylabel("best CRF", color=INK, fontsize=8)
                ax.grid(True, color=GRID, linewidth=0.8)
                ax.set_axisbelow(True)
                if run is not None:
                    ax.scatter(oracle.X[idx, r], np.full(len(idx), lo - 0.06), marker="|", s=90,
                               c=order, cmap=ORDER_RAMP, norm=order_norm, linewidths=1.5, zorder=3)
            else:                                          # heatmap: x = feature c, y = feature r
                xs, ys, Z, present = projected_landscape(oracle, c, r)
                ex, ey = _edges(xs), _edges(ys)
                none_feasible = present & np.isnan(Z)
                ax.pcolormesh(ex, ey, np.ma.masked_where(~none_feasible, np.ones_like(Z)),
                              cmap=ListedColormap([NO_FEASIBLE]), shading="flat")
                ax.pcolormesh(ex, ey, np.ma.masked_invalid(Z), cmap=CRF_RAMP, norm=crf_norm, shading="flat")
                ax.scatter(opt[c], opt[r], marker="*", s=190, color="white", edgecolors="#0b0b0b", linewidths=1.0, zorder=6)
                ax.set_xlim(ex[0], ex[-1])
                ax.set_ylim(ey[0], ey[-1])
                if run is not None:
                    px, py = oracle.X[idx, c], oracle.X[idx, r]
                    ax.plot(px, py, color=PATH_COLOR, linewidth=0.9, alpha=0.7, zorder=3)
                    ax.scatter(px[feas], py[feas], c=order[feas], cmap=ORDER_RAMP, norm=order_norm, s=42,
                               edgecolors="white", linewidths=0.7, zorder=4)
                    if trunc.any():
                        ax.scatter(px[trunc], py[trunc], marker=TRUNCATED_MARKER, s=110, facecolors="none",
                                   edgecolors=INFEASIBLE, linewidths=1.4, zorder=5)
                    if (~feas).any():
                        ax.scatter(px[~feas], py[~feas], marker="x", s=48, color=INFEASIBLE, linewidths=1.6, zorder=5)
                    ax.scatter(px[best_i], py[best_i], s=170, facecolors="none", edgecolors="#0b0b0b", linewidths=1.6, zorder=6)
                    for k in labelled:
                        ax.annotate(str(k + 1), (px[k], py[k]), xytext=(4, 3), textcoords="offset points",
                                    fontsize=7, color="#0b0b0b", zorder=7)
            _style(ax)
            if r == d - 1:
                ax.set_xlabel(AXIS_LABELS[FEATURES[c]], color=INK, fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
            if c == 0 and r != 0:
                ax.set_ylabel(AXIS_LABELS[FEATURES[r]], color=INK, fontsize=9)
            elif c != 0 and r != c:
                ax.tick_params(labelleft=False)

    # Colour bars and legend in the empty upper-right cells.
    cax = axes[0, 2].inset_axes([0.05, 0.62, 0.9, 0.09])
    fig.colorbar(ScalarMappable(norm=crf_norm, cmap=CRF_RAMP), cax=cax, orientation="horizontal")
    cax.set_title("best CRF over the other two variables (max-projection)", fontsize=8.5, color=INK, loc="left")
    cax.tick_params(labelsize=7.5, colors=INK)
    handles = [
        Line2D([], [], marker="*", color="white", markeredgecolor="#0b0b0b", markersize=13, linestyle="none", label="known optimum"),
        Line2D([], [], marker="s", color=NO_FEASIBLE, markersize=12, linestyle="none", label="no feasible method (outside bounds)"),
        Line2D([], [], color=INK, linestyle="--", label="optimum (diagonal panels)"),
    ]
    if run is not None:
        cax2 = axes[0, 2].inset_axes([0.05, 0.2, 0.9, 0.09])
        fig.colorbar(ScalarMappable(norm=order_norm, cmap=ORDER_RAMP), cax=cax2, orientation="horizontal")
        cax2.set_title("injection order (ticks on the diagonal mark each injection)", fontsize=8.5, color=INK, loc="left")
        cax2.tick_params(labelsize=7.5, colors=INK)
        handles += [
            Line2D([], [], color=PATH_COLOR, linewidth=1, label="path from one injection to the next"),
            Line2D([], [], marker="o", color="#eb6834", markeredgecolor="white", markersize=8, linestyle="none", label="feasible injection"),
            Line2D([], [], marker=TRUNCATED_MARKER, color="none", markeredgecolor=INFEASIBLE, markersize=10, markeredgewidth=1.4,
                   linestyle="none", label="truncated run (peaks missing)"),
            Line2D([], [], marker="x", color=INFEASIBLE, markersize=8, markeredgewidth=1.6, linestyle="none", label="infeasible injection (outside bounds)"),
            Line2D([], [], marker="o", color="none", markeredgecolor="#0b0b0b", markersize=12, markeredgewidth=1.6, linestyle="none", label="best injection of the run"),
        ]
    axes[1, 3].legend(handles=handles, loc="center", fontsize=8.5, frameon=False)

    if run is None:
        title = f"{oracle.set_name} set · CRF landscape of {oracle.n_candidates} grid methods · optimum CRF {oracle.optimum:.3f}"
    else:
        best = run.best_curve()[-1]
        title = (f"{oracle.set_name} set · {LABELS[run.method]} · seed {run.seed} · {run.budget} injections · "
                 f"best CRF {best:.3f} (regret {oracle.optimum - best:.3f})")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color="#0b0b0b")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_heatmaps(oracle: TableOracle, path: str | Path) -> Path:
    """Landscape only: the six pairwise max-projection heatmaps in a 2 x 3 grid, no profiles, no runs."""
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    opt = _optimum_point(oracle)
    crf_norm = Normalize(0.0, 1.0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for ax, (c, r) in zip(axes.ravel(), pairs):
        xs, ys, Z, present = projected_landscape(oracle, c, r)
        ex, ey = _edges(xs), _edges(ys)
        none_feasible = present & np.isnan(Z)
        ax.pcolormesh(ex, ey, np.ma.masked_where(~none_feasible, np.ones_like(Z)),
                      cmap=ListedColormap([NO_FEASIBLE]), shading="flat")
        ax.pcolormesh(ex, ey, np.ma.masked_invalid(Z), cmap=CRF_RAMP, norm=crf_norm, shading="flat")
        ax.scatter(opt[c], opt[r], marker="*", s=260, color="white", edgecolors="#0b0b0b", linewidths=1.2, zorder=6)
        ax.set_xlim(ex[0], ex[-1])
        ax.set_ylim(ey[0], ey[-1])
        ax.set_xlabel(AXIS_LABELS[FEATURES[c]], color=INK, fontsize=10)
        ax.set_ylabel(AXIS_LABELS[FEATURES[r]], color=INK, fontsize=10)
        ax.set_title(f"{FEATURES[c]} × {FEATURES[r]}", loc="left", fontsize=11, color="#0b0b0b")
        _style(ax)
        ax.tick_params(labelsize=9)
    cbar = fig.colorbar(ScalarMappable(norm=crf_norm, cmap=CRF_RAMP), ax=axes, location="right", shrink=0.6, pad=0.01)
    cbar.set_label("best CRF over the other two variables", color=INK, fontsize=10)
    cbar.ax.tick_params(labelsize=9, colors=INK)
    handles = [
        Line2D([], [], marker="*", color="white", markeredgecolor="#0b0b0b", markersize=15, linestyle="none", label="known optimum"),
        Line2D([], [], marker="s", color=NO_FEASIBLE, markersize=13, linestyle="none", label="no feasible method (outside bounds)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle(f"{oracle.set_name} set · CRF landscape of {oracle.n_candidates} grid methods · optimum CRF {oracle.optimum:.3f}",
                 x=0.01, ha="left", fontsize=13, color="#0b0b0b")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_walks(oracle: TableOracle, runs: dict[str, SearchRun], path: str | Path, pair: tuple[int, int] = (1, 2),
               label_every: int = 5) -> Path:
    """One panel per method on the same projected landscape (default phi1 x tG), each with that method's run overlaid.

    The compact companion of the corner plots: the three searches side by side on the one projection
    where the ridge is easiest to see. Markers are injections coloured by order, the ring is the best one.
    """
    c, r = pair
    xs, ys, Z, present = projected_landscape(oracle, c, r)
    ex, ey = _edges(xs), _edges(ys)
    opt = _optimum_point(oracle)
    fig, axes = plt.subplots(1, len(runs), figsize=(5.4 * len(runs) + 0.8, 5.2), layout="constrained", squeeze=False)
    order_norm = None
    for ax, (method, run) in zip(axes[0], runs.items()):
        none_feasible = present & np.isnan(Z)
        ax.pcolormesh(ex, ey, np.ma.masked_where(~none_feasible, np.ones_like(Z)),
                      cmap=ListedColormap([NO_FEASIBLE]), shading="flat")
        ax.pcolormesh(ex, ey, np.ma.masked_invalid(Z), cmap=CRF_RAMP, norm=Normalize(0.0, 1.0), shading="flat")
        ax.scatter(opt[c], opt[r], marker="*", s=230, color="white", edgecolors="#0b0b0b", linewidths=1.0, zorder=6)
        idx = np.array(run.indices())
        order = np.arange(1, len(idx) + 1)
        order_norm = Normalize(1, run.budget)
        feas = np.array([o.feasible for o in run.observations])
        trunc = feas & (run.n_missing_per_run() > 0)
        best_i = int(np.nanargmax(np.where(feas, run.crf_per_run(), -np.inf)))
        px, py = oracle.X[idx, c], oracle.X[idx, r]
        ax.plot(px, py, color=PATH_COLOR, linewidth=1.0, alpha=0.75, zorder=3)
        ax.scatter(px[feas], py[feas], c=order[feas], cmap=ORDER_RAMP, norm=order_norm, s=64,
                   edgecolors="white", linewidths=0.7, zorder=4)
        if trunc.any():
            ax.scatter(px[trunc], py[trunc], marker=TRUNCATED_MARKER, s=150, facecolors="none",
                       edgecolors=INFEASIBLE, linewidths=1.4, zorder=5)
        ax.scatter(px[best_i], py[best_i], s=240, facecolors="none", edgecolors="#0b0b0b", linewidths=1.8, zorder=6)
        for k in range(len(idx)):
            if k == 0 or (k + 1) % label_every == 0:
                ax.annotate(str(k + 1), (px[k], py[k]), xytext=(5, 4), textcoords="offset points",
                            fontsize=8, color="#0b0b0b", zorder=7)
        ax.set_xlim(ex[0], ex[-1])
        ax.set_ylim(ey[0], ey[-1])
        ax.set_xlabel(AXIS_LABELS[FEATURES[c]], color=INK, fontsize=9)
        ax.set_ylabel(AXIS_LABELS[FEATURES[r]], color=INK, fontsize=9)
        best = float(np.nanmax(np.where(feas, run.crf_per_run(), -np.inf)))
        hit = run.runs_to_optimum()
        ax.set_title(f"{LABELS.get(method, method)} · seed {run.seed} · best {best:.3f}"
                     + (f" · optimum at {hit}" if hit else ""), loc="left", fontsize=10, color=INK)
        _style(ax)
    if order_norm is not None:
        fig.colorbar(ScalarMappable(norm=order_norm, cmap=ORDER_RAMP), ax=axes[0].tolist(), shrink=0.85,
                     pad=0.02, label="injection order")
    fig.suptitle(f"{oracle.set_name} · where each search walked · best CRF over the other two variables "
                 f"· optimum {oracle.optimum:.3f} (star) · ring = best injection · hollow triangle = truncated",
                 x=0.01, ha="left", fontsize=11, color=INK)
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_set(set_name: str, seed: int = 0, budget: int = 25, out_dir: Path | None = None,
             verbose: bool = True, methods: tuple[str, ...] = ("random", "bo", "llm"), walks: bool = True) -> list[Path]:
    """Landscape, one corner plot per method with its seed-``seed`` run, and the compact walks figure.

    Random search and BO are re-run (deterministic in the seed); the LLM run is loaded from its
    transcript in the set's results root and skipped with a note when that seed has none.
    ``out_dir`` defaults to the ``results/`` folder of the set's own data root.
    """
    from .llm_repeats import load_run

    oracle = TableOracle(set_name)
    out_dir = Path(out_dir) if out_dir is not None else results_dir(set_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, SearchRun] = {}
    for method in methods:
        if method == "llm":
            run = load_run(oracle, seed, budget)
            if run is None:
                if verbose:
                    print(f"   (no finished LLM transcript for {set_name} seed {seed}: LLM walk skipped)")
                continue
        else:
            run = {"random": random_search, "bo": bo_search}[method](oracle, budget=budget, seed=seed)
        runs[method] = run
    paths = [plot_landscape(oracle, out_dir / f"{set_name}_landscape.png")]
    for method, run in runs.items():
        paths.append(plot_landscape(oracle, out_dir / f"{set_name}_trajectory_{method}.png", run))
    if walks and runs:
        paths.append(plot_walks(oracle, runs, out_dir / f"{set_name}_walks.png"))
    if verbose:
        for p in paths:
            print(f"-> {p}")
    return paths


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Draw the CRF landscape and search trajectories.")
    p.add_argument("--set", default="hard", help="easy | medium | hard | all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=25)
    p.add_argument("--out", type=Path, default=None, help="default: the results/ folder of each set's data root")
    p.add_argument("--heatmaps-only", action="store_true",
                   help="write only <set>_landscape_heatmaps.png: the six pairwise heatmaps, no profiles, no runs")
    a = p.parse_args(argv)
    for name in (SET_NAMES if a.set == "all" else (a.set,)):
        if a.heatmaps_only:
            out = Path(a.out) if a.out is not None else results_dir(name)
            out.mkdir(parents=True, exist_ok=True)
            print(f"-> {plot_heatmaps(TableOracle(name), out / f'{name}_landscape_heatmaps.png')}")
        else:
            plot_set(name, a.seed, a.budget, a.out)


if __name__ == "__main__":
    _main()
