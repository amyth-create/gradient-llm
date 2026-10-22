"""Scatter of log k_w versus S for the analyte sets, with clusters of closely eluting analytes circled.

    python -m sim.plot_analytes                      # results/analyte_sets.png
    python -m sim.plot_analytes --sets medium hard   # a subset

Two analytes are "close" when their LSS lines are nearly parallel and nearly
coincident: |d log k_w| <= 0.3 and |dS| <= 1.0, the same criterion the easy-set
generator rejects and the medium/hard generators add on purpose. Clusters are
the connected components of that relation. This figure shows the hidden analyte
parameters, so it is diagnostic output for the simulator's owner, not something
an optimizer may see.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from .config import AnalyteSet  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SET_NAMES = ("easy", "medium", "hard")
CLOSE_DLOG_KW = 0.3
CLOSE_DS = 1.0
POINT = "#2a78d6"
INK = "#52514e"
GRID = "#e4e3df"


def clusters(log_kw: np.ndarray, S: np.ndarray, d_log_kw: float = CLOSE_DLOG_KW, dS: float = CLOSE_DS) -> list[list[int]]:
    """Connected components of the 'closely eluting' relation; only groups of two or more."""
    n = len(log_kw)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if abs(log_kw[i] - log_kw[j]) <= d_log_kw and abs(S[i] - S[j]) <= dS:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def plot_analyte_sets(sets: list[AnalyteSet], path: str | Path, pad_S: float = 0.55, pad_log_kw: float = 0.28) -> Path:
    n = len(sets)
    ncol = n if n <= 3 else 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.4 * nrow), layout="constrained",
                             sharex=True, sharey=True, squeeze=False)
    for ax, aset in zip(axes.ravel(), sets):
        log_kw, S, _ = aset.arrays()
        groups = clusters(log_kw, S)
        for g in groups:
            cx, cy = S[g].mean(), log_kw[g].mean()
            w = np.ptp(S[g]) + pad_S
            h = np.ptp(log_kw[g]) + pad_log_kw
            ax.add_patch(Ellipse((cx, cy), w, h, fill=False, edgecolor="#0b0b0b", linestyle="--", linewidth=1.3, zorder=2))
        ax.scatter(S, log_kw, s=46, color=POINT, edgecolors="white", linewidths=0.8, zorder=3)
        for i, (x, y) in enumerate(zip(S, log_kw)):
            ax.annotate(str(i + 1), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8, color="#0b0b0b", zorder=4)
        in_clusters = sum(len(g) for g in groups)
        ax.set_title(f"{aset.name} · {len(aset)} analytes · {len(groups)} close pair{'s' if len(groups) != 1 else ''}"
                     f" ({in_clusters} analytes)", loc="left", fontsize=10.5, color="#0b0b0b")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK)
    for ax in axes.ravel()[len(sets):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("S  (solvent-strength slope, −d log10 k / dφ)", color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("log10 k_w  (retention in water at 30 °C)", color=INK)
    all_S = np.concatenate([a.arrays()[1] for a in sets])
    all_kw = np.concatenate([a.arrays()[0] for a in sets])
    axes[0, 0].set_xlim(all_S.min() - 0.6, all_S.max() + 0.6)
    axes[0, 0].set_ylim(all_kw.min() - 0.35, all_kw.max() + 0.35)
    handles = [Line2D([], [], marker="o", color=POINT, markeredgecolor="white", linestyle="none", markersize=7, label="analyte (numbered by row in the set)"),
               Line2D([], [], color="#0b0b0b", linestyle="--", label=f"cluster of closely eluting analytes: |Δlog k_w| ≤ {CLOSE_DLOG_KW}, |ΔS| ≤ {CLOSE_DS}")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.suptitle("Analyte parameters of the sets used in the search comparison", x=0.01, ha="left", fontsize=12, color="#0b0b0b")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Scatter log k_w vs S for analyte sets, circling close pairs.")
    p.add_argument("--sets", nargs="+", default=list(SET_NAMES))
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "analyte_sets.png")
    a = p.parse_args(argv)
    path = plot_analyte_sets([AnalyteSet.load(s) for s in a.sets], a.out)
    for s in a.sets:
        aset = AnalyteSet.load(s)
        log_kw, S, _ = aset.arrays()
        for g in clusters(log_kw, S):
            members = ", ".join(f"#{i + 1} (log k_w {log_kw[i]:.2f}, S {S[i]:.2f})" for i in g)
            print(f"{s}: cluster {members}")
    print(f"-> {path}")


if __name__ == "__main__":
    _main()
