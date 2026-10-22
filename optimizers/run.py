"""Run random search, BO and the LLM search on one or all analyte sets; print the injections, plot efficiency and regret.

    python -m optimizers.run --set hard --budget 25 --seeds 10          # random + BO
    python -m optimizers.run --set hard --methods random bo llm         # + the LLM search (one seed, 25 API calls)
    python -m optimizers.run --set all --methods random bo llm --llm-seeds 3
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from sim.config import results_dir

from .bo import bo_search
from .llm_search import llm_search, write_transcript
from .oracle import SearchRun, TableOracle, format_run
from .plot import plot_comparison
from .random_search import random_search

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SET_NAMES = ("easy", "medium", "hard")
METHODS = {"random": random_search, "bo": bo_search, "llm": llm_search}
DEFAULT_METHODS = ("random", "bo")


def summarise(runs: list[SearchRun], tol: float = 0.01) -> dict:
    best = np.array([r.best_curve()[-1] for r in runs])
    hits = [r.runs_to_optimum(tol) for r in runs]
    found = [h for h in hits if h is not None]
    return {
        "method": runs[0].method,
        "mean_best": float(np.nanmean(best)),
        "min_best": float(np.nanmin(best)),
        "found_frac": len(found) / len(runs),
        "median_runs_to_optimum": float(np.median(found)) if found else float("nan"),
        "mean_final_regret": float(np.nanmean([r.regret_curve()[-1] for r in runs])),
    }


def write_runs_csv(runs: list[SearchRun], path: Path) -> Path:
    rows = [row for r in runs for row in r.to_rows()]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def run_set(set_name: str, budget: int, seeds: int, out_dir: Path | None = None, shown_seed: int = 0,
            verbose: bool = True, methods: tuple[str, ...] = DEFAULT_METHODS, llm_seeds: int = 1,
            llm_n_init: int = 0, llm_from_transcripts: bool = False) -> dict[str, list[SearchRun]]:
    """``out_dir`` defaults to the ``results/`` folder of the set's own data root.

    With ``llm_from_transcripts`` the LLM campaigns are replayed from the finished transcripts in the
    set's results root (``llm_transcripts/<set>_seed<k>.jsonl``, e.g. written by ``optimizers.llm_repeats``)
    instead of calling the API; a missing or incomplete transcript is an error, not a silent gap.
    """
    out_dir = Path(out_dir) if out_dir is not None else results_dir(set_name)
    oracle = TableOracle(set_name)
    runs_by_method: dict[str, list[SearchRun]] = {}
    for method in methods:
        fn = METHODS[method]
        n_seeds = llm_seeds if method == "llm" else seeds
        t = time.perf_counter()
        if method == "llm" and llm_from_transcripts:
            from .llm_repeats import load_run, transcript_path
            runs = []
            for s in range(n_seeds):
                run = load_run(oracle, s, budget)
                if run is None:
                    raise FileNotFoundError(f"no finished transcript for {set_name} seed {s} "
                                            f"(expected {transcript_path(set_name, s)} with {budget} injections)")
                runs.append(run)
            runs_by_method[method] = runs
        elif method == "llm":
            runs = []
            for s in range(n_seeds):
                transcript: list[dict] = []
                runs.append(fn(oracle, budget=budget, seed=s, n_init=llm_n_init, transcript=transcript, verbose=verbose))
                out_dir.mkdir(parents=True, exist_ok=True)
                write_transcript(transcript, out_dir / "llm_transcripts" / f"{set_name}_seed{s}.jsonl")
            runs_by_method[method] = runs
        else:
            runs_by_method[method] = [fn(oracle, budget=budget, seed=s) for s in range(n_seeds)]
        elapsed = time.perf_counter() - t
        if verbose:
            shown = next((r for r in runs_by_method[method] if r.seed == shown_seed), runs_by_method[method][0])
            print(format_run(shown))
            print(f"   ({n_seeds} seeds in {elapsed:.1f} s)\n")
    if verbose:
        print(f"== {set_name}: summary over {seeds} seeds ({llm_seeds} for llm), budget {budget}, optimum {oracle.optimum:.3f}")
        print(f"{'method':8s} {'mean best':>9s} {'worst seed':>10s} {'found optimum':>13s} {'median injections':>17s} {'mean regret':>11s}")
        for method, runs in runs_by_method.items():
            s = summarise(runs)
            med = f"{s['median_runs_to_optimum']:.0f}" if np.isfinite(s["median_runs_to_optimum"]) else "-"
            print(f"{method:8s} {s['mean_best']:9.3f} {s['min_best']:10.3f} {s['found_frac']:12.0%} {med:>17s} {s['mean_final_regret']:11.3f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = plot_comparison(runs_by_method, out_dir / f"{set_name}_search.png", shown_seed=shown_seed)
    csv_path = write_runs_csv([r for runs in runs_by_method.values() for r in runs], out_dir / f"{set_name}_runs.csv")
    if verbose:
        print(f"   plot -> {png}\n   runs -> {csv_path}\n")
    return runs_by_method


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Random search vs Bayesian optimisation over the method tables.")
    p.add_argument("--set", default="hard", help="easy | medium | hard | all")
    p.add_argument("--budget", type=int, default=25, help="injections per run")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--shown-seed", type=int, default=0, help="seed whose injection log is printed and plotted")
    p.add_argument("--out", type=Path, default=None, help="default: the results/ folder of each set's data root")
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=list(METHODS),
                   help="which optimisers to run; 'llm' calls the Azure deployment once per injection")
    p.add_argument("--llm-seeds", type=int, default=1, help="repeats of the LLM search (each costs `budget` API calls)")
    p.add_argument("--llm-n-init", type=int, default=0, help="warm-start injections shared with random search and BO")
    p.add_argument("--llm-from-transcripts", action="store_true",
                   help="replay the LLM campaigns from finished transcripts in the set's results root; no API calls")
    a = p.parse_args(argv)
    sets = SET_NAMES if a.set == "all" else (a.set,)
    for name in sets:
        run_set(name, a.budget, a.seeds, a.out, a.shown_seed, methods=tuple(a.methods),
                llm_seeds=a.llm_seeds, llm_n_init=a.llm_n_init, llm_from_transcripts=a.llm_from_transcripts)


if __name__ == "__main__":
    _main()
