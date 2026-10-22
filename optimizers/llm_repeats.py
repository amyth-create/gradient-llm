"""Repeated LLM-search campaigns, run concurrently and resumably, so the LLM gets the same 50 repeats as random search and BO.

    .venv/bin/python -m optimizers.llm_repeats --repeats 50 --workers 6
    .venv/bin/python -m optimizers.llm_repeats --sets hard --repeats 50 --summary-only   # no API calls, just the tables and figure

Each campaign is one ``llm_search`` with ``n_init=0`` (the seed only labels the
repeat; proposals are nondeterministic). A finished campaign is persisted as
``results/llm_transcripts/<set>_seed<k>.jsonl``; on the next invocation any
seed whose transcript already holds ``budget`` injections is loaded from disk
instead of re-run, so the batch can be interrupted and resumed. A campaign that
fails (API error, unparseable output) is retried up to ``--attempts`` times and
otherwise skipped with a message; it can be re-run later.

Outputs, in ``results/``:

- ``llm_crf_over_runs.csv``: every injection of every LLM campaign, same columns as the other repeats CSVs
- ``crf_over_runs.png`` / ``crf_over_runs.csv``: random search, BO and the LLM search overlaid, 50 repeats each (``--figure``)
- the per-injection and checkpoint tables printed by ``optimizers.repeats``, for all three methods
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sim.config import results_dir

from .check_azure import load_settings, make_client
from .efficiency import CHECKPOINTS, checkpoint_table
from .llm_search import llm_search, write_transcript
from .oracle import SearchRun, TableOracle
from .plot import LABELS
from .repeats import METHODS as BASE_METHODS, per_injection_table, plot_methods_over_runs, repeat_search, write_runs_csv

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TRANSCRIPT_DIR = RESULTS_DIR / "llm_transcripts"
SET_NAMES = ("easy", "medium", "hard")
_print_lock = threading.Lock()


def _say(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def transcript_path(set_name: str, seed: int) -> Path:
    """``results/llm_transcripts/<set>_seed<k>.jsonl`` in the set's own data root."""
    return results_dir(set_name) / "llm_transcripts" / f"{set_name}_seed{seed}.jsonl"


def load_run(oracle: TableOracle, seed: int, budget: int) -> SearchRun | None:
    """Rebuild the SearchRun of a finished campaign from its transcript; None if absent or incomplete."""
    path = transcript_path(oracle.set_name, seed)
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) < budget:
        return None
    run = oracle.new_run("llm", seed, budget)
    for row in rows[:budget]:
        oracle.query(run, int(row["index"]))
    return run


def run_campaign(oracle: TableOracle, seed: int, budget: int, client, model: str, attempts: int) -> SearchRun | None:
    for attempt in range(1, attempts + 1):
        transcript: list[dict] = []
        t = time.perf_counter()
        try:
            run = llm_search(oracle, budget=budget, seed=seed, n_init=0, client=client, model=model, transcript=transcript)
        except Exception as e:  # noqa: BLE001 - any API or parsing failure: report, back off, retry
            _say(f"  {oracle.set_name:6s} seed {seed:2d}  attempt {attempt}/{attempts} failed after "
                 f"{len(transcript)} injections: {type(e).__name__}: {str(e)[:160]}")
            time.sleep(min(60, 5 * attempt))
            continue
        write_transcript(transcript, transcript_path(oracle.set_name, seed))
        hit = run.runs_to_optimum()
        _say(f"  {oracle.set_name:6s} seed {seed:2d}  best {run.best_curve()[-1]:.3f}  "
             f"{'optimum at ' + str(hit) if hit else 'not found':16s}  {time.perf_counter() - t:5.0f} s")
        return run
    _say(f"  {oracle.set_name:6s} seed {seed:2d}  GAVE UP after {attempts} attempts")
    return None


def collect(sets: tuple[str, ...], repeats: int, budget: int, workers: int, attempts: int,
            summary_only: bool) -> dict[str, list[SearchRun]]:
    oracles = {s: TableOracle(s) for s in sets}
    runs: dict[str, dict[int, SearchRun]] = {s: {} for s in sets}
    todo: list[tuple[str, int]] = []
    for s in sets:
        for seed in range(repeats):
            run = load_run(oracles[s], seed, budget)
            if run is not None:
                runs[s][seed] = run
            else:
                todo.append((s, seed))
    loaded = sum(len(v) for v in runs.values())
    _say(f"{loaded} finished campaigns loaded from "
         f"{', '.join(str(results_dir(s) / 'llm_transcripts') for s in sets)}, {len(todo)} to run")
    if todo and not summary_only:
        settings = load_settings()
        client = make_client(settings, max_retries=5, timeout=180.0)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_campaign, oracles[s], seed, budget, client, settings.deployment, attempts): (s, seed)
                       for s, seed in todo}
            done = 0
            for fut in as_completed(futures):
                s, seed = futures[fut]
                run = fut.result()
                done += 1
                if run is not None:
                    runs[s][seed] = run
                if done % 10 == 0 or done == len(todo):
                    _say(f"-- {done}/{len(todo)} campaigns done, {time.perf_counter() - t0:.0f} s elapsed")
    return {s: [runs[s][k] for k in sorted(runs[s])] for s in sets}


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Repeated LLM-search campaigns, concurrent and resumable.")
    p.add_argument("--sets", nargs="+", default=list(SET_NAMES))
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--budget", type=int, default=25)
    p.add_argument("--workers", type=int, default=6, help="concurrent campaigns (each is one API call at a time)")
    p.add_argument("--attempts", type=int, default=3, help="tries per campaign before giving up on it")
    p.add_argument("--summary-only", action="store_true", help="no API calls: summarise the transcripts on disk")
    p.add_argument("--figure", action="store_true", help="also rerun random search and BO and write the three-method figure")
    a = p.parse_args(argv)
    sets = tuple(a.sets)

    llm_runs = collect(sets, a.repeats, a.budget, a.workers, a.attempts, a.summary_only)
    n = {s: len(r) for s, r in llm_runs.items()}
    _say(f"\ncampaigns per set: {n}")
    if any(v == 0 for v in n.values()):
        return 1
    # Aggregate files go to the sets' own data root when they share one (e.g. hplcsim/results), else results/.
    roots = {results_dir(s) for s in sets}
    out_root = roots.pop() if len(roots) == 1 else RESULTS_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = write_runs_csv([r for s in sets for r in llm_runs[s]], out_root / "llm_crf_over_runs.csv")
    _say(f"-> {csv_path}")
    _say(f"\n{LABELS['llm']}: mean CRF of injection n (share truncated, share infeasible)")
    _say(per_injection_table(llm_runs))
    _say(f"\n{LABELS['llm']}: mean best CRF after n injections (share of repeats within 1 % of the optimum)")
    _say(checkpoint_table(llm_runs, checkpoints=tuple(c for c in CHECKPOINTS if c <= a.budget)))

    if a.figure:
        all_runs: dict[str, dict[str, list[SearchRun]]] = {}
        for s in sets:
            all_runs[s] = {m: repeat_search(m, s, a.repeats, a.budget) for m in BASE_METHODS}
            all_runs[s]["llm"] = llm_runs[s]
        for m in BASE_METHODS:
            by_set = {s: all_runs[s][m] for s in sets}
            _say(f"\n{LABELS[m]}: mean best CRF after n injections (share within 1 % of the optimum)")
            _say(checkpoint_table(by_set, checkpoints=tuple(c for c in CHECKPOINTS if c <= a.budget)))
        png = plot_methods_over_runs(all_runs, out_root / "crf_over_runs.png", n_init=8)
        csv_all = write_runs_csv([r for s in sets for m in all_runs[s] for r in all_runs[s][m]], out_root / "crf_over_runs.csv")
        _say(f"-> {png}\n-> {csv_all}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
