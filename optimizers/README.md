# Search strategies over the method tables

Three optimisers that treat `tables/<set>.csv` as a discrete oracle: each query is
one injection, the budget is fixed, and regret is measured against the stored
optimum in `configs/analytes/<set>.optimum.json`.

```bash
.venv/bin/python -m optimizers.run --set hard --budget 25 --seeds 10
.venv/bin/python -m optimizers.run --set all
.venv/bin/python -m optimizers.run --set hard --methods random bo llm        # + the LLM search (25 API calls per seed)
.venv/bin/python -m optimizers.run --set all --methods random bo llm --llm-seeds 3
.venv/bin/python -m optimizers.repeats --method bo --repeats 50     # BO 50 times on each set
.venv/bin/python -m optimizers.repeats --methods random bo          # both overlaid: results/crf_over_runs.png
.venv/bin/python -m optimizers.llm_repeats --repeats 50 --workers 6 --figure   # 50 LLM campaigns per set (3 750 API calls), then all three overlaid
.venv/bin/python -m optimizers.llm_repeats --sets hplcsim_beh hplcsim_sb --repeats 10 --workers 6   # the real-compound sets; transcripts land in hplcsim/results/
.venv/bin/python -m optimizers.run --set hplcsim_sb --methods random bo llm --llm-seeds 10 --llm-from-transcripts   # replay those campaigns, no API calls
.venv/bin/python -m optimizers.run --set hplcsim_sb --methods random bo llm   # real compounds (see README: import_hplcsim)
```

`--set` takes any analyte set that has a table and an optimum, so the two
real-compound sets `hplcsim_beh` and `hplcsim_sb` (21 and 22 measured
compounds fitted from the HPLC Simulator library) run unchanged; `--set all`
still means the three generated sets.

Prints every injection of the shown seed for both methods (with the number of
peaks missing because the run ended at 42 min), a summary over seeds, and writes
`results/<set>_search.png` (three panels: CRF per injection, best CRF so far,
regret) plus `results/<set>_runs.csv` with every injection of every seed. A
truncated run (peaks missing) is drawn as a red triangle at its observed CRF; it
is an ordinary, poorly scored injection, not a failure.

| module             | what it does                                                                 |
|--------------------|------------------------------------------------------------------------------|
| `oracle.py`        | `TableOracle` (rows, unit-cube features, budget, regret) and `SearchRun` logs |
| `random_search.py` | seeded permutation of the rows                                               |
| `bo.py`            | GP (ARD Matern-5/2, hyperparameters refit each step) + expected improvement over all unqueried rows; truncated runs enter at their observed CRF, an out-of-bounds injection (none in the default tables) at a penalty of -0.5 |
| `plot.py`          | the three-panel figure                                                       |
| `llm_search.py`    | one chat call per injection against the Azure OpenAI deployment in `.env`: the model reads the retention table (with pairwise R, CRF and its terms) and proposes the next method as JSON; the harness snaps it to the grid, never charges a repeat, and writes `results/llm_transcripts/<set>_seed<k>.jsonl` |
| `llm_system_prompt.txt` | what the model is told: instrument, grid, CRF rule, linear-solvent-strength theory, how to work; generic across sets |
| `check_azure.py`   | `python -m optimizers.check_azure` verifies the deployment in `.env`; `make_client` is the client factory the search uses |
| `run.py`           | CLI, summary statistics, CSV export; `--methods` picks the optimisers, `--llm-seeds` and `--llm-n-init` size the LLM runs; `--llm-from-transcripts` replays finished campaigns from the set's `results/llm_transcripts/` instead of calling the API |
| `repeats.py`       | one method run many times per set (seed = repeat number): CRF of injection n averaged over repeats, with the share of infeasible injections; writes `results/<method>_crf_over_runs.{png,csv}` |
| `llm_repeats.py`   | the LLM search run many times per set, concurrently and resumably (a finished campaign is its transcript on disk; `--summary-only` makes no API calls); writes `llm_crf_over_runs.csv` and, with `--figure`, the three-method `crf_over_runs.{png,csv}` into the sets' data root (`results/`, or `hplcsim/results/` when every set given lives there) |

BO's first five injections are the same rows random search draws for that seed,
so the two curves share a starting point and diverge only through the surrogate.

```python
from optimizers import TableOracle, random_search, bo_search
oracle = TableOracle("hard")
run = bo_search(oracle, budget=25, seed=0)
run.best_curve(); run.regret_curve(); run.runs_to_optimum()
```

The LLM search plugs in the same way (`llm_search(oracle, budget=25, seed=0)`):
the model sees only what derives from the retention table and the public
configs, never the analyte parameters or the optimum; `--llm-n-init k` gives it
the same first `k` rows random search and BO draw for that seed. Design notes in
`llm_search_design.md`.

## Landscape and trajectories

```bash
.venv/bin/python -m optimizers.landscape --set hard --seed 0
```

Writes `<set>_landscape.png`, `<set>_trajectory_{random,bo,llm}.png` and `<set>_walks.png` into the set's results root. The LLM trajectory is replayed from `llm_transcripts/<set>_seed<k>.jsonl` and skipped with a note when that seed has no finished transcript; the walks figure puts the three seed-`k` runs side by side on the phi1 x tG projection.
The search space is four-dimensional, so the landscape is a corner plot: each
lower-triangle panel shows, for one pair of variables, the best CRF over the other
two (a max-projection: "is there any good method with these two values?"), grey
cells have no feasible method at all, and the diagonal shows the best CRF against
each variable alone. The white star is the known optimum. Trajectory figures overlay
the shown seed's injections coloured by order, joined in sequence, with truncated
runs (peaks missing) ringed by a red triangle, out-of-bounds injections (none on the
default grid) as red crosses, the best injection ringed, and ticks on the diagonal
marking where each injection fell. The max-projection is deliberately optimistic:
a point can sit on a dark cell and still score poorly if its other two variables
are off, which is exactly what the hard set's two near pairs do.
