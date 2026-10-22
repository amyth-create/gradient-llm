# Method -> CRF tables

One CSV per analyte set, one row per method on a round-number grid inside the
instrument bounds. Regenerate or change the resolution with

```bash
.venv/bin/python -m sim.table                                   # defaults below
.venv/bin/python -m sim.table --phi-step 0.10 --tG-step 5 --out tables/coarse
```

Default grid: phi0 0.05–0.60 and phi1 0.15–0.95 in steps of 0.05 (phi1 >= phi0 + 0.10),
tG 2–40 min in steps of 2, T 25–60 °C in steps of 5. That is 22 080 methods per set.

| set        | rows   | feasible | truncated (peaks missing) | rows with CRF >= 0.99 | best grid CRF | stored optimum |
|------------|--------|----------|---------------------------|-----------------------|---------------|----------------|
| easy       | 22 080 | 100 %    | 14.3 %                    | 1 112                 | 1.0           | 1.0            |
| medium     | 22 080 | 100 %    | 3.5 %                     | 500                   | 1.0           | 1.0            |
| hard       | 22 080 | 100 %    | 12.7 %                    | 82                    | 1.0           | 1.0            |
| hplcsim_beh | 22 080 | 100 %   | 11.0 %                    | 0 (11 within 1 % of optimum) | 0.9378 | 0.9419         |
| hplcsim_sb | 22 080 | 100 %    | 16.2 %                    | 0 (15 within 1 % of optimum) | 0.9438 | 0.9448         |

The two `hplcsim_*` sets are real compounds (21 and 22 of them) fitted from the
HPLC Simulator library by `python -m sim.import_hplcsim`; see the main README.
Their tables live in `hplcsim/tables/`, not here: `hplcsim/` is a second data
root with the same layout, and `python -m sim.table --sets hplcsim_beh hplcsim_sb`
writes there because that is where their YAMLs are.

The run ends at t_end = t_max - t_eq = 42 min. Analytes still on the column then
are not observed: such rows stay feasible with a finite (poor) CRF, a shorter
retention table and `n_missing > 0`. No row of the default grid is infeasible;
the `feasible` / `reason` columns exist for out-of-bounds methods only.
Pressure is not part of the study.

## Columns

| column               | meaning                                                                 |
|----------------------|-------------------------------------------------------------------------|
| method_id            | row index, stable for a given grid                                      |
| phi0, phi1, tG, T    | the method: initial / final organic fraction, gradient time (min), °C   |
| feasible             | True / False; False only for a method outside the bounds (never on the default grid) |
| reason               | empty, or why the method is infeasible (bounds)                         |
| crf                  | the score over the observed peaks; empty only when infeasible           |
| resolution_term      | sum over observed adjacent pairs of min(R / 1.5, 1), divided by n - 1; a lost peak is one pair with zero credit, so in [0, 1] |
| time_term            | 0.5 * max(0, (t_last - 10) / 10) with t_last = 42 (t_end) whenever a peak is missing; subtracted from the resolution term |
| min_resolution       | worst resolution among observed adjacent pairs; empty when fewer than two peaks were observed |
| n_unresolved_pairs   | observed adjacent pairs with R < 1.5 (add n_missing for the pairs the CRF scores as unresolved) |
| t_first, t_last      | first and last observed retention time (min); empty when nothing was observed |
| run_time             | min(t_R of the last analyte, 42) + 3 min re-equilibration               |
| n_missing            | analytes that had not eluted when the run ended at 42 min               |
| n_observed           | rows in the retention table, n - n_missing                              |
| retention_table      | JSON list of [t_R, sigma] of the observed peaks sorted by t_R, no identities; empty when infeasible |

## Using the tables as a discrete oracle

```python
from sim.table import load_table, lookup
t = load_table("tables/hard.csv")

# random search: sample rows
import numpy as np
rng = np.random.default_rng(0)
picks = rng.choice(len(t["crf"]), size=25, replace=False)
best = np.nanmax(t["crf"][picks])

# Bayesian optimisation: the rows are the candidate set
X = np.column_stack([t["phi0"], t["phi1"], t["tG"], t["T"]])     # features
y = t["crf"]                                                     # finite on the whole default grid

# LLM search: snap a proposed method to the nearest grid row
row = lookup(t, phi0=0.07, phi1=0.5, tG=6.3, T=53)
row["crf"], row["retention_table"]
```

The retention table is what a reasoning agent is allowed to see; the analyte
parameters are not in these files. Regret against the stored optimum is
`optimum_crf - crf`; a method within 1 % of the optimum counts as found.
