# Gradient LLM

LLM-driven HPLC gradient method search, benchmarked against random search and
Bayesian optimisation on a deterministic HPLC gradient simulator. The simulator
(`sim/`), the optimizers (`optimizers/`, see [optimizers/README.md](optimizers/README.md)),
precomputed method tables (`tables/`), real-compound sets (`hplcsim/`), a real
oligolactide dataset (`real data/`) and the write-up
(`HPLC Gradient Search Report.pdf`) all live in this repository.
The LLM search needs an Azure OpenAI deployment: copy `.env.example` to `.env`
and fill it in (never commit `.env`).

## HPLC Gradient Simulator

A deterministic function from one linear-gradient method to one retention table
and one CRF score. It stands in for an HPLC instrument: optimizers query it one
method at a time and never see the analyte parameters behind it. Built to the
spec in `HPLC Gradient Simulator — Spec.md`.

### Setup

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy pydantic pyyaml pytest
.venv/bin/pip install matplotlib        # optional: sim.chromatogram.plot and the --out PNG
```

### Use

```python
from sim import Simulator, Method

sim = Simulator.from_configs("medium")          # configs/analytes/medium.yaml
result = sim.run(Method(phi0=0.05, phi1=0.60, tG=8.0, T=45.0))
result.retention_table   # [(t_R [min], sigma [min]), ...] sorted, no identities; only peaks seen by t_end
result.crf               # in [-1.6, 1]; nan only if infeasible (outside the bounds)
result.n_missing         # analytes still on the column when the run ended at t_end = t_max - t_eq = 42 min
result.run_time          # min(last peak, t_end) + re-equilibration (min)
result.feasible          # False -> result.reason says why (bounds only)
sim.bounds()             # box over (phi0, dphi, tG, T) that the optimizer may see
```

Vectorised evaluation for grid searches: `sim.run_batch(X)` with `X` of shape
`(M, 4)` over `(phi0, dphi, tG, T)`; 216 000 points take about a second.

Command line:

```bash
.venv/bin/python -m sim hard --phi0 0.05 --phi1 0.48 --tG 5.9 --T 55        # one method, JSON result
.venv/bin/python -m sim.chromatogram hard --phi0 0.05 --phi1 0.5 --tG 6 --T 55 --out chrom.png   # detector trace
.venv/bin/python -m sim.optimum easy medium hard     # rewrite <set>.optimum.json
.venv/bin/python -m sim.generate --seed 7 --out /tmp/sets       # a fresh family of sets
.venv/bin/python -m sim.import_hplcsim --table               # real-compound sets hplcsim_beh / hplcsim_sb (+ optima, tables)
.venv/bin/pytest
```

### Layout

```
sim/
  config.py     pydantic models: InstrumentConfig, Analyte(Set), CRFConfig, Method, Result
  retention.py  log_k(phi, T, analytes), t0(instrument)
  gradient.py   phi_in(t), t_r_closed (LSS, vectorised), t_r_integrated (RK4 reference)
  width.py      sigma from plate count, k at elution, band compression, extra-column term
  pressure.py   standalone Darcy/Kozeny-Carman pressure model; not used by the simulator
  crf.py        resolution(table), crf(table, cfg), crf_truncated(...) for a run that ended before the last peak
  simulator.py  Simulator.run / run_batch; bounds feasibility, truncation at t_end; the only public entry point
  chromatogram.py  detector trace from a retention table: trace, apparent_peaks, plot; CLI
  optimum.py    216 000-point grid + Nelder-Mead; writes <set>.optimum.json
  generate.py   seeded analyte-set generator with the rejection rules
  import_hplcsim.py  fits the HPLC Simulator's measured compounds to log_kw / S / dH -> hplcsim/configs/analytes/
configs/        instrument.yaml (incl. detector block), crf.yaml, analytes/{easy,medium,hard}.yaml (+ .optimum.json)
hplcsim/        a second data root with the same layout (configs/analytes, tables, results) for the two
                real-compound sets hplcsim_beh / hplcsim_sb, plus configs/external/ with the HPLC Simulator
                compound library (43 compounds, CC BY-NC-SA 3.0); see hplcsim/README.md
examples/       hard_optimum.png: the hard set at a CRF 1.0 method, as the detector would show it
tests/          one module per sim module; every equation pinned to a hand-computed case
```

Downstream code imports only `Simulator`, `Method`, `Result`, `InstrumentConfig`.

### Chromatogram

`sim.chromatogram` turns a retention table into the detector trace an operator
would see: a sum of Gaussians, one per row, sampled at the detector rate from
the `detector:` block of `configs/instrument.yaml` (10 Hz by default, 0.1 s
between samples), with optional seeded white noise and a linear baseline drift.
The table carries no identities or amounts, so every peak has area 1.0 unless
`areas=` says otherwise; height is then set by width alone, and a late, broad
peak comes out visibly shorter than an early, sharp one. With the shipped
config and no extra arguments the trace is deterministic and noise-free.

```python
from sim import Simulator, Method
from sim.chromatogram import from_result, apparent_peaks, plot

sim = Simulator.from_configs("hard")
res = sim.run(Method(phi0=0.05, phi1=0.5, tG=6, T=55))
t, signal = from_result(res, sim.instrument)                       # noise-free, deterministic
t, noisy = from_result(res, sim.instrument, noise_sd=0.5, drift=2.0, seed=0)
peaks = apparent_peaks(t, signal)                                  # [(t_apex, height), ...]
plot(t, signal, path="chrom.png", peaks=peaks)                     # needs matplotlib
```

`apparent_peaks` is the operator's view, not the table's: it counts local
maxima that are at least 5 % of the tallest sample and stand that far above
the valleys beside them, so two analytes closer than about 2 sigma fuse into
one visible peak and the count can be lower than the table's fixed row count.
On a noisy trace keep that floor above the noise band (or smooth first), since
an isolated noise spike counts like any other peak.
`from_result` raises `ValueError` for an infeasible result (no run, no trace).
Plotting is the only part of `sim` that needs matplotlib
(`pip install -e ".[plot]"` or `.venv/bin/pip install matplotlib`).

Command line:

```bash
.venv/bin/python -m sim.chromatogram hard --phi0 0.05 --phi1 0.5 --tG 6 --T 55 --out examples/hard_optimum.png
```

prints the apparent peak count with the apex list and saves the PNG when
`--out` is given; `--noise SD`, `--drift RISE` and `--seed N` (default 0) add
the detector artefacts. [examples/hard_optimum.png](examples/hard_optimum.png)
is that command's output: ten peaks, the last one broad and clearly shorter
than the early ones.

### Where the implementation departs from the spec text

Each of these follows the spec's stated intent or its tests where the formula
as written did not.

1. **Sign of the temperature term.** The spec writes
   `log k = log k_w + dH (1/T - 1/T_ref) - S phi` and says dH is negative so
   that retention falls as T rises. With negative dH that expression makes
   retention *rise* with T. The code uses `dH (1/T_ref - 1/T)`, which is the
   van 't Hoff form and gives the stated behaviour (temperature-direction test).
2. **Regime 3 (analyte still on the column when the gradient ends).** The spec's
   `t_R = t_D + t_G + t_0 (1 - x_G)(1 + k_1)` mixes the inlet-time and real-time
   frames and is off by `x_G t_0` (up to ~0.45 min, enough to fail the 1 %
   agreement with the integrator). The code derives the exact expression from
   the same migration integral: `t_R = t_D + t_G + t_0 + (1 - x_G) t_0 k_1`,
   which matches the RK4 reference to numerical precision.
3. **Pressure removed from the study.** The spec's pressure model and P_max check are
   not part of the simulator: `Result` has no pressure field, feasibility checks only
   bounds, and no table or optimizer sees pressure. With the defaults the
   limit never bound anyway (worst case about 530 bar against 600). The model survives
   as the standalone `sim/pressure.py`, unit-tested, with the viscosity coefficient
   0.87 instead of the spec's 1.2 so its maximum sits at phi = 0.20 as the spec intends.
4. **Run time: truncation instead of infeasibility.** The spec makes a method whose
   last peak plus re-equilibration exceeds t_max infeasible (nan CRF, empty table).
   The code instead ends the run at `t_end = t_max - t_eq` (42 min) and reports what
   the detector saw by then: peaks eluting later are not observed, the method stays
   feasible, `retention_table` holds only the observed peaks, `n_missing` counts the
   rest, and `run_time` is `min(t_R,last, t_end) + t_eq`. The CRF (`crf_truncated`)
   credits adjacent observed pairs as usual, gives each missing analyte one pair with
   zero credit (the denominator stays n - 1, so a lost peak scores exactly like a
   fully co-eluting one) and charges the time term at `min(t_R,last observed, t_end)`,
   i.e. at t_end whenever anything is missing; its floor is
   `-w_t (t_end - t_target) / t_target = -1.6`. Why: the time term already penalises
   late runs, so the cut-off was a cliff in an otherwise smooth landscape (a run
   ending at 41.9 min scored about -1.6 while 42.1 min scored nan); a truncated
   chromatogram is what an operator actually sees when the run ends; and scoring a
   missing peak as co-eluting closes the loophole of a method looking good because
   its hardest peak never showed up. Only the bounds make a method infeasible.
5. **dphi box.** `bounds()` exposes `dphi in [0.10, 0.90]`. Since phi1 must also
   stay below 0.95, the box is a superset of the feasible region and the
   `phi1 <= 0.95` constraint is re-checked in `run()` (reason "bounds").

### Difficulty of the generated sets

All three stored optima reach CRF 1.0 (the spec floors are 0.9 / 0.9 / 0.7), so
difficulty shows up as the size of the region an optimizer must find, not as a
lower ceiling. Share of feasible grid points with CRF >= 0.99:

| set        | n  | CRF >= 0.99 |
|------------|----|-------------|
| easy       | 6  | 5.7 %       |
| medium     | 8  | 2.1 %       |
| hard       | 10 | 0.4 %       |

The spec's fourth set, rp_variant (the medium set with S permuted among analytes
and dH scaled by 1.5), was dropped from the study: permuting S breaks up the near
pairs, so it came out easier than medium instead of being a same-difficulty
transfer test.

### Real-compound sets from the HPLC Simulator library

The generated sets are random draws, so an optimizer that has seen them is not
proof of anything about real chemistry. `sim/import_hplcsim.py` adds two sets
built from measured compounds: the library of the HPLC Simulator (Boswell,
Stoll, Carr et al., *J. Chem. Educ.* 2013, 90, 198; copied from
github.com/pgboswell/hplc-simulator into `hplcsim/configs/external/`, CC BY-NC-SA 3.0).
Everything from it lives under `hplcsim/`, a second data root with the project's
layout; `sim.config.set_root` finds a set by name in whichever root holds its
YAML, and its optimum, table and results stay in that root.
Each compound there has a measured surface `log10 k = (w + m T) + (v + n T) phi`
per organic modifier. The importer fits that surface to the simulator's
`log_kw + dH (1/T_ref - 1/T) - S phi` by weighted least squares over the
method box (full weight where 0.5 <= k <= 50, the band a gradient actually
moves) and writes the three parameters plus a `<set>.import.json` with the
per-compound residuals. The fit is close for all but the two weakly retained
compounds whose slope drifts strongly with temperature (caffeine: RMSE 0.12 in
log k inside the window, 2-acetylfuran 0.05); every other compound is within
0.03 RMSE and 0.07 max.

| set          | source phase / modifier         | n  | optimum CRF | grid rows within 1 % of it |
|--------------|---------------------------------|----|-------------|----------------------------|
| hplcsim_beh  | Waters Acquity BEH C18 / ACN    | 21 | 0.942       | 11 (0.05 %)                |
| hplcsim_sb   | Agilent Zorbax SB-C18 / ACN     | 22 | 0.945       | 15 (0.07 %)                |

Neither set reaches CRF 1.0 within 42 min, and the region within 1 % of the
optimum is an order of magnitude smaller than the hard set's, so they are the
tougher benchmark. With a budget of 25 injections (10 seeds) random search ends
at a mean best CRF of 0.86 (beh) / 0.84 (sb) and BO at 0.92 / 0.89, each finding
the optimum region in one seed out of ten; the LLM search (10 campaigns each,
`optimizers.llm_repeats`) reached within 1 % of the optimum in every campaign,
median 6.5 (beh) and 6 (sb) injections, range 3 to 13, and all ten per set ended
on the same best grid method at 0.938 / 0.944 (`hplcsim/results/hplcsim_*_search.png`,
`_runs.csv`, `_walks.png`; replay with `optimizers.run ... --llm-from-transcripts`).
`--solvent methanol`, `--ids` / `--names` and `--name` give methanol variants
and subsets (e.g. a ten-compound mixture); `--table` also writes the table.

### Precomputed tables

`tables/<set>.csv` holds 22 080 methods per set on a round-number grid with the
CRF, feasibility, diagnostics and retention table for each, ready for random
search, Bayesian optimisation or LLM-driven search as a discrete oracle. See
[tables/README.md](tables/README.md); regenerate with `python -m sim.table`
(`--sets hplcsim_beh hplcsim_sb` for the real-compound sets, whose tables land in
`hplcsim/tables/`).
