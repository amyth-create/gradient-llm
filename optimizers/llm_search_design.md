# LLM-driven search over the method tables: design

Third optimiser beside `random_search` and `bo_search`. Same contract: a
`TableOracle`, a budget, a seed, one `oracle.query` per injection, a `SearchRun`
back, so `optimizers.run` can print, plot and score it unchanged.

## 1. Shape of the loop

**Harness-driven, one Claude call per injection.** The Python harness owns the
budget, the snapping, the logging and the conversation; the model only ever
answers one question: *given everything so far, which method next?*

```
for k in 1..budget:
    user msg   = rendered observation of injection k-1 + running log + "propose injection k"
    assistant  = {"phi0", "phi1", "tG", "T", "rationale"}      (structured JSON output)
    harness    : snap to the grid -> oracle.query(run, index) -> next user msg
```

Why not let the model drive a tool loop (`inject(...)` as a tool, model loops
until done)?

- The harness-driven loop maps 1:1 onto `oracle.query`, so seeds, budgets and
  `SearchRun` logging stay identical to random search and BO.
- Opus 5.5 rejects forced `tool_choice`, and a model-driven loop can end its turn
  early; with structured outputs every call is guaranteed to yield one proposal.
- Nothing is lost: analysis the model might want from a tool (pairwise R,
  elution phi, time terms) is cheap to precompute and hand over in the message.

The tool-use variant is still worth a branch later (section 7).

**Append-only conversation.** Every `response.content` (thinking blocks
included) goes back verbatim; the running log is re-rendered inside each *new*
user message, never by editing an old one. That keeps the cached prefix valid
and satisfies the preserved-thinking check on 5.x models.

**Snapping and repeats.** The prompt states the grid, so proposals should land on
it; the harness snaps anyway (same scaled distance as `sim.table.lookup`) and
tells the model when snapping changed the method. A proposal that snaps to an
already-queried row is *not* charged: the harness replies "that is the method
from injection k, propose another" and re-asks, up to three times, then falls
back to the nearest unqueried row so the run always completes the budget.

**Warm start.** `n_init` random injections from `random_order(oracle, seed)`,
the same rows BO starts from, go into the first user message as history. Run
two configurations: `n_init=0` (chemist from a blank page) and `n_init=8`
(matched to BO's design). Seeds then mean the same thing across all three
methods; with `n_init=0` the seed only labels the repeat.

## 2. What crosses the boundary

Everything the model sees derives from the retention table plus the public
instrument and CRF configs. Nothing from the analyte YAML, the optimum JSON or
the CSV beyond the queried row.

| sent to the model | not sent |
|---|---|
| method as run (and the pre-snap proposal if it differed) | `method_id`, set name, difficulty label |
| feasible / reason string | the stored optimum, regret, "distance to optimum" |
| retention table: row, t_R, sigma, R to next row | analyte parameters (log kw, S, dH) |
| CRF, resolution term, time term, min R, number of pairs with R < 1.5, t_last, run time | `t_last` / `min_resolution` of an *infeasible* row (the instrument would have aborted; the table happens to store them) |
| budget used / remaining, best CRF so far and at which injection | anything about other analyte sets |

The harness computes the derived columns itself from the retention table with
`sim.crf.resolution`, `resolution_term`, `time_term` and `CRFConfig.load()`;
`TableOracle` only needs to expose what it already loads.

Memorisation guard: the system prompt must stay generic (LSS theory, the
scoring rule, method-development practice). No "optima tend to be around
phi1 0.5 and 55 C". Check by running the same prompt on a fresh
`python -m sim.generate --seed N` set.

## 3. System prompt (draft)

Stable text, cached with one `cache_control` breakpoint. Only the per-injection
user messages vary.

```text
You are developing a reversed-phase UHPLC gradient method for an unknown mixture.
You choose the method; the instrument runs it and returns the chromatographic
result. Your goal is the highest CRF score reachable within a fixed number of
injections.

## Instrument (fixed for the campaign)
- Column 100 x 2.1 mm, 1.8 um particles, N = 15 000 plates. Flow fixed at 0.5 mL/min.
- Dead time t0 = 0.45 min. Dwell (gradient delay) tD = 0.20 min: the programmed
  gradient reaches the column inlet at 0.20 min and the outlet 0.45 min after that,
  so at the outlet the gradient runs from 0.65 min to tG + 0.65 min.
- Extra-column dispersion sigma_ec = 0.02 min; no peak is narrower than that.
- Run-time limit: last peak + 3 min re-equilibration must be within 45 min, i.e. the
  last peak must elute by 42 min. Otherwise the run fails and returns no data.

## Method (what you choose)
A linear gradient from phi0 to phi1 (organic fraction) over tG minutes, then a hold
at phi1 until the last peak elutes, at column temperature T.
The instrument only runs grid values; anything else is rounded to the nearest one:
- phi0: 0.05 to 0.60 in steps of 0.05
- phi1: 0.15 to 0.95 in steps of 0.05, and phi1 >= phi0 + 0.10
- tG:   2 to 40 min in steps of 2
- T:    25 to 60 C in steps of 5

## What a run returns
For a feasible run: the retention table, one row per analyte sorted by retention
time, with t_R (min), sigma (peak standard deviation, min) and the resolution to
the next row, R = (t_R,next - t_R) / (2 (sigma + sigma_next)); plus the CRF and
its two terms. The number of rows equals the number of analytes and never
changes: co-elution shows up as R near 0, not as a missing row. Rows carry no
identities. The same compound can be row 3 in one run and row 5 in another when
the elution order changes, so compare runs by pattern, not by row number.
For an infeasible run you get only the reason.

## The score
CRF = mean over adjacent pairs of min(R / 1.5, 1)  -  0.5 * max(0, (t_last - 10) / 10)
- A pair earns full credit at R >= 1.5 and nothing beyond it, so the score is lost
  at the worst pairs.
- A last peak after 10 min costs 0.05 per extra minute.
- 1.0 is the maximum: every pair at R >= 1.5 with the last peak within 10 min.
  The best method is therefore the shortest fully resolved one.

## Retention model (what the chemistry lets you infer)
Reversed-phase retention follows the linear-solvent-strength model: for each
compound log10 k = log kw - S * phi at a given temperature, with compound-specific
log kw (roughly 1.5 to 4.5) and S (roughly 3 to 10). Raising temperature lowers
retention for every compound by a compound-specific amount, so T shifts
selectivity modestly (it can reorder a close pair) and shortens the run.
Consequences you can use:
- The organic fraction at which a peak leaves the column is about
  phi_e = phi0 + (phi1 - phi0) * (t_R - 0.65) / tG, clamped to [phi0, phi1].
  A peak at or before about 0.6 min is essentially unretained: lower phi0.
  A peak after tG + 0.65 min eluted during the hold at phi1: raise phi1 or lengthen tG.
- Trimming the gradient window to the elution window (phi0 a little below phi_e of
  the first peak, phi1 a little above phi_e of the last) keeps every peak inside
  the gradient and spends the run time where the peaks are.
- Gradient steepness b = t0 * S * (phi1 - phi0) / tG sets resolution. Shallower
  (longer tG, narrower window) spreads pairs whose S differ and widens peaks a
  little; steeper compresses and shortens. A pair whose members have nearly equal
  S barely responds to slope and needs T or a different phi window instead.
- Two runs that differ only in tG show which pairs respond to slope (their R
  changes a lot) and which do not.

## How to work
- Propose exactly one method per injection. Each injection spends one unit of
  budget; a proposal that rounds to a method already run is wasted, so stay on the
  grid and do not repeat.
- Early injections buy information: where the elution window is, which pairs are
  critical, how they respond to slope and temperature. Later injections exploit it.
  Keep the best method in mind and refine around it as the budget runs out.
- Once every pair is at R >= 1.5, the remaining gain is time: shorten tG, narrow the
  window or raise T until a pair starts to fall below 1.5, then back off one step.
- Avoid infeasible runs: a shallow gradient ending at a low phi1 can hold the last
  compound on the column past 42 min.

Answer with JSON only:
{"phi0": <number>, "phi1": <number>, "tG": <number>, "T": <number>,
 "rationale": "<one or two sentences>"}
```

Notes on the draft:

- It gives theory and practice, not a script. Opus 5.5 and Fable do worse with
  over-prescriptive step lists; the "How to work" section is deliberately short.
- The elution-window and steepness rules are standard method-development
  practice and hold for any analyte set, so they are fair game.
- The number of analytes is not stated; the first feasible table reveals it.
- Ablation to run: delete the "Retention model" section and measure the drop.
  That number is the value of the domain prior, which is the interesting result
  for the write-up.

## 4. Per-injection user message (template)

Rendered by the harness; the only variable text in the conversation.

```text
Injection 7 of 25 run, 18 remaining. Best so far: CRF 0.812 at injection 5.

Result of injection 7: phi0 0.10, phi1 0.60, tG 8, T 45  (as proposed)
feasible: yes
CRF 0.734 = resolution term 0.834 - time term 0.100
t_last 12.0 min, run time 15.0 min
peaks (sorted by t_R):
 row   t_R    sigma   R_to_next
  1   1.37   0.022     5.9
  2   2.08   0.022     ...
  ...
pairs below 1.5: (4,5) R 0.61, (6,7) R 1.12

All injections so far:
 #   phi0  phi1   tG   T   feasible   CRF    minR   t_last   worst pair
 1   0.05  0.95   20  40   yes       0.512   0.21    18.4    (4,5)
 ...
 7   0.10  0.60    8  45   yes       0.734   0.61    12.0    (4,5)

Propose injection 8.
```

An infeasible result renders as `feasible: no, reason: run time: last peak +
re-equilibration exceeds t_max` and nothing else. A snapped proposal renders as
`(proposed 0.07/0.50/6.3/53, run as 0.05/0.50/6/55)`. A rejected repeat renders
as `That rounds to the method you ran at injection 3; it was not run and cost
nothing. Propose a different method.`

## 5. Harness sketch (`optimizers/llm_search.py`)

`anthropic` is not in `.venv` yet: `.venv/bin/pip install anthropic`. Credentials
come from `ANTHROPIC_API_KEY` or an `ant auth login` profile; the zero-arg client
finds either.

```python
"""LLM-driven search: one Claude call per injection, structured JSON proposals."""
from __future__ import annotations
import json
import numpy as np
import anthropic
from sim.config import CRFConfig, InstrumentConfig
from sim.crf import resolution, resolution_term, time_term
from .oracle import SearchRun, TableOracle
from .random_search import random_order

MODEL = "claude-opus-5-5"
SYSTEM = open(__file__.replace("llm_search.py", "llm_system_prompt.txt")).read()
SCHEMA = {
    "type": "object",
    "properties": {"phi0": {"type": "number"}, "phi1": {"type": "number"},
                   "tG": {"type": "number"}, "T": {"type": "number"},
                   "rationale": {"type": "string"}},
    "required": ["phi0", "phi1", "tG", "T", "rationale"],
    "additionalProperties": False,
}


def snap(oracle: TableOracle, p: dict, exclude: set[int]) -> int:
    """Nearest grid row by box-scaled distance (same metric as sim.table.lookup)."""
    scale = oracle.X.max(axis=0) - oracle.X.min(axis=0)
    d = (((oracle.X - np.array([p["phi0"], p["phi1"], p["tG"], p["T"]])) / scale) ** 2).sum(axis=1)
    d[list(exclude)] = np.inf if exclude else 0  # only used for the forced fallback
    return int(np.argmin(d))


def render_turn(oracle, run, cfg, last_proposal=None) -> str:
    ...  # section 4; derived columns from oracle.retention_table[idx] via sim.crf


def llm_search(oracle: TableOracle, budget: int = 25, seed: int = 0, n_init: int = 0,
               effort: str = "high", transcript: list | None = None) -> SearchRun:
    client = anthropic.Anthropic()
    cfg = CRFConfig.load()
    run = oracle.new_run("llm", seed, budget)
    for idx in random_order(oracle, seed)[:min(n_init, budget)]:
        oracle.query(run, int(idx))
    messages = [{"role": "user", "content": render_turn(oracle, run, cfg)}]
    retries = 0
    while len(run) < budget:
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},   # summaries go to the transcript
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        if resp.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"stopped: {resp.stop_reason}")
        messages.append({"role": "assistant", "content": resp.content})     # append-only
        proposal = json.loads(next(b.text for b in resp.content if b.type == "text"))
        idx = snap(oracle, proposal, set())
        if idx in run.indices():
            retries += 1
            if retries <= 3:
                k = run.indices().index(idx) + 1
                messages.append({"role": "user", "content":
                    f"That rounds to the method you ran at injection {k}; it was not run "
                    f"and cost nothing. Propose a different method."})
                continue
            idx = snap(oracle, proposal, set(run.indices()))                  # forced fallback
        retries = 0
        oracle.query(run, idx)
        if transcript is not None:
            transcript.append({"injection": len(run), "proposal": proposal, "row": idx,
                               "usage": resp.usage.to_dict()})
        if len(run) < budget:
            messages.append({"role": "user", "content": render_turn(oracle, run, cfg, proposal)})
    return run
```

Points worth keeping when this becomes real code:

- `fallbacks="default"` is the server-side refusal fallback the API skill
  recommends by default; harmless here, drop it if it gets in the way.
- Effort defaults to `medium` on Opus 5.5; set it explicitly and sweep
  `medium`/`high` once. The per-turn problem is small, `high` should be plenty.
- Sampling parameters are rejected on 5.x models, so repeats are nondeterministic;
  that is what the seeds loop is for.
- Register `"llm": llm_search` in `optimizers/run.py` `METHODS`; `plot_comparison`
  takes a dict keyed by method, so a third series should drop in. Write the
  transcript (proposal, rationale, thinking summary, usage) to
  `results/<set>_llm_seed<k>.jsonl` for the demo.
- Opus 5.5 reads only its own thinking blocks, so keep one model per run.

## 6. Evaluation protocol

Same as the other two: budget 25, 10 seeds, three sets, regret against the
stored optimum, "found" = within 1 %. Add:

| run | purpose |
|---|---|
| `n_init=0`, full prompt | the headline: chemist from a blank page |
| `n_init=8`, full prompt | matched to BO's initial design |
| `n_init=0`, prompt without the retention-model section | value of the domain prior |
| fresh `sim.generate --seed N` set | memorisation check on the prompt |

Cost per 25-injection run, Opus 5.5, rough bucket:

| item | estimate |
|---|---|
| system prompt | ~1.5 k tokens, cached after the first call |
| history at injection 25 | ~10 k tokens, mostly cache reads |
| thinking + output at `high` | ~1 to 3 k tokens per call |
| one run | about 1 to 2 USD |
| full benchmark (3 sets x 10 seeds x 4 configs) | low hundreds of USD at most; start with 3 seeds |

## 7. Variants after the baseline

1. **Tool-use loop.** Give the model `inject(phi0, phi1, tG, T)` plus a free
   `predict(...)` helper that fits LSS parameters to peaks the model has matched
   across runs. Needs peak-matching logic the model has to supply; interesting,
   fragile.
2. **LLM over BO candidates.** BO proposes its top-k expected-improvement rows,
   the model picks one with the chromatograms in view. Cheap way to add a domain
   prior to the surrogate.
3. **Vision.** Hand the model the `sim.chromatogram` trace PNG instead of the
   table. Closer to what an operator sees, strictly less information; good demo,
   worse optimiser.
