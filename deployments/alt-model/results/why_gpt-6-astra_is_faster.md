# Why gpt-6-astra reaches the hard-set optimum faster than gpt-5.4

Same harness, prompt, sets, budget (25) and seeds; 50 campaigns per model on the hard set.
Transcripts: `results/llm_transcripts/hard_seed*.jsonl` (gpt-6-astra) and
`deployments/alt-model/results/llm_transcripts/hard_seed*.jsonl` (gpt-5.4).

| | gpt-6-astra | gpt-5.4 |
|---|---|---|
| first injection within 1 % of the optimum, median (mean) | 4 (3.7) | 10 (10.1) |
| opening method, modal | 0.05 -> 0.95, 10 or 12 min, 40 C (50/50) | 0.05 -> 0.95, 20 min, 40 C (49/50) |
| CRF of injection 1, mean | 0.931 | 0.854 |
| injections from first best >= 0.90 to the optimum, mean | 2.7 | 7.7 |
| polishing steps changing one parameter only | 85 % | 32 % |
| polishing steps changing temperature | 34 % | 60 % |
| polishing steps scoring below the previous injection | 16 % | 46 % |
| polishing steps reversing the model's own previous change | 5 | 78 |
| proposals that repeated an already-run method (not charged) | 0 | 14 |
| truncated runs (a compound held past 42 min) | 0 | 2, both in seed 0 |
| rationales quoting an R value / a numeric prediction | 15 % / 12 % | 5 % / 6 % |
| completion tokens per visible word, median | 4.0 | 1.3 |
| calls with completion tokens > 2x the visible text | 84 % | 0 % |
| latency per call, median | 8.6 s | 2.8 s |
| optima found at 55 C | 15 of 50 | 0 of 50 |

## 1. gpt-6-astra deliberates before answering; gpt-5.4 answers directly

Neither run sets `reasoning_effort`, so each model uses its default. gpt-6-astra's
completion tokens are four times its visible JSON and its calls take three times as
long: it is spending hidden reasoning tokens on every proposal. gpt-5.4 returns
little more than the JSON itself. Everything below follows from that.

## 2. gpt-6-astra reads the score and the retention model quantitatively

Both models get the same prompt, which states the 10-minute time target and the
phi_e formula for where a peak leaves the column. gpt-6-astra opens with a
10 or 12-minute gradient, so its first injection already has no time penalty and
scores 0.92 to 0.95; from there the nearest CRF 1.0 method in the table differs in
one parameter. gpt-5.4 opens at 20 minutes, whose last peak at 12.6 min costs 0.13,
and the nearest CRF 1.0 method differs in two parameters.

In the polishing phase gpt-6-astra predicts outcomes before running them ("the last
peak is expected near 9.6 minutes, preserving the zero time penalty"; "raise phi0
and narrow the window while keeping the gradient slope close to injection 2") and
names the pair it is targeting by its R value. gpt-5.4's rationales are qualitative
("may shave enough time", "probes whether", "a distinct unexplored direction") and
in the last ten injections become a coverage strategy: "the last nearby unexplored
composition point".

## 3. One variable at a time, and no dead temperatures

gpt-6-astra changes one parameter per step in 85 % of polishing moves, almost always
tG or phi0 at fixed temperature, so each result attributes cleanly and the next step
is monotone. gpt-5.4 changes two or more parameters in 68 % of moves and reverses its
own previous change 78 times, so nearly half its polishing injections score below the
one before.

Temperature is the expensive trap on this set: the table holds CRF 1.0 methods only
at 40 C and 55 C. gpt-5.4 spends 60 % of its polishing steps on temperature, mostly
35, 45, 50 and 30 C, none of which contain an optimum, and all 50 of its optima are at
40 C. gpt-6-astra moves temperature in 34 % of steps and, when it does, jumps to 55 C
and finds an optimum there in 15 campaigns.

## What this means for the comparison

The harness and prompt are sufficient for both models to reach the optimum within 25
injections. The speed difference is the model's: gpt-6-astra's default deliberation
turns the prompt's retention model and score into arithmetic and single-variable
steps; gpt-5.4 behaves like a sensible but heuristic chromatographer who tries things.

A cleaner test of "is it the model or the thinking" is to rerun gpt-5.4 with
`reasoning_effort` set (the search already accepts it; the repeats runner does not yet
expose it) in a third deployment folder.
