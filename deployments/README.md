# Deployments: the same search on a different model

One folder per Azure OpenAI deployment to compare against the main one in `../.env`
(gpt-6-astra, whose results are the committed `results/` and `hplcsim/results/`).
The purpose is to tell whether the LLM search's performance is down to the model
itself: the harness, prompt, sets, budget and seeds stay identical; only the
deployment changes.

```
deployments/
  .env.example        template for a new folder
  run.sh              wrapper: deployments/run.sh <name> <python.module> [args...]
  alt-model/
    .env              AZURE_* for this deployment (gitignored)
    results/          everything this deployment writes (created on first run)
      llm_transcripts/<set>_seed<k>.jsonl
      llm_crf_over_runs.csv, crf_over_runs.png, ...
```

## Add a deployment

1. `mkdir deployments/<name>` and copy `deployments/.env.example` to `deployments/<name>/.env`
   (or copy `deployments/alt-model/.env`, which already carries the resource's endpoint and key).
2. Fill in `AZURE_DEPLOYMENT` with the deployment name from the Azure portal.
3. Check it answers and handles structured output:

```bash
deployments/run.sh <name> optimizers.check_azure
```

## Run the comparison

Same command as for the main model, through the wrapper. 50 repeats per set,
budget 25, is what the committed results use (3 sets x 50 x 25 = 3750 API calls);
start with one set to see the cost and latency first.

```bash
deployments/run.sh <name> optimizers.llm_repeats --sets hard --repeats 50 --workers 6
```

```bash
deployments/run.sh <name> optimizers.llm_repeats --repeats 50 --workers 6 --figure
```

The run is resumable: interrupt it and rerun the same command, finished campaigns are
loaded from `deployments/<name>/results/llm_transcripts/`. `--figure` also reruns
random search and BO (deterministic, no API calls) so the three-method figure for
this model lands in the same folder.

## Compare

Print the per-injection and checkpoint tables without API calls, for each model:

```bash
.venv/bin/python -m optimizers.llm_repeats --summary-only
```

```bash
deployments/run.sh <name> optimizers.llm_repeats --summary-only
```

The row-level data is `llm_crf_over_runs.csv` in `results/` (main model) and in
`deployments/<name>/results/` (this one), same columns, so the two can be read
side by side. If the second model reaches the same "mean best CRF after n
injections" and the same share of repeats within 1 % of the optimum, the
performance is a property of the harness and prompt, not the model.

## How the wrapper works

`run.sh` only sets two environment variables and runs `.venv/bin/python -m <module>`
from the project root:

| variable           | read by                              | effect                                                                 |
|--------------------|--------------------------------------|------------------------------------------------------------------------|
| `HPLC_ENV_FILE`    | `optimizers.check_azure.load_settings` | the `.env` to read instead of `<project>/.env`                       |
| `HPLC_RESULTS_DIR` | `sim.config.results_dir`             | where every set's results go instead of `<set root>/results`           |

Without `HPLC_RESULTS_DIR` the resumable runner would find the main model's
transcripts in `results/llm_transcripts/` and load them instead of calling the
new deployment. Any `AZURE_*` variable exported in the shell is unset by the
wrapper first, because the `.env` loader never overrides an exported variable.
