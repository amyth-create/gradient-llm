#!/usr/bin/env bash
# Run a project module against one deployment folder:
#   deployments/run.sh <name> <python.module> [args...]
# e.g.
#   deployments/run.sh alt-model optimizers.check_azure
#   deployments/run.sh alt-model optimizers.llm_repeats --sets hard --repeats 50 --workers 6
#   deployments/run.sh alt-model optimizers.llm_repeats --summary-only
#
# It exports HPLC_ENV_FILE=deployments/<name>/.env (read by optimizers.check_azure.load_settings)
# and HPLC_RESULTS_DIR=deployments/<name>/results (honoured by sim.config.results_dir), so the
# campaigns, transcripts, CSVs and figures of this deployment stay in its own folder and the
# resumable LLM runner never picks up another model's transcripts. Any AZURE_* variable exported
# in the shell is unset first so the folder's .env always wins.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: deployments/run.sh <name> <python.module> [args...]" >&2
    exit 2
fi
name="$1"; shift
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(dirname "$here")"
folder="$here/$name"

if [ ! -f "$folder/.env" ]; then
    echo "no env file at $folder/.env (copy deployments/.env.example there and fill it in)" >&2
    exit 2
fi

unset AZURE_ENDPOINT AZURE_DEPLOYMENT AZURE_API_KEY AZURE_API_VERSION
export HPLC_ENV_FILE="$folder/.env"
export HPLC_RESULTS_DIR="$folder/results"
mkdir -p "$HPLC_RESULTS_DIR"

cd "$project"
exec .venv/bin/python -m "$@"
