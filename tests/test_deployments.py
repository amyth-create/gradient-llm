"""The two overrides deployments/run.sh relies on: HPLC_ENV_FILE and HPLC_RESULTS_DIR."""
from pathlib import Path

from optimizers import llm_repeats
from optimizers.check_azure import ENV_FILE, default_env_file, load_settings
from sim.config import results_dir, set_root

AZURE_VARS = ("AZURE_ENDPOINT", "AZURE_DEPLOYMENT", "AZURE_API_KEY", "AZURE_API_VERSION")


def test_env_file_defaults_to_the_project_env(monkeypatch):
    monkeypatch.delenv("HPLC_ENV_FILE", raising=False)
    assert default_env_file() == ENV_FILE


def test_env_file_override_is_read_by_load_settings(tmp_path, monkeypatch):
    for k in AZURE_VARS:
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / "alt" / ".env"
    env.parent.mkdir()
    env.write_text("AZURE_ENDPOINT=https://other.openai.azure.com/openai/v1\n"
                   "AZURE_DEPLOYMENT=other-model\nAZURE_API_KEY=k2\nAZURE_API_VERSION=\n")
    monkeypatch.setenv("HPLC_ENV_FILE", str(env))
    assert default_env_file() == env
    s = load_settings()
    assert s.deployment == "other-model"
    assert s.endpoint == "https://other.openai.azure.com"
    assert s.surface == "v1"


def test_results_dir_override_moves_every_set_and_the_transcripts(tmp_path, monkeypatch):
    monkeypatch.delenv("HPLC_RESULTS_DIR", raising=False)
    assert results_dir("hard") == set_root("hard") / "results"
    monkeypatch.setenv("HPLC_RESULTS_DIR", str(tmp_path / "alt" / "results"))
    assert results_dir("hard") == tmp_path / "alt" / "results"
    assert results_dir("easy") == tmp_path / "alt" / "results"
    assert llm_repeats.transcript_path("hard", 3) == tmp_path / "alt" / "results" / "llm_transcripts" / "hard_seed3.jsonl"
    assert not Path(tmp_path / "alt").exists()  # resolving a path writes nothing
