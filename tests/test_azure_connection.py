"""Live check of the Azure OpenAI deployment in .env. Skipped unless AZURE_LIVE_TEST=1.

    AZURE_LIVE_TEST=1 .venv/bin/pytest tests/test_azure_connection.py -s
"""
import os

import pytest

from optimizers.check_azure import MissingSetting, check, load_settings


def test_azure_deployment_responds():
    if os.environ.get("AZURE_LIVE_TEST") != "1":
        pytest.skip("set AZURE_LIVE_TEST=1 to call the live Azure deployment")
    try:
        settings = load_settings()
    except MissingSetting as e:
        pytest.skip(str(e))
    report = check(settings, verbose=False)
    assert report.ok, report.summary
    assert report.structured_ok is not False, report.summary


def test_load_settings_reduces_a_pasted_target_uri(tmp_path, monkeypatch):
    for k in ("AZURE_ENDPOINT", "AZURE_DEPLOYMENT", "AZURE_API_KEY", "AZURE_API_VERSION"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "AZURE_ENDPOINT=https://my-res.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview\n"
        "AZURE_DEPLOYMENT=\nAZURE_API_KEY=abc\nAZURE_API_VERSION=\n")
    s = load_settings(env)
    assert s.endpoint == "https://my-res.openai.azure.com"
    assert s.deployment == "gpt-4o"
    assert s.api_version == "2025-01-01-preview"
    assert s.api_key == "abc"
    assert s.surface == "classic"


def test_load_settings_detects_the_v1_surface(tmp_path, monkeypatch):
    for k in ("AZURE_ENDPOINT", "AZURE_DEPLOYMENT", "AZURE_API_KEY", "AZURE_API_VERSION"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text("AZURE_ENDPOINT=https://my-res.openai.azure.com/openai/v1\n"
                   "AZURE_DEPLOYMENT=gpt-x\nAZURE_API_KEY=abc\nAZURE_API_VERSION=2026-09-03\n")
    s = load_settings(env)
    assert s.surface == "v1"
    assert s.endpoint == "https://my-res.openai.azure.com"
    assert s.base_url == "https://my-res.openai.azure.com/openai/v1/"
    assert any("not used" in n for n in s.notes)


def test_load_settings_reports_missing(tmp_path, monkeypatch):
    for k in ("AZURE_ENDPOINT", "AZURE_DEPLOYMENT", "AZURE_API_KEY", "AZURE_API_VERSION"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text("AZURE_ENDPOINT=https://x.openai.azure.com\nAZURE_DEPLOYMENT=\nAZURE_API_KEY=\n")
    with pytest.raises(MissingSetting, match="AZURE_DEPLOYMENT, AZURE_API_KEY"):
        load_settings(env)
