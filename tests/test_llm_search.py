"""Offline tests of the LLM search harness: snapping, rendering, and the loop driven by a scripted client."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from optimizers import TableOracle
from optimizers.check_azure import Settings
from optimizers.llm_search import Renderer, _scale, llm_search, snap
from sim.config import CRFConfig, InstrumentConfig


@pytest.fixture(scope="module")
def oracle() -> TableOracle:
    return TableOracle("easy")


@pytest.fixture(scope="module")
def instrument() -> InstrumentConfig:
    return InstrumentConfig.load()


class ScriptedClient:
    """Answers each chat call with the next scripted content; records every request."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        content = self.replies.pop(0)
        choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def proposal(phi0, phi1, tG, T, why="because") -> str:
    return json.dumps({"phi0": phi0, "phi1": phi1, "tG": tG, "T": T, "rationale": why})


def test_snap_rounds_to_the_grid_and_can_skip_rows(oracle, instrument):
    scale = _scale(instrument)
    i = snap(oracle, {"phi0": 0.07, "phi1": 0.5, "tG": 6.3, "T": 53}, scale)
    assert oracle.X[i].tolist() == [0.05, 0.5, 6.0, 55.0]
    j = snap(oracle, {"phi0": 0.07, "phi1": 0.5, "tG": 6.3, "T": 53}, scale, exclude={i})
    assert j != i


def test_renderer_reproduces_the_table_crf(oracle, instrument):
    r = Renderer(oracle, instrument, CRFConfig.load())
    assert r.n_analytes == 6
    rng = np.random.default_rng(0)
    for i in rng.choice(oracle.n_candidates, 500, replace=False):
        d = r.derived(int(i))
        assert d["res_term"] - d["time_term"] == pytest.approx(oracle.crf[i], abs=1e-5)
        assert len(d["t"]) + d["n_missing"] == 6


def test_renderer_text_hides_identities_and_shows_missing_peaks(oracle, instrument):
    r = Renderer(oracle, instrument, CRFConfig.load())
    trunc = int(np.flatnonzero(oracle.n_missing > 0)[0])
    text = r.result_block(1, trunc, None)
    assert "still on the column" in text and "of 6" in text
    assert "method_id" not in text and "log_kw" not in text
    first = r.turn(oracle.new_run("llm", 0, 5), [])
    assert "contains 6 compounds" in first and first.endswith("Propose injection 1.")


def test_loop_spends_the_budget_and_does_not_charge_repeats(oracle):
    replies = [
        proposal(0.05, 0.95, 10, 40),          # injection 1
        proposal(0.05, 0.95, 10, 40),          # repeat: not charged, asked again
        "not json at all",                     # parse error: asked again
        proposal(0.05, 0.95, 6, 40),           # injection 2
        proposal(0.06, 0.94, 6.4, 41),         # rounds to injection 2 (repeat 1)
        proposal(0.05, 0.95, 6, 40),           # repeat 2
        proposal(0.05, 0.95, 6, 40),           # repeat 3
        proposal(0.05, 0.95, 6, 40),           # repeat 4 -> forced nearest unqueried row = injection 3
    ]
    client = ScriptedClient(replies)
    settings = Settings("https://x.openai.azure.com", "dep", "k", "", "v1")
    transcript = []
    run = llm_search(oracle, budget=3, seed=0, settings=settings, client=client, transcript=transcript)
    assert len(run) == 3 and len(set(run.indices())) == 3
    assert client.requests[0]["model"] == "dep"
    assert client.requests[0]["messages"][0]["role"] == "system"
    assert "reverse-phase HPLC" in client.requests[0]["messages"][0]["content"]
    assert transcript[0]["repeat_attempts"] == 0 and transcript[0]["parse_errors"] == 0
    assert transcript[1]["repeat_attempts"] == 1 and transcript[1]["parse_errors"] == 1
    assert transcript[2]["repeat_attempts"] == 4           # three re-asks, then the forced fallback
    # the conversation is append-only: every request is a prefix of the next
    for a, b in zip(client.requests, client.requests[1:]):
        assert b["messages"][: len(a["messages"])] == a["messages"]
    assert not client.replies
