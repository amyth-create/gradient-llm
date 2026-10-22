"""LLM-driven search over the method tables: one chat call per injection, structured JSON proposals.

    from optimizers import TableOracle, llm_search
    run = llm_search(TableOracle("hard"), budget=25, seed=0)

The harness owns the budget. Each turn it sends the result of the last injection
(the retention table with pairwise resolution, the CRF and its terms), a
one-line log of every injection so far, and asks for the next method. The
model answers with ``{"phi0", "phi1", "tG", "T", "rationale"}`` enforced by a
json_schema response format; the harness snaps that to the nearest grid row and
spends one injection on it through ``TableOracle.query``. A proposal that snaps
to a row already run is not charged: the model is asked again, up to
``max_repeats`` times, then the nearest unqueried row is taken so the run always
completes its budget.

The model sees only what derives from the retention table and the public
configs: never the analyte parameters, the optimum, or any row it has not run.
The conversation is append-only. The client comes from ``check_azure.make_client``
and the deployment name is the ``model`` on every call.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sim.config import CRFConfig, InstrumentConfig
from sim.crf import resolution, resolution_term_truncated, time_term_truncated

from .check_azure import PROPOSAL_SCHEMA, Settings, load_settings, make_client
from .oracle import SearchRun, TableOracle
from .random_search import random_order

PROMPT_FILE = Path(__file__).resolve().parent / "llm_system_prompt.txt"
RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "method_proposal", "strict": True, "schema": PROPOSAL_SCHEMA}}
FIELDS = ("phi0", "phi1", "tG", "T")


def load_prompt(path: str | Path = PROMPT_FILE) -> str:
    return Path(path).read_text()


# --------------------------------------------------------------------------- #
# Snapping
# --------------------------------------------------------------------------- #
def _scale(instrument: InstrumentConfig) -> np.ndarray:
    b = instrument.method_bounds
    return np.array([b.phi0[1] - b.phi0[0], b.phi1_max - b.phi0[0], b.tG[1] - b.tG[0], b.T[1] - b.T[0]])


def snap(oracle: TableOracle, proposal: dict, scale: np.ndarray, exclude: set[int] | None = None) -> int:
    """Nearest grid row by box-scaled distance (the metric of ``sim.table.lookup``), skipping ``exclude``."""
    p = np.array([float(proposal[f]) for f in FIELDS])
    d = (((oracle.X - p) / scale) ** 2).sum(axis=1)
    if exclude:
        d[list(exclude)] = np.inf
    return int(np.argmin(d))


def _same(oracle: TableOracle, index: int, proposal: dict) -> bool:
    return all(abs(float(proposal[f]) - oracle.X[index, i]) < 1e-6 for i, f in enumerate(FIELDS))


# --------------------------------------------------------------------------- #
# Rendering what the model may see
# --------------------------------------------------------------------------- #
class Renderer:
    """Turns oracle rows into the text the model sees; derived numbers come from the retention table only."""

    def __init__(self, oracle: TableOracle, instrument: InstrumentConfig, cfg: CRFConfig):
        self.oracle = oracle
        self.cfg = cfg
        self.t_end = instrument.t_end
        self.t_eq = instrument.t_eq_min
        self.n_analytes = int(len(oracle.retention_table[0]) + oracle.n_missing[0])

    def derived(self, index: int) -> dict:
        table = self.oracle.retention_table[index]
        t = np.array([r[0] for r in table], dtype=float)
        s = np.array([r[1] for r in table], dtype=float)
        n_missing = int(self.oracle.n_missing[index])
        obs = np.ones(len(t), dtype=bool)
        res = float(resolution_term_truncated(t, s, obs, self.n_analytes, self.cfg))
        tt = float(time_term_truncated(t, obs, self.n_analytes, self.t_end, self.cfg))
        R = resolution(t, s) if len(t) >= 2 else np.zeros(0)
        t_last = float(t[-1]) if len(t) else float("nan")
        run_time = (min(t_last, self.t_end) if len(t) else self.t_end) + self.t_eq
        if n_missing:
            run_time = self.t_end + self.t_eq
        return {"t": t, "sigma": s, "R": R, "res_term": res, "time_term": tt, "n_missing": n_missing,
                "t_last": t_last, "run_time": run_time, "crf": float(self.oracle.crf[index])}

    @staticmethod
    def _pairs_below(R: np.ndarray, target: float) -> str:
        bad = [f"({j + 1},{j + 2}) R {R[j]:.2f}" for j in range(len(R)) if R[j] < target]
        return ", ".join(bad) if bad else "none"

    def result_block(self, k: int, index: int, proposal: dict | None) -> str:
        o = self.oracle
        d = self.derived(index)
        m = o.X[index]
        method = f"phi0 {m[0]:.2f}, phi1 {m[1]:.2f}, tG {m[2]:.0f} min, T {m[3]:.0f} C"
        if proposal is not None and not _same(o, index, proposal):
            note = (f"  (proposed {proposal['phi0']}/{proposal['phi1']}/{proposal['tG']}/{proposal['T']}, "
                    f"rounded to the grid and run as above)")
        else:
            note = "  (as proposed)" if proposal is not None else ""
        lines = [f"Result of injection {k}: {method}{note}"]
        if not o.feasible[index]:
            lines.append(f"not run: {o.reason[index]}")
            return "\n".join(lines)
        seen = len(d["t"])
        missing = (f"; {d['n_missing']} still on the column when the run ended at {self.t_end:.0f} min"
                   if d["n_missing"] else "")
        lines.append(f"observed peaks: {seen} of {self.n_analytes}{missing}")
        lines.append(f"CRF {d['crf']:.3f} = resolution term {d['res_term']:.3f} - time term {d['time_term']:.3f}")
        if seen:
            lines.append(f"t_last {d['t_last']:.2f} min (last observed peak), run time {d['run_time']:.1f} min")
            lines.append("peaks (sorted by t_R):")
            lines.append(" row     t_R   sigma   R_to_next")
            for j in range(seen):
                r_next = f"{d['R'][j]:8.2f}" if j < len(d["R"]) else "       -"
                lines.append(f" {j + 1:3d}  {d['t'][j]:6.2f}  {d['sigma'][j]:6.4f}  {r_next}")
            lines.append(f"pairs below {self.cfg.R_target:.1f}: {self._pairs_below(d['R'], self.cfg.R_target)}")
        return "\n".join(lines)

    def log_block(self, run: SearchRun) -> str:
        lines = ["All injections so far:",
                 "  #   phi0  phi1   tG   T   seen    CRF   minR  t_last  worst pair"]
        for ob in run.observations:
            d = self.derived(ob.index)
            seen = f"{len(d['t'])}/{self.n_analytes}"
            min_r = f"{d['R'].min():5.2f}" if len(d["R"]) else "    -"
            worst = f"({int(np.argmin(d['R'])) + 1},{int(np.argmin(d['R'])) + 2})" if len(d["R"]) else "-"
            t_last = f"{d['t_last']:6.1f}" if np.isfinite(d["t_last"]) else "     -"
            lines.append(f"{ob.run:3d}   {ob.phi0:.2f}  {ob.phi1:.2f}  {ob.tG:3.0f}  {ob.T:2.0f}  {seen:>5s}  "
                         f"{ob.crf:6.3f}  {min_r}  {t_last}  {worst}")
        return "\n".join(lines)

    def turn(self, run: SearchRun, new: list[tuple[int, int, dict | None]]) -> str:
        """The user message after the injections in ``new`` ([(k, index, proposal)]) have been run."""
        k = len(run)
        parts = []
        if k == 0:
            parts.append(f"No injections yet. The mixture contains {self.n_analytes} compounds. "
                         f"You have {run.budget} injections.")
        else:
            best_k = int(np.nanargmax(run.crf_per_run())) + 1
            best = run.observations[best_k - 1].crf
            parts.append(f"Injection {k} of {run.budget} run, {run.budget - k} remaining. "
                         f"Best so far: CRF {best:.3f} at injection {best_k}.")
            for kk, index, proposal in new:
                parts.append(self.result_block(kk, index, proposal))
            parts.append(self.log_block(run))
        parts.append(f"Propose injection {k + 1}.")
        return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def _ask(client, model: str, messages: list[dict], max_completion_tokens: int, extra: dict) -> tuple[str, object, float]:
    t = time.perf_counter()
    resp = client.chat.completions.create(model=model, messages=messages, response_format=RESPONSE_FORMAT,
                                          max_completion_tokens=max_completion_tokens, **extra)
    dt = time.perf_counter() - t
    choice = resp.choices[0]
    content = choice.message.content or ""
    if not content.strip() and choice.finish_reason == "length":
        raise RuntimeError("the model ran out of output tokens before answering; raise max_completion_tokens")
    if not content.strip():
        raise RuntimeError(f"empty reply (finish_reason {choice.finish_reason})")
    return content, resp.usage, dt


def _parse(content: str) -> dict:
    data = json.loads(content)
    for f in FIELDS:
        data[f] = float(data[f])
    data.setdefault("rationale", "")
    return data


def llm_search(oracle: TableOracle, budget: int = 25, seed: int = 0, n_init: int = 0, *,
               settings: Settings | None = None, client=None, model: str | None = None,
               reasoning_effort: str | None = None, max_repeats: int = 3, max_completion_tokens: int = 4096,
               prompt_path: str | Path = PROMPT_FILE, transcript: list | None = None,
               verbose: bool = False) -> SearchRun:
    """One chat call per injection until the budget is spent; returns the SearchRun like the other optimisers.

    ``n_init`` warm-start injections are the rows ``random_order(oracle, seed)`` gives
    random search and BO, so seeds are comparable across methods. ``transcript``
    (a list) receives one dict per injection with the proposal, the snapped row,
    the usage and the latency.
    """
    settings = settings or load_settings()
    client = client or make_client(settings, max_retries=2, timeout=120.0)
    model = model or settings.deployment
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    instrument = InstrumentConfig.load()
    renderer = Renderer(oracle, instrument, CRFConfig.load())
    scale = _scale(instrument)

    run = oracle.new_run("llm", seed, budget)
    new: list[tuple[int, int, dict | None]] = []
    for idx in random_order(oracle, seed)[:min(n_init, budget)]:
        oracle.query(run, int(idx))
        new.append((len(run), int(idx), None))
    messages = [{"role": "system", "content": load_prompt(prompt_path)},
                {"role": "user", "content": renderer.turn(run, new)}]

    while len(run) < budget:
        attempts, parse_errors = 0, 0
        while True:
            content, usage, dt = _ask(client, model, messages, max_completion_tokens, extra)
            messages.append({"role": "assistant", "content": content})
            try:
                proposal = _parse(content)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                parse_errors += 1
                if parse_errors > max_repeats:
                    raise RuntimeError(f"unparseable proposal after {parse_errors} tries: {content[:200]!r}") from e
                messages.append({"role": "user", "content":
                                 "That was not valid JSON with numeric phi0, phi1, tG, T and a rationale. "
                                 "Answer with the JSON object only."})
                continue
            index = snap(oracle, proposal, scale)
            if index in run.indices():
                attempts += 1
                if attempts <= max_repeats:
                    k_prev = run.indices().index(index) + 1
                    messages.append({"role": "user", "content":
                                     f"That rounds to the method you ran at injection {k_prev}; it was not run "
                                     f"and cost nothing. Propose a different method."})
                    continue
                index = snap(oracle, proposal, scale, exclude=set(run.indices()))   # forced fallback
            break
        obs = oracle.query(run, index)
        if transcript is not None:
            transcript.append({
                "set": oracle.set_name, "seed": seed, "injection": obs.run, "index": index,
                "proposal": {f: proposal[f] for f in FIELDS}, "rationale": proposal.get("rationale", ""),
                "run_as": {f: float(oracle.X[index, i]) for i, f in enumerate(FIELDS)},
                "crf": obs.crf, "n_missing": obs.n_missing, "best": obs.best, "regret": obs.regret,
                "repeat_attempts": attempts, "parse_errors": parse_errors, "latency_s": round(dt, 2),
                "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                          "completion_tokens": getattr(usage, "completion_tokens", None)},
            })
        if verbose:
            print(f"  llm {obs.run:3d}  {obs.phi0:.2f} {obs.phi1:.2f} {obs.tG:4.0f} {obs.T:3.0f}  "
                  f"crf {obs.crf:6.3f}  best {obs.best:6.3f}  missing {obs.n_missing}  "
                  f"({dt:.1f} s)  {proposal.get('rationale', '')[:90]}", flush=True)
        if len(run) < budget:
            messages.append({"role": "user", "content": renderer.turn(run, [(obs.run, index, proposal)])})
    return run


def write_transcript(transcript: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for row in transcript:
            fh.write(json.dumps(row) + "\n")
    return path
