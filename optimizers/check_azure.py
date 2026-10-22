"""Check that the Azure OpenAI deployment named in .env answers.

    .venv/bin/python -m optimizers.check_azure            # reads <project>/.env
    .venv/bin/python -m optimizers.check_azure --env path/to/other.env
    HPLC_ENV_FILE=deployments/<name>/.env .venv/bin/python -m optimizers.check_azure   # a second deployment

Loads the AZURE_* settings, then makes two small calls against the deployment:
a plain chat completion, and a structured-output (json_schema) completion of
the exact shape the LLM-driven search will rely on. Exit code 0 when the
endpoint, key and deployment all work; 1 otherwise. The key is never printed.

Two endpoint shapes are accepted:
  v1 surface   https://<resource>.openai.azure.com/openai/v1   (no api-version; recommended)
  classic      https://<resource>.openai.azure.com             (needs AZURE_API_VERSION, e.g. 2024-10-21)
``make_client`` returns the matching OpenAI client; the LLM search uses it too.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_FILE_VAR = "HPLC_ENV_FILE"
DEFAULT_API_VERSION = "2024-10-21"
REQUIRED = ("AZURE_ENDPOINT", "AZURE_DEPLOYMENT", "AZURE_API_KEY")

# The proposal the LLM search asks for on every injection.
PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "phi0": {"type": "number"},
        "phi1": {"type": "number"},
        "tG": {"type": "number"},
        "T": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["phi0", "phi1", "tG", "T", "rationale"],
    "additionalProperties": False,
}


class MissingSetting(RuntimeError):
    """A required AZURE_* variable is absent or empty."""


@dataclass(frozen=True)
class Settings:
    endpoint: str          # origin only, e.g. https://<resource>.openai.azure.com
    deployment: str
    api_key: str
    api_version: str       # classic surface only; ignored on v1
    surface: str           # "v1" or "classic"
    notes: tuple[str, ...] = ()

    @property
    def base_url(self) -> str:
        return f"{self.endpoint}/openai/v1/"


def default_env_file() -> Path:
    """``$HPLC_ENV_FILE`` when it is set (a second deployment kept under ``deployments/``), else ``<project>/.env``."""
    return Path(os.environ.get(ENV_FILE_VAR) or ENV_FILE).expanduser()


def load_settings(env_file: Path | None = None) -> Settings:
    """Read the AZURE_* variables from ``env_file`` (default ``default_env_file()``).

    The .env file never overrides a variable already exported in the environment.
    """
    env_file = Path(env_file) if env_file is not None else default_env_file()
    load_dotenv(env_file, override=False)
    raw = {k: os.environ.get(k, "").strip().strip('"').strip("'")
           for k in (*REQUIRED, "AZURE_API_VERSION")}
    notes: list[str] = []
    endpoint, deployment, api_version = raw["AZURE_ENDPOINT"], raw["AZURE_DEPLOYMENT"], raw["AZURE_API_VERSION"]
    surface = "classic"

    if endpoint:
        u = urlparse(endpoint if "://" in endpoint else "https://" + endpoint)
        parts = [p for p in u.path.split("/") if p]
        if parts[:2] == ["openai", "v1"]:
            surface = "v1"
            if api_version:
                notes.append("endpoint is the /openai/v1 surface: AZURE_API_VERSION is not used")
        else:
            # A pasted classic "target URI" carries the deployment and api-version; recover them when the vars are blank.
            if "deployments" in parts and parts.index("deployments") + 1 < len(parts) and not deployment:
                deployment = parts[parts.index("deployments") + 1]
                notes.append(f"AZURE_DEPLOYMENT taken from the endpoint path: {deployment}")
            qv = parse_qs(u.query).get("api-version")
            if qv and not api_version:
                api_version = qv[0]
                notes.append(f"AZURE_API_VERSION taken from the endpoint query: {api_version}")
            if parts or u.query:
                notes.append(f"AZURE_ENDPOINT had a path or query; using the origin {u.scheme}://{u.netloc}")
        if u.scheme != "https":
            notes.append("AZURE_ENDPOINT is not https")
        endpoint = f"{u.scheme}://{u.netloc}"

    missing = [k for k, v in zip(REQUIRED, (endpoint, deployment, raw["AZURE_API_KEY"])) if not v]
    if missing:
        raise MissingSetting(f"{', '.join(missing)} not set; fill in {env_file}")
    if surface == "classic" and not api_version:
        api_version = DEFAULT_API_VERSION
        notes.append(f"AZURE_API_VERSION blank; using {DEFAULT_API_VERSION}")
    return Settings(endpoint, deployment, raw["AZURE_API_KEY"], api_version, surface, tuple(notes))


def make_client(settings: Settings, **kwargs):
    """The OpenAI client for this endpoint shape. ``model=settings.deployment`` on every call."""
    from openai import AzureOpenAI, OpenAI
    if settings.surface == "v1":
        return OpenAI(base_url=settings.base_url, api_key=settings.api_key, **kwargs)
    return AzureOpenAI(azure_endpoint=settings.endpoint, api_key=settings.api_key,
                       api_version=settings.api_version, **kwargs)


@dataclass
class Report:
    ok: bool = False
    structured_ok: bool | None = None       # None = not reached
    lines: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return "\n".join(self.lines)


def check(settings: Settings, verbose: bool = True) -> Report:
    """Two live calls against the deployment; returns a Report with ok=True when the first succeeds."""
    from openai import (APIConnectionError, APIStatusError, AuthenticationError, BadRequestError,
                        NotFoundError, PermissionDeniedError, RateLimitError)

    rep = Report()

    def say(line: str) -> None:
        rep.lines.append(line)
        if verbose:
            print(line, flush=True)

    if settings.surface == "v1":
        say(f"endpoint     {settings.base_url}  (v1 surface, no api-version)")
    else:
        say(f"endpoint     {settings.endpoint}  (classic surface, api-version {settings.api_version})")
    say(f"deployment   {settings.deployment}")
    say(f"api key      set ({len(settings.api_key)} chars)")
    for n in settings.notes:
        say(f"note         {n}")

    client = make_client(settings, max_retries=1, timeout=60.0)

    # 1. plain chat completion: proves endpoint, key, deployment name (and api version on classic).
    t = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=settings.deployment,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            max_completion_tokens=1024,          # reasoning models spend some of this before answering
        )
    except AuthenticationError as e:
        say(f"FAIL  authentication (HTTP {e.status_code}): the API key was rejected by {settings.endpoint}")
        return rep
    except PermissionDeniedError as e:
        say(f"FAIL  permission denied (HTTP {e.status_code}): {e.message}")
        return rep
    except NotFoundError as e:
        hint = ("no deployment '{d}' at this endpoint".format(d=settings.deployment) if settings.surface == "v1" else
                "no deployment '{d}' at this endpoint, or api-version {v} is not an Azure REST version "
                "(it should look like 2024-10-21; leave it blank for the /openai/v1 endpoint)"
                .format(d=settings.deployment, v=settings.api_version))
        say(f"FAIL  not found (HTTP 404): {hint}: {e.message}")
        return rep
    except RateLimitError as e:
        say(f"FAIL  rate limited (HTTP 429): the deployment exists but has no quota right now: {e.message}")
        return rep
    except BadRequestError as e:
        say(f"FAIL  bad request (HTTP 400): {e.message}")
        return rep
    except APIStatusError as e:
        say(f"FAIL  HTTP {e.status_code}: {e.message}")
        return rep
    except APIConnectionError as e:
        say(f"FAIL  could not connect to {settings.endpoint}: {e}")
        return rep
    dt = time.perf_counter() - t
    choice = resp.choices[0] if resp.choices else None
    text = (choice.message.content or "").strip() if choice else ""
    usage = resp.usage
    tokens = f"{usage.prompt_tokens} in / {usage.completion_tokens} out" if usage else "usage not reported"
    if not text:
        say(f"WARN  chat completion returned no text (finish_reason {choice.finish_reason if choice else 'n/a'}), "
            f"model {resp.model}, {tokens}")
    else:
        say(f"PASS  chat completion in {dt:.1f} s: model {resp.model}, {tokens}, reply {text!r}")
    rep.ok = True

    # 2. structured output in the shape the LLM search uses.
    t = time.perf_counter()
    content = ""
    try:
        resp2 = client.chat.completions.create(
            model=settings.deployment,
            messages=[
                {"role": "system", "content": "You choose HPLC gradient methods. Answer with JSON matching the schema."},
                {"role": "user", "content": "Propose any one method on this grid: phi0 0.05-0.60 step 0.05; "
                                            "phi1 0.15-0.95 step 0.05 with phi1 >= phi0 + 0.10; tG 2-40 min step 2; "
                                            "T 25-60 C step 5. One-sentence rationale."},
            ],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "method_proposal", "strict": True, "schema": PROPOSAL_SCHEMA}},
            max_completion_tokens=2048,
        )
        content = resp2.choices[0].message.content or ""
        data = json.loads(content)
        missing = [k for k in PROPOSAL_SCHEMA["required"] if k not in data]
        if missing:
            rep.structured_ok = False
            say(f"WARN  structured output came back as JSON but without {missing}: {content[:200]}")
        else:
            rep.structured_ok = True
            dt = time.perf_counter() - t
            method = {k: data[k] for k in ("phi0", "phi1", "tG", "T")}
            say(f"PASS  structured output in {dt:.1f} s: {json.dumps(method)}  rationale: {data['rationale'][:120]!r}")
    except BadRequestError as e:
        rep.structured_ok = False
        say(f"WARN  json_schema response_format rejected by this deployment: {e.message}. "
            f"The LLM search will have to use json_object mode or prompt-only JSON.")
    except json.JSONDecodeError as e:
        rep.structured_ok = False
        say(f"WARN  structured output was not valid JSON ({e}); first 200 chars: {content[:200]!r}")
    except (APIStatusError, APIConnectionError) as e:
        rep.structured_ok = False
        say(f"WARN  structured output call failed: {e}")
    return rep


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Check the Azure OpenAI deployment in .env.")
    p.add_argument("--env", type=Path, default=None,
                   help="env file to read (default: $HPLC_ENV_FILE, else <project>/.env)")
    a = p.parse_args(argv)
    try:
        settings = load_settings(a.env)
    except MissingSetting as e:
        print(f"FAIL  {e}")
        return 1
    rep = check(settings)
    print("result       " + ("OK: endpoint, key and deployment all respond" if rep.ok else "FAILED, see above"))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(_main())
