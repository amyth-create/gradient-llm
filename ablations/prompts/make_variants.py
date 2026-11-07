"""Build the prompt variants for the hard-set prompt study from the full system prompt.

Each variant is the full prompt (optimizers/llm_system_prompt.txt) with whole sections removed or
one section replaced, so the files differ from the original only where the variant says. Run:

    .venv/bin/python ablations/prompts/make_variants.py

Variants
  full        the prompt used in the main campaigns (not written here; the original file)
  no_theory   "## Retention model" removed (same file as optimizers/llm_system_prompt_no_theory.txt)
  no_howto    "## How to work" removed; retention model and score kept
  no_score    the CRF formula and its explanation replaced by "higher is better"; the one
              "How to work" bullet that restates the rule (resolve every pair, then trade time) removed
  minimal     retention model and "How to work" removed and the score reduced to "higher is better":
              instrument, method bounds, what a run returns and the answer format only
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "optimizers" / "llm_system_prompt.txt"
OUT = Path(__file__).resolve().parent

SCORE_SHORT = """## The score
Each run returns a CRF between -1.6 and 1.0; higher is better and 1.0 is the maximum.
"""

HOWTO_SCORE_BULLET = re.compile(
    r"- Once every pair is at R >= 1\.5, the remaining gain is time:.*?\n(?=- |\n|Answer)", re.S)


def split_sections(text: str) -> tuple[str, dict[str, str], str]:
    """Preamble, {heading: section text including its heading}, tail (the answer format)."""
    parts = re.split(r"(?m)^(?=## )", text)
    pre, sections = parts[0], {}
    for p in parts[1:]:
        heading = p.splitlines()[0][3:].split(" (")[0].strip()
        sections[heading] = p
    # the answer format is the end of the last section
    last = list(sections)[-1]
    body, tail = sections[last].split("Answer with JSON only:")
    sections[last] = body
    return pre, sections, "Answer with JSON only:" + tail


def build(drop: set[str], score_short: bool, drop_howto_score_bullet: bool) -> str:
    pre, sections, tail = split_sections(FULL.read_text())
    out = [pre]
    for heading, body in sections.items():
        if heading in drop:
            continue
        if heading == "The score" and score_short:
            body = SCORE_SHORT + "\n"
        if heading == "How to work" and drop_howto_score_bullet:
            body = HOWTO_SCORE_BULLET.sub("", body)
        out.append(body)
    return "".join(out).rstrip("\n") + "\n\n" + tail


VARIANTS = {
    "no_theory": dict(drop={"Retention model"}, score_short=False, drop_howto_score_bullet=False),
    "no_howto": dict(drop={"How to work"}, score_short=False, drop_howto_score_bullet=False),
    "no_score": dict(drop=set(), score_short=True, drop_howto_score_bullet=True),
    "minimal": dict(drop={"Retention model", "How to work"}, score_short=True, drop_howto_score_bullet=False),
}

if __name__ == "__main__":
    full_lines = len(FULL.read_text().splitlines())
    print(f"full: {full_lines} lines")
    for name, spec in VARIANTS.items():
        text = build(**spec)
        (OUT / f"{name}.txt").write_text(text)
        print(f"{name}: {len(text.splitlines())} lines -> {OUT / (name + '.txt')}")
