#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastokens",
#     "transformers>=4.45",
#     "huggingface-hub>=0.24",
# ]
# ///
"""Three-way parity audit: fastokens vs HuggingFace AutoTokenizer (reference).

For each model + sample, encode the same text three ways:
1. fastokens._native.Tokenizer.from_model(...).encode
2. transformers.AutoTokenizer.from_pretrained(..., use_fast=True).encode
   (this is HuggingFace's reference Rust tokenizers crate)

We can't run the IREE tokenizer from Python, but the IREE side is already
parity-validated against elixir-nx/tokenizers (which wraps the same HF Rust
crate) — so if fastokens disagrees with HF, fastokens is the diverging library
relative to the canonical reference, and IREE matches HF.

We report: for each model, first divergence index between fast and HF, plus
final-ID-count delta and length of common prefix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results" / "longbench"
TEXTS_DIR = RESULTS_DIR / "texts"
INPUTS_PATH = RESULTS_DIR / "inputs.jsonl"

# One sample per model to keep runtime reasonable; pick 16K bucket.
TARGET_BUCKET = 16384


def read_text(sha: str) -> str:
    return (TEXTS_DIR / f"{sha}.txt").read_text(encoding="utf-8")


def fast_ids(tok, text: str) -> list[int]:
    enc = tok.encode(text)
    if isinstance(enc, list):
        return list(enc)
    if hasattr(enc, "ids"):
        return list(enc.ids)
    return list(enc)


def hf_ids(tok, text: str) -> list[int]:
    # AutoTokenizer.encode() applies special tokens by default; we want raw
    # text→ids, so use add_special_tokens=False to match IREE's bench config.
    return tok.encode(text, add_special_tokens=False)


def first_divergence(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n  # one is a prefix of the other


def main() -> int:
    from fastokens._native import Tokenizer  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    rows: list[dict] = []
    with INPUTS_PATH.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("bucket") == TARGET_BUCKET and row.get("sample_idx") == 0:
                rows.append(row)

    print(f'{"model":50s} {"fast":>6} {"hf":>6} {"common":>7} {"first_diff":>11} {"verdict":>20}')
    for row in rows:
        model = row["model"]
        sha = row["sha1"]
        text = read_text(sha)
        try:
            ftok = Tokenizer.from_model(model)
            htok = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=False)
        except Exception as exc:  # noqa: BLE001
            print(f'{model:50s} {"?":>6} {"?":>6} {"?":>7} {"?":>11} {"LOAD_FAILED " + str(exc)[:50]:>20}')
            continue

        f_ids = fast_ids(ftok, text)
        h_ids = hf_ids(htok, text)
        diff = first_divergence(f_ids, h_ids)

        if diff == len(f_ids) == len(h_ids):
            verdict = "EXACT MATCH"
        elif diff == len(f_ids) and diff == len(h_ids):
            verdict = "EXACT MATCH"
        else:
            verdict = "diverges" + (" (lengths agree)" if len(f_ids) == len(h_ids) else "")

        print(
            f'{model:50s} {len(f_ids):>6} {len(h_ids):>6} {diff:>7} '
            f'{diff:>11} {verdict:>20}'
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
