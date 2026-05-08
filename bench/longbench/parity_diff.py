#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastokens",
#     "transformers>=4.45",
#     "huggingface-hub>=0.24",
# ]
# ///
"""Position-by-position parity diff: HF AutoTokenizer (ground truth) vs IREE.

Reads each <model>__iree.json in results/longbench/parity/, encodes the same
text via HuggingFace `AutoTokenizer.from_pretrained(..., use_fast=True)`, and
reports:
- exact match? (same length AND same IDs)
- first divergence index
- the surrounding context tokens (decoded) at the divergence point
- count delta at end

We also re-encode via fastokens to confirm fastokens still matches HF (it did
in the prior parity_check, but verifying for these specific samples too).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results" / "longbench"
TEXTS_DIR = RESULTS_DIR / "texts"
PARITY_DIR = RESULTS_DIR / "parity"


def main() -> int:
    from fastokens._native import Tokenizer  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    files = sorted(PARITY_DIR.glob("*__iree.json"))
    if not files:
        sys.exit("no IREE parity dumps found; run parity_iree_dump.exs first")

    for path in files:
        dump = json.loads(path.read_text())
        model = dump["model"]
        sha = dump["sha1"]
        iree_ids = dump["ids"]
        text = (TEXTS_DIR / f"{sha}.txt").read_text(encoding="utf-8")

        try:
            htok = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=False)
            hf_ids = htok.encode(text, add_special_tokens=False)
        except Exception as exc:  # noqa: BLE001
            print(f"\n=== {model} ===")
            print(f"  HF load failed: {exc}")
            continue

        try:
            ftok = Tokenizer.from_model(model)
            f_raw = ftok.encode(text)
            fast_ids = list(f_raw if isinstance(f_raw, list) else f_raw.ids)
        except Exception as exc:  # noqa: BLE001
            fast_ids = None

        # First divergence index between iree and hf.
        n = min(len(iree_ids), len(hf_ids))
        first_diff = next((i for i in range(n) if iree_ids[i] != hf_ids[i]), n)

        match = first_diff == len(iree_ids) == len(hf_ids)

        print(f"\n=== {model} ===")
        print(f"  sample sha:      {sha[:12]}")
        print(f"  HF ids:          {len(hf_ids)} (ground truth)")
        print(f"  IREE ids:        {len(iree_ids)}  (Δ={len(iree_ids) - len(hf_ids):+d})")
        if fast_ids is not None:
            print(f"  fastokens ids:   {len(fast_ids)} (Δ vs HF: {len(fast_ids) - len(hf_ids):+d})")
            fast_match = fast_ids == hf_ids
            print(f"  fastokens=HF?    {'YES' if fast_match else 'NO'}")
        print(f"  IREE=HF?         {'YES (exact)' if match else 'NO'}")

        if not match:
            print(f"  first diff at:   index {first_diff}")
            ctx_lo = max(0, first_diff - 2)
            ctx_hi = min(len(iree_ids), len(hf_ids), first_diff + 3)
            print(f"  HF[{ctx_lo}:{ctx_hi}]   = {hf_ids[ctx_lo:ctx_hi]}")
            print(f"  IREE[{ctx_lo}:{ctx_hi}] = {iree_ids[ctx_lo:ctx_hi]}")
            # Decode the surrounding text
            try:
                hf_chunk = htok.decode(hf_ids[ctx_lo:ctx_hi])
                iree_chunk = htok.decode(iree_ids[ctx_lo:min(ctx_hi, len(iree_ids))])
                print(f"  HF text chunk:   {hf_chunk!r}")
                print(f"  IREE text chunk: {iree_chunk!r}")
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
