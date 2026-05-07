#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastokens",
#     "huggingface-hub>=0.24",
#     "datasets>=3.0",
#     "transformers>=4.45",
# ]
# ///
"""Long-context tokenizer benchmark — fastokens side.

For each model in ``MODELS`` the script loads the LongBench-v2 dataset, picks
``SAMPLES_PER_BUCKET`` items whose token count (per that model's tokenizer)
falls closest to each target bucket, and times ``fastokens._native.Tokenizer.encode``
with ``WARMUP`` warmups + ``ITERS`` timed iterations.

Outputs (under ``bench/results/longbench/``):
- ``inputs.jsonl``      — one row per (model, bucket, sample) with sha1 → text
- ``texts/<sha1>.txt``  — deduplicated input texts
- ``fastokens.json``    — timings keyed by model + bucket + sample

The Elixir runner ``bench/longbench_compare.exs`` consumes the same inputs,
ensuring both libraries time the *exact same* string.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS: list[str] = [
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3.5-397B-A17B",
    "MiniMaxAI/MiniMax-M2.1",
    "MiniMaxAI/MiniMax-M2.5",
    "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "zai-org/GLM-4.7",
    "zai-org/GLM-5",
]

# Token-count buckets (target lengths the LongBench prompts will be selected against).
BUCKETS: list[int] = [1_024, 4_096, 16_384, 65_536, 100_000]
BUCKET_TOLERANCE = (0.75, 1.5)  # sample must lie within [0.75x, 1.5x] of target
SAMPLES_PER_BUCKET = 3

WARMUP = 3
ITERS = 10

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results" / "longbench"
TEXTS_DIR = RESULTS_DIR / "texts"
INPUTS_PATH = RESULTS_DIR / "inputs.jsonl"
TIMINGS_PATH = RESULTS_DIR / "fastokens.json"

# Filter applied to MODELS via env var (comma-separated repos).
MODEL_FILTER = os.environ.get("MODEL_FILTER")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class InputRow:
    model: str
    bucket: int
    sample_idx: int
    sha1: str
    natural_tokens: int
    longbench_id: str


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sha1_of(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def write_text(text: str) -> str:
    digest = sha1_of(text)
    path = TEXTS_DIR / f"{digest}.txt"
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    return digest


def filtered_models() -> list[str]:
    if not MODEL_FILTER:
        return MODELS
    wanted = {m.strip() for m in MODEL_FILTER.split(",") if m.strip()}
    return [m for m in MODELS if m in wanted]


def load_longbench() -> list[dict]:
    from datasets import load_dataset

    log("Loading LongBench-v2 (zai-org/LongBench-v2)…")
    ds = load_dataset("zai-org/LongBench-v2", split="train")
    items = []
    for row in ds:
        ctx = row.get("context") or ""
        if not ctx:
            continue
        items.append({"_id": row.get("_id", ""), "context": ctx})
    log(f"  {len(items)} contexts loaded")
    return items


def pick_samples_for_buckets(
    token_counts: list[tuple[int, int]],  # [(item_idx, token_count), …]
) -> dict[int, list[int]]:
    """For each bucket return up to SAMPLES_PER_BUCKET item indices.

    Items are picked greedily by closeness to the bucket target, restricted to
    a tolerance window. Items already used by a smaller bucket are not reused.
    """
    used: set[int] = set()
    selection: dict[int, list[int]] = {b: [] for b in BUCKETS}

    for bucket in BUCKETS:
        lo = bucket * BUCKET_TOLERANCE[0]
        hi = bucket * BUCKET_TOLERANCE[1]
        candidates = [
            (idx, count) for idx, count in token_counts
            if idx not in used and lo <= count <= hi
        ]
        candidates.sort(key=lambda pair: abs(pair[1] - bucket))
        chosen = [idx for idx, _ in candidates[:SAMPLES_PER_BUCKET]]
        selection[bucket] = chosen
        used.update(chosen)
    return selection


def _ids_sha(ids) -> str:
    """sha256 of a token-id list, for parity checking without storing full IDs."""
    if hasattr(ids, "ids"):
        ids = ids.ids
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    payload = ",".join(str(int(v)) for v in ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def time_encode(tokenizer, text: str) -> dict:
    """Return per-iteration ms timings + token count + ids_sha for parity check."""
    # Warmup — also asserts encode actually runs to completion.
    for _ in range(WARMUP):
        tokenizer.encode(text)
    samples_ms: list[float] = []
    token_count = 0
    ids_sha = ""
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        ids = tokenizer.encode(text)
        elapsed_ns = time.perf_counter_ns() - t0
        samples_ms.append(elapsed_ns / 1_000_000)
        token_count = _id_count(ids)
        ids_sha = _ids_sha(ids)
    return {
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "p95_ms": _p95(samples_ms),
        "tokens": token_count,
        "ids_sha": ids_sha,
    }


def _id_count(ids) -> int:
    # fastokens.encode may return a list[int] or an object with .ids — handle both.
    if isinstance(ids, list):
        return len(ids)
    if hasattr(ids, "ids"):
        return len(ids.ids)
    if hasattr(ids, "input_ids"):
        return len(ids.input_ids)
    return len(list(ids))


def _p95(samples: list[float]) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    k = max(0, int(round(0.95 * (len(s) - 1))))
    return s[k]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    from fastokens._native import Tokenizer  # type: ignore

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    longbench = load_longbench()
    if not longbench:
        log("LongBench-v2 returned no contexts; aborting.")
        return 1

    models = filtered_models()
    log(f"Benchmarking {len(models)} models")

    inputs: list[InputRow] = []
    timings: dict[str, dict] = {}

    for model in models:
        log(f"\n=== {model} ===")
        try:
            tok = Tokenizer.from_model(model)
        except Exception as exc:  # noqa: BLE001 — surface any load failure
            log(f"  load failed: {exc!r}")
            timings[model] = {"error": f"load failed: {exc!r}"}
            continue

        # 1. Pre-tokenize each context to discover its natural length.
        log(f"  measuring lengths over {len(longbench)} contexts…")
        token_counts: list[tuple[int, int]] = []
        for idx, item in enumerate(longbench):
            try:
                ids = tok.encode(item["context"])
                token_counts.append((idx, _id_count(ids)))
            except Exception as exc:  # noqa: BLE001
                log(f"    skipped item {idx}: {exc!r}")

        if not token_counts:
            log("  no contexts tokenized; skipping model.")
            timings[model] = {"error": "no contexts tokenized"}
            continue

        # 2. Pick representative samples per bucket.
        selection = pick_samples_for_buckets(token_counts)
        bucket_results: dict[str, list[dict]] = {}
        for bucket, idxs in selection.items():
            log(f"  bucket {bucket}: {len(idxs)} sample(s)")
            samples_out: list[dict] = []
            for sample_idx, item_idx in enumerate(idxs):
                ctx = longbench[item_idx]["context"]
                natural = next(c for i, c in token_counts if i == item_idx)
                sha = write_text(ctx)
                inputs.append(
                    InputRow(
                        model=model,
                        bucket=bucket,
                        sample_idx=sample_idx,
                        sha1=sha,
                        natural_tokens=natural,
                        longbench_id=longbench[item_idx]["_id"],
                    )
                )
                t = time_encode(tok, ctx)
                samples_out.append(
                    {
                        "sample_idx": sample_idx,
                        "sha1": sha,
                        "natural_tokens": natural,
                        **t,
                    }
                )
                log(
                    f"    sample {sample_idx}: tokens={t['tokens']} "
                    f"median={t['median_ms']:.2f}ms p95={t['p95_ms']:.2f}ms"
                )
            bucket_results[str(bucket)] = samples_out

        timings[model] = {"buckets": bucket_results}

    # Persist artifacts.
    with INPUTS_PATH.open("w", encoding="utf-8") as fh:
        for row in inputs:
            fh.write(json.dumps(asdict(row)) + "\n")
    TIMINGS_PATH.write_text(json.dumps(timings, indent=2), encoding="utf-8")

    log(f"\nWrote {INPUTS_PATH} ({len(inputs)} rows)")
    log(f"Wrote {TIMINGS_PATH}")
    log(f"Texts cached under {TEXTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
