#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.26",
# ]
# ///
"""Plot batch + decode results from fastokens_batch.json + iree_batch.json.

Outputs (under ``bench/results/longbench/``):
- ``batch_encode_aggregate.png``      — fastokens vs IREE encode latency vs batch size
- ``batch_decode_aggregate.png``      — fastokens vs IREE decode latency vs batch size
- ``decode_single_aggregate.png``     — single decode latency curves over all 5 buckets
- ``per_model/<model>_batch.png``     — per-model batch encode/decode + speedups
- ``per_model/<model>_decode.png``    — per-model single decode curves
- ``batch_decode_summary.md``         — table of every (kind, model, params) row
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results" / "longbench"
FAST_PATH = RESULTS_DIR / "fastokens_batch.json"
IREE_PATH = RESULTS_DIR / "iree_batch.json"
PER_MODEL_DIR = RESULTS_DIR / "per_model"
SUMMARY_PATH = RESULTS_DIR / "batch_decode_summary.md"

BATCH_BUCKETS = [1024, 4096]
BATCH_SIZES = [8, 32, 128]
SINGLE_BUCKETS = [1024, 4096, 16384, 65536, 100_000]


def slugify(name: str) -> str:
    return name.replace("/", "__")


def load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"missing {path}")
    return json.loads(path.read_text())


def median_ms(samples_ms: list[float]) -> float:
    return statistics.median(samples_ms) if samples_ms else float("nan")


def aggregate_single(buckets: dict) -> dict[int, dict]:
    """Aggregate per-iteration timings across all samples in each bucket."""
    out: dict[int, dict] = {}
    for b_str, samples in buckets.items():
        all_iter: list[float] = []
        token_counts = []
        for s in samples:
            all_iter.extend(s["samples_ms"])
            token_counts.append(s["tokens"])
        if all_iter:
            out[int(b_str)] = {
                "median_ms": median_ms(all_iter),
                "p95_ms": float(np.percentile(all_iter, 95)),
                "tokens": int(round(median_ms(token_counts))),
            }
    return out


def aggregate_batch(batch_dict: dict) -> dict[tuple[int, int], dict]:
    """Returns {(bucket, batch_size): timing}."""
    out: dict[tuple[int, int], dict] = {}
    for key, entry in batch_dict.items():
        bucket = entry.get("bucket")
        batch_size = entry.get("batch_size")
        if bucket is None or batch_size is None:
            bucket_str, bs_str = key.split("|")
            bucket, batch_size = int(bucket_str), int(bs_str)
        out[(int(bucket), int(batch_size))] = {
            "median_ms": entry["median_ms"],
            "p95_ms": entry["p95_ms"],
            "total_tokens": entry.get("total_tokens", 0),
        }
    return out


def plot_per_model_batch(model: str, fast: dict, iree: dict) -> None:
    PER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fast_b = aggregate_batch(fast.get("batch_encode", {}))
    iree_b = aggregate_batch(iree.get("batch_encode", {}))
    fast_d = aggregate_batch(fast.get("batch_decode", {}))
    iree_d = aggregate_batch(iree.get("batch_decode", {}))

    if not (fast_b or fast_d):
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    for col, bucket in enumerate(BATCH_BUCKETS):
        # encode (top row)
        ax = axes[0, col]
        fast_xs, fast_ys = [], []
        iree_xs, iree_ys = [], []
        for bs in BATCH_SIZES:
            if (bucket, bs) in fast_b:
                fast_xs.append(bs)
                fast_ys.append(fast_b[(bucket, bs)]["median_ms"])
            if (bucket, bs) in iree_b:
                iree_xs.append(bs)
                iree_ys.append(iree_b[(bucket, bs)]["median_ms"])
        if fast_xs:
            ax.plot(fast_xs, fast_ys, marker="o", color="#FF914D", label="fastokens")
        if iree_xs:
            ax.plot(iree_xs, iree_ys, marker="s", color="#5A9BF6", label="IREE.Tokenizers")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels([str(b) for b in BATCH_SIZES])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Encode latency (ms, median)")
        ax.set_title(f"Batch encode @ {bucket} tokens/prompt")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        # decode (bottom row)
        ax = axes[1, col]
        fast_xs, fast_ys = [], []
        iree_xs, iree_ys = [], []
        for bs in BATCH_SIZES:
            if (bucket, bs) in fast_d:
                fast_xs.append(bs)
                fast_ys.append(fast_d[(bucket, bs)]["median_ms"])
            if (bucket, bs) in iree_d:
                iree_xs.append(bs)
                iree_ys.append(iree_d[(bucket, bs)]["median_ms"])
        if fast_xs:
            ax.plot(fast_xs, fast_ys, marker="o", color="#FF914D", label="fastokens")
        if iree_xs:
            ax.plot(iree_xs, iree_ys, marker="s", color="#5A9BF6", label="IREE.Tokenizers")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels([str(b) for b in BATCH_SIZES])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Decode latency (ms, median)")
        ax.set_title(f"Batch decode @ {bucket} tokens/prompt")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle(model, fontsize=13)
    fig.tight_layout()
    fig.savefig(PER_MODEL_DIR / f"{slugify(model)}_batch.png", dpi=120)
    plt.close(fig)


def plot_per_model_decode(model: str, fast: dict, iree: dict) -> None:
    PER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fast_d = aggregate_single(fast.get("single_decode", {}))
    iree_d = aggregate_single(iree.get("single_decode", {}))
    common = sorted(set(fast_d) & set(iree_d))
    if not common:
        return

    xs = [iree_d[b]["tokens"] for b in common]
    fast_ms = [fast_d[b]["median_ms"] for b in common]
    iree_ms = [iree_d[b]["median_ms"] for b in common]

    fig, (ax_lat, ax_speed) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_lat.plot(xs, fast_ms, marker="o", color="#FF914D", label="fastokens")
    ax_lat.plot(xs, iree_ms, marker="s", color="#5A9BF6", label="IREE.Tokenizers")
    ax_lat.set_xscale("log")
    ax_lat.set_yscale("log")
    ax_lat.set_xlabel("Input tokens")
    ax_lat.set_ylabel("Decode latency (ms, median)")
    ax_lat.set_title(f"{model}\nsingle decode latency vs token count")
    ax_lat.grid(True, which="both", alpha=0.3)
    ax_lat.legend()

    speedups = [fa / ir if ir > 0 else float("nan") for fa, ir in zip(fast_ms, iree_ms)]
    bars = ax_speed.bar(
        [str(b) for b in common],
        speedups,
        color=["#35C296" if s >= 1 else "#E15554" for s in speedups],
    )
    ax_speed.axhline(1.0, color="#7F8796", linewidth=0.8, linestyle="--")
    ax_speed.set_ylabel("fastokens / IREE.Tokenizers (median)")
    ax_speed.set_xlabel("Bucket (tokens)")
    ax_speed.set_title("Decode speedup (>1 → IREE faster)")
    for bar, val in zip(bars, speedups):
        ax_speed.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}×",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(PER_MODEL_DIR / f"{slugify(model)}_decode.png", dpi=120)
    plt.close(fig)


def plot_batch_aggregate(fast_data: dict, iree_data: dict, op: str, output: Path) -> None:
    """One panel per bucket; lines = models × {fastokens, IREE}; x = batch size."""
    bucket_panels = BATCH_BUCKETS
    fig, axes = plt.subplots(1, len(bucket_panels), figsize=(7 * len(bucket_panels), 5))
    if len(bucket_panels) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab20")
    models = sorted(set(fast_data.keys()) & set(iree_data.keys()))
    models = [m for m in models if "error" not in fast_data[m] and "error" not in iree_data[m]]

    field = "batch_encode" if op == "encode" else "batch_decode"
    title_op = "encode" if op == "encode" else "decode"

    for col, bucket in enumerate(bucket_panels):
        ax = axes[col]
        for i, model in enumerate(models):
            color = cmap(i % 20)
            fb = aggregate_batch(fast_data[model].get(field, {}))
            ib = aggregate_batch(iree_data[model].get(field, {}))
            fast_xs = [bs for bs in BATCH_SIZES if (bucket, bs) in fb]
            fast_ys = [fb[(bucket, bs)]["median_ms"] for bs in fast_xs]
            iree_xs = [bs for bs in BATCH_SIZES if (bucket, bs) in ib]
            iree_ys = [ib[(bucket, bs)]["median_ms"] for bs in iree_xs]
            if fast_xs:
                ax.plot(fast_xs, fast_ys, marker="o", linestyle="-", color=color, alpha=0.85,
                        label=f"{model} (fastokens)")
            if iree_xs:
                ax.plot(iree_xs, iree_ys, marker="s", linestyle="--", color=color, alpha=0.85,
                        label=f"{model} (IREE)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels([str(b) for b in BATCH_SIZES])
        ax.set_xlabel("Batch size")
        ax.set_ylabel(f"{title_op.capitalize()} latency (ms, median)")
        ax.set_title(f"Batch {title_op} @ {bucket} tokens/prompt")
        ax.grid(True, which="both", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Batch {title_op} latency — fastokens vs IREE.Tokenizers", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_decode_aggregate(fast_data: dict, iree_data: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab20")
    models = sorted(set(fast_data.keys()) & set(iree_data.keys()))
    models = [m for m in models if "error" not in fast_data[m] and "error" not in iree_data[m]]

    for i, model in enumerate(models):
        color = cmap(i % 20)
        fd = aggregate_single(fast_data[model].get("single_decode", {}))
        id_ = aggregate_single(iree_data[model].get("single_decode", {}))
        common = sorted(set(fd) & set(id_))
        if not common:
            continue
        xs = [id_[b]["tokens"] for b in common]
        fast_ys = [fd[b]["median_ms"] for b in common]
        iree_ys = [id_[b]["median_ms"] for b in common]
        ax.plot(xs, fast_ys, marker="o", linestyle="-", color=color, alpha=0.85,
                label=f"{model} (fastokens)")
        ax.plot(xs, iree_ys, marker="s", linestyle="--", color=color, alpha=0.85,
                label=f"{model} (IREE)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input tokens")
    ax.set_ylabel("Decode latency (ms, median)")
    ax.set_title("Single decode latency — fastokens vs IREE.Tokenizers")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def write_summary(fast_data: dict, iree_data: dict) -> None:
    lines: list[str] = []
    lines.append("# LongBench-v2 — batch encode + decode")
    lines.append("")
    lines.append(
        "Synthetic 1K and 4K prompts are produced by truncating the natural "
        "16K LongBench contexts (encode → slice → decode → re-encode for ground "
        "truth token count). Batches reuse the same three samples cycled to the "
        "target batch size."
    )
    lines.append("")

    models = sorted(set(fast_data.keys()) | set(iree_data.keys()))

    # Single decode table
    lines.append("## Single decode (median ms)")
    lines.append("")
    lines.append("| Model | 1K | 4K | 16K | 64K | 100K |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for model in models:
        fd = aggregate_single(fast_data.get(model, {}).get("single_decode", {}))
        id_ = aggregate_single(iree_data.get(model, {}).get("single_decode", {}))
        cells = [model]
        for bucket in SINGLE_BUCKETS:
            if bucket in fd and bucket in id_:
                f, i = fd[bucket]["median_ms"], id_[bucket]["median_ms"]
                speed = f / i if i > 0 else float("nan")
                cells.append(f"{f:.2f} / {i:.2f} ({speed:.2f}×)")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cells: fastokens / IREE.Tokenizers (speedup = fast/iree).")
    lines.append("")

    # Batch encode table
    for op, field in [("encode", "batch_encode"), ("decode", "batch_decode")]:
        lines.append(f"## Batch {op} (median ms)")
        lines.append("")
        lines.append(
            "| Model | 1K×8 | 1K×32 | 1K×128 | 4K×8 | 4K×32 | 4K×128 |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for model in models:
            f_b = aggregate_batch(fast_data.get(model, {}).get(field, {}))
            i_b = aggregate_batch(iree_data.get(model, {}).get(field, {}))
            cells = [model]
            for bucket in BATCH_BUCKETS:
                for bs in BATCH_SIZES:
                    if (bucket, bs) in f_b and (bucket, bs) in i_b:
                        f = f_b[(bucket, bs)]["median_ms"]
                        i = i_b[(bucket, bs)]["median_ms"]
                        speed = f / i if i > 0 else float("nan")
                        cells.append(f"{f:.2f} / {i:.2f} ({speed:.2f}×)")
                    else:
                        cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    fast = load(FAST_PATH)
    iree = load(IREE_PATH)

    plot_batch_aggregate(fast, iree, "encode", RESULTS_DIR / "batch_encode_aggregate.png")
    plot_batch_aggregate(fast, iree, "decode", RESULTS_DIR / "batch_decode_aggregate.png")
    plot_decode_aggregate(fast, iree, RESULTS_DIR / "decode_single_aggregate.png")

    for model in sorted(set(fast.keys()) & set(iree.keys())):
        if "error" in fast.get(model, {}) or "error" in iree.get(model, {}):
            continue
        plot_per_model_batch(model, fast[model], iree[model])
        plot_per_model_decode(model, fast[model], iree[model])

    write_summary(fast, iree)

    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {RESULTS_DIR / 'batch_encode_aggregate.png'}")
    print(f"Wrote {RESULTS_DIR / 'batch_decode_aggregate.png'}")
    print(f"Wrote {RESULTS_DIR / 'decode_single_aggregate.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
