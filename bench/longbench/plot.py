#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.8",
#     "numpy>=1.26",
# ]
# ///
"""Join fastokens.json + iree.json and emit graphs + a summary markdown.

Reads the JSONs produced by ``run_fastokens.py`` and ``longbench_compare.exs``
under ``bench/results/longbench/`` and writes:

- ``per_model/<model_slug>_latency.png`` — encode latency vs token bucket
- ``per_model/<model_slug>_speedup.png`` — fastokens÷iree per bucket
- ``aggregate_latency.png``               — all models, log-y latency curves
- ``aggregate_speedup_heatmap.png``       — speedup matrix (model × bucket)
- ``summary.md``                          — table per model + skipped notes
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
FASTOKENS_PATH = RESULTS_DIR / "fastokens.json"
IREE_PATH = RESULTS_DIR / "iree.json"
PER_MODEL_DIR = RESULTS_DIR / "per_model"
AGGREGATE_LATENCY_PATH = RESULTS_DIR / "aggregate_latency.png"
AGGREGATE_HEATMAP_PATH = RESULTS_DIR / "aggregate_speedup_heatmap.png"
SUMMARY_PATH = RESULTS_DIR / "summary.md"


def slugify(name: str) -> str:
    return name.replace("/", "__")


def load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"missing {path}")
    return json.loads(path.read_text())


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def aggregate_bucket(samples: list[dict]) -> dict:
    """Combine all samples in a bucket into one (median over per-iter samples)."""
    all_iter = []
    token_counts = []
    for s in samples:
        all_iter.extend(s["samples_ms"])
        token_counts.append(s["tokens"])
    if not all_iter:
        return {}
    return {
        "median_ms": median(all_iter),
        "p95_ms": float(np.percentile(all_iter, 95)),
        "tokens": int(round(median(token_counts))),
        "n_samples": len(samples),
        "n_iters": len(all_iter),
    }


def per_model_chart(
    model: str,
    fast_buckets: dict[str, dict],
    iree_buckets: dict[str, dict],
) -> None:
    PER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    common = sorted(
        (int(b) for b in fast_buckets.keys() & iree_buckets.keys()),
    )
    if not common:
        return

    xs = [fast_buckets[str(b)]["tokens"] for b in common]
    fast_ms = [fast_buckets[str(b)]["median_ms"] for b in common]
    iree_ms = [iree_buckets[str(b)]["median_ms"] for b in common]

    fig, (ax_lat, ax_speed) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_lat.plot(xs, fast_ms, marker="o", label="fastokens", color="#FF914D")
    ax_lat.plot(xs, iree_ms, marker="s", label="IREE.Tokenizers", color="#5A9BF6")
    ax_lat.set_xscale("log")
    ax_lat.set_yscale("log")
    ax_lat.set_xlabel("Input tokens")
    ax_lat.set_ylabel("Encode latency (ms, median)")
    ax_lat.set_title(f"{model}\nlatency vs context length")
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
    ax_speed.set_xlabel("Bucket (target tokens)")
    ax_speed.set_title("Speedup (>1 → IREE faster)")
    for bar, val in zip(bars, speedups):
        ax_speed.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}×",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out = PER_MODEL_DIR / f"{slugify(model)}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


def aggregate_latency_chart(rows: list[dict]) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab20")
    for i, row in enumerate(rows):
        color = cmap(i % 20)
        ax.plot(
            row["xs"], row["fast_ms"],
            marker="o", linestyle="-", color=color, alpha=0.85,
            label=f"{row['model']} (fastokens)",
        )
        ax.plot(
            row["xs"], row["iree_ms"],
            marker="s", linestyle="--", color=color, alpha=0.85,
            label=f"{row['model']} (IREE)",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input tokens")
    ax.set_ylabel("Encode latency (ms, median)")
    ax.set_title("Long-context tokenizer latency — fastokens vs IREE.Tokenizers")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(AGGREGATE_LATENCY_PATH, dpi=130)
    plt.close(fig)


def aggregate_heatmap(rows: list[dict], all_buckets: list[int]) -> None:
    if not rows or not all_buckets:
        return
    matrix = np.full((len(rows), len(all_buckets)), np.nan)
    for r_idx, row in enumerate(rows):
        for c_idx, bucket in enumerate(all_buckets):
            if bucket in row["bucket_to_speedup"]:
                matrix[r_idx, c_idx] = row["bucket_to_speedup"][bucket]

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 2.5))
    cmap = plt.get_cmap("RdYlGn")
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    vmax = max(2.0, float(np.nanpercentile(finite, 95)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)

    ax.set_xticks(range(len(all_buckets)))
    ax.set_xticklabels([f"{b // 1024}K" if b >= 1024 else str(b) for b in all_buckets])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row["model"] for row in rows], fontsize=8)
    ax.set_xlabel("Bucket (target tokens)")
    ax.set_title("IREE.Tokenizers speedup vs fastokens (>1 → IREE faster)")

    for r_idx in range(matrix.shape[0]):
        for c_idx in range(matrix.shape[1]):
            v = matrix[r_idx, c_idx]
            if np.isfinite(v):
                ax.text(
                    c_idx, r_idx, f"{v:.2f}×",
                    ha="center", va="center",
                    color="black" if v < vmax * 0.6 else "white",
                    fontsize=8,
                )

    fig.colorbar(im, ax=ax, label="speedup ×")
    fig.tight_layout()
    fig.savefig(AGGREGATE_HEATMAP_PATH, dpi=130)
    plt.close(fig)


def write_summary(rows: list[dict], skipped: list[tuple[str, str]]) -> None:
    lines: list[str] = []
    lines.append("# LongBench-v2 — fastokens vs IREE.Tokenizers")
    lines.append("")
    lines.append("Encode latency over LongBench-v2 contexts, bucketed by token count.")
    lines.append(
        "Each cell is the median across {samples per bucket × iterations per sample}; "
        "speedup = fastokens median / IREE.Tokenizers median."
    )
    lines.append("")

    for row in rows:
        lines.append(f"## {row['model']}")
        lines.append("")
        lines.append("| Bucket | Median tokens | fastokens median | IREE.Tokenizers median | Speedup |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for bucket in row["xs_buckets"]:
            f = row["fast_by_bucket"][bucket]
            i = row["iree_by_bucket"][bucket]
            speed = f["median_ms"] / i["median_ms"] if i["median_ms"] > 0 else float("nan")
            lines.append(
                f"| {bucket} | {i['tokens']} | {f['median_ms']:.2f} ms | "
                f"{i['median_ms']:.2f} ms | {speed:.2f}× |"
            )
        lines.append("")

    if skipped:
        lines.append("## Skipped models")
        lines.append("")
        for model, reason in skipped:
            lines.append(f"- `{model}` — {reason}")
        lines.append("")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    fast = load_json(FASTOKENS_PATH)
    iree = load_json(IREE_PATH)

    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    seen_buckets: set[int] = set()

    models = sorted(set(fast.keys()) | set(iree.keys()))

    for model in models:
        fast_entry = fast.get(model, {})
        iree_entry = iree.get(model, {})

        if "error" in fast_entry or "error" in iree_entry:
            reason = fast_entry.get("error") or iree_entry.get("error")
            skipped.append((model, reason))
            continue

        fast_buckets_raw = fast_entry.get("buckets", {})
        iree_buckets_raw = iree_entry.get("buckets", {})

        fast_by_bucket = {}
        iree_by_bucket = {}
        for b_str, samples in fast_buckets_raw.items():
            agg = aggregate_bucket(samples)
            if agg:
                fast_by_bucket[int(b_str)] = agg
        for b_str, samples in iree_buckets_raw.items():
            agg = aggregate_bucket(samples)
            if agg:
                iree_by_bucket[int(b_str)] = agg

        common = sorted(set(fast_by_bucket) & set(iree_by_bucket))
        if not common:
            skipped.append((model, "no overlapping buckets between fastokens and IREE"))
            continue

        seen_buckets.update(common)

        per_model_chart(
            model,
            {str(b): fast_by_bucket[b] for b in common},
            {str(b): iree_by_bucket[b] for b in common},
        )

        rows.append({
            "model": model,
            "xs_buckets": common,
            "xs": [iree_by_bucket[b]["tokens"] for b in common],
            "fast_ms": [fast_by_bucket[b]["median_ms"] for b in common],
            "iree_ms": [iree_by_bucket[b]["median_ms"] for b in common],
            "fast_by_bucket": fast_by_bucket,
            "iree_by_bucket": iree_by_bucket,
            "bucket_to_speedup": {
                b: fast_by_bucket[b]["median_ms"] / iree_by_bucket[b]["median_ms"]
                for b in common
                if iree_by_bucket[b]["median_ms"] > 0
            },
        })

    aggregate_latency_chart(rows)
    aggregate_heatmap(rows, sorted(seen_buckets))
    write_summary(rows, skipped)

    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {AGGREGATE_LATENCY_PATH}")
    print(f"Wrote {AGGREGATE_HEATMAP_PATH}")
    print(f"Per-model charts under {PER_MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
