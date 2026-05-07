# Benchmarks

This directory contains a small benchmark harness that compares `IREE.Tokenizers`
against the published `tokenizers` package.

## Run

```bash
cd bench
mix deps.get
mix run compare.exs
mix run sentencepiece_compare.exs
MODEL_FILTER="Qwen/Qwen3.5-9B" mix run model_matrix_graphs.exs
```

The benchmark uses the local minimal BPE fixture and exercises short, medium,
and long encode/decode workloads.

`compare.exs` generates:
- `bench/results/tokenizers_compare_encode.svg`
- `bench/results/tokenizers_compare_decode.svg`
- `bench/results/tokenizers_compare.md`

`sentencepiece_compare.exs` generates:
- `bench/results/sentencepiece_compare_encode.svg`
- `bench/results/sentencepiece_compare_decode.svg`
- `bench/results/sentencepiece_compare.md`

`model_matrix_graphs.exs` benchmarks a curated list of public model repos and
generates latency and speedup SVG charts similar to the ZML blog post. Set
`HF_TOKEN` if any benchmark target requires authentication.

## LongBench-v2: fastokens vs IREE.Tokenizers

Long-context comparison against [`crusoecloud/fastokens`](https://github.com/crusoecloud/fastokens),
exercising prompts drawn from [`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2).
Each model is evaluated over five token-count buckets — 1K, 4K, 16K, 64K,
100K — with three samples per bucket and 3 warmup + 10 timed iterations per
sample.

The harness is split in two so each library times the *exact same* byte
strings:

```bash
# 1. Time fastokens, persist the bucketed prompt corpus + texts/<sha>.txt cache.
./bench/longbench/run_fastokens.py

# 2. Time IREE.Tokenizers on the same inputs.
cd bench && mix run longbench_compare.exs

# 3. Join both result files and emit charts + summary.
./bench/longbench/plot.py
```

Outputs land under `bench/results/longbench/`:

- `inputs.jsonl`, `texts/<sha1>.txt` — shared prompt corpus
- `fastokens.json`, `iree.json` — per-(model, bucket, sample) timings
- `per_model/<model>.png` — per-model latency + speedup charts
- `aggregate_latency.png`, `aggregate_speedup_heatmap.png`, `summary.md`

`MODEL_FILTER="deepseek-ai/DeepSeek-V3,Qwen/Qwen3-Next-80B-A3B-Instruct"` works
as a comma-separated filter on the Python side; the Elixir runner simply
processes whatever models appear in `inputs.jsonl`. The Python script picks up
the cached HF token automatically; the Elixir runner reads `HF_TOKEN`,
falling back to `~/.cache/huggingface/token`.

## Parity validation

`validate_parity.exs` is a regression runner that checks encoder, decoder,
`encode_batch/3`, and `EncodeStream` output of `IREE.Tokenizers` against
`elixir-nx/tokenizers` (the Rust-backed Hugging Face `tokenizers` crate),
using real public tokenizers from the Hugging Face Hub. It exercises 19
representative inputs per model, in both `add_special_tokens: true/false`
modes, including long (100–200 KB) ASCII / CJK / mixed sequences and emoji
with ZWJ/skin-tone modifiers.

```bash
cd bench
mix run validate_parity.exs                               # full matrix
MODEL_FILTER="Qwen/Qwen2.5-7B-Instruct" mix run validate_parity.exs  # one model
HF_TOKEN=hf_... mix run validate_parity.exs              # gated repos
```

The report is written to `bench/results/parity_report.md`. Historical upstream
notes and the local fixes/workarounds that closed earlier gaps are documented in
[`docs/UPSTREAM_BUGS.md`](../docs/UPSTREAM_BUGS.md); consult the latest parity
report for the live status on the current branch. The selected matrix is green
on this branch.
