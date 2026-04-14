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

The report is written to `bench/results/parity_report.md`. Known failing
cases are documented in [`docs/UPSTREAM_BUGS.md`](../docs/UPSTREAM_BUGS.md);
they trace into the vendored IREE tokenizer C runtime and must be fixed
upstream rather than patched in this package.
