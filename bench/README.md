# Benchmarks

This directory contains a small benchmark harness that compares `IREE.Tokenizers`
against the published `tokenizers` package.

## Run

```bash
cd bench
mix deps.get
mix run compare.exs
MODEL_FILTER="Qwen/Qwen3.5-9B" mix run model_matrix_graphs.exs
```

The benchmark uses the local minimal BPE fixture and exercises short, medium,
and long encode/decode workloads.

`compare.exs` generates:
- `bench/results/tokenizers_compare_encode.svg`
- `bench/results/tokenizers_compare_decode.svg`
- `bench/results/tokenizers_compare.md`

`model_matrix_graphs.exs` benchmarks a curated list of public model repos and
generates latency and speedup SVG charts similar to the ZML blog post. Set
`HF_TOKEN` if any benchmark target requires authentication.
