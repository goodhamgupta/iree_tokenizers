# Benchmarks

This directory contains a small benchmark harness that compares `IREE.Tokenizers`
against the published `tokenizers` package.

## Run

```bash
cd bench
mix deps.get
mix run compare.exs
mix run gpt2_graphs.exs
MODEL_FILTER="Qwen/Qwen3.5-9B" mix run model_matrix_graphs.exs
```

The benchmark uses the local minimal BPE fixture and exercises short, medium,
and long encode/decode workloads.

`gpt2_graphs.exs` benchmarks GPT-2 encode/decode throughput on a batch of 100
prompts and writes SVG charts plus a Markdown summary into `bench/results/`.

`model_matrix_graphs.exs` benchmarks a curated list of public model repos and
generates latency and speedup SVG charts similar to the ZML blog post. Set
`HF_TOKEN` if any benchmark target requires authentication.
