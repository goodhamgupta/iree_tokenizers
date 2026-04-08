# Benchmarks

This directory contains a small benchmark harness that compares `IREE.Tokenizers`
against the published `tokenizers` package.

## Run

```bash
cd bench
mix deps.get
mix run compare.exs
```

The benchmark uses the local minimal BPE fixture and exercises short, medium,
and long encode/decode workloads.
