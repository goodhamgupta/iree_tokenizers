---
name: iree-tokenizers-benchmarking
description: Use when running IREE.Tokenizers benchmarks, refreshing bench/results artifacts, or updating README performance claims while preserving parity-first reporting.
version: 1.0.0
author: Hermes Agent
license: Apache-2.0
metadata:
  hermes:
    tags: [benchmarking, performance, elixir, tokenizers, parity]
    related_skills: [iree-tokenizers-parity]
---

# IREE.Tokenizers benchmarking workflow

## Overview

Benchmark claims in this repo must be evidence-backed. The benchmark harness is
under `bench/` and writes markdown/SVG artifacts to `bench/results/`. README
performance summaries should be derived from those artifacts or from a fresh run
performed during the task.

Correctness comes first: never report latency for outputs that are not equivalent
between `IREE.Tokenizers` and the reference package.

## When to use

Use this skill when:

- the user asks how fast the package is
- updating README benchmark tables, charts, or speedup language
- refreshing `bench/results/*.md` or `bench/results/*.svg`
- adding or removing model rows from the benchmark matrix
- comparing `IREE.Tokenizers` with `elixir-nx/tokenizers`
- investigating whether an optimization materially improves performance

Use `iree-tokenizers-parity` first if the task touches correctness, new model
coverage, batch behavior, or stream behavior.

## Benchmark scripts

Set up the benchmark app:

```bash
cd bench
mix deps.get
```

Generic local BPE fixture comparison:

```bash
mix run compare.exs
```

Outputs:

```text
bench/results/tokenizers_compare.md
bench/results/tokenizers_compare_encode.svg
bench/results/tokenizers_compare_decode.svg
```

SentencePiece `.model` comparison:

```bash
mix run sentencepiece_compare.exs
```

Outputs:

```text
bench/results/sentencepiece_compare.md
bench/results/sentencepiece_compare_encode.svg
bench/results/sentencepiece_compare_decode.svg
```

Curated model latency/speedup matrix:

```bash
mix run model_matrix_graphs.exs
```

Outputs:

```text
bench/results/model_matrix.md
bench/results/model_matrix_latency.svg
bench/results/model_matrix_speedup.svg
```

Limit model-matrix iteration:

```bash
MODEL_FILTER="Qwen/Qwen3.5-9B" mix run model_matrix_graphs.exs
MODEL_FILTER="zai-org/GLM-5.1" mix run model_matrix_graphs.exs
```

If a target is gated or rate-limited:

```bash
HF_TOKEN=*** mix run model_matrix_graphs.exs
```

## Parity gates before reporting speed

Before quoting speedups, confirm the benchmark script itself validated equivalent
outputs for the row/workload. Existing scripts follow these rules:

- `compare.exs` reports only shared token sequences and decoded strings.
- `sentencepiece_compare.exs` validates representative inputs before recording
  direct `.model` latency numbers.
- `model_matrix_graphs.exs` skips rows whose benchmark corpus does not produce
  equivalent outputs, excludes embedding models, and reports stream numbers only
  when stream output matches IREE one-shot output on that corpus.

If a script is changed, preserve those gates.

## How to update README performance summaries

1. Run the relevant script or read the checked-in artifact if the task is docs
   only.
2. Use exact numbers from the generated markdown tables.
3. State that benchmark numbers are machine-dependent.
4. Separate correctness claims from speed claims.
5. Link to the artifact path, not only to chart images.
6. Do not generalize from one workload to all tokenizers.

Useful artifact paths:

```text
bench/results/model_matrix.md
bench/results/tokenizers_compare.md
bench/results/sentencepiece_compare.md
bench/results/parity_report.md
```

## Current checked-in result shape

At the time this skill was written, the checked-in artifacts showed:

- model matrix: IREE one-shot speedups of 1.6x-5.6x and stream speedups of
  5.4x-14.0x on published rows
- local BPE fixture: medium/long encode about 1.3x faster and medium/long decode
  about 10x faster
- SentencePiece `.model`: T5-small encode 1.97x faster, LLaMA encode 1.18x
  faster, LLaMA decode 1.81x faster

Always re-read the artifacts before repeating these numbers; they may change.

## Adding a model to the matrix

1. Identify whether the model is a tokenizer family representative or a duplicate
   of an existing tokenizer family.
2. Avoid embedding-only models in the published latency matrix unless the task is
   explicitly about embeddings.
3. Ensure the repo has a usable tokenizer asset path.
4. Run the model with `MODEL_FILTER` first.
5. Confirm output equivalence before accepting the row.
6. Update markdown and SVG artifacts together.
7. Update README only after artifacts exist.

## Evaluating optimization claims

For a performance change:

1. Capture a baseline with the exact command and artifact/terminal output.
2. Make the change.
3. Re-run the same command under the same model filter/workload.
4. Report before/after numbers and whether the delta is material.
5. Run parity checks if the optimization changes encode/decode semantics, buffer
   sizing, stream behavior, or native runtime calls.

Avoid phrases like "much faster" unless the benchmark result supports them.

## Common pitfalls

1. Benchmarking divergent outputs. That measures apples vs oranges.
2. Updating README numbers without updating or referencing `bench/results`.
3. Using full model downloads when tokenizer assets are enough.
4. Forgetting `HF_TOKEN` for gated benchmark targets.
5. Reporting stream speedups for a model where stream output was not verified
   against one-shot output.
6. Treating one local run as universal. Mention machine dependence.
7. Leaving stale chart images after markdown numbers change.

## Verification checklist

- [ ] Relevant benchmark script ran or checked-in artifact was re-read
- [ ] Output equivalence/parity gate confirmed for reported rows
- [ ] Markdown and SVG artifacts are consistent when regenerated
- [ ] README performance claims match artifact numbers
- [ ] Machine dependence and scope are stated
- [ ] `bench/results/parity_report.md` checked for correctness claims
