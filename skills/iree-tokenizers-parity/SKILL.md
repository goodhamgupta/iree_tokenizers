---
name: iree-tokenizers-parity
description: Use when debugging, fixing, or verifying tokenizer correctness in this repo against elixir-nx/tokenizers, including one-shot, batch, stream, decode, padding/truncation, and new Hugging Face model checks.
version: 1.0.0
author: Hermes Agent
license: Apache-2.0
metadata:
  hermes:
    tags: [elixir, rustler, tokenizer, parity, huggingface, iree]
    related_skills: [iree-tokenizers-benchmarking, iree-tokenizers-vendor-refresh]
---

# IREE.Tokenizers parity workflow

## Overview

This repo is parity-first. A change is not complete just because it compiles or
is faster: token IDs, decoded text, batch behavior, stream behavior, masks,
offsets, and tokenizer JSON defaults must keep matching the reference package
where this project claims support.

The reference implementation for parity is `elixir-nx/tokenizers`, which wraps
the Hugging Face Rust `tokenizers` crate.

## When to use

Use this skill when:

- a tokenizer produces different IDs or decoded text from `elixir-nx/tokenizers`
- batch or stream output differs from one-shot output
- padding/truncation defaults from `tokenizer.json` look wrong
- a new Hugging Face model needs tokenizer-only verification
- a README/doc claim says a model family is green and you need to verify it
- a native/vendor change touched tokenizer behavior

Do not use this for pure README wording edits unless the wording makes a parity
claim; for performance claims, use `iree-tokenizers-benchmarking` too.

## Baseline commands

Start from repo root:

```bash
jj status
mix deps.get
mix test
cargo test --manifest-path native/iree_tokenizers_native/Cargo.toml
```

Run the full selected parity matrix:

```bash
cd bench
mix deps.get
mix run validate_parity.exs
```

Run one model while iterating:

```bash
cd bench
MODEL_FILTER="Qwen/Qwen2.5-7B-Instruct" mix run validate_parity.exs
```

The selected report is written to:

```text
bench/results/parity_report.md
```

## Optional pretrained integration suites

These tests are skipped unless their environment variables are set:

```bash
RUN_PRETRAINED_BATCH_INTEGRATION=1 mix test test/iree_tokenizers/batch_integration_test.exs
RUN_PRETRAINED_STREAM_INTEGRATION=1 mix test test/iree_tokenizers/stream_integration_test.exs
RUN_SENTENCEPIECE_INTEGRATION=1 mix test test/iree_tokenizers/sentencepiece_integration_test.exs
```

Use them when a change touches pretrained loading, batch, stream, SentencePiece,
BPE merge behavior, decode behavior, or tokenizer JSON config.

## Where to add regressions

- Synthetic tokenizer JSON fixtures: `test/fixtures/`
- Small compatibility regressions: `test/iree_tokenizers/compatibility_test.exs`
- Constructor/config/default behavior: `test/iree_tokenizers/tokenizer_test.exs`
- Batch regressions: `test/iree_tokenizers/batch_integration_test.exs`
- Stream regressions: `test/iree_tokenizers/stream_integration_test.exs`
- SentencePiece `.model` and Llama/Phi/T5 parity:
  `test/iree_tokenizers/sentencepiece_integration_test.exs`
- Historical bug narrative: `docs/UPSTREAM_BUGS.md`
- Matrix evidence: `bench/results/parity_report.md`

Prefer the smallest failing fixture/test first, then run broader suites.

## How to interpret failures

1. `ids_equal=false`
   - Encoder/model/pre-tokenizer/post-processor behavior is wrong.
   - Check tokenizer JSON parsing, special/added-token matching, BPE merge logic,
     Unigram/SentencePiece conversion, BERT normalization/segmentation, and
     tokenizer JSON default transformations.

2. `ids_equal=true` but decoded output differs
   - Decode behavior is wrong. Check byte-level decode, SentencePiece-style BPE
     decode strategies, special-token skipping, and UTF-8 boundary handling.

3. One-shot matches but batch differs
   - Keep one-shot semantics authoritative. `encode_batch/3` intentionally routes
     through `encode/3` for parity; do not reintroduce a native batch fast path
     without proving it is parity-clean on mixed short/long batches.

4. One-shot matches but stream differs
   - Check whether the tokenizer family needs buffered finalize. Current Rust
     heuristics buffer Unigram, SentencePiece-style BPE, null-pretokenizer BPE,
     and Metaspace+ByteFallback BPE cases that can diverge at chunk seams.
   - Also check the Elixir stream wrapper when tokenizer JSON padding/truncation
     defaults are involved.

5. Padding/truncation differs
   - Check `lib/iree/tokenizers/tokenizer.ex` default config parsing.
   - Preserve `offsets: nil` when offsets were not requested.
   - Preserve special-token boundaries when truncating with `add_special_tokens: true`.

## Known parity-sensitive implementation surfaces

Elixir:

- `lib/iree/tokenizers/tokenizer.ex`
  - `from_pretrained/2` path fallbacks and cache behavior
  - `encode/3` default truncation and transformations
  - `encode_batch/3` parity-preserving per-input path
  - tokenizer JSON padding/truncation parsing
- `lib/iree/tokenizers/encode_stream.ex`
  - config-driven buffered fallback
  - finalized stream error behavior
- `lib/iree/tokenizers/encoding.ex`
  - pad/truncate/transform semantics

Rust:

- `native/iree_tokenizers_native/src/tokenizer.rs`
  - buffer sizing and bounded retries
  - metadata inference for stream/decode strategies
  - SentencePiece BPE decode handling
  - token metadata, UNK source text, offsets
- `native/iree_tokenizers_native/src/stream.rs`
  - native vs buffered stream state
  - stream feed buffer sizing
  - finalize behavior and post-finalize errors
- `native/iree_tokenizers_native/src/sentencepiece.rs`
  - `.model` to tokenizer JSON conversion

Vendored C patches:

- byte-level decoder UTF-8 behavior
- added/special-token matching and state reset after emission
- BPE long-input and repeated-punctuation behavior
- BERT whitespace/control-character classification
- Metaspace / segmenter finalize behavior

## Tokenizer-only verification for a new HF model

Do not download model weights when only tokenizer parity is needed. Download only
tokenizer metadata:

```bash
hf download ORG/MODEL \
  --include 'tokenizer*' \
  --include 'special_tokens_map.json' \
  --include 'config.json' \
  --exclude '*.safetensors' \
  --exclude '*.bin' \
  --exclude '*.pt' \
  --exclude '*.gguf' \
  --local-dir /tmp/model-tokenizer-only
```

Then load the local tokenizer file through both implementations and compare:

- ordinary ASCII
- whitespace-heavy text
- Unicode Latin
- CJK
- emoji and ZWJ emoji
- code snippets
- JSON and markdown
- URLs
- repeated punctuation
- control characters
- long repeated ASCII/CJK/mixed inputs
- literal added/special token strings from the tokenizer JSON

Check both `add_special_tokens: true` and `false`; check IDs, token strings,
type IDs, decode parity, batch IDs, and stream-vs-one-shot parity.

## Report template

Use a compact, evidence-backed summary:

```text
Checked MODEL with tokenizer-only assets.
Reference: elixir-nx/tokenizers.
Inputs: N cases x add_special_tokens true/false.
Results: one-shot ids/decode/token/type parity PASS/FAIL, batch PASS/FAIL,
stream-vs-one-shot PASS/FAIL.
Artifacts updated: bench/results/parity_report.md (if applicable).
Commands run: ...
Known caveats: ...
```

## Common pitfalls

1. Treating `docs/UPSTREAM_BUGS.md` as live status. It preserves historical
   failures; verify current code and `bench/results/parity_report.md`.
2. Fixing only the Elixir wrapper when the bug is in Rust strategy selection or
   vendored C behavior.
3. Reintroducing a native encode batch path for speed without proving mixed
   short/long parity.
4. Forgetting `add_special_tokens: true` and `false`; many bugs only appear in
   one mode.
5. Checking IDs but not decode; byte-level decode bugs can leave IDs identical.
6. Checking one-shot but not stream; chunk-boundary bugs can silently change
   tokenization.
7. Downloading model weights for tokenizer-only work. Use `hf download` include
   and exclude patterns.

## Verification checklist

- [ ] `jj status` reviewed before editing
- [ ] Small failing regression added or identified
- [ ] Targeted test passes
- [ ] `mix test` passes for local code changes
- [ ] `cargo test --manifest-path native/iree_tokenizers_native/Cargo.toml`
      passes for native changes
- [ ] Relevant pretrained integration suite passes when behavior touches a
      pretrained path
- [ ] `bench/results/parity_report.md` updated when matrix evidence changes
- [ ] README/docs updated only with evidence-backed claims
