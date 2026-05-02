---
name: iree-tokenizers-vendor-refresh
description: Use when refreshing the vendored IREE tokenizer runtime, updating native Rust dependencies, or touching FFI surfaces that can overwrite or invalidate local parity fixes.
version: 1.0.0
author: Hermes Agent
license: Apache-2.0
metadata:
  hermes:
    tags: [iree, vendor, rust, c, ffi, parity]
    related_skills: [iree-tokenizers-parity]
---

# IREE tokenizer vendor refresh workflow

## Overview

The native crate builds against a curated vendored subset of IREE's tokenizer C
runtime under `native/iree_tokenizers_native/vendor/iree_tokenizer_src`. Refreshes
can silently overwrite local C patches that are required for Hugging Face parity.
Treat vendor updates as correctness-sensitive native changes, not routine file
copies.

## When to use

Use this skill when:

- updating `native/iree_tokenizers_native/vendor/IREE_COMMIT`
- running `scripts/update_iree_bundle.sh`
- changing files under `native/iree_tokenizers_native/vendor/iree_tokenizer_src`
- updating Rust FFI mirrors in `native/iree_tokenizers_native/src/ffi.rs`
- updating Rust dependencies that affect native tests or tokenizer loading
- investigating a parity regression after a vendor/dependency update

## Source of truth

Pinned commit file:

```text
native/iree_tokenizers_native/vendor/IREE_COMMIT
```

Refresh script:

```text
scripts/update_iree_bundle.sh
```

The script contains `EXPECTED_IREE_COMMIT`. Update that value intentionally
before running a refresh against a different upstream checkout.

## Refresh procedure

1. Start clean or understand existing changes:

```bash
jj status
```

2. Inspect the current pin:

```bash
cat native/iree_tokenizers_native/vendor/IREE_COMMIT
```

3. Prepare an upstream IREE checkout at the exact commit you want.

4. Update `EXPECTED_IREE_COMMIT` in `scripts/update_iree_bundle.sh` if the pin is
   changing.

5. Run the refresh from repo root:

```bash
scripts/update_iree_bundle.sh /path/to/iree-checkout
```

6. Inspect changed source lists:

```bash
cat native/iree_tokenizers_native/sources/base_sources.txt
cat native/iree_tokenizers_native/sources/tokenizer_sources.txt
```

7. Inspect changed C headers and FFI-relevant structs before compiling.

## Known local patch surfaces to re-check

After any vendor refresh, inspect these areas because previous fixes have lived
there:

- `iree/tokenizer/decoder/byte_level.c`
  - byte-level UTF-8 decode fidelity for emoji/CJK-adjacent token boundaries
- `iree/tokenizer/format/huggingface/tokenizer_json.c`
  - added-token matching should include all added tokens where appropriate and
    route by normalization behavior, not only by `special=true`
- `iree/tokenizer/special_tokens.c`
  - `rstrip` matching must consume trailing whitespace when parity requires it
- `iree/tokenizer/tokenizer.c`
  - state reset after special/added token emission
  - long-input segment finalize behavior
  - output capacity / pending-state behavior
- `iree/tokenizer/model/bpe_encode.c`
  - long-input BPE seeding and merge-produced token handling
- `iree/tokenizer/model/bpe_backtrack.c`
  - repeated punctuation / suffix blocking behavior
- `iree/tokenizer/normalizer/bert.c`
  - BERT whitespace/control-character classification
- `iree/tokenizer/segmenter/bert.c`
  - must stay aligned with BERT normalizer classification
- `iree/tokenizer/format/huggingface/segmenter_json.c`
  - Metaspace split/whitespace flags for T5/SentencePiece-style tokenizers

Do not assume upstream now carries the local fix unless you verify behavior.

## Rust FFI checks

When upstream C structs change, update Rust mirrors exactly in:

```text
native/iree_tokenizers_native/src/ffi.rs
```

Then update all initializers in:

```text
native/iree_tokenizers_native/src/tokenizer.rs
native/iree_tokenizers_native/src/stream.rs
native/iree_tokenizers_native/src/sentencepiece.rs
```

Search for direct struct initializers after extending resources or FFI structs.
For example, if `TokenizerResource` gains a field, update every
`TokenizerResource { ... }`, including test-only initializers.

## Dependency update notes

When updating native Rust dependencies:

```bash
cargo update --manifest-path native/iree_tokenizers_native/Cargo.toml --dry-run --verbose
```

Then run:

```bash
cargo fmt --manifest-path native/iree_tokenizers_native/Cargo.toml
cargo test --manifest-path native/iree_tokenizers_native/Cargo.toml
mix test
```

If updating `ureq`, check test helper APIs carefully. Previous major versions
changed response body access patterns.

## Validation ladder after refresh

Run in this order:

```bash
cargo fmt --manifest-path native/iree_tokenizers_native/Cargo.toml
cargo test --manifest-path native/iree_tokenizers_native/Cargo.toml
mix format
mix test
```

Then run behavior-specific pretrained suites:

```bash
RUN_PRETRAINED_BATCH_INTEGRATION=1 mix test test/iree_tokenizers/batch_integration_test.exs
RUN_PRETRAINED_STREAM_INTEGRATION=1 mix test test/iree_tokenizers/stream_integration_test.exs
RUN_SENTENCEPIECE_INTEGRATION=1 mix test test/iree_tokenizers/sentencepiece_integration_test.exs
```

Finally run the selected parity matrix:

```bash
cd bench
MODEL_FILTER='Qwen/Qwen2.5-7B-Instruct,openai-community/gpt2,microsoft/Phi-3-mini-4k-instruct,google-t5/t5-small (json),google-t5/t5-small (spiece),google-bert/bert-base-uncased,sentence-transformers/all-MiniLM-L6-v2' mix run validate_parity.exs
```

## Common failure signatures

- `RESOURCE_EXHAUSTED` or process killed on long T5/SentencePiece inputs:
  inspect Metaspace segmenter/finalize behavior and Rust bounded capacity retry
  logic before increasing buffers blindly.
- IDs match but decode differs on Qwen/GPT-2/emoji/CJK:
  inspect byte-level decoder behavior.
- Stream differs from one-shot on Phi/Llama/Gemma-style tokenizers:
  inspect stream strategy inference and buffered finalize behavior.
- BERT control-character cases differ:
  inspect normalizer and segmenter whitespace classification together.
- Compile errors about missing fields in Rust initializers:
  search all direct resource/FFI struct initializers, including test-only code.

## Common pitfalls

1. Hand-copying only a few upstream files instead of using the refresh script.
2. Updating `IREE_COMMIT` without updating `EXPECTED_IREE_COMMIT`.
3. Trusting `mix test` alone. Native tests and pretrained parity suites catch
   different failure classes.
4. Forgetting that a vendor refresh can remove local C patches.
5. Updating FFI field order incorrectly. Rust mirrors must match C exactly.
6. Running benchmarks before restoring parity.

## Verification checklist

- [ ] `jj status` reviewed before refresh
- [ ] `EXPECTED_IREE_COMMIT` and `vendor/IREE_COMMIT` match the intended commit
- [ ] `sources/base_sources.txt` and `sources/tokenizer_sources.txt` inspected
- [ ] Known local patch surfaces reviewed
- [ ] Rust FFI mirrors and initializers checked
- [ ] `cargo fmt` and `cargo test` passed
- [ ] `mix format` and `mix test` passed
- [ ] Relevant pretrained suites passed
- [ ] Selected parity matrix passed or failures documented with root-cause notes
