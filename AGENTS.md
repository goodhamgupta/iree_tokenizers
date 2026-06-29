# AGENTS.md

Scope: this file applies to the whole repository.

## Project in one paragraph

`IREE.Tokenizers` is an Elixir package that exposes fast inference-time
LLM tokenization APIs backed by a Rustler NIF and the vendored IREE tokenizer C
runtime. It loads Hugging Face `tokenizer.json`, OpenAI `.tiktoken`, and
SentencePiece `.model` assets from files, buffers, or the Hugging Face Hub. The
project is parity-first: performance claims and implementation changes must not
regress token IDs, decoded text, batch behavior, or streaming behavior against
`elixir-nx/tokenizers` unless the limitation is explicitly documented.

## Before you start

1. Check the working copy with `jj status` first. This repo uses jj; prefer `jj`
   for status/diff/history. Do not push to a remote unless the user explicitly
   asks.
2. Read any relevant repo-local skills under `skills/*/SKILL.md` before touching
   matching areas:
   - `skills/iree-tokenizers-parity/SKILL.md`
   - `skills/iree-tokenizers-benchmarking/SKILL.md`
   - `skills/iree-tokenizers-vendor-refresh/SKILL.md`
3. Treat checked-in code, tests, and current `bench/results/*.md` artifacts as
   authoritative. Historical prose in `docs/UPSTREAM_BUGS.md` preserves old bug
   shapes and can lag the live branch state.
4. Do not edit generated build output such as `_build/`, `bench/_build/`,
   `native/iree_tokenizers_native/target/`, or `priv/native/*.so` unless the
   task is explicitly about release artifacts.
5. For PRs that change package behavior, update the `@version` in `mix.exs`
   using SemVer: patch for bug fixes/parity fixes, minor for backward-compatible
   features, and major for breaking API or behavior changes.

## Repository map

- `lib/iree/tokenizers.ex` - package-level module documentation.
- `lib/iree/tokenizers/tokenizer.ex` - main public API, loading, HF downloads,
  caching, encode/decode, batch encode, tokenizer JSON defaults.
- `lib/iree/tokenizers/encoding.ex` - encoding struct and pad/truncate helpers.
- `lib/iree/tokenizers/encode_stream.ex` and `decode_stream.ex` - BEAM-facing
  stream wrappers.
- `lib/iree/tokenizers/native.ex` - RustlerPrecompiled NIF declaration.
- `native/iree_tokenizers_native/src/` - Rust NIF bridge and tokenizer resource
  implementation.
- `native/iree_tokenizers_native/vendor/iree_tokenizer_src/` - curated vendored
  IREE C runtime bundle.
- `test/iree_tokenizers/` - unit, compatibility, and gated pretrained tests.
- `bench/` - reproducible parity and benchmark harnesses.
- `bench/results/` - checked-in generated reports/charts used by README claims.
- `docs/UPSTREAM_BUGS.md` - historical failure modes and local fix notes.
- `scripts/update_iree_bundle.sh` - refreshes the vendored IREE source bundle.

## Development environment

`.tool-versions` currently pins:

- Erlang 28.0
- Elixir 1.19.5

The native crate uses Rust 2021 and Rustler. In `:dev` and `:test`, the package
forces a local source build of the NIF. Release builds use `rustler_precompiled`
when a matching target artifact exists.

## Normal validation commands

Run the smallest relevant check first, then broaden.

General local checks from repo root:

```bash
mix deps.get
mix test
cargo test --manifest-path native/iree_tokenizers_native/Cargo.toml
```

Formatting:

```bash
mix format
cargo fmt --manifest-path native/iree_tokenizers_native/Cargo.toml
```

Targeted Elixir test:

```bash
mix test path/to/test.exs:LINE
```

Optional pretrained suites:

```bash
RUN_PRETRAINED_BATCH_INTEGRATION=1 mix test test/iree_tokenizers/batch_integration_test.exs
RUN_PRETRAINED_STREAM_INTEGRATION=1 mix test test/iree_tokenizers/stream_integration_test.exs
RUN_SENTENCEPIECE_INTEGRATION=1 mix test test/iree_tokenizers/sentencepiece_integration_test.exs
```

Full selected parity matrix:

```bash
cd bench
mix deps.get
mix run validate_parity.exs
```

Targeted parity iteration:

```bash
cd bench
MODEL_FILTER="Qwen/Qwen2.5-7B-Instruct" mix run validate_parity.exs
```

## Correctness rules

- Prefer tests that compare against `elixir-nx/tokenizers` for any tokenizer
  behavior change involving IDs, token strings, decoded text, masks, offsets,
  padding/truncation, batch, or stream output.
- Do not assume an upstream/runtime bug is still live. Verify against current
  code and `bench/results/parity_report.md`.
- If `ids_equal=true` but decoded text differs, treat it as a decoder/parity bug,
  not a harmless formatting issue.
- If one-shot encode matches but batch or stream differs, preserve one-shot
  semantics first. Existing code intentionally routes `encode_batch/3` through
  `encode/3` for parity.
- For `tokenizer.json` defaults, preserve automatic padding/truncation behavior
  and avoid rebuilding `offsets` as an empty list when offsets were not requested.
- For stream fixes, verify both stream-vs-one-shot parity and post-finalize error
  behavior (`stream already finalized`).

## Benchmark/documentation rules

- Only claim performance improvements that are backed by checked-in benchmark
  artifacts or a fresh benchmark run in the task.
- Benchmark rows should be reported only when the compared libraries produce
  equivalent outputs for the workload. Do not benchmark divergent outputs as if
  they are comparable latency rows.
- When updating README performance summaries, cross-check:
  - `bench/results/model_matrix.md`
  - `bench/results/tokenizers_compare.md`
  - `bench/results/sentencepiece_compare.md`
  - `bench/results/parity_report.md`
- Be explicit that benchmark numbers are machine-dependent.

## Native/vendor rules

- The vendored IREE commit is recorded in
  `native/iree_tokenizers_native/vendor/IREE_COMMIT`.
- Use `scripts/update_iree_bundle.sh /path/to/iree` for vendor refreshes; do not
  hand-copy random subsets of C files.
- After any vendor refresh, inspect known local patch surfaces before trusting
  tests:
  - byte-level decoder UTF-8 handling
  - Hugging Face added/special token matching
  - BPE long-input and repeated-punctuation paths
  - BERT normalizer/segmenter whitespace classification
  - stream finalize/chunk-boundary behavior
- Run Rust tests, Elixir tests, and relevant pretrained parity suites after a
  vendor refresh.

## Style preferences

- Keep the Elixir API thin and direct. Avoid compatibility wrapper layers whose
  only purpose is to preserve old monkeypatch paths.
- Prefer small, focused regression fixtures in `test/fixtures/` and targeted
  tests in `test/iree_tokenizers/`.
- Keep README user-facing: what the package does, how to use it, results, and
  implementation. Keep deep bug archaeology in `docs/UPSTREAM_BUGS.md` or a
  skill.
- Use exact paths and commands in docs so future agents and users can reproduce
  claims.
