# Upstream IREE Tokenizer Bugs

This document tracks parity failures originally surfaced by the
`IREE.Tokenizers` Elixir parity harness (`bench/validate_parity.exs`) against
our vendored IREE tokenizer snapshot.

Update on this branch:
- Fixed locally in this package:
  - byte-level BPE decode corruption for emoji / CJK-adjacent sequences
  - SentencePiece / Llama-family `EncodeStream` chunk-boundary divergence
  - Unigram / SentencePiece `EncodeStream.finalize/1` no-progress failure
  - Phi-3 repeated punctuation and long-input BPE encode divergences
- Still unresolved here (documented below and/or in the parity report):
  - `gpt2` batch encode deadlock
  - BERT control-character whitespace classification mismatch
  - BERT batch-path emoji mismatch
  - Phi-3 `special_token_literal` added-token parity mismatch
  - T5 batch encode mismatches on long inputs
  - tokenizer.json `padding.max_length` is still not auto-applied

Some of the remaining issues likely need true upstream fixes. Others are known
HF-compatibility gaps that still need explicit local handling.

Pinned vendored commit: `71af3a5e41a8e265330bc693194c708cf6df4724`
(see `native/iree_tokenizers_native/vendor/IREE_COMMIT`).

Reference implementation for parity checks: [`elixir-nx/tokenizers`][hf]
(the Rust-backed HF `tokenizers` crate). Everywhere this document says
"HF", it means that reference.

[hf]: https://github.com/elixir-nx/tokenizers

## How to reproduce any of these

```bash
cd bench
mix deps.get
MODEL_FILTER="<label-from-below>" mix run validate_parity.exs
cat results/parity_report.md
```

---

## 1. Byte-level BPE decoder mangles multibyte UTF-8 for emoji / CJK-adjacent sequences [fixed locally]

**Affected models (all byte-level BPE):**

- `deepseek-ai/DeepSeek-R1` — cases `emoji`, `long_mixed`
- `Qwen/Qwen2.5-7B-Instruct` — cases `emoji`, `long_cjk`, `long_mixed`
- `openai-community/gpt2` — cases `unicode_cjk`, `emoji`, `long_cjk`, `long_mixed`

**Shape of the failure**

`IREE.Tokenizers.Tokenizer.encode/3` produces an identical token id list
to HF (`ids_equal = true`) — so the encoder is fine. But
`IREE.Tokenizers.Tokenizer.decode/3` yields a different byte string than
HF's decoder, and the IREE-decoded string is **not** a round-trip of the
original input, while HF's is.

Minimal reproducer:

```elixir
{:ok, t} = IREE.Tokenizers.Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
text = "🚀🌍 Let's go! 👩‍💻 👨‍👩‍👧‍👦 🇺🇸 🏳️‍🌈"
{:ok, enc} = IREE.Tokenizers.Tokenizer.encode(t, text, add_special_tokens: false)
{:ok, decoded} = IREE.Tokenizers.Tokenizer.decode(t, enc.ids)
decoded == text  # => false
```

**Where it lives:**
`native/iree_tokenizers_native/vendor/iree_tokenizer_src/iree/tokenizer/decoder/byte_level.c`

The byte-level BPE decoder maps GPT-2 style byte-to-unicode sentinels back
to real bytes, then reassembles UTF-8. The reassembly step drops or
corrupts sequences when a token boundary lands inside a multibyte
codepoint that is also part of a grapheme cluster (ZWJ family emoji, skin
tone modifiers, Han characters followed by ASCII punctuation).

**Priority:** 🔴 critical. Hits the three most-downloaded tokenizer
families on the Hub (DeepSeek, Qwen, GPT-2) on realistic prompts.

---

## 2. `encode_batch/3` deadlocks for `gpt2` on mixed-length input lists

**Affected models:** `openai-community/gpt2`.

**Shape of the failure**

Calling `IREE.Tokenizers.Tokenizer.encode_batch/3` on a list containing
both short and long inputs (the parity harness uses 18 inputs ranging from
1 byte to 184 KB) returns:

```
{:error,
 {:internal,
  "iree/tokenizer/tokenizer.c:2642: INTERNAL; encode deadlock: \
   no progress despite partial segment handling \
   (logical_capacity=8192 bytes, used=8192 bytes, \
    pending_input=3291 bytes, has_partial=0)"}}
```

Per-case `encode/3` on each of the same inputs individually succeeds, so
the deadlock is specific to the batch code path.

**Where it lives:**
`native/iree_tokenizers_native/vendor/iree_tokenizer_src/iree/tokenizer/tokenizer.c:2642`

The logical buffer fills up (`used == logical_capacity == 8192`) while
3291 bytes are still pending in the input stream. The partial-segment
code cannot grow the buffer and returns `INTERNAL`.

**Priority:** 🔴 critical. Any batch of long + short inputs will hit it.

---

## 3. `EncodeStream.finalize/1` fails for Unigram / SentencePiece models [fixed locally]

**Affected models:** `google-t5/t5-small` (both `:huggingface_json` and
`:sentencepiece_model` load paths). Any T5 / mT5 / FLAN-T5 derivative will
likely hit the same bug because the tokenizer is Unigram-based.

**Shape of the failure**

```elixir
{:error,
 {:invalid_argument,
  "encode stream finalize made no progress while pending \
   data remained; this is an upstream tokenizer runtime \
   limitation for some Unigram/SentencePiece models..."}}
```

This error is raised by `native/iree_tokenizers_native/src/stream.rs:127`
after the vendored C runtime returns `has_pending=true` but yields zero
tokens on finalize. The Rust layer cannot recover: it would have to
know which bytes were already committed (consumed vs. tokenized are
different in the C contract) in order to avoid double-encoding.

**Where it lives:** the Unigram / SentencePiece encode-state finalize path
in the vendored C runtime — likely under
`iree/tokenizer/encoder/` or `iree/tokenizer/model.c`. Exact file TBD.

**Priority:** 🔴 high. `EncodeStream` is unusable for every T5-family
tokenizer.

**Workaround (documented in the Rust error message):** fall back to
one-shot `IREE.Tokenizers.Tokenizer.encode/3` on the full input.

---

## 4. `EncodeStream` produces different tokens than `encode/3` on SentencePiece/Llama tokenizers [fixed locally]

**Affected models:** `microsoft/Phi-3-mini-4k-instruct` and, by
extension, other Llama / Mistral family tokenizers that use a
SentencePiece-style BPE with leading-space metaspace.

**Shape of the failure**

Feeding the same input through `EncodeStream.feed/2` in 16 KB chunks
produces a different token id list (and a different total count) than
calling `encode/3` on the concatenated input:

```
streamed  = 20_496 ids
oneshot   = 20_481 ids
first_diff = %{index: 2823, iree: 1775, hf: 13750}
```

The stream over-tokenizes: tokens that should merge across a chunk
boundary are emitted as shorter tokens instead. This is a chunk-seam
merge bug — the stream path does not hold back a trailing suffix long
enough to let the next chunk complete the merge.

**Where it lives:** the streaming feed path in the vendored C runtime,
likely the same module that owns the merge boundary decision for
BPE / Metaspace.

**Priority:** 🟡 medium. Stream *errors* are loud; this one is silent
and will produce subtly wrong logits unless the caller also verifies
against a one-shot encode.

---

## 5. Llama-SPM tokenizers over-tokenize on repeated punctuation and long mixed inputs [mostly fixed locally]

**Affected model:** `microsoft/Phi-3-mini-4k-instruct`.

The long-input and repeated-punctuation cases are fixed locally on this branch.
The remaining unresolved parity gap in this model family is the
`special_token_literal` case, which appears to be tied to added-token metadata
semantics rather than the long-input BPE path documented below.

**Shape of the failure**

Multiple cases diverge from HF at the encoder level:

| Case | iree ids | hf ids | First diff |
|---|---:|---:|---|
| `special_token_literal` (66 B) | 27 | 26 | index 3, iree `322`, hf `29871` |
| `repeated_punct` (23 B) | 11 | 11 | index 2, iree `13626`, hf `1577` |
| `long_repeat_ascii` (184 KB) | 49153 | 49153 | index 5, iree `12500`, hf `432` |
| `long_mixed` (108 KB) | **45056** | **43008** | index 16, iree `5777`, hf `274` |

The `long_mixed` case over-produces by ~5 %. `repeated_punct` has matching
length but completely different tokens from index 2 onward, which implies
the BPE merge priorities for `!!!`, `???`, `...`, `,,,`, `;;;`, `:::` are
being resolved differently.

**Where it lives:** vendored SentencePiece / BPE encoder in
`iree/tokenizer/model.c` or a related file.

**Priority:** 🟡 medium. Phi-3 is popular and the divergence is stable
(reproduces on every run) which implies a table / ordering issue rather
than a race.

---

## 6. BERT BasicTokenizer does not treat `\v`, `\f`, `\b` as whitespace

**Affected model:** `google-bert/bert-base-uncased` (and likely every
WordPiece tokenizer that uses BasicTokenizer's normalizer).

**Shape of the failure**

Input `"bell\x07tab\ttab vertical\vform\ftab back\bspace"` produces:

- IREE: 11 / 9 tokens (`add_special_tokens` true / false)
- HF:   12 / 10 tokens
- First diff at index 6, iree `2433`, hf `14192`

HF's BasicTokenizer treats `\v`, `\f`, and `\b` as whitespace-equivalent
and splits on them; IREE keeps them attached to the preceding word.

**Where it lives:**
`native/iree_tokenizers_native/vendor/iree_tokenizer_src/iree/tokenizer/normalizer.c`
(BERT normalizer / whitespace classification).

**Priority:** 🟢 low. Only affects literal control characters in input,
which is rare in practice.

---

## 7. BERT `encode_batch/3` disagrees with single `encode/3` on the `emoji` case

**Affected model:** `google-bert/bert-base-uncased`.

**Shape of the failure**

When the 18-input batch is fed through `encode_batch/3`, the emoji entry
(index 10 in the batch) disagrees with HF's batch result even though the
**same** input passed through `encode/3` single matches HF. So it is a
batch-path-only divergence.

**Where it lives:** the batch encode pre-tokenization path in the
vendored C runtime.

**Priority:** 🟢 low.

---

## 8. Tokenizer-level `padding` config in `tokenizer.json` is ignored

**Affected model:** `sentence-transformers/all-MiniLM-L6-v2` (and every
other model that sets `padding.max_length` in its `tokenizer.json`).

**Shape of the failure**

`tokenizer.json` sets `padding: max_length=128`. HF's `tokenizers`
auto-applies this; `encode/3` returns 128 ids for every input. IREE
returns the raw unpadded id list, so the parity harness reports `0/19`
matching cases even though the prefix of the IREE output is byte-for-byte
correct.

This is arguably a documented design choice — the README steers users
toward `IREE.Tokenizers.Encoding.Transformation` for padding. But it is a
real parity gap with HF and should at minimum be called out in the
README so that users who migrate from HF are not surprised.

**Where it lives:** the loader / post-processor setup; `tokenizer.rs`
and / or the vendored config parser.

**Priority:** 🟢 low. Workaround is `Encoding.Transformation.pad/2`.

---

## Reporting upstream

When filing these upstream, include:
- The IREE commit this package is pinned to (above).
- The exact error message or divergence, copied from
  `bench/results/parity_report.md`.
- A link to the parity harness section of the bench README so reviewers
  can reproduce independently.
- The `elixir-nx/tokenizers` version used as ground truth
  (see `bench/mix.lock`).
