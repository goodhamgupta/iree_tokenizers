# IREE.Tokenizers

Fast Hugging Face `tokenizer.json` bindings for Elixir backed by the IREE tokenizer runtime.

## Features

- Load tokenizer definitions from a local `tokenizer.json` buffer or file
- Download and cache `tokenizer.json` files from the Hugging Face Hub
- One-shot encode/decode and batched encode/decode
- Token offsets and type IDs
- Vocab lookup helpers
- Streaming encode/decode

## Scope

V1 is intentionally inference-only.

- Supported:
  - Hugging Face `tokenizer.json`
  - BPE
  - WordPiece
  - Unigram
- Deferred:
  - `.tiktoken`
  - SentencePiece `.model`
  - pair-sequence encode input
  - training and tokenizer mutation APIs

## Local Development

```bash
mix deps.get
IREE_TOKENIZERS_BUILD=1 mix test
```

The local development flow forces a source build of the Rust NIF.

## Example

```elixir
{:ok, tokenizer} = IREE.Tokenizers.Tokenizer.from_file("tokenizer.json")

{:ok, encoding} =
  IREE.Tokenizers.Tokenizer.encode(tokenizer, "Hello world", add_special_tokens: false)

encoding.ids

{:ok, text} =
  IREE.Tokenizers.Tokenizer.decode(tokenizer, encoding.ids, skip_special_tokens: false)
```

## Vendored IREE Bundle

The native crate builds against a curated vendored source bundle under
`native/iree_tokenizers_native/vendor/iree_tokenizer_src`.

To refresh that bundle from a pinned upstream IREE checkout:

```bash
scripts/update_iree_bundle.sh /path/to/iree
```
