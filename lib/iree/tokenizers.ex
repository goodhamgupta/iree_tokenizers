defmodule IREE.Tokenizers do
  @moduledoc """
  Fast Hugging Face `tokenizer.json`, OpenAI `.tiktoken`, and SentencePiece
  `.model` bindings for Elixir backed by the IREE tokenizer runtime.

  The main entrypoint is `IREE.Tokenizers.Tokenizer`.

  Supported load formats:

  - Hugging Face `tokenizer.json`
  - OpenAI `.tiktoken`
  - SentencePiece `.model`

  Supported runtime capabilities:

  - one-shot encode/decode
  - batched encode/decode
  - streaming encode/decode
  - token offsets and type IDs
  - vocabulary lookup helpers

  The library is intentionally inference-focused. Pair-sequence encoding,
  tokenizer training, and full mutation parity with `elixir-nx/tokenizers`
  are not yet complete.
  """
end
