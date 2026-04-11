defmodule IREE.Tokenizers.EncodeStream do
  @moduledoc """
  Streaming encoder state.

  Use this when you want to feed a tokenizer incrementally from multiple
  binary chunks while preserving the same output you would get from one-shot
  encoding of the full input.
  """

  defstruct [:resource]

  @typedoc """
  Mutable streaming encode state owned by the NIF.
  """
  @type t :: %__MODULE__{resource: reference()}

  @doc """
  Creates a new encode stream for the given tokenizer.

  Options:

  - `:add_special_tokens` - whether post-processing special tokens should be
    emitted during finalization, defaults to `true`
  - `:max_chunk_bytes` - maximum chunk size expected by `feed/2`, defaults to
    `65536`
  """
  @spec new(IREE.Tokenizers.Tokenizer.t(), keyword()) :: {:ok, t()} | {:error, {atom(), binary()}}
  def new(tokenizer, opts \\ []) do
    opts = Keyword.validate!(opts, add_special_tokens: true, max_chunk_bytes: 65_536)
    IREE.Tokenizers.Native.encode_stream_new(tokenizer, opts)
  end

  @doc """
  Feeds a binary chunk into the stream and returns any newly produced token IDs.
  """
  @spec feed(t(), binary()) :: {:ok, [integer()]} | {:error, {atom(), binary()}}
  def feed(%__MODULE__{} = stream, chunk) when is_binary(chunk) do
    IREE.Tokenizers.Native.encode_stream_feed(stream, chunk)
  end

  def feed(%__MODULE__{}, _chunk),
    do: {:error, {:invalid_argument, "expected a binary chunk"}}

  @doc """
  Flushes any remaining state and returns the final token IDs.
  """
  @spec finalize(t()) :: {:ok, [integer()]} | {:error, {atom(), binary()}}
  def finalize(%__MODULE__{} = stream) do
    IREE.Tokenizers.Native.encode_stream_finalize(stream)
  end
end
