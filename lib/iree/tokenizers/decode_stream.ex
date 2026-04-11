defmodule IREE.Tokenizers.DecodeStream do
  @moduledoc """
  Streaming decoder state.

  Use this when token IDs arrive incrementally and you want to decode them
  into text while preserving the same result as one-shot decode.
  """

  defstruct [:resource]

  @typedoc """
  Mutable streaming decode state owned by the NIF.
  """
  @type t :: %__MODULE__{resource: reference()}

  @doc """
  Creates a new decode stream for the given tokenizer.

  Options:

  - `:skip_special_tokens` - whether to suppress special tokens from output,
    defaults to `true`
  """
  @spec new(IREE.Tokenizers.Tokenizer.t(), keyword()) :: {:ok, t()} | {:error, {atom(), binary()}}
  def new(tokenizer, opts \\ []) do
    opts = Keyword.validate!(opts, skip_special_tokens: true)
    IREE.Tokenizers.Native.decode_stream_new(tokenizer, opts)
  end

  @doc """
  Feeds a list of token IDs into the stream and returns any newly produced text.
  """
  @spec feed(t(), [integer()]) :: {:ok, binary()} | {:error, {atom(), binary()}}
  def feed(%__MODULE__{} = stream, ids) when is_list(ids) do
    IREE.Tokenizers.Native.decode_stream_feed(stream, ids)
  end

  def feed(%__MODULE__{}, _ids),
    do: {:error, {:invalid_argument, "expected a list of token ids"}}

  @doc """
  Flushes any remaining decode state and returns the final text chunk.
  """
  @spec finalize(t()) :: {:ok, binary()} | {:error, {atom(), binary()}}
  def finalize(%__MODULE__{} = stream) do
    IREE.Tokenizers.Native.decode_stream_finalize(stream)
  end
end
