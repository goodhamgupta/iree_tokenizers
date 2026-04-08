defmodule IREE.Tokenizers.EncodeStream do
  @moduledoc """
  Streaming encoder state.
  """

  defstruct [:resource]

  @type t :: %__MODULE__{resource: reference()}

  @spec new(IREE.Tokenizers.Tokenizer.t(), keyword()) :: {:ok, t()} | {:error, {atom(), binary()}}
  def new(tokenizer, opts \\ []) do
    opts = Keyword.validate!(opts, add_special_tokens: true)
    IREE.Tokenizers.Native.encode_stream_new(tokenizer, opts)
  end

  @spec feed(t(), binary()) :: {:ok, [integer()]} | {:error, {atom(), binary()}}
  def feed(%__MODULE__{} = stream, chunk) when is_binary(chunk) do
    IREE.Tokenizers.Native.encode_stream_feed(stream, chunk)
  end

  def feed(%__MODULE__{}, _chunk),
    do: {:error, {:invalid_argument, "expected a binary chunk"}}

  @spec finalize(t()) :: {:ok, [integer()]} | {:error, {atom(), binary()}}
  def finalize(%__MODULE__{} = stream) do
    IREE.Tokenizers.Native.encode_stream_finalize(stream)
  end
end
