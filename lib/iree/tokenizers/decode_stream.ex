defmodule IREE.Tokenizers.DecodeStream do
  @moduledoc """
  Streaming decoder state.
  """

  defstruct [:resource]

  @type t :: %__MODULE__{resource: reference()}

  @spec new(IREE.Tokenizers.Tokenizer.t(), keyword()) :: {:ok, t()} | {:error, {atom(), binary()}}
  def new(tokenizer, opts \\ []) do
    opts = Keyword.validate!(opts, skip_special_tokens: true)
    IREE.Tokenizers.Native.decode_stream_new(tokenizer, opts)
  end

  @spec feed(t(), [integer()]) :: {:ok, binary()} | {:error, {atom(), binary()}}
  def feed(%__MODULE__{} = stream, ids) when is_list(ids) do
    IREE.Tokenizers.Native.decode_stream_feed(stream, ids)
  end

  def feed(%__MODULE__{}, _ids),
    do: {:error, {:invalid_argument, "expected a list of token ids"}}

  @spec finalize(t()) :: {:ok, binary()} | {:error, {atom(), binary()}}
  def finalize(%__MODULE__{} = stream) do
    IREE.Tokenizers.Native.decode_stream_finalize(stream)
  end
end
