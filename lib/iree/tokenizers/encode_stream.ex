defmodule IREE.Tokenizers.EncodeStream do
  @moduledoc """
  Streaming encoder state.

  Use this when you want to feed a tokenizer incrementally from multiple
  binary chunks while preserving the same output you would get from one-shot
  encoding of the full input.
  """

  alias IREE.Tokenizers.{ComponentRegistry, Tokenizer}

  defstruct [:resource]

  @typedoc """
  Mutable streaming encode state owned by the NIF or a local buffered fallback.
  """
  @type t :: %__MODULE__{resource: reference() | {:buffered, pid()}}

  @doc """
  Creates a new encode stream for the given tokenizer.

  Options:

  - `:add_special_tokens` - whether post-processing special tokens should be
    emitted during finalization, defaults to `true`
  - `:max_chunk_bytes` - maximum chunk size expected by `feed/2`, defaults to
    `65536`
  """
  @spec new(Tokenizer.t(), keyword()) :: {:ok, t()} | {:error, {atom(), binary()}}
  def new(tokenizer, opts \\ []) do
    opts = Keyword.validate!(opts, add_special_tokens: true, max_chunk_bytes: 65_536)

    if buffered_fallback_required?(tokenizer) do
      {:ok, agent} =
        Agent.start_link(fn ->
          %{
            tokenizer: tokenizer,
            add_special_tokens: opts[:add_special_tokens],
            chunks: [],
            finalized?: false
          }
        end)

      {:ok, %__MODULE__{resource: {:buffered, agent}}}
    else
      IREE.Tokenizers.Native.encode_stream_new(tokenizer, opts)
    end
  end

  @doc """
  Feeds a binary chunk into the stream and returns any newly produced token IDs.
  """
  @spec feed(t(), binary()) :: {:ok, [integer()]} | {:error, {atom(), binary()}}
  def feed(%__MODULE__{resource: {:buffered, agent}}, chunk) when is_binary(chunk) do
    if Process.alive?(agent) do
      case Agent.get(agent, & &1.finalized?) do
        true ->
          {:error, {:invalid_argument, "stream already finalized"}}

        false ->
          Agent.update(agent, fn state -> %{state | chunks: [chunk | state.chunks]} end)
          {:ok, []}
      end
    else
      {:error, {:invalid_argument, "stream already finalized"}}
    end
  end

  def feed(%__MODULE__{} = stream, chunk) when is_binary(chunk) do
    IREE.Tokenizers.Native.encode_stream_feed(stream, chunk)
  end

  def feed(%__MODULE__{}, _chunk),
    do: {:error, {:invalid_argument, "expected a binary chunk"}}

  @doc """
  Flushes any remaining state and returns the final token IDs.
  """
  @spec finalize(t()) :: {:ok, [integer()]} | {:error, {atom(), binary()}}
  def finalize(%__MODULE__{resource: {:buffered, agent}}) do
    if Process.alive?(agent) do
      state =
        Agent.get_and_update(agent, fn state ->
          current = state
          {current, %{state | finalized?: true, chunks: []}}
        end)

      if state.finalized? do
        {:error, {:invalid_argument, "stream already finalized"}}
      else
        buffered_input = state.chunks |> Enum.reverse() |> IO.iodata_to_binary()
        Agent.stop(agent)

        case Tokenizer.encode(state.tokenizer, buffered_input,
               add_special_tokens: state.add_special_tokens
             ) do
          {:ok, encoding} -> {:ok, encoding.ids}
          {:error, _reason} = error -> error
        end
      end
    else
      {:error, {:invalid_argument, "stream already finalized"}}
    end
  end

  def finalize(%__MODULE__{} = stream) do
    IREE.Tokenizers.Native.encode_stream_finalize(stream)
  end

  defp buffered_fallback_required?(tokenizer) do
    components = ComponentRegistry.get(tokenizer.resource)

    components[:default_truncation] != nil or
      components[:default_encoding_transformations] not in [nil, []]
  end
end
