defmodule IREE.Tokenizers.Encoding do
  @moduledoc """
  Result returned by encoding operations.

  This module intentionally mirrors the most useful `Tokenizers.Encoding`
  helpers so callers can inspect token IDs, offsets, masks, and derived
  metadata without dealing with the NIF directly.
  """

  defstruct ids: [],
            type_ids: [],
            offsets: nil,
            attention_mask: [],
            special_tokens_mask: [],
            tokens: []

  @typedoc """
  An encoded token sequence with optional offsets and derived masks.
  """
  @type t :: %__MODULE__{
          ids: [integer()],
          type_ids: [non_neg_integer()],
          offsets: nil | [{non_neg_integer(), non_neg_integer()}],
          attention_mask: [non_neg_integer()],
          special_tokens_mask: [non_neg_integer()],
          tokens: [binary()]
        }

  alias IREE.Tokenizers.Encoding.Transformation

  @doc """
  Returns the number of tokens in the encoding.
  """
  @spec get_length(t()) :: non_neg_integer()
  def get_length(%__MODULE__{ids: ids}), do: length(ids)

  @doc """
  Alias for `get_length/1`.
  """
  @spec n_tokens(t()) :: non_neg_integer()
  def n_tokens(encoding), do: get_length(encoding)

  @doc """
  Returns the number of sequences represented by the encoding.

  The current IREE-backed implementation only emits single-sequence encodings.
  """
  @spec get_n_sequences(t()) :: non_neg_integer()
  def get_n_sequences(%__MODULE__{ids: []}), do: 0
  def get_n_sequences(_encoding), do: 1

  @doc """
  Returns the token IDs.
  """
  @spec get_ids(t()) :: [integer()]
  def get_ids(%__MODULE__{ids: ids}), do: ids

  @doc """
  Returns the token IDs packed into a little-endian `u32` binary.
  """
  @spec get_u32_ids(t()) :: binary()
  def get_u32_ids(%__MODULE__{ids: ids}), do: u32_binary(ids)

  @doc """
  Returns the type IDs.
  """
  @spec get_type_ids(t()) :: [integer()]
  def get_type_ids(%__MODULE__{type_ids: ids}), do: ids

  @doc """
  Returns the type IDs packed into a little-endian `u32` binary.
  """
  @spec get_u32_type_ids(t()) :: binary()
  def get_u32_type_ids(%__MODULE__{type_ids: ids}), do: u32_binary(ids)

  @doc """
  Returns the attention mask.
  """
  @spec get_attention_mask(t()) :: [integer()]
  def get_attention_mask(%__MODULE__{attention_mask: mask}), do: mask

  @doc """
  Returns the attention mask packed into a little-endian `u32` binary.
  """
  @spec get_u32_attention_mask(t()) :: binary()
  def get_u32_attention_mask(%__MODULE__{attention_mask: mask}), do: u32_binary(mask)

  @doc """
  Returns the special-tokens mask.
  """
  @spec get_special_tokens_mask(t()) :: [integer()]
  def get_special_tokens_mask(%__MODULE__{special_tokens_mask: mask}), do: mask

  @doc """
  Returns the special-tokens mask packed into a little-endian `u32` binary.
  """
  @spec get_u32_special_tokens_mask(t()) :: binary()
  def get_u32_special_tokens_mask(%__MODULE__{special_tokens_mask: mask}), do: u32_binary(mask)

  @doc """
  Returns the token strings corresponding to the encoding.
  """
  @spec get_tokens(t()) :: [binary()]
  def get_tokens(%__MODULE__{tokens: tokens}), do: tokens

  @doc """
  Returns word IDs for each token.

  The current implementation does not track word IDs and returns `nil` entries.
  """
  @spec get_word_ids(t()) :: [nil]
  def get_word_ids(%__MODULE__{ids: ids}), do: List.duplicate(nil, length(ids))

  @doc """
  Returns sequence IDs for each token, with special tokens represented as `nil`.
  """
  @spec get_sequence_ids(t()) :: [non_neg_integer() | nil]
  def get_sequence_ids(%__MODULE__{type_ids: type_ids, special_tokens_mask: special_mask}) do
    Enum.zip(type_ids, special_mask)
    |> Enum.map(fn
      {_type_id, 1} -> nil
      {type_id, _} -> type_id
    end)
  end

  @doc """
  Returns byte offsets for each token.
  """
  @spec get_offsets(t()) :: [{integer(), integer()}]
  def get_offsets(%__MODULE__{offsets: nil}), do: []
  def get_offsets(%__MODULE__{offsets: offsets}), do: offsets

  @doc """
  Returns overflowing encodings, if any.

  The current implementation does not emit overflowing pieces and always
  returns an empty list.
  """
  @spec get_overflowing(t()) :: [t()]
  def get_overflowing(_encoding), do: []

  @doc """
  Replaces all sequence IDs in the encoding with the given value.
  """
  @spec set_sequence_id(t(), non_neg_integer()) :: t()
  def set_sequence_id(%__MODULE__{} = encoding, id) when is_integer(id) and id >= 0 do
    %{encoding | type_ids: List.duplicate(id, get_length(encoding))}
  end

  @doc """
  Pads the encoding to `target_length`.

  Supported options:

  - `:direction` - `:left` or `:right`, defaults to `:right`
  - `:pad_id` - token ID used for padding, defaults to `0`
  - `:pad_type_id` - type ID used for padding, defaults to `0`
  - `:pad_token` - token string used for padding, defaults to `"[PAD]"`
  """
  @spec pad(t(), non_neg_integer(), keyword()) :: t()
  def pad(%__MODULE__{} = encoding, target_length, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        direction: :right,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]"
      )

    current = get_length(encoding)
    pad_count = max(target_length - current, 0)
    if pad_count == 0, do: encoding, else: do_pad(encoding, pad_count, opts)
  end

  @doc """
  Truncates the encoding to `max_length`.

  Supported options:

  - `:direction` - `:left` or `:right`, defaults to `:right`
  - `:stride` - accepted for compatibility, currently not applied
  """
  @spec truncate(t(), non_neg_integer(), keyword()) :: t()
  def truncate(%__MODULE__{} = encoding, max_length, opts \\ []) do
    opts = Keyword.validate!(opts, stride: 0, direction: :right)
    current = get_length(encoding)

    if current <= max_length,
      do: encoding,
      else: do_truncate(encoding, max_length, opts[:direction])
  end

  @doc """
  Applies a list of transformations in order.
  """
  @spec transform(t(), [Transformation.t()]) :: t()
  def transform(%__MODULE__{} = encoding, transformations) when is_list(transformations) do
    Enum.reduce(transformations, encoding, fn
      {:pad, {target_length, opts}}, acc -> pad(acc, target_length, opts)
      {:truncate, {max_length, opts}}, acc -> truncate(acc, max_length, opts)
      {:set_sequence_id, id}, acc -> set_sequence_id(acc, id)
    end)
  end

  defp do_pad(encoding, pad_count, opts) do
    padding = %__MODULE__{
      ids: List.duplicate(opts[:pad_id], pad_count),
      type_ids: List.duplicate(opts[:pad_type_id], pad_count),
      offsets: maybe_offsets(encoding.offsets, pad_count),
      attention_mask: List.duplicate(0, pad_count),
      special_tokens_mask: List.duplicate(1, pad_count),
      tokens: List.duplicate(opts[:pad_token], pad_count)
    }

    case opts[:direction] do
      :left -> concat(padding, encoding)
      :right -> concat(encoding, padding)
    end
  end

  defp do_truncate(encoding, max_length, :right) do
    slice(encoding, 0, max_length)
  end

  defp do_truncate(encoding, max_length, :left) do
    slice(encoding, get_length(encoding) - max_length, max_length)
  end

  defp concat(left, right) do
    %__MODULE__{
      ids: left.ids ++ right.ids,
      type_ids: left.type_ids ++ right.type_ids,
      offsets: concat_offsets(left.offsets, right.offsets),
      attention_mask: left.attention_mask ++ right.attention_mask,
      special_tokens_mask: left.special_tokens_mask ++ right.special_tokens_mask,
      tokens: left.tokens ++ right.tokens
    }
  end

  defp slice(encoding, offset, length) do
    %__MODULE__{
      ids: Enum.slice(encoding.ids, offset, length),
      type_ids: Enum.slice(encoding.type_ids, offset, length),
      offsets: if(encoding.offsets, do: Enum.slice(encoding.offsets, offset, length), else: nil),
      attention_mask: Enum.slice(encoding.attention_mask, offset, length),
      special_tokens_mask: Enum.slice(encoding.special_tokens_mask, offset, length),
      tokens: Enum.slice(encoding.tokens, offset, length)
    }
  end

  defp maybe_offsets(nil, _pad_count), do: nil
  defp maybe_offsets(_offsets, pad_count), do: List.duplicate({0, 0}, pad_count)

  defp concat_offsets(nil, nil), do: nil
  defp concat_offsets(left, right), do: (left || []) ++ (right || [])

  defp u32_binary(list) do
    for value <- list, into: <<>>, do: <<value::unsigned-little-32>>
  end
end

defimpl Inspect, for: IREE.Tokenizers.Encoding do
  import Inspect.Algebra

  def inspect(%IREE.Tokenizers.Encoding{} = encoding, opts) do
    attrs = [
      n_tokens: length(encoding.ids),
      ids: encoding.ids,
      type_ids: encoding.type_ids,
      offsets: encoding.offsets,
      attention_mask: encoding.attention_mask,
      special_tokens_mask: encoding.special_tokens_mask
    ]

    concat(["#IREE.Tokenizers.Encoding<", to_doc(attrs, opts), ">"])
  end
end
