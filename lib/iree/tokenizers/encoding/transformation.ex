defmodule IREE.Tokenizers.Encoding.Transformation do
  @moduledoc """
  Helpers for building transformation lists passed to
  `IREE.Tokenizers.Encoding.transform/2` or the `:encoding_transformations`
  option on encode functions.
  """

  @typedoc """
  A single encoding transformation.
  """
  @type t ::
          {:pad, {non_neg_integer(), keyword()}}
          | {:truncate, {non_neg_integer(), keyword()}}
          | {:set_sequence_id, non_neg_integer()}

  @doc """
  Builds a padding transformation.
  """
  @spec pad(non_neg_integer(), keyword()) :: t()
  def pad(target_length, opts \\ []), do: {:pad, {target_length, opts}}

  @doc """
  Builds a truncation transformation.
  """
  @spec truncate(non_neg_integer(), keyword()) :: t()
  def truncate(max_length, opts \\ []), do: {:truncate, {max_length, opts}}

  @doc """
  Builds a transformation that replaces all sequence IDs with the given value.
  """
  @spec set_sequence_id(non_neg_integer()) :: t()
  def set_sequence_id(id), do: {:set_sequence_id, id}
end
