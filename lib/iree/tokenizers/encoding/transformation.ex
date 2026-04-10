defmodule IREE.Tokenizers.Encoding.Transformation do
  @moduledoc """
  Helpers for building encoding transformation lists.
  """

  @type t ::
          {:pad, {non_neg_integer(), keyword()}}
          | {:truncate, {non_neg_integer(), keyword()}}
          | {:set_sequence_id, non_neg_integer()}

  @spec pad(non_neg_integer(), keyword()) :: t()
  def pad(target_length, opts \\ []), do: {:pad, {target_length, opts}}

  @spec truncate(non_neg_integer(), keyword()) :: t()
  def truncate(max_length, opts \\ []), do: {:truncate, {max_length, opts}}

  @spec set_sequence_id(non_neg_integer()) :: t()
  def set_sequence_id(id), do: {:set_sequence_id, id}
end
