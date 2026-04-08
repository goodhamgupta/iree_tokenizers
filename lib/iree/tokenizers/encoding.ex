defmodule IREE.Tokenizers.Encoding do
  @moduledoc """
  Result returned by encoding operations.
  """

  defstruct ids: [], type_ids: [], offsets: nil

  @type t :: %__MODULE__{
          ids: [integer()],
          type_ids: [non_neg_integer()],
          offsets: nil | [{non_neg_integer(), non_neg_integer()}]
        }
end

defimpl Inspect, for: IREE.Tokenizers.Encoding do
  import Inspect.Algebra

  def inspect(%IREE.Tokenizers.Encoding{} = encoding, opts) do
    attrs = [
      n_tokens: length(encoding.ids),
      ids: encoding.ids,
      type_ids: encoding.type_ids,
      offsets: encoding.offsets
    ]

    concat(["#IREE.Tokenizers.Encoding<", to_doc(attrs, opts), ">"])
  end
end
