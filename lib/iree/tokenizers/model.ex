defmodule IREE.Tokenizers.Model do
  @moduledoc """
  Pure Elixir model specification used to build IREE-backed tokenizers.

  These structs are not tokenizers by themselves. They are small declarative
  descriptions that can be passed to `IREE.Tokenizers.Tokenizer.init/1`.

  Currently supported model constructors live in:

  - `IREE.Tokenizers.Model.BPE`
  - `IREE.Tokenizers.Model.WordPiece`
  - `IREE.Tokenizers.Model.Unigram`
  """

  defstruct [:type, spec: %{}, info: %{}]

  @typedoc """
  A model specification that can be turned into an IREE tokenizer with
  `IREE.Tokenizers.Tokenizer.init/1`.
  """
  @type t :: %__MODULE__{
          type: binary(),
          spec: map(),
          info: map()
        }

  @doc """
  Returns metadata about the model specification.

  The exact keys vary by model type, but always include `"model_type"`.
  """
  @spec info(t()) :: map()
  def info(%__MODULE__{info: info}), do: info
end

defimpl Inspect, for: IREE.Tokenizers.Model do
  import Inspect.Algebra

  def inspect(model, opts) do
    attrs =
      model
      |> IREE.Tokenizers.Model.info()
      |> Keyword.new(fn {k, v} -> {String.to_atom(k), v} end)

    concat(["#IREE.Tokenizers.Model<", to_doc(attrs, opts), ">"])
  end
end
