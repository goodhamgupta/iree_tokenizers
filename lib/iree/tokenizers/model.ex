defmodule IREE.Tokenizers.Model do
  @moduledoc """
  Pure Elixir model spec used to build IREE-backed tokenizers.
  """

  defstruct [:type, spec: %{}, info: %{}]

  @type t :: %__MODULE__{
          type: binary(),
          spec: map(),
          info: map()
        }

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
