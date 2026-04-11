defmodule IREE.Tokenizers.Model.Unigram do
  @moduledoc """
  Unigram model spec compatible with `IREE.Tokenizers.Tokenizer.init/1`.
  """

  alias IREE.Tokenizers.Model

  @type options :: [
          byte_fallback: boolean(),
          unk_id: integer()
        ]

  @spec init([{String.t(), number()}], options()) :: {:ok, Model.t()}
  def init(vocab, options \\ []) when is_list(vocab) do
    opts =
      Keyword.validate!(options,
        byte_fallback: false,
        unk_id: 0
      )

    spec = %{
      "vocab" => Enum.map(vocab, fn {token, score} -> [token, score] end),
      "unk_id" => opts[:unk_id],
      "byte_fallback" => opts[:byte_fallback]
    }

    info = %{
      "model_type" => "Unigram",
      "vocab_size" => length(vocab),
      "unk_id" => opts[:unk_id],
      "byte_fallback" => opts[:byte_fallback]
    }

    {:ok, %Model{type: "Unigram", spec: spec, info: info}}
  end

  @spec empty() :: {:ok, Model.t()}
  def empty, do: init([])
end
