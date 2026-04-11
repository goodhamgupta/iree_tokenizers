defmodule IREE.Tokenizers.Model.Unigram do
  @moduledoc """
  Unigram model specification compatible with `IREE.Tokenizers.Tokenizer.init/1`.

  This model shape is also used internally when SentencePiece Unigram
  tokenizers are translated into the IREE-backed runtime format.
  """

  alias IREE.Tokenizers.Model

  @typedoc """
  Options for Unigram model construction.
  """
  @type options :: [
          byte_fallback: boolean(),
          unk_id: integer()
        ]

  @spec init([{String.t(), number()}], options()) :: {:ok, Model.t()}
  @doc """
  Builds a Unigram model specification from an in-memory scored vocabulary.
  """
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
  @doc """
  Returns an empty Unigram model specification.
  """
  def empty, do: init([])
end
