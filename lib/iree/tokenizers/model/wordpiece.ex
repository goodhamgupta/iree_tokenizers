defmodule IREE.Tokenizers.Model.WordPiece do
  @moduledoc """
  WordPiece model spec compatible with `IREE.Tokenizers.Tokenizer.init/1`.
  """

  alias IREE.Tokenizers.Model

  @type options :: [
          unk_token: String.t(),
          max_input_chars_per_word: number(),
          continuing_subword_prefix: String.t()
        ]

  @spec init(%{String.t() => integer()}, options()) :: {:ok, Model.t()}
  def init(vocab, options \\ []) when is_map(vocab) do
    opts =
      Keyword.validate!(options,
        unk_token: "[UNK]",
        max_input_chars_per_word: 100,
        continuing_subword_prefix: "##"
      )

    spec = %{
      "vocab" => Map.new(vocab),
      "unk_token" => opts[:unk_token],
      "max_input_chars_per_word" => opts[:max_input_chars_per_word],
      "continuing_subword_prefix" => opts[:continuing_subword_prefix]
    }

    info = %{
      "model_type" => "WordPiece",
      "vocab_size" => map_size(vocab),
      "unk_token" => opts[:unk_token],
      "max_input_chars_per_word" => opts[:max_input_chars_per_word],
      "continuing_subword_prefix" => opts[:continuing_subword_prefix]
    }

    {:ok, %Model{type: "WordPiece", spec: spec, info: info}}
  end

  @spec empty() :: {:ok, Model.t()}
  def empty, do: init(%{})

  @spec from_file(String.t(), options()) :: {:ok, Model.t()} | {:error, term()}
  def from_file(vocab_path, options \\ []) do
    with {:ok, vocab_text} <- File.read(vocab_path) do
      vocab =
        vocab_text
        |> String.split("\n", trim: true)
        |> Enum.with_index()
        |> Map.new(fn {token, id} -> {token, id} end)

      init(vocab, options)
    end
  end
end
