defmodule IREE.Tokenizers.Model.BPE do
  @moduledoc """
  BPE model spec compatible with `IREE.Tokenizers.Tokenizer.init/1`.
  """

  alias IREE.Tokenizers.Model

  @type options :: [
          cache_capacity: number(),
          dropout: float(),
          unk_token: String.t(),
          continuing_subword_prefix: String.t(),
          end_of_word_suffix: String.t(),
          fuse_unk: boolean(),
          byte_fallback: boolean()
        ]

  @spec init(%{String.t() => integer()}, [{String.t(), String.t()}], options()) ::
          {:ok, Model.t()}
  def init(vocab, merges, opts \\ []) when is_map(vocab) and is_list(merges) do
    opts =
      Keyword.validate!(opts,
        cache_capacity: 10_000,
        dropout: nil,
        unk_token: nil,
        continuing_subword_prefix: nil,
        end_of_word_suffix: nil,
        fuse_unk: false,
        byte_fallback: false
      )

    spec = %{
      "vocab" => Map.new(vocab),
      "merges" => Enum.map(merges, fn {left, right} -> [left, right] end),
      "unk_token" => opts[:unk_token],
      "continuing_subword_prefix" => opts[:continuing_subword_prefix],
      "end_of_word_suffix" => opts[:end_of_word_suffix],
      "fuse_unk" => opts[:fuse_unk],
      "byte_fallback" => opts[:byte_fallback]
    }

    info = %{
      "model_type" => "BPE",
      "vocab_size" => map_size(vocab),
      "merge_count" => length(merges),
      "unk_token" => opts[:unk_token],
      "continuing_subword_prefix" => opts[:continuing_subword_prefix],
      "end_of_word_suffix" => opts[:end_of_word_suffix],
      "fuse_unk" => opts[:fuse_unk],
      "byte_fallback" => opts[:byte_fallback]
    }

    {:ok, %Model{type: "BPE", spec: spec, info: info}}
  end

  @spec empty() :: {:ok, Model.t()}
  def empty, do: init(%{}, [])

  @spec from_file(String.t(), String.t(), options()) :: {:ok, Model.t()} | {:error, term()}
  def from_file(vocab_path, merges_path, opts \\ []) do
    with {:ok, vocab_json} <- File.read(vocab_path),
         {:ok, vocab} <- Jason.decode(vocab_json),
         true <- is_map(vocab) or {:error, :invalid_vocab},
         {:ok, merges_text} <- File.read(merges_path) do
      merges =
        merges_text
        |> String.split("\n", trim: true)
        |> Enum.reject(&String.starts_with?(&1, "#"))
        |> Enum.map(fn line ->
          case String.split(line, ~r/\s+/, trim: true) do
            [left, right] -> {left, right}
            _ -> raise ArgumentError, "invalid merges line: #{inspect(line)}"
          end
        end)

      init(vocab, merges, opts)
    else
      {:error, reason} -> {:error, reason}
      false -> {:error, :invalid_vocab}
      error -> {:error, error}
    end
  rescue
    error in [ArgumentError, Jason.DecodeError] -> {:error, error}
  end
end
