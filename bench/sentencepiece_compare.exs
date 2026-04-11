Mix.Task.run("app.start")

alias IREE.Tokenizers.Tokenizer, as: IREETokenizer
alias IREETokenizersBench.Support
alias Tokenizers.Encoding, as: HFEncoding
alias Tokenizers.Tokenizer, as: ElixirTokenizers

results_dir = Path.expand("results", __DIR__)
File.mkdir_p!(results_dir)

models = [
  %{
    label: "T5-small (SentencePiece Unigram)",
    repo: "google-t5/t5-small",
    text: "translate English to German: The house is wonderful."
  },
  %{
    label: "LLaMA tokenizer (SentencePiece BPE)",
    repo: "hf-internal-testing/llama-tokenizer",
    text: "The quick brown fox jumps over the lazy dog."
  }
]

rows =
  Enum.map(models, fn model ->
    {:ok, iree_tokenizer} =
      IREETokenizer.from_pretrained(model.repo, format: :sentencepiece_model)

    {:ok, tokenizers_tokenizer} = ElixirTokenizers.from_pretrained(model.repo)

    {:ok, iree_encoding} = IREETokenizer.encode(iree_tokenizer, model.text, add_special_tokens: false)
    {:ok, tokenizers_encoding} =
      ElixirTokenizers.encode(tokenizers_tokenizer, model.text, add_special_tokens: false)

    tokenizers_ids = HFEncoding.get_ids(tokenizers_encoding)

    if iree_encoding.ids != tokenizers_ids do
      raise "encode mismatch for #{model.repo}"
    end

    {:ok, iree_decoded} =
      IREETokenizer.decode(iree_tokenizer, iree_encoding.ids, skip_special_tokens: false)

    {:ok, tokenizers_decoded} =
      ElixirTokenizers.decode(tokenizers_tokenizer, tokenizers_ids, skip_special_tokens: false)

    if iree_decoded != tokenizers_decoded do
      raise "decode mismatch for #{model.repo}"
    end

    iree_encode_ms =
      Support.time_ms(fn ->
        IREETokenizer.encode(iree_tokenizer, model.text, add_special_tokens: false)
      end)

    tokenizers_encode_ms =
      Support.time_ms(fn ->
        ElixirTokenizers.encode(tokenizers_tokenizer, model.text, add_special_tokens: false)
      end)

    iree_decode_ms =
      Support.time_ms(fn ->
        IREETokenizer.decode(iree_tokenizer, iree_encoding.ids, skip_special_tokens: false)
      end)

    tokenizers_decode_ms =
      Support.time_ms(fn ->
        ElixirTokenizers.decode(tokenizers_tokenizer, tokenizers_ids, skip_special_tokens: false)
      end)

    %{
      label: model.label,
      subtitle: "#{byte_size(model.text)} bytes, #{length(iree_encoding.ids)} ids",
      repo: model.repo,
      bytes: byte_size(model.text),
      ids: length(iree_encoding.ids),
      iree_encode_ms: iree_encode_ms,
      tokenizers_encode_ms: tokenizers_encode_ms,
      encode_speedup: tokenizers_encode_ms / iree_encode_ms,
      iree_decode_ms: iree_decode_ms,
      tokenizers_decode_ms: tokenizers_decode_ms,
      decode_speedup: tokenizers_decode_ms / iree_decode_ms
    }
  end)

encode_chart = Path.join(results_dir, "sentencepiece_compare_encode.svg")
decode_chart = Path.join(results_dir, "sentencepiece_compare_decode.svg")
summary_path = Path.join(results_dir, "sentencepiece_compare.md")

Support.render_dual_series_svg(
  encode_chart,
  "SentencePiece model encode latency",
  "IREE .model loader vs elixir-nx/tokenizers tokenizer.json, lower is better",
  Enum.map(rows, fn row ->
    %{
      label: row.label,
      subtitle: row.subtitle,
      iree_ms: row.iree_encode_ms,
      tokenizers_ms: row.tokenizers_encode_ms,
      speedup: row.encode_speedup
    }
  end),
  %{key: :iree_ms, label: "IREE.Tokenizers (.model)", color: "#5A9BF6", formatter: &Support.format_ms/1},
  %{key: :tokenizers_ms, label: "elixir-nx/tokenizers", color: "#FF914D", formatter: &Support.format_ms/1}
)

Support.render_dual_series_svg(
  decode_chart,
  "SentencePiece model decode latency",
  "IREE .model loader vs elixir-nx/tokenizers tokenizer.json, lower is better",
  Enum.map(rows, fn row ->
    %{
      label: row.label,
      subtitle: row.subtitle,
      iree_ms: row.iree_decode_ms,
      tokenizers_ms: row.tokenizers_decode_ms,
      speedup: row.decode_speedup
    }
  end),
  %{key: :iree_ms, label: "IREE.Tokenizers (.model)", color: "#5A9BF6", formatter: &Support.format_ms/1},
  %{key: :tokenizers_ms, label: "elixir-nx/tokenizers", color: "#FF914D", formatter: &Support.format_ms/1}
)

summary = """
# SentencePiece `.model` comparison against elixir-nx/tokenizers

## Encode latency

| Model | Repo | Input bytes | Output ids | IREE `.model` | `tokenizers` | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
#{Enum.map_join(rows, "\n", fn row ->
  "| #{row.label} | #{row.repo} | #{row.bytes} | #{row.ids} | #{Support.format_ms(row.iree_encode_ms)} | #{Support.format_ms(row.tokenizers_encode_ms)} | #{Float.round(row.encode_speedup, 2)}x |"
end)}

## Decode latency

| Model | Repo | Input bytes | Output ids | IREE `.model` | `tokenizers` | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
#{Enum.map_join(rows, "\n", fn row ->
  "| #{row.label} | #{row.repo} | #{row.bytes} | #{row.ids} | #{Support.format_ms(row.iree_decode_ms)} | #{Support.format_ms(row.tokenizers_decode_ms)} | #{Float.round(row.decode_speedup, 2)}x |"
end)}
"""

File.write!(summary_path, summary)

IO.puts("Wrote benchmark artifacts:")
IO.puts("  #{encode_chart}")
IO.puts("  #{decode_chart}")
IO.puts("  #{summary_path}")
