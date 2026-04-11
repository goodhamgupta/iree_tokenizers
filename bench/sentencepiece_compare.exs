Mix.Task.run("app.start")

alias IREE.Tokenizers.Tokenizer, as: IREETokenizer
alias IREETokenizersBench.Support
alias Tokenizers.Tokenizer, as: ElixirTokenizers

results_dir = Path.expand("results", __DIR__)
File.mkdir_p!(results_dir)

models = [
  %{
    label: "T5-small (SentencePiece Unigram)",
    repo: "google-t5/t5-small",
    benchmark_text: "translate English to German: The house is wonderful.",
    validation_texts: [
      "translate English to German: The house is wonderful.",
      " translate English to German: The house is wonderful.",
      "translate  English\n to German: The house is wonderful."
    ]
  },
  %{
    label: "LLaMA tokenizer (SentencePiece BPE)",
    repo: "hf-internal-testing/llama-tokenizer",
    benchmark_text: "The quick brown fox jumps over the lazy dog.",
    validation_texts: [
      "The quick brown fox jumps over the lazy dog.",
      " The quick brown fox jumps over the lazy dog.",
      "The  quick brown fox\n jumps over the lazy dog."
    ]
  }
]

{rows, skipped} =
  Enum.reduce(models, {[], []}, fn model, {rows, skipped} ->
    {:ok, iree_tokenizer} =
      IREETokenizer.from_pretrained(model.repo, format: :sentencepiece_model)

    {:ok, tokenizers_tokenizer} = ElixirTokenizers.from_pretrained(model.repo)

    validations =
      Enum.map(model.validation_texts, fn text ->
        {:ok, comparison} =
          Support.encode_comparison(iree_tokenizer, tokenizers_tokenizer, text,
            add_special_tokens: false
          )

        {text, comparison}
      end)

    mismatches =
      Enum.reject(validations, fn {_text, comparison} ->
        Support.equivalent_outputs?(comparison)
      end)

    case mismatches do
      [] ->
        benchmark_text = model.benchmark_text

        {:ok, benchmark_comparison} =
          Support.encode_comparison(iree_tokenizer, tokenizers_tokenizer, benchmark_text,
            add_special_tokens: false
          )

        iree_encode_ms =
          Support.time_ms(fn ->
            IREETokenizer.encode(iree_tokenizer, benchmark_text, add_special_tokens: false)
          end)

        tokenizers_encode_ms =
          Support.time_ms(fn ->
            ElixirTokenizers.encode(tokenizers_tokenizer, benchmark_text,
              add_special_tokens: false
            )
          end)

        iree_decode_ms =
          Support.time_ms(fn ->
            IREETokenizer.decode(iree_tokenizer, benchmark_comparison.iree_ids,
              skip_special_tokens: false
            )
          end)

        tokenizers_decode_ms =
          Support.time_ms(fn ->
            ElixirTokenizers.decode(
              tokenizers_tokenizer,
              benchmark_comparison.iree_ids,
              skip_special_tokens: false
            )
          end)

        row = %{
          label: model.label,
          subtitle:
            "#{byte_size(benchmark_text)} bytes, #{length(benchmark_comparison.iree_ids)} ids",
          repo: model.repo,
          bytes: byte_size(benchmark_text),
          ids: length(benchmark_comparison.iree_ids),
          iree_encode_ms: iree_encode_ms,
          tokenizers_encode_ms: tokenizers_encode_ms,
          encode_speedup: tokenizers_encode_ms / iree_encode_ms,
          iree_decode_ms: iree_decode_ms,
          tokenizers_decode_ms: tokenizers_decode_ms,
          decode_speedup: tokenizers_decode_ms / iree_decode_ms
        }

        {[row | rows], skipped}

      [{text, comparison} | _] ->
        reason =
          "validation mismatch on #{inspect(text)} (ids_equal=#{comparison.ids_equal}, decoded_equal=#{comparison.decoded_equal})"

        {rows, [%{label: model.label, repo: model.repo, reason: reason} | skipped]}
    end
  end)

rows = Enum.reverse(rows)
skipped = Enum.reverse(skipped)

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
  %{
    key: :iree_ms,
    label: "IREE.Tokenizers (.model)",
    color: "#5A9BF6",
    formatter: &Support.format_ms/1
  },
  %{
    key: :tokenizers_ms,
    label: "elixir-nx/tokenizers",
    color: "#FF914D",
    formatter: &Support.format_ms/1
  }
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
  %{
    key: :iree_ms,
    label: "IREE.Tokenizers (.model)",
    color: "#5A9BF6",
    formatter: &Support.format_ms/1
  },
  %{
    key: :tokenizers_ms,
    label: "elixir-nx/tokenizers",
    color: "#FF914D",
    formatter: &Support.format_ms/1
  }
)

summary = """
# SentencePiece `.model` comparison against elixir-nx/tokenizers

## Encode latency

| Model | Repo | Input bytes | Output ids | IREE `.model` | `tokenizers` | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
#{Enum.map_join(rows, "\n", fn row -> "| #{row.label} | #{row.repo} | #{row.bytes} | #{row.ids} | #{Support.format_ms(row.iree_encode_ms)} | #{Support.format_ms(row.tokenizers_encode_ms)} | #{Float.round(row.encode_speedup, 2)}x |" end)}

## Decode latency

| Model | Repo | Input bytes | Output ids | IREE `.model` | `tokenizers` | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
#{Enum.map_join(rows, "\n", fn row -> "| #{row.label} | #{row.repo} | #{row.bytes} | #{row.ids} | #{Support.format_ms(row.iree_decode_ms)} | #{Support.format_ms(row.tokenizers_decode_ms)} | #{Float.round(row.decode_speedup, 2)}x |" end)}

#{if skipped == [] do
  ""
else
  """

  ## Skipped

  #{Enum.map_join(skipped, "\n", fn entry -> "- #{entry.label} (#{entry.repo}): #{entry.reason}" end)}
  """
end}
"""

File.write!(summary_path, summary)

IO.puts("Wrote benchmark artifacts:")
IO.puts("  #{encode_chart}")
IO.puts("  #{decode_chart}")
IO.puts("  #{summary_path}")
