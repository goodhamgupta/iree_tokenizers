Mix.Task.run("app.start")

alias IREE.Tokenizers.Tokenizer, as: IREETokenizer
alias IREETokenizersBench.Support
alias Tokenizers.Tokenizer, as: ElixirTokenizers

results_dir = Path.expand("results", __DIR__)
File.mkdir_p!(results_dir)

fixture = Path.expand("../test/fixtures/bpe_bytelevel_minimal.json", __DIR__)

{:ok, iree_tokenizer} = IREETokenizer.from_file(fixture)
{:ok, tokenizers_tokenizer} = ElixirTokenizers.from_file(fixture)

short = "Hello world"

medium =
  Enum.join(
    List.duplicate(
      "The tokenizer converts text into tokens, and benchmark coverage should include both encode and decode workloads. ",
      16
    )
  )

long =
  Enum.join(
    List.duplicate(
      "Large language model inference pipelines often spend meaningful time in tokenization before the model even begins execution. ",
      256
    )
  )

texts = [
  {"short", short},
  {"medium", medium},
  {"long", long}
]

Benchee.run(
  texts
  |> Enum.flat_map(fn {label, text} ->
    [
      {"iree encode #{label}",
       fn -> IREETokenizer.encode(iree_tokenizer, text, add_special_tokens: false) end},
      {"tokenizers encode #{label}",
       fn -> ElixirTokenizers.encode(tokenizers_tokenizer, text, add_special_tokens: false) end}
    ]
  end)
  |> Map.new(),
  time: 1,
  warmup: 0.5,
  memory_time: 0.5,
  before_each: fn input -> input end,
  formatters: [
    Benchee.Formatters.Console
  ]
)

encode_rows =
  Enum.map(texts, fn {label, text} ->
    {:ok, comparison} =
      Support.encode_comparison(iree_tokenizer, tokenizers_tokenizer, text,
        add_special_tokens: false,
        track_offsets: true
      )

    equivalent? = Support.equivalent_outputs?(comparison)

    mismatch_reason =
      if equivalent? do
        nil
      else
        "encode outputs diverged (ids_equal=#{comparison.ids_equal}, decoded_equal=#{comparison.decoded_equal})"
      end

    base = %{
      label: String.capitalize(label),
      subtitle:
        "#{byte_size(text)} bytes input, #{length(comparison.iree_ids)} / #{length(comparison.tokenizers_ids)} output ids",
      bytes: byte_size(text),
      iree_ids: comparison.iree_ids,
      tokenizers_ids: comparison.tokenizers_ids,
      comparable_decode: equivalent?,
      mismatch_reason: mismatch_reason
    }

    if equivalent? do
      iree_ms =
        Support.time_ms(fn ->
          IREETokenizer.encode(iree_tokenizer, text, add_special_tokens: false)
        end)

      tokenizers_ms =
        Support.time_ms(fn ->
          ElixirTokenizers.encode(tokenizers_tokenizer, text, add_special_tokens: false)
        end)

      Map.merge(base, %{
        iree_ms: iree_ms,
        tokenizers_ms: tokenizers_ms,
        speedup: tokenizers_ms / iree_ms
      })
    else
      Map.merge(base, %{iree_ms: nil, tokenizers_ms: nil, speedup: nil})
    end
  end)

comparable_encode_rows = Enum.filter(encode_rows, & &1.comparable_decode)
skipped_encode_rows = Enum.reject(encode_rows, & &1.comparable_decode)

Benchee.run(
  encode_rows
  |> Enum.filter(& &1.comparable_decode)
  |> Enum.flat_map(fn row ->
    [
      {"iree decode #{String.downcase(row.label)}",
       fn -> IREETokenizer.decode(iree_tokenizer, row.iree_ids, skip_special_tokens: false) end},
      {"tokenizers decode #{String.downcase(row.label)}",
       fn ->
         ElixirTokenizers.decode(
           tokenizers_tokenizer,
           row.iree_ids,
           skip_special_tokens: false
         )
       end}
    ]
  end)
  |> Map.new(),
  time: 1,
  warmup: 0.5,
  memory_time: 0.5,
  formatters: [
    Benchee.Formatters.Console
  ]
)

decode_rows =
  encode_rows
  |> Enum.filter(& &1.comparable_decode)
  |> Enum.map(fn row ->
    iree_ms =
      Support.time_ms(fn ->
        IREETokenizer.decode(iree_tokenizer, row.iree_ids, skip_special_tokens: false)
      end)

    tokenizers_ms =
      Support.time_ms(fn ->
        ElixirTokenizers.decode(
          tokenizers_tokenizer,
          row.iree_ids,
          skip_special_tokens: false
        )
      end)

    %{
      label: row.label,
      subtitle: "#{length(row.iree_ids)} shared ids decoded",
      iree_ms: iree_ms,
      tokenizers_ms: tokenizers_ms,
      speedup: tokenizers_ms / iree_ms
    }
  end)

skipped_decode_rows = skipped_encode_rows

encode_chart = Path.join(results_dir, "tokenizers_compare_encode.svg")
decode_chart = Path.join(results_dir, "tokenizers_compare_decode.svg")
summary_path = Path.join(results_dir, "tokenizers_compare.md")

Support.render_dual_series_svg(
  encode_chart,
  "IREE vs tokenizers encode latency",
  "Local minimal BPE fixture, lower is better",
  comparable_encode_rows,
  %{key: :iree_ms, label: "IREE.Tokenizers", color: "#5A9BF6", formatter: &Support.format_ms/1},
  %{
    key: :tokenizers_ms,
    label: "elixir-nx/tokenizers",
    color: "#FF914D",
    formatter: &Support.format_ms/1
  }
)

Support.render_dual_series_svg(
  decode_chart,
  "IREE vs tokenizers decode latency",
  "Local minimal BPE fixture, lower is better",
  decode_rows,
  %{key: :iree_ms, label: "IREE.Tokenizers", color: "#5A9BF6", formatter: &Support.format_ms/1},
  %{
    key: :tokenizers_ms,
    label: "elixir-nx/tokenizers",
    color: "#FF914D",
    formatter: &Support.format_ms/1
  }
)

summary = """
# Fixture comparison against elixir-nx/tokenizers

Local fixture: `test/fixtures/bpe_bytelevel_minimal.json`

## Encode latency (shared token sequences only)

| Workload | Input bytes | IREE.Tokenizers | elixir-nx/tokenizers | Speedup |
| --- | ---: | ---: | ---: | ---: |
#{Enum.map_join(comparable_encode_rows, "\n", fn row -> "| #{row.label} | #{row.bytes} | #{Support.format_ms(row.iree_ms)} | #{Support.format_ms(row.tokenizers_ms)} | #{Float.round(row.speedup, 2)}x |" end)}

## Decode latency (shared ID sequences only)

| Workload | Shared ids | IREE.Tokenizers | elixir-nx/tokenizers | Speedup |
| --- | ---: | ---: | ---: | ---: |
#{Enum.map_join(decode_rows, "\n", fn row -> "| #{row.label} | #{String.replace_suffix(row.subtitle, " shared ids decoded", "")} | #{Support.format_ms(row.iree_ms)} | #{Support.format_ms(row.tokenizers_ms)} | #{Float.round(row.speedup, 2)}x |" end)}

#{if skipped_encode_rows == [] do
  ""
else
  """
  ## Skipped workloads

  #{Enum.map_join(skipped_encode_rows, "\n", fn row -> "- #{row.label}: #{row.mismatch_reason}" end)}
  """
end}

## Notes

- Latency is only reported for workloads where IREE.Tokenizers and
  elixir-nx/tokenizers produced the same token ids and decoded strings for
  the shared input. Divergent workloads are listed under "Skipped workloads"
  rather than being benchmarked as separate rows.
"""

File.write!(summary_path, summary)

IO.puts("Wrote benchmark artifacts:")
IO.puts("  #{encode_chart}")
IO.puts("  #{decode_chart}")
IO.puts("  #{summary_path}")
