Mix.Task.run("app.start")

alias IREE.Tokenizers.Tokenizer, as: IREETokenizer
alias IREETokenizersBench.Support
alias Tokenizers.Encoding, as: HFEncoding
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
    {:ok, iree_encoding} =
      IREETokenizer.encode(iree_tokenizer, text, add_special_tokens: false, track_offsets: true)

    {:ok, tokenizers_encoding} =
      ElixirTokenizers.encode(tokenizers_tokenizer, text, add_special_tokens: false)

    iree_ids = iree_encoding.ids
    hf_ids = HFEncoding.get_ids(tokenizers_encoding)

    iree_ms =
      Support.time_ms(fn ->
        IREETokenizer.encode(iree_tokenizer, text, add_special_tokens: false)
      end)

    tokenizers_ms =
      Support.time_ms(fn ->
        ElixirTokenizers.encode(tokenizers_tokenizer, text, add_special_tokens: false)
      end)

    %{
      label: String.capitalize(label),
      subtitle:
        "#{byte_size(text)} bytes input, #{length(iree_ids)} / #{length(hf_ids)} output ids",
      bytes: byte_size(text),
      iree_ms: iree_ms,
      tokenizers_ms: tokenizers_ms,
      speedup: tokenizers_ms / iree_ms,
      iree_ids: iree_ids,
      tokenizers_ids: hf_ids
    }
  end)

Benchee.run(
  encode_rows
  |> Enum.flat_map(fn row ->
    [
      {"iree decode #{String.downcase(row.label)}",
       fn -> IREETokenizer.decode(iree_tokenizer, row.iree_ids, skip_special_tokens: false) end},
      {"tokenizers decode #{String.downcase(row.label)}",
       fn ->
         ElixirTokenizers.decode(
           tokenizers_tokenizer,
           row.tokenizers_ids,
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
  Enum.map(encode_rows, fn row ->
    iree_ms =
      Support.time_ms(fn ->
        IREETokenizer.decode(iree_tokenizer, row.iree_ids, skip_special_tokens: false)
      end)

    tokenizers_ms =
      Support.time_ms(fn ->
        ElixirTokenizers.decode(
          tokenizers_tokenizer,
          row.tokenizers_ids,
          skip_special_tokens: false
        )
      end)

    %{
      label: row.label,
      subtitle: "#{length(row.iree_ids)} / #{length(row.tokenizers_ids)} ids decoded",
      iree_ms: iree_ms,
      tokenizers_ms: tokenizers_ms,
      speedup: tokenizers_ms / iree_ms
    }
  end)

encode_chart = Path.join(results_dir, "tokenizers_compare_encode.svg")
decode_chart = Path.join(results_dir, "tokenizers_compare_decode.svg")
summary_path = Path.join(results_dir, "tokenizers_compare.md")

Support.render_dual_series_svg(
  encode_chart,
  "IREE vs tokenizers encode latency",
  "Local minimal BPE fixture, lower is better",
  encode_rows,
  %{key: :iree_ms, label: "IREE.Tokenizers", color: "#5A9BF6", formatter: &Support.format_ms/1},
  %{key: :tokenizers_ms, label: "elixir-nx/tokenizers", color: "#FF914D", formatter: &Support.format_ms/1}
)

Support.render_dual_series_svg(
  decode_chart,
  "IREE vs tokenizers decode latency",
  "Local minimal BPE fixture, lower is better",
  decode_rows,
  %{key: :iree_ms, label: "IREE.Tokenizers", color: "#35C296", formatter: &Support.format_ms/1},
  %{key: :tokenizers_ms, label: "elixir-nx/tokenizers", color: "#FF914D", formatter: &Support.format_ms/1}
)

summary = """
# Fixture comparison against elixir-nx/tokenizers

Local fixture: `test/fixtures/bpe_bytelevel_minimal.json`

## Encode latency

| Workload | Input bytes | IREE.Tokenizers | elixir-nx/tokenizers | Speedup |
| --- | ---: | ---: | ---: | ---: |
#{Enum.map_join(encode_rows, "\n", fn row ->
  "| #{row.label} | #{row.bytes} | #{Support.format_ms(row.iree_ms)} | #{Support.format_ms(row.tokenizers_ms)} | #{Float.round(row.speedup, 2)}x |"
end)}

## Decode latency

| Workload | IREE / tokenizers ids | IREE.Tokenizers | elixir-nx/tokenizers | Speedup |
| --- | ---: | ---: | ---: | ---: |
#{Enum.map_join(decode_rows, "\n", fn row ->
  "| #{row.label} | #{String.replace_suffix(row.subtitle, " ids decoded", "")} | #{Support.format_ms(row.iree_ms)} | #{Support.format_ms(row.tokenizers_ms)} | #{Float.round(row.speedup, 2)}x |"
end)}

## Notes

- Encode latency compares the same input text for both libraries.
- Decode latency compares each library decoding its own encoded ID sequence for the same input text.
- The minimal local BPE fixture diverges in output token counts on longer inputs, so latency is a more faithful cross-library comparison than tokens/sec for this specific fixture.
"""

File.write!(summary_path, summary)

IO.puts("Wrote benchmark artifacts:")
IO.puts("  #{encode_chart}")
IO.puts("  #{decode_chart}")
IO.puts("  #{summary_path}")
