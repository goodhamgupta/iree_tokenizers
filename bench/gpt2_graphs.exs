Mix.Task.run("app.start")

alias IREETokenizersBench.Support
alias IREE.Tokenizers.Tokenizer, as: IREETokenizer
alias Tokenizers.Tokenizer, as: ElixirTokenizers

results_dir = Path.expand("results", __DIR__)
File.mkdir_p!(results_dir)

IO.puts("Loading GPT-2 tokenizers...")
{:ok, iree_tokenizer} = IREETokenizer.from_pretrained("gpt2", Support.pretrained_opts())

{:ok, tokenizers_tokenizer} =
  ElixirTokenizers.from_pretrained("gpt2", Support.pretrained_opts())

batch =
  for index <- 1..100 do
    "#{index}. Tokenization performance matters for real-time inference, large context windows, and developer tooling. " <>
      "This benchmark measures GPT-2 throughput on a fixed batch of prompts in Elixir. "
  end

IO.puts("Preparing decode inputs...")

{:ok, iree_encodings} =
  IREETokenizer.encode_batch(iree_tokenizer, batch, add_special_tokens: false)

{:ok, tokenizers_encodings} =
  ElixirTokenizers.encode_batch(tokenizers_tokenizer, batch, add_special_tokens: false)

iree_decode_batch = Enum.map(iree_encodings, & &1.ids)
tokenizers_decode_batch = Enum.map(tokenizers_encodings, &Tokenizers.Encoding.get_ids(&1))

IO.puts("Benchmarking encode throughput...")

encode_results = [
  Support.benchmark_throughput(
    "iree",
    Support.count_iree_ids(iree_encodings),
    fn -> IREETokenizer.encode_batch(iree_tokenizer, batch, add_special_tokens: false) end,
    1.0,
    2.0
  ),
  Support.benchmark_throughput(
    "tokenizers",
    Support.count_hf_ids(tokenizers_encodings),
    fn ->
      ElixirTokenizers.encode_batch(tokenizers_tokenizer, batch, add_special_tokens: false)
    end,
    1.0,
    2.0
  )
]

IO.puts("Benchmarking decode throughput...")

decode_results = [
  Support.benchmark_throughput(
    "iree",
    Support.count_id_lists(iree_decode_batch),
    fn ->
      IREETokenizer.decode_batch(iree_tokenizer, iree_decode_batch, skip_special_tokens: false)
    end,
    1.0,
    2.0
  ),
  Support.benchmark_throughput(
    "tokenizers",
    Support.count_id_lists(tokenizers_decode_batch),
    fn ->
      ElixirTokenizers.decode_batch(
        tokenizers_tokenizer,
        tokenizers_decode_batch,
        skip_special_tokens: false
      )
    end,
    1.0,
    2.0
  )
]

Support.render_throughput_svg(
  Path.join(results_dir, "gpt2_batch100_encode.svg"),
  "GPT-2 · batch of 100 · encode throughput · higher is better",
  [
    %{
      label: "iree",
      value: Enum.find(encode_results, &(&1.label == "iree")).tokens_per_second,
      color: "#5A9BF6"
    },
    %{
      label: "tokenizers",
      value: Enum.find(encode_results, &(&1.label == "tokenizers")).tokens_per_second,
      color: "#E23A37"
    }
  ]
)

Support.render_throughput_svg(
  Path.join(results_dir, "gpt2_batch100_decode.svg"),
  "GPT-2 · batch of 100 · decode throughput · higher is better",
  [
    %{
      label: "iree",
      value: Enum.find(decode_results, &(&1.label == "iree")).tokens_per_second,
      color: "#35C296"
    },
    %{
      label: "tokenizers",
      value: Enum.find(decode_results, &(&1.label == "tokenizers")).tokens_per_second,
      color: "#E23A37"
    }
  ]
)

File.write!(
  Path.join(results_dir, "summary.md"),
  """
  # GPT-2 Batch-of-100 Throughput

  Compared against the published `tokenizers` Elixir package (`elixir-nx/tokenizers`).

  ## Encode

  #{Enum.map_join(encode_results, "\n", fn result -> "- #{result.label}: #{Support.format_tokens_per_second(result.tokens_per_second)}" end)}

  ## Decode

  #{Enum.map_join(decode_results, "\n", fn result -> "- #{result.label}: #{Support.format_tokens_per_second(result.tokens_per_second)}" end)}
  """
)

IO.puts("Generated benchmark artifacts:")
IO.puts("  #{Path.join(results_dir, "gpt2_batch100_encode.svg")}")
IO.puts("  #{Path.join(results_dir, "gpt2_batch100_decode.svg")}")
IO.puts("  #{Path.join(results_dir, "summary.md")}")
