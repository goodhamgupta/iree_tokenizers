Mix.Task.run("app.start")

alias IREE.Tokenizers.Tokenizer, as: IREETokenizer
alias Tokenizers.Tokenizer, as: ElixirTokenizers

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

texts = %{
  "short" => short,
  "medium" => medium,
  "long" => long
}

Benchee.run(
  Enum.flat_map(texts, fn {label, text} ->
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

encoded_inputs =
  texts
  |> Enum.map(fn {label, text} ->
    {:ok, iree_encoding} = IREETokenizer.encode(iree_tokenizer, text, add_special_tokens: false)

    {:ok, tokenizers_encoding} =
      ElixirTokenizers.encode(tokenizers_tokenizer, text, add_special_tokens: false)

    {label, iree_encoding.ids, Tokenizers.Encoding.get_ids(tokenizers_encoding)}
  end)

Benchee.run(
  encoded_inputs
  |> Enum.flat_map(fn {label, iree_ids, tokenizers_ids} ->
    [
      {"iree decode #{label}",
       fn -> IREETokenizer.decode(iree_tokenizer, iree_ids, skip_special_tokens: false) end},
      {"tokenizers decode #{label}",
       fn ->
         ElixirTokenizers.decode(
           tokenizers_tokenizer,
           tokenizers_ids,
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
