defmodule IREETokenizers.BatchIntegrationTest do
  use ExUnit.Case, async: false

  alias IREE.Tokenizers.Tokenizer, as: IREETokenizer
  alias Tokenizers.Encoding, as: HFEncoding
  alias Tokenizers.Tokenizer, as: HFTokenizer

  @moduletag integration: true
  @moduletag skip:
               if(System.get_env("RUN_PRETRAINED_BATCH_INTEGRATION") in ["1", "true"],
                 do: false,
                 else:
                   "set RUN_PRETRAINED_BATCH_INTEGRATION=1 to run pretrained batch integration tests"
               )

  test "gpt2 batch encode matches per-item Hugging Face parity on mixed-length inputs" do
    inputs = harness_batch_inputs()

    {:ok, iree_tokenizer} = IREETokenizer.from_pretrained("openai-community/gpt2")
    {:ok, hf_tokenizer} = HFTokenizer.from_pretrained("openai-community/gpt2")

    assert_batch_encoding_parity(iree_tokenizer, hf_tokenizer, inputs)
  end

  test "bert one-shot encode matches Hugging Face on control-character whitespace regression" do
    input = "bell\x07tab\ttab vertical\vform\ftab back\bspace"

    {:ok, iree_tokenizer} = IREETokenizer.from_pretrained("google-bert/bert-base-uncased")
    {:ok, hf_tokenizer} = HFTokenizer.from_pretrained("google-bert/bert-base-uncased")

    {:ok, iree_encoding} = IREETokenizer.encode(iree_tokenizer, input, add_special_tokens: false)
    {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: false)

    assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
  end

  test "bert batch encode matches per-item Hugging Face parity on emoji regression corpus" do
    inputs = [
      "Hello, world!",
      "日本語 한국어 中文 ไทย עברית العربية",
      "🚀🌍 Let's go! 👩‍💻 👨‍👩‍👧‍👦 🇺🇸 🏳️‍🌈"
    ]

    {:ok, iree_tokenizer} = IREETokenizer.from_pretrained("google-bert/bert-base-uncased")
    {:ok, hf_tokenizer} = HFTokenizer.from_pretrained("google-bert/bert-base-uncased")

    assert_batch_encoding_parity(iree_tokenizer, hf_tokenizer, inputs)
  end

  test "t5 batch encode matches per-item Hugging Face parity on long regression inputs" do
    inputs = harness_batch_inputs()

    {:ok, iree_tokenizer} = IREETokenizer.from_pretrained("google-t5/t5-small")
    {:ok, hf_tokenizer} = HFTokenizer.from_pretrained("google-t5/t5-small")

    assert_batch_encoding_parity(iree_tokenizer, hf_tokenizer, inputs)
  end

  test "t5 sentencepiece batch encode matches per-item Hugging Face parity on long regression inputs" do
    inputs = harness_batch_inputs()

    {:ok, iree_tokenizer} =
      IREETokenizer.from_pretrained("google-t5/t5-small", format: :sentencepiece_model)

    {:ok, hf_tokenizer} = HFTokenizer.from_pretrained("google-t5/t5-small")

    assert_batch_encoding_parity(iree_tokenizer, hf_tokenizer, inputs)
  end

  test "byte-level bpe keeps merges after mixed CJK and ASCII runs" do
    input = "日本語のトークナイザーはUnicodeをうまく扱えますか？ 中文分词 한국어 테스트. "

    {:ok, iree_tokenizer} = IREETokenizer.from_pretrained("LiquidAI/LFM2.5-230M")
    {:ok, hf_tokenizer} = HFTokenizer.from_pretrained("LiquidAI/LFM2.5-230M")

    {:ok, iree_encoding} = IREETokenizer.encode(iree_tokenizer, input, add_special_tokens: false)
    {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: false)

    assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
  end

  test "byte-level ignore_merges bpe keeps repeated-punctuation segmentation" do
    input = "!!! ??? ... ,,, ;;; :::"

    {iree_tokenizer, hf_tokenizer} =
      case System.get_env("NEMOTRON_TOKENIZER_JSON") do
        nil ->
          {:ok, iree_tokenizer} =
            IREETokenizer.from_pretrained("nvidia/Nemotron-3-Embed-8B-BF16")

          {:ok, hf_tokenizer} = HFTokenizer.from_pretrained("nvidia/Nemotron-3-Embed-8B-BF16")
          {iree_tokenizer, hf_tokenizer}

        path ->
          {:ok, iree_tokenizer} = IREETokenizer.from_file(path)
          {:ok, hf_tokenizer} = HFTokenizer.from_file(path)
          {iree_tokenizer, hf_tokenizer}
      end

    {:ok, iree_encoding} = IREETokenizer.encode(iree_tokenizer, input, add_special_tokens: false)
    {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: false)

    assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
  end

  test "byte-level bpe preserves merge rank for repeated punctuation" do
    input = "!!! ??? ... ,,, ;;; :::"
    tokenizer_json = System.get_env("GLM_TOKENIZER_JSON")

    {iree_tokenizer, hf_tokenizer} =
      if tokenizer_json do
        {:ok, iree_tokenizer} = IREETokenizer.from_file(tokenizer_json)
        {:ok, hf_tokenizer} = HFTokenizer.from_file(tokenizer_json)
        {iree_tokenizer, hf_tokenizer}
      else
        {:ok, iree_tokenizer} = IREETokenizer.from_pretrained("zai-org/GLM-5.2")
        {:ok, hf_tokenizer} = HFTokenizer.from_pretrained("zai-org/GLM-5.2")
        {iree_tokenizer, hf_tokenizer}
      end

    for add_special_tokens <- [true, false] do
      {:ok, iree_encoding} =
        IREETokenizer.encode(iree_tokenizer, input, add_special_tokens: add_special_tokens)

      {:ok, hf_encoding} =
        HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: add_special_tokens)

      assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
    end
  end

  test "byte-level bpe preserves merge rank for Nemotron repeated punctuation" do
    input = "!!! ??? ... ,,, ;;; :::"
    tokenizer_json = System.get_env("NEMOTRON_1B_TOKENIZER_JSON")

    {iree_tokenizer, hf_tokenizer} =
      if tokenizer_json do
        {:ok, iree_tokenizer} = IREETokenizer.from_file(tokenizer_json)
        {:ok, hf_tokenizer} = HFTokenizer.from_file(tokenizer_json)
        {iree_tokenizer, hf_tokenizer}
      else
        {:ok, iree_tokenizer} =
          IREETokenizer.from_pretrained("nvidia/Nemotron-3-Embed-1B-BF16")

        {:ok, hf_tokenizer} = HFTokenizer.from_pretrained("nvidia/Nemotron-3-Embed-1B-BF16")
        {iree_tokenizer, hf_tokenizer}
      end

    for add_special_tokens <- [true, false] do
      {:ok, iree_encoding} =
        IREETokenizer.encode(iree_tokenizer, input, add_special_tokens: add_special_tokens)

      {:ok, hf_encoding} =
        HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: add_special_tokens)

      assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)

      {:ok, [iree_batch_encoding]} =
        IREETokenizer.encode_batch(iree_tokenizer, [input],
          add_special_tokens: add_special_tokens
        )

      {:ok, [hf_batch_encoding]} =
        HFTokenizer.encode_batch(hf_tokenizer, [input], add_special_tokens: add_special_tokens)

      assert iree_batch_encoding.ids == HFEncoding.get_ids(hf_batch_encoding)
    end
  end

  defp assert_batch_encoding_parity(iree_tokenizer, hf_tokenizer, inputs) do
    {:ok, iree_encodings} =
      IREETokenizer.encode_batch(iree_tokenizer, inputs, add_special_tokens: false)

    {:ok, hf_encodings} =
      HFTokenizer.encode_batch(hf_tokenizer, inputs, add_special_tokens: false)

    assert Enum.map(iree_encodings, & &1.ids) == Enum.map(hf_encodings, &HFEncoding.get_ids/1)
  end

  defp harness_batch_inputs do
    long_repeat = String.duplicate("the quick brown fox jumps over the lazy dog. ", 4096)
    cjk_long = String.duplicate("日本語のトークナイザーはUnicodeをうまく扱えますか？ 中文分词 한국어 테스트. ", 1024)
    mixed_long = String.duplicate("Tokenization 日本語 🚀 déjà vu naïve café.\n\t", 2048)

    [
      "a",
      "Hello, world!",
      "   leading\t\ttabs\n\nnewlines   trailing   ",
      "naïve café résumé coöperate façade",
      "日本語 한국어 中文 ไทย עברית العربية",
      "🚀🌍 Let's go! 👩‍💻 👨‍👩‍👧‍👦 🇺🇸 🏳️‍🌈",
      "def f(x):\n    return [i**2 for i in range(x) if i % 2 == 0]\n",
      "fn main() { let v: Vec<u32> = (0..10).filter(|n| n % 3 == 0).collect(); println!(\"{:?}\", v); }",
      "{\"name\": \"Alice\", \"age\": 30, \"tags\": [\"admin\", \"user\"]}",
      "Try <|endoftext|> and <s> </s> <pad> <unk> [CLS] [SEP] in one line",
      "bell\x07tab\ttab vertical\vform\ftab back\bspace",
      "0 1 12 123 1234567890 3.1415926535 -42 +7 0xFF 1e-9",
      "See https://example.com/path?q=hello%20world&n=42#frag and ftp://a.b/c",
      "# Title\n\n- item **bold**\n- `code`\n\n> quote\n\n```py\nprint(1)\n```",
      "!!! ??? ... ,,, ;;; :::",
      long_repeat,
      cjk_long,
      mixed_long
    ]
  end
end
