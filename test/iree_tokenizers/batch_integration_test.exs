defmodule IREETokenizers.BatchIntegrationTest do
  use ExUnit.Case, async: false

  alias IREE.Tokenizers.EncodeStream
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

  test "split plus ByteLevel use_regex false keeps literal whitespace tokens" do
    inputs = [
      "   leading\t\ttabs\n\nnewlines   trailing   ",
      "def f(x):\n    return [i**2 for i in range(x) if i % 2 == 0]\n",
      "bell\x07tab\ttab vertical\vform\ftab back\bspace",
      "# Title\n\n- item **bold**\n- `code`\n\n> quote\n\n```py\nprint(1)\n```"
    ]

    tokenizer_json = System.get_env("MOTIF_TOKENIZER_JSON")

    {iree_tokenizer, hf_tokenizer} =
      if tokenizer_json do
        {:ok, iree_tokenizer} = IREETokenizer.from_file(tokenizer_json)
        {:ok, hf_tokenizer} = HFTokenizer.from_file(tokenizer_json)
        {iree_tokenizer, hf_tokenizer}
      else
        {:ok, iree_tokenizer} = IREETokenizer.from_pretrained("Motif-Technologies/Motif-3")
        {:ok, hf_tokenizer} = HFTokenizer.from_pretrained("Motif-Technologies/Motif-3")
        {iree_tokenizer, hf_tokenizer}
      end

    for add_special_tokens <- [true, false] do
      {:ok, iree_encodings} =
        IREETokenizer.encode_batch(iree_tokenizer, inputs, add_special_tokens: add_special_tokens)

      {:ok, hf_encodings} =
        HFTokenizer.encode_batch(hf_tokenizer, inputs, add_special_tokens: add_special_tokens)

      assert Enum.map(iree_encodings, & &1.ids) == Enum.map(hf_encodings, &HFEncoding.get_ids/1)
    end
  end

  test "gpt split plus ByteLevel keeps whitespace branch priority" do
    inputs = [
      "   leading\t\ttabs\n\nnewlines   trailing   ",
      "def f(x):\n    return [i**2 for i in range(x) if i % 2 == 0]\n"
    ]

    tokenizer_json =
      System.get_env("MINICPM5_TOKENIZER_JSON") || System.get_env("QWEN38_TOKENIZER_JSON")

    {iree_tokenizer, hf_tokenizer} =
      if tokenizer_json do
        {:ok, iree_tokenizer} = IREETokenizer.from_file(tokenizer_json)
        {:ok, hf_tokenizer} = HFTokenizer.from_file(tokenizer_json)
        {iree_tokenizer, hf_tokenizer}
      else
        {:ok, iree_tokenizer} = IREETokenizer.from_pretrained("empero-ai/Qwen3.8-9B-Distill")
        {:ok, hf_tokenizer} = HFTokenizer.from_pretrained("empero-ai/Qwen3.8-9B-Distill")
        {iree_tokenizer, hf_tokenizer}
      end

    for add_special_tokens <- [true, false] do
      for input <- inputs do
        {:ok, iree_encoding} =
          IREETokenizer.encode(iree_tokenizer, input, add_special_tokens: add_special_tokens)

        {:ok, hf_encoding} =
          HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: add_special_tokens)

        assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
      end

      {:ok, iree_encodings} =
        IREETokenizer.encode_batch(iree_tokenizer, inputs, add_special_tokens: add_special_tokens)

      {:ok, hf_encodings} =
        HFTokenizer.encode_batch(hf_tokenizer, inputs, add_special_tokens: add_special_tokens)

      assert Enum.map(iree_encodings, & &1.ids) == Enum.map(hf_encodings, &HFEncoding.get_ids/1)
    end
  end

  @tag skip:
         if(System.get_env("WEMM_TOKENIZER_JSON"),
           do: false,
           else: "set WEMM_TOKENIZER_JSON to run the WeMM tokenizer regression"
         )
  test "WeMM regex normalizer preserves one-shot, batch, and stream parity" do
    tokenizer_json = System.fetch_env!("WEMM_TOKENIZER_JSON")
    {:ok, iree_tokenizer} = IREETokenizer.from_file(tokenizer_json)
    {:ok, hf_tokenizer} = HFTokenizer.from_file(tokenizer_json)

    inputs = [
      "   leading\t\ttabs\n\nnewlines   trailing   ",
      "# Title\n\n- item **bold**\n- `code`\n\n> quote\n\n```py\nprint(1)\n```"
    ]

    for add_special_tokens <- [true, false] do
      for input <- inputs do
        {:ok, iree_encoding} =
          IREETokenizer.encode(iree_tokenizer, input,
            add_special_tokens: add_special_tokens,
            track_offsets: true
          )

        {:ok, hf_encoding} =
          HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: add_special_tokens)

        assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
        assert iree_encoding.tokens == HFEncoding.get_tokens(hf_encoding)
        assert iree_encoding.type_ids == HFEncoding.get_type_ids(hf_encoding)
        assert length(iree_encoding.offsets) == length(iree_encoding.ids)
        assert length(HFEncoding.get_offsets(hf_encoding)) == length(iree_encoding.ids)

        for skip_special_tokens <- [true, false] do
          assert {:ok, iree_decoded} =
                   IREETokenizer.decode(iree_tokenizer, iree_encoding.ids,
                     skip_special_tokens: skip_special_tokens
                   )

          assert {:ok, hf_decoded} =
                   HFTokenizer.decode(hf_tokenizer, HFEncoding.get_ids(hf_encoding),
                     skip_special_tokens: skip_special_tokens
                   )

          assert iree_decoded == hf_decoded
        end

        {:ok, stream} =
          EncodeStream.new(iree_tokenizer,
            add_special_tokens: add_special_tokens,
            max_chunk_bytes: 1
          )

        prefix_ids =
          for <<byte <- input>>, reduce: [] do
            ids ->
              {:ok, chunk_ids} = EncodeStream.feed(stream, <<byte>>)
              ids ++ chunk_ids
          end

        assert {:ok, suffix_ids} = EncodeStream.finalize(stream)
        assert prefix_ids ++ suffix_ids == HFEncoding.get_ids(hf_encoding)
      end

      {:ok, iree_batch} =
        IREETokenizer.encode_batch(iree_tokenizer, inputs,
          add_special_tokens: add_special_tokens,
          track_offsets: true
        )

      {:ok, hf_batch} =
        HFTokenizer.encode_batch(hf_tokenizer, inputs, add_special_tokens: add_special_tokens)

      for {iree_encoding, hf_encoding} <- Enum.zip(iree_batch, hf_batch) do
        assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
        assert iree_encoding.tokens == HFEncoding.get_tokens(hf_encoding)
        assert iree_encoding.type_ids == HFEncoding.get_type_ids(hf_encoding)
        assert length(iree_encoding.offsets) == length(iree_encoding.ids)
        assert length(HFEncoding.get_offsets(hf_encoding)) == length(iree_encoding.ids)
      end

      iree_ids = Enum.map(iree_batch, & &1.ids)
      hf_ids = Enum.map(hf_batch, &HFEncoding.get_ids/1)

      for skip_special_tokens <- [true, false] do
        assert {:ok, iree_decoded} =
                 IREETokenizer.decode_batch(iree_tokenizer, iree_ids,
                   skip_special_tokens: skip_special_tokens
                 )

        assert {:ok, hf_decoded} =
                 HFTokenizer.decode_batch(hf_tokenizer, hf_ids,
                   skip_special_tokens: skip_special_tokens
                 )

        assert iree_decoded == hf_decoded
      end
    end
  end

  test "superwhisper and DeepSeek GPT split variants keep whitespace branch priority" do
    inputs = [
      "   leading\t\ttabs\n\nnewlines   trailing   ",
      "def f(x):\n    return [i**2 for i in range(x) if i % 2 == 0]\n",
      "a   1",
      "a  日",
      "a   日本語",
      "a   b",
      "a   ",
      "   ",
      "   a",
      "   1",
      "   日",
      "1   a",
      "日   a",
      "a   123   b",
      "a   日本語   b",
      "a\t\t1",
      "a\n\n日",
      "a \t  日",
      "a\t\t!",
      "a\t\t?",
      "a\n\n!",
      "a  !"
    ]

    # DeepSeek V4 Flash and Pro currently publish byte-identical tokenizer JSON,
    # so one representative asset covers the shared regression in issues #52/#53.
    tokenizers = [
      {"superwhisper/s1-mini", "SUPERWHISPER_TOKENIZER_JSON"},
      {"deepseek-ai/DeepSeek-V4-Flash-0731", "DEEPSEEK_V4_TOKENIZER_JSON"}
    ]

    for {repo, tokenizer_json_env} <- tokenizers do
      tokenizer_json = System.get_env(tokenizer_json_env)

      {iree_tokenizer, hf_tokenizer} =
        if tokenizer_json do
          {:ok, iree_tokenizer} = IREETokenizer.from_file(tokenizer_json)
          {:ok, hf_tokenizer} = HFTokenizer.from_file(tokenizer_json)
          {iree_tokenizer, hf_tokenizer}
        else
          {:ok, iree_tokenizer} = IREETokenizer.from_pretrained(repo)
          {:ok, hf_tokenizer} = HFTokenizer.from_pretrained(repo)
          {iree_tokenizer, hf_tokenizer}
        end

      for add_special_tokens <- [true, false] do
        for input <- inputs do
          {:ok, iree_encoding} =
            IREETokenizer.encode(iree_tokenizer, input,
              add_special_tokens: add_special_tokens,
              track_offsets: true
            )

          {:ok, hf_encoding} =
            HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: add_special_tokens)

          assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
          assert iree_encoding.tokens == HFEncoding.get_tokens(hf_encoding)
          assert iree_encoding.type_ids == HFEncoding.get_type_ids(hf_encoding)
          assert iree_encoding.offsets == HFEncoding.get_offsets(hf_encoding)

          assert {:ok, iree_decoded} =
                   IREETokenizer.decode(iree_tokenizer, iree_encoding.ids,
                     skip_special_tokens: false
                   )

          assert {:ok, hf_decoded} =
                   HFTokenizer.decode(hf_tokenizer, HFEncoding.get_ids(hf_encoding),
                     skip_special_tokens: false
                   )

          assert iree_decoded == hf_decoded

          {:ok, stream} =
            EncodeStream.new(iree_tokenizer,
              add_special_tokens: add_special_tokens,
              max_chunk_bytes: 4
            )

          prefix_ids =
            input
            |> String.codepoints()
            |> Enum.flat_map(fn chunk ->
              {:ok, ids} = EncodeStream.feed(stream, chunk)
              ids
            end)

          {:ok, suffix_ids} = EncodeStream.finalize(stream)
          assert prefix_ids ++ suffix_ids == iree_encoding.ids
        end

        {:ok, iree_encodings} =
          IREETokenizer.encode_batch(iree_tokenizer, inputs,
            add_special_tokens: add_special_tokens
          )

        {:ok, hf_encodings} =
          HFTokenizer.encode_batch(hf_tokenizer, inputs, add_special_tokens: add_special_tokens)

        assert Enum.map(iree_encodings, & &1.ids) ==
                 Enum.map(hf_encodings, &HFEncoding.get_ids/1)
      end
    end
  end

  test "GLM 5.3 and PhoneLLM GPT split variants keep whitespace branch priority" do
    inputs = [
      "   leading\t\ttabs\n\nnewlines   trailing   ",
      "def f(x):\n    return [i**2 for i in range(x) if i % 2 == 0]\n",
      "  a"
    ]

    tokenizers = [
      {"zai-org/GLM-5.3", "GLM53_TOKENIZER_JSON"},
      {"pipecat-ai/phonellm-alpha-1", "PHONELLM_TOKENIZER_JSON"}
    ]

    for {repo, tokenizer_json_env} <- tokenizers do
      {iree_tokenizer, hf_tokenizer} =
        case System.get_env(tokenizer_json_env) do
          nil ->
            {:ok, iree_tokenizer} = IREETokenizer.from_pretrained(repo)
            {:ok, hf_tokenizer} = HFTokenizer.from_pretrained(repo)
            {iree_tokenizer, hf_tokenizer}

          tokenizer_json ->
            {:ok, iree_tokenizer} = IREETokenizer.from_file(tokenizer_json)
            {:ok, hf_tokenizer} = HFTokenizer.from_file(tokenizer_json)
            {iree_tokenizer, hf_tokenizer}
        end

      for add_special_tokens <- [true, false] do
        for input <- inputs do
          {:ok, iree_encoding} =
            IREETokenizer.encode(iree_tokenizer, input, add_special_tokens: add_special_tokens)

          {:ok, hf_encoding} =
            HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: add_special_tokens)

          assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
          assert iree_encoding.tokens == HFEncoding.get_tokens(hf_encoding)
          assert iree_encoding.type_ids == HFEncoding.get_type_ids(hf_encoding)
          assert iree_encoding.offsets == nil

          assert {:ok, iree_decoded} =
                   IREETokenizer.decode(iree_tokenizer, iree_encoding.ids,
                     skip_special_tokens: false
                   )

          assert {:ok, hf_decoded} =
                   HFTokenizer.decode(hf_tokenizer, HFEncoding.get_ids(hf_encoding),
                     skip_special_tokens: false
                   )

          assert iree_decoded == hf_decoded

          {:ok, stream} =
            EncodeStream.new(iree_tokenizer,
              add_special_tokens: add_special_tokens,
              max_chunk_bytes: 4
            )

          prefix_ids =
            input
            |> String.codepoints()
            |> Enum.flat_map(fn chunk ->
              {:ok, ids} = EncodeStream.feed(stream, chunk)
              ids
            end)

          assert {:ok, suffix_ids} = EncodeStream.finalize(stream)
          assert prefix_ids ++ suffix_ids == iree_encoding.ids
        end

        {:ok, iree_encodings} =
          IREETokenizer.encode_batch(iree_tokenizer, inputs,
            add_special_tokens: add_special_tokens
          )

        {:ok, hf_encodings} =
          HFTokenizer.encode_batch(hf_tokenizer, inputs, add_special_tokens: add_special_tokens)

        for {iree_encoding, hf_encoding} <- Enum.zip(iree_encodings, hf_encodings) do
          assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
          assert iree_encoding.tokens == HFEncoding.get_tokens(hf_encoding)
          assert iree_encoding.type_ids == HFEncoding.get_type_ids(hf_encoding)
          assert iree_encoding.offsets == nil
        end
      end
    end
  end

  test "Granite 4.2 direct ByteLevel regex keeps whitespace branch priority" do
    tokenizer_json = System.get_env("GRANITE_TOKENIZER_JSON")

    {iree_tokenizer, hf_tokenizer} =
      if tokenizer_json do
        {:ok, iree_tokenizer} = IREETokenizer.from_file(tokenizer_json)
        {:ok, hf_tokenizer} = HFTokenizer.from_file(tokenizer_json)
        {iree_tokenizer, hf_tokenizer}
      else
        {:ok, iree_tokenizer} =
          IREETokenizer.from_pretrained("ibm-granite/granite-4.2-30b")

        {:ok, hf_tokenizer} =
          HFTokenizer.from_pretrained("ibm-granite/granite-4.2-30b")

        {iree_tokenizer, hf_tokenizer}
      end

    inputs = [
      "   leading\t\ttabs\n\nnewlines   trailing   ",
      "def f(x):\n    return [i**2 for i in range(x) if i % 2 == 0]\n",
      "# Title\n\n- item **bold**\n- `code`\n\n> quote\n\n```py\nprint(1)\n```"
    ]

    for add_special_tokens <- [true, false] do
      for input <- inputs do
        {:ok, iree_encoding} =
          IREETokenizer.encode(iree_tokenizer, input,
            add_special_tokens: add_special_tokens,
            track_offsets: true
          )

        {:ok, hf_encoding} =
          HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: add_special_tokens)

        assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
        assert iree_encoding.tokens == HFEncoding.get_tokens(hf_encoding)
        assert iree_encoding.type_ids == HFEncoding.get_type_ids(hf_encoding)
        assert iree_encoding.offsets == HFEncoding.get_offsets(hf_encoding)

        assert {:ok, iree_decoded} =
                 IREETokenizer.decode(iree_tokenizer, iree_encoding.ids,
                   skip_special_tokens: false
                 )

        assert {:ok, hf_decoded} =
                 HFTokenizer.decode(hf_tokenizer, HFEncoding.get_ids(hf_encoding),
                   skip_special_tokens: false
                 )

        assert iree_decoded == input
        assert hf_decoded == input
      end

      {:ok, iree_batch} =
        IREETokenizer.encode_batch(iree_tokenizer, inputs,
          add_special_tokens: add_special_tokens,
          track_offsets: true
        )

      {:ok, hf_batch} =
        HFTokenizer.encode_batch(hf_tokenizer, inputs, add_special_tokens: add_special_tokens)

      for {iree_encoding, hf_encoding} <- Enum.zip(iree_batch, hf_batch) do
        assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
        assert iree_encoding.tokens == HFEncoding.get_tokens(hf_encoding)
        assert iree_encoding.type_ids == HFEncoding.get_type_ids(hf_encoding)
        assert iree_encoding.offsets == HFEncoding.get_offsets(hf_encoding)
      end
    end
  end

  test "special token prefix false positive preserves metaspace BPE span" do
    input = "Try <|endoftext|> and <s> </s> <pad> <unk> [CLS] [SEP] in one line"
    tokenizer_json = System.get_env("BITCPM_TOKENIZER_JSON")

    {iree_tokenizer, hf_tokenizer} =
      if tokenizer_json do
        {:ok, iree_tokenizer} = IREETokenizer.from_file(tokenizer_json)
        {:ok, hf_tokenizer} = HFTokenizer.from_file(tokenizer_json)
        {iree_tokenizer, hf_tokenizer}
      else
        {:ok, iree_tokenizer} = IREETokenizer.from_pretrained("openbmb/BitCPM-CANN-8B")
        {:ok, hf_tokenizer} = HFTokenizer.from_pretrained("openbmb/BitCPM-CANN-8B")
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

  test "Supra Router preserves exact ByteLevel merge priority on long input" do
    tokenizer_json = System.get_env("SUPRA_TOKENIZER_JSON")

    {iree_tokenizer, hf_tokenizer} =
      if tokenizer_json do
        {:ok, iree_tokenizer} = IREETokenizer.from_file(tokenizer_json)
        {:ok, hf_tokenizer} = HFTokenizer.from_file(tokenizer_json)
        {iree_tokenizer, hf_tokenizer}
      else
        {:ok, iree_tokenizer} = IREETokenizer.from_pretrained("SupraLabs/Supra-Router-51M")
        {:ok, hf_tokenizer} = HFTokenizer.from_pretrained("SupraLabs/Supra-Router-51M")
        {iree_tokenizer, hf_tokenizer}
      end

    inputs = [
      "xxxxxxxxxxxxxxxxUnicode",
      String.duplicate(
        "日本語のトークナイザーはUnicodeをうまく扱えますか？ 中文分词 한국어 테스트. ",
        1024
      )
    ]

    for add_special_tokens <- [true, false] do
      for input <- inputs do
        {:ok, iree_encoding} =
          IREETokenizer.encode(iree_tokenizer, input, add_special_tokens: add_special_tokens)

        {:ok, hf_encoding} =
          HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: add_special_tokens)

        assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
      end

      {:ok, iree_encodings} =
        IREETokenizer.encode_batch(iree_tokenizer, inputs, add_special_tokens: add_special_tokens)

      {:ok, hf_encodings} =
        HFTokenizer.encode_batch(hf_tokenizer, inputs, add_special_tokens: add_special_tokens)

      assert Enum.map(iree_encodings, & &1.ids) ==
               Enum.map(hf_encodings, &HFEncoding.get_ids/1)
    end

    long_cjk = List.last(inputs)
    <<head::binary-size(16_384), tail::binary>> = long_cjk
    {:ok, one_shot} = IREETokenizer.encode(iree_tokenizer, long_cjk, add_special_tokens: false)
    {:ok, stream} = EncodeStream.new(iree_tokenizer, add_special_tokens: false)
    {:ok, head_ids} = EncodeStream.feed(stream, head)
    {:ok, tail_ids} = EncodeStream.feed(stream, tail)
    {:ok, final_ids} = EncodeStream.finalize(stream)
    assert head_ids ++ tail_ids ++ final_ids == one_shot.ids
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
