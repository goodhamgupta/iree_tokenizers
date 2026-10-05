defmodule IREETokenizers.CompatibilityTest do
  use ExUnit.Case, async: true

  alias IREE.Tokenizers.{Encoding, EncodeStream, Tokenizer}
  alias Tokenizers.Encoding, as: HFEncoding
  alias Tokenizers.Tokenizer, as: HFTokenizer

  test "matches official tokenizers on shared bpe fixture outputs" do
    fixture = fixture_path("bpe_bytelevel_minimal.json")
    {:ok, iree_tokenizer} = Tokenizer.from_file(fixture)
    {:ok, hf_tokenizer} = HFTokenizer.from_file(fixture)

    {:ok, iree_encoding} =
      Tokenizer.encode(iree_tokenizer, "Hello world",
        add_special_tokens: false,
        track_offsets: true
      )

    {:ok, hf_encoding} =
      HFTokenizer.encode(hf_tokenizer, "Hello world", add_special_tokens: false)

    assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
    assert Encoding.get_type_ids(iree_encoding) == HFEncoding.get_type_ids(hf_encoding)
    assert Encoding.get_offsets(iree_encoding) == HFEncoding.get_offsets(hf_encoding)

    assert Encoding.get_attention_mask(iree_encoding) ==
             HFEncoding.get_attention_mask(hf_encoding)

    assert Encoding.get_special_tokens_mask(iree_encoding) ==
             HFEncoding.get_special_tokens_mask(hf_encoding)

    assert Tokenizer.get_vocab_size(iree_tokenizer) == HFTokenizer.get_vocab_size(hf_tokenizer)

    assert Tokenizer.token_to_id(iree_tokenizer, "hello") ==
             HFTokenizer.token_to_id(hf_tokenizer, "hello")

    assert Tokenizer.id_to_token(iree_tokenizer, 109) ==
             HFTokenizer.id_to_token(hf_tokenizer, 109)

    assert {:ok, iree_text} =
             Tokenizer.decode(iree_tokenizer, Encoding.get_ids(iree_encoding),
               skip_special_tokens: false
             )

    assert {:ok, hf_text} =
             HFTokenizer.decode(hf_tokenizer, HFEncoding.get_ids(hf_encoding),
               skip_special_tokens: false
             )

    assert iree_text == hf_text
  end

  test "byte-level decode preserves multibyte UTF-8 split across merged tokens" do
    fixture = fixture_path("bytelevel_utf8_split.json")
    {:ok, iree_tokenizer} = Tokenizer.from_file(fixture)
    {:ok, hf_tokenizer} = HFTokenizer.from_file(fixture)

    {:ok, iree_encoding} = Tokenizer.encode(iree_tokenizer, "🚀", add_special_tokens: false)
    {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, "🚀", add_special_tokens: false)

    assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)

    assert {:ok, iree_text} =
             Tokenizer.decode(iree_tokenizer, Encoding.get_ids(iree_encoding),
               skip_special_tokens: false
             )

    assert {:ok, hf_text} =
             HFTokenizer.decode(hf_tokenizer, HFEncoding.get_ids(hf_encoding),
               skip_special_tokens: false
             )

    assert iree_text == "🚀"
    assert hf_text == "🚀"
  end

  test "word-cache BPE path preserves lower-rank overlapping merge order" do
    fixture = fixture_path("bpe_word_cache_overlap.json")
    {:ok, iree_tokenizer} = Tokenizer.from_file(fixture)
    {:ok, hf_tokenizer} = HFTokenizer.from_file(fixture)

    {:ok, iree_encoding} = Tokenizer.encode(iree_tokenizer, " ,,,", add_special_tokens: false)
    {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, " ,,,", add_special_tokens: false)

    assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
    assert Encoding.get_tokens(iree_encoding) == HFEncoding.get_tokens(hf_encoding)
    assert Encoding.get_tokens(iree_encoding) == ["▁,", ",,"]
  end

  test "BPE repeated suffix merges preserve rank across alternate decompositions" do
    fixture = fixture_path("bpe_repeated_suffix_overlap.json")
    {:ok, tokenizer} = Tokenizer.from_file(fixture)
    {:ok, reference} = HFTokenizer.from_file(fixture)

    # Use an ASCII stand-in for metaspace to also compare source offsets.
    # Both _, + , and _ + ,, produce _,,. The trailing comma must merge
    # with its neighbor first, even though both sides of that merge are equal.
    inputs = [" ,,,", " ;;;", ",,, ;;;"] ++ Enum.map(1..16, &String.duplicate(" ", &1))

    for special <- [true, false] do
      for input <- inputs do
        {:ok, actual} =
          Tokenizer.encode(tokenizer, input, add_special_tokens: special, track_offsets: true)

        {:ok, expected} = HFTokenizer.encode(reference, input, add_special_tokens: special)
        assert actual.ids == HFEncoding.get_ids(expected)
        assert actual.tokens == HFEncoding.get_tokens(expected)
        assert actual.offsets == HFEncoding.get_offsets(expected)
        assert actual.type_ids == HFEncoding.get_type_ids(expected)
        assert actual.attention_mask == HFEncoding.get_attention_mask(expected)
        assert actual.special_tokens_mask == HFEncoding.get_special_tokens_mask(expected)

        assert Tokenizer.decode(tokenizer, actual.ids) ==
                 HFTokenizer.decode(reference, actual.ids)

        {:ok, stream} =
          EncodeStream.new(tokenizer, add_special_tokens: special, max_chunk_bytes: 1)

        prefix =
          for <<byte <- input>>, reduce: [] do
            ids ->
              {:ok, chunk} = EncodeStream.feed(stream, <<byte>>)
              ids ++ chunk
          end

        assert {:ok, suffix} = EncodeStream.finalize(stream)
        assert prefix ++ suffix == actual.ids

        assert {:error, {:invalid_argument, "stream already finalized"}} =
                 EncodeStream.finalize(stream)

        assert {:error, {:invalid_argument, "stream already finalized"}} =
                 EncodeStream.feed(stream, " ")
      end

      {:ok, actual} = Tokenizer.encode_batch(tokenizer, inputs, add_special_tokens: special)
      {:ok, expected} = HFTokenizer.encode_batch(reference, inputs, add_special_tokens: special)
      assert Enum.map(actual, & &1.ids) == Enum.map(expected, &HFEncoding.get_ids/1)
      assert Enum.all?(actual, &is_nil(&1.offsets))
    end
  end

  test "byte-level BPE path preserves lower-rank emoji merge order" do
    fixture = fixture_path("bpe_bytelevel_emoji_merge_rank.json")
    {:ok, iree_tokenizer} = Tokenizer.from_file(fixture)
    {:ok, hf_tokenizer} = HFTokenizer.from_file(fixture)

    {:ok, iree_encoding} = Tokenizer.encode(iree_tokenizer, " 👩", add_special_tokens: false)
    {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, " 👩", add_special_tokens: false)

    assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
    assert Encoding.get_tokens(iree_encoding) == HFEncoding.get_tokens(hf_encoding)
    assert Encoding.get_tokens(iree_encoding) == ["Ġ", "ðŁĳ©"]
  end

  test "direct ByteLevel regex preserves lookahead whitespace branch priority" do
    fixture = fixture_path("bpe_bytelevel_gpt_whitespace.json")
    {:ok, iree_tokenizer} = Tokenizer.from_file(fixture)
    {:ok, hf_tokenizer} = HFTokenizer.from_file(fixture)

    cases = [
      {"  a", [0, 2], ["Ġ", "Ġa"], [{0, 1}, {1, 3}]},
      {"    return", [4, 16], ["ĠĠĠ", "Ġreturn"], [{0, 3}, {3, 10}]},
      {"\n\n-", [19, 19, 21], ["Ċ", "Ċ", "-"], [{0, 1}, {1, 2}, {2, 3}]},
      {"\t\ta", [17, 17, 1], ["ĉ", "ĉ", "a"], [{0, 1}, {1, 2}, {2, 3}]},
      {"a  ", [1, 3], ["a", "ĠĠ"], [{0, 1}, {1, 3}]}
    ]

    for add_special_tokens <- [true, false] do
      for {input, expected_ids, expected_tokens, expected_offsets} <- cases do
        {:ok, iree_encoding} =
          Tokenizer.encode(iree_tokenizer, input,
            add_special_tokens: add_special_tokens,
            track_offsets: true
          )

        {:ok, hf_encoding} =
          HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: add_special_tokens)

        assert Encoding.get_ids(iree_encoding) == expected_ids
        assert Encoding.get_tokens(iree_encoding) == expected_tokens
        assert Encoding.get_offsets(iree_encoding) == expected_offsets
        assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
        assert Encoding.get_tokens(iree_encoding) == HFEncoding.get_tokens(hf_encoding)
        assert Encoding.get_offsets(iree_encoding) == HFEncoding.get_offsets(hf_encoding)

        assert {:ok, iree_decoded} =
                 Tokenizer.decode(iree_tokenizer, expected_ids, skip_special_tokens: false)

        assert {:ok, hf_decoded} =
                 HFTokenizer.decode(hf_tokenizer, expected_ids, skip_special_tokens: false)

        assert iree_decoded == input
        assert hf_decoded == input

        {:ok, stream} =
          EncodeStream.new(iree_tokenizer,
            add_special_tokens: add_special_tokens,
            max_chunk_bytes: 1
          )

        streamed_prefix =
          for <<byte <- input>>, reduce: [] do
            ids ->
              {:ok, chunk_ids} = EncodeStream.feed(stream, <<byte>>)
              ids ++ chunk_ids
          end

        assert {:ok, streamed_suffix} = EncodeStream.finalize(stream)
        assert streamed_prefix ++ streamed_suffix == expected_ids
      end

      inputs = Enum.map(cases, &elem(&1, 0))

      {:ok, iree_batch} =
        Tokenizer.encode_batch(iree_tokenizer, inputs,
          add_special_tokens: add_special_tokens,
          track_offsets: true
        )

      {:ok, hf_batch} =
        HFTokenizer.encode_batch(hf_tokenizer, inputs, add_special_tokens: add_special_tokens)

      for {{iree_encoding, hf_encoding}, {_input, ids, tokens, offsets}} <-
            Enum.zip(Enum.zip(iree_batch, hf_batch), cases) do
        assert Encoding.get_ids(iree_encoding) == ids
        assert Encoding.get_tokens(iree_encoding) == tokens
        assert Encoding.get_offsets(iree_encoding) == offsets
        assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
        assert Encoding.get_tokens(iree_encoding) == HFEncoding.get_tokens(hf_encoding)
        assert Encoding.get_offsets(iree_encoding) == HFEncoding.get_offsets(hf_encoding)
      end
    end
  end

  test "regex Replace end anchor follows Oniguruma line-end semantics" do
    fixture = fixture_path("bpe_regex_end_anchor_normalizer.json")
    {:ok, iree_tokenizer} = Tokenizer.from_file(fixture)
    {:ok, hf_tokenizer} = HFTokenizer.from_file(fixture)

    cases = [
      {"interior LF run", "x\n\nx", [0, 3, 0], ["x", "Ċ", "x"], nil, "x\nx"},
      {"EOF LF", "x\n", [0], ["x"], [{0, 1}], "x"},
      {"non-LF continuation", "x\nx", [0, 3, 0], ["x", "Ċ", "x"], [{0, 1}, {1, 2}, {2, 3}],
       "x\nx"},
      {"interior CRLF", "x\r\n\r\ny", [0, 2, 3, 2, 3, 1], ["x", "č", "Ċ", "č", "Ċ", "y"],
       [{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}], "x\r\n\r\ny"},
      {"EOF CRLF", "x\r\n", [0, 2], ["x", "č"], [{0, 1}, {1, 2}], "x\r"}
    ]

    for add_special_tokens <- [true, false] do
      for {_name, input, base_ids, base_tokens, base_offsets, normalized} <- cases do
        expected_ids = if add_special_tokens, do: base_ids ++ [5], else: base_ids

        expected_tokens =
          if add_special_tokens, do: base_tokens ++ ["<embedding>"], else: base_tokens

        {:ok, iree_encoding} =
          Tokenizer.encode(iree_tokenizer, input,
            add_special_tokens: add_special_tokens,
            track_offsets: true
          )

        {:ok, hf_encoding} =
          HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: add_special_tokens)

        assert Encoding.get_ids(iree_encoding) == expected_ids
        assert Encoding.get_tokens(iree_encoding) == expected_tokens
        assert Encoding.get_type_ids(iree_encoding) == List.duplicate(0, length(expected_ids))
        assert length(Encoding.get_offsets(iree_encoding)) == length(expected_ids)
        assert length(HFEncoding.get_offsets(hf_encoding)) == length(expected_ids)
        assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
        assert Encoding.get_tokens(iree_encoding) == HFEncoding.get_tokens(hf_encoding)
        assert Encoding.get_type_ids(iree_encoding) == HFEncoding.get_type_ids(hf_encoding)

        if base_offsets do
          expected_offsets =
            if add_special_tokens, do: base_offsets ++ [{0, 0}], else: base_offsets

          assert Encoding.get_offsets(iree_encoding) == expected_offsets
          assert Encoding.get_offsets(iree_encoding) == HFEncoding.get_offsets(hf_encoding)
        end

        assert {:ok, iree_decoded_keep} =
                 Tokenizer.decode(iree_tokenizer, expected_ids, skip_special_tokens: false)

        assert {:ok, hf_decoded_keep} =
                 HFTokenizer.decode(hf_tokenizer, expected_ids, skip_special_tokens: false)

        expected_keep = if add_special_tokens, do: normalized <> "<embedding>", else: normalized
        assert iree_decoded_keep == expected_keep
        assert hf_decoded_keep == expected_keep

        assert {:ok, iree_decoded_skip} =
                 Tokenizer.decode(iree_tokenizer, expected_ids, skip_special_tokens: true)

        assert {:ok, hf_decoded_skip} =
                 HFTokenizer.decode(hf_tokenizer, expected_ids, skip_special_tokens: true)

        assert iree_decoded_skip == normalized
        assert hf_decoded_skip == normalized
      end

      inputs = Enum.map(cases, &elem(&1, 1))

      {:ok, iree_batch} =
        Tokenizer.encode_batch(iree_tokenizer, inputs,
          add_special_tokens: add_special_tokens,
          track_offsets: true
        )

      {:ok, hf_batch} =
        HFTokenizer.encode_batch(hf_tokenizer, inputs, add_special_tokens: add_special_tokens)

      for {{iree_encoding, hf_encoding}, {_name, _input, _ids, _tokens, offsets, _decoded}} <-
            Enum.zip(Enum.zip(iree_batch, hf_batch), cases) do
        assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
        assert Encoding.get_tokens(iree_encoding) == HFEncoding.get_tokens(hf_encoding)
        assert Encoding.get_type_ids(iree_encoding) == HFEncoding.get_type_ids(hf_encoding)
        assert length(Encoding.get_offsets(iree_encoding)) == length(iree_encoding.ids)

        if offsets do
          assert Encoding.get_offsets(iree_encoding) == HFEncoding.get_offsets(hf_encoding)
        end
      end

      stream_chunks = [
        ["x", "\n", "\n", "x"],
        ["x\n", "\nx"],
        ["x\n\n", "x"]
      ]

      {:ok, one_shot} =
        Tokenizer.encode(iree_tokenizer, "x\n\nx", add_special_tokens: add_special_tokens)

      for chunks <- stream_chunks do
        {:ok, stream} =
          EncodeStream.new(iree_tokenizer,
            add_special_tokens: add_special_tokens,
            max_chunk_bytes: 1
          )

        prefix_ids =
          Enum.flat_map(chunks, fn chunk ->
            {:ok, ids} = EncodeStream.feed(stream, chunk)
            ids
          end)

        assert {:ok, suffix_ids} = EncodeStream.finalize(stream)
        assert prefix_ids ++ suffix_ids == Encoding.get_ids(one_shot)

        assert {:error, {:invalid_argument, "stream already finalized"}} =
                 EncodeStream.finalize(stream)
      end
    end
  end

  test "regex end-anchor changes preserve start-anchor behavior" do
    fixture = fixture_path("bpe_regex_end_anchor_normalizer.json")

    json =
      fixture
      |> File.read!()
      |> Jason.decode!()
      |> put_in(["normalizer", "pattern", "Regex"], "^x")
      |> Jason.encode!()

    {:ok, iree_tokenizer} = Tokenizer.from_buffer(json)
    {:ok, hf_tokenizer} = HFTokenizer.from_buffer(json)

    for input <- ["xxx", "yxx"] do
      {:ok, iree_encoding} =
        Tokenizer.encode(iree_tokenizer, input, add_special_tokens: false)

      {:ok, hf_encoding} =
        HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: false)

      assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
      assert Encoding.get_tokens(iree_encoding) == HFEncoding.get_tokens(hf_encoding)

      assert {:ok, iree_decoded} =
               Tokenizer.decode(iree_tokenizer, Encoding.get_ids(iree_encoding),
                 skip_special_tokens: false
               )

      assert {:ok, hf_decoded} =
               HFTokenizer.decode(hf_tokenizer, HFEncoding.get_ids(hf_encoding),
                 skip_special_tokens: false
               )

      assert iree_decoded == hf_decoded
    end
  end

  test "deferred end anchor preserves alternation priority across chunks" do
    fixture = fixture_path("bpe_regex_end_anchor_normalizer.json")
    base = fixture |> File.read!() |> Jason.decode!()

    patterns = [
      {"x$|x\\n",
       [
         {"x\n", [3], "\n"},
         {"yx\n", [1, 3], "y\n"},
         {"xy", [0, 1], "xy"},
         {"yxy", [1, 0, 1], "yxy"},
         {"x", [], ""},
         {"yx", [1], "y"}
       ]},
      {"x\\n|x$",
       [
         {"x\n", [], ""},
         {"yx\n", [1], "y"},
         {"xy", [0, 1], "xy"},
         {"yxy", [1, 0, 1], "yxy"},
         {"x", [], ""},
         {"yx", [1], "y"}
       ]}
    ]

    for {pattern, cases} <- patterns do
      json =
        base
        |> Map.put("normalizer", nil)
        |> put_in(["pre_tokenizer", "pretokenizers", Access.at(0), "pattern", "Regex"], pattern)
        |> put_in(["pre_tokenizer", "pretokenizers", Access.at(0), "behavior"], "Removed")
        |> Jason.encode!()

      {:ok, iree_tokenizer} = Tokenizer.from_buffer(json)
      {:ok, hf_tokenizer} = HFTokenizer.from_buffer(json)

      for add_special_tokens <- [true, false], {input, base_ids, decoded} <- cases do
        expected_ids = if add_special_tokens, do: base_ids ++ [5], else: base_ids
        expected_decoded = if add_special_tokens, do: decoded <> "<embedding>", else: decoded

        {:ok, iree_encoding} =
          Tokenizer.encode(iree_tokenizer, input, add_special_tokens: add_special_tokens)

        {:ok, hf_encoding} =
          HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: add_special_tokens)

        assert Encoding.get_ids(iree_encoding) == expected_ids
        assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
        assert Encoding.get_tokens(iree_encoding) == HFEncoding.get_tokens(hf_encoding)
        assert Encoding.get_type_ids(iree_encoding) == HFEncoding.get_type_ids(hf_encoding)

        assert {:ok, iree_decoded} =
                 Tokenizer.decode(iree_tokenizer, expected_ids, skip_special_tokens: false)

        assert {:ok, hf_decoded} =
                 HFTokenizer.decode(hf_tokenizer, expected_ids, skip_special_tokens: false)

        assert iree_decoded == expected_decoded
        assert hf_decoded == expected_decoded

        seam_chunkings =
          if byte_size(input) > 1 do
            for seam <- 1..(byte_size(input) - 1) do
              [
                binary_part(input, 0, seam),
                binary_part(input, seam, byte_size(input) - seam)
              ]
            end
          else
            []
          end

        chunkings = Enum.uniq([[input], String.codepoints(input)] ++ seam_chunkings)

        for chunks <- chunkings do
          {:ok, stream} =
            EncodeStream.new(iree_tokenizer,
              add_special_tokens: add_special_tokens,
              max_chunk_bytes: 1
            )

          prefix_ids =
            Enum.flat_map(chunks, fn chunk ->
              {:ok, ids} = EncodeStream.feed(stream, chunk)
              ids
            end)

          assert {:ok, suffix_ids} = EncodeStream.finalize(stream)
          assert prefix_ids ++ suffix_ids == expected_ids

          assert {:error, {:invalid_argument, "stream already finalized"}} =
                   EncodeStream.finalize(stream)
        end
      end
    end
  end

  test "exact byte-level BPE preserves future lower-rank merge priority" do
    fixture = fixture_path("bpe_bytelevel_window_frontier.json")
    {:ok, iree_tokenizer} = Tokenizer.from_file(fixture)
    {:ok, hf_tokenizer} = HFTokenizer.from_file(fixture)

    # The legacy bounded path sees `ic + o` before the lower-rank future chain
    # `o + d`, `od + e`. The exact path must apply one global merge order.
    input = "xxxxxxxxxxxUnicode"

    {:ok, iree_encoding} = Tokenizer.encode(iree_tokenizer, input, add_special_tokens: false)
    {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: false)

    assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
    assert Encoding.get_tokens(iree_encoding) == HFEncoding.get_tokens(hf_encoding)
    assert Enum.take(Encoding.get_tokens(iree_encoding), -3) == ["Un", "ic", "ode"]

    # Force the Rust wrapper's bounded output retry. Exact-path allocations are
    # released with the failed native state and rebuilt without double-free.
    retry_input = String.duplicate("x", 256)

    {:ok, iree_retry} =
      Tokenizer.encode(iree_tokenizer, retry_input,
        add_special_tokens: false,
        track_offsets: true
      )

    {:ok, hf_retry} =
      HFTokenizer.encode(hf_tokenizer, retry_input, add_special_tokens: false)

    assert Encoding.get_ids(iree_retry) == HFEncoding.get_ids(hf_retry)
    assert Encoding.get_offsets(iree_retry) == HFEncoding.get_offsets(hf_retry)

    newline_input = String.duplicate("\n", 4096)

    {:ok, iree_newlines} =
      Tokenizer.encode(iree_tokenizer, newline_input, add_special_tokens: false)

    {:ok, hf_newlines} =
      HFTokenizer.encode(hf_tokenizer, newline_input, add_special_tokens: false)

    assert Encoding.get_ids(iree_newlines) == HFEncoding.get_ids(hf_newlines)
  end

  test "byte-level BPE applies one global rank order across long dependency chains" do
    fixture = fixture_path("bpe_bytelevel_rank_chain.json")
    {:ok, iree_tokenizer} = Tokenizer.from_file(fixture)
    {:ok, hf_tokenizer} = HFTokenizer.from_file(fixture)

    {:ok, iree_encoding} =
      Tokenizer.encode(iree_tokenizer, "abcdefghi", add_special_tokens: false)

    {:ok, hf_encoding} =
      HFTokenizer.encode(hf_tokenizer, "abcdefghi", add_special_tokens: false)

    assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
    assert Encoding.get_tokens(iree_encoding) == ["a", "bc", "de", "fg", "hi"]

    assert {:ok, empty} = Tokenizer.encode(iree_tokenizer, "", add_special_tokens: false)
    assert Encoding.get_ids(empty) == []
  end

  test "byte-level streams buffer before exact finalize even with a tiny ring" do
    fixture = fixture_path("bpe_bytelevel_rank_chain.json")

    json =
      fixture
      |> File.read!()
      |> Jason.decode!()
      |> put_in(["model", "vocab", String.duplicate("z", 1000)], 17)
      |> Jason.encode!()

    {:ok, tokenizer} = Tokenizer.from_buffer(json)
    input = "abcdefghi"
    {:ok, one_shot} = Tokenizer.encode(tokenizer, input, add_special_tokens: false)
    {:ok, stream} = EncodeStream.new(tokenizer, add_special_tokens: false, max_chunk_bytes: 1)

    for <<byte <- input>> do
      assert {:ok, []} = EncodeStream.feed(stream, <<byte>>)
    end

    assert {:ok, streamed_ids} = EncodeStream.finalize(stream)
    assert streamed_ids == Encoding.get_ids(one_shot)
  end

  test "loads BPE tokenizer.json whose unk_token is absent from vocab (issue #9)" do
    # Laguna-XS.2 declares `unk_token: "[UNK]"` but never adds `[UNK]` to
    # vocab. HF's reference loader treats that as a soft failure (UNK just
    # unreachable). The vendored C BPE path previously raised
    # `NOT_FOUND; unk_token '[UNK]' not found in vocabulary`; the patch in
    # `format/huggingface/model_json.c` now leaves the id INVALID and
    # continues. This fixture pins that behaviour.
    fixture = fixture_path("bpe_unk_token_not_in_vocab.json")

    assert {:ok, tokenizer} = Tokenizer.from_file(fixture)
    assert {:ok, encoding} = Tokenizer.encode(tokenizer, "hello world", add_special_tokens: false)
    assert Encoding.get_ids(encoding) != []
  end

  test "loads tokenizer.json with negative-lookahead Split pre_tokenizer (issue #9)" do
    # Reproduces the Laguna-XS.2 load failure: the vendored C regex parser
    # rejected `(?:\r?\n)+(?!\r?\n)` with "unbalanced parentheses in
    # lookahead" because the lookahead body is `\r?\n` (more than a single
    # atom). The Rust-side sanitizer drops the redundant lookahead before
    # handing the JSON to the C runtime.
    fixture = fixture_path("lookahead_pre_tokenizer_minimal.json")

    assert {:ok, tokenizer} = Tokenizer.from_file(fixture)
    assert {:ok, encoding} = Tokenizer.encode(tokenizer, "hello world", add_special_tokens: false)
    assert is_list(Encoding.get_ids(encoding))
    assert Encoding.get_ids(encoding) != []
  end

  test "loads tokenizer.json with Cohere positive-lookahead digit Split pre_tokenizer (issue #20)" do
    fixture = fixture_path("positive_lookahead_digit_split_minimal.json")

    assert {:ok, tokenizer} = Tokenizer.from_file(fixture)

    assert {:ok, encoding} =
             Tokenizer.encode(tokenizer, "hello 1234567", add_special_tokens: false)

    assert is_list(Encoding.get_ids(encoding))
    assert Encoding.get_ids(encoding) != []
  end

  test "DeepSeek-style Sequence preserves whitespace at digit and CJK parent boundaries" do
    fixture = fixture_path("deepseek_sequence_whitespace_boundary.json")
    {:ok, iree_tokenizer} = Tokenizer.from_file(fixture)
    {:ok, hf_tokenizer} = HFTokenizer.from_file(fixture)

    cases = [
      {"a   b", ["a", "ĠĠ", "Ġb"]},
      {"a   1", ["a", "ĠĠĠ", "1"]},
      {"a  日", ["a", "ĠĠ", "æĹ¥"]},
      {"a   日", ["a", "ĠĠĠ", "æĹ¥"]},
      {"a   ", ["a", "ĠĠĠ"]},
      {"a\t\t!", ["a", "ĉ", "ĉ", "!"]},
      {"a\t\t?", ["a", "ĉ", "ĉ", "?"]},
      {"a\n\n!", ["a", "ĊĊ", "!"]},
      {"a  !", ["a", "Ġ", "Ġ!"]}
    ]

    for add_special_tokens <- [true, false] do
      for {input, expected_tokens} <- cases do
        {:ok, iree_encoding} =
          Tokenizer.encode(iree_tokenizer, input,
            add_special_tokens: add_special_tokens,
            track_offsets: true
          )

        {:ok, hf_encoding} =
          HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: add_special_tokens)

        assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
        assert Encoding.get_tokens(iree_encoding) == expected_tokens
        assert Encoding.get_tokens(iree_encoding) == HFEncoding.get_tokens(hf_encoding)
        assert Encoding.get_type_ids(iree_encoding) == HFEncoding.get_type_ids(hf_encoding)
        assert Encoding.get_offsets(iree_encoding) == HFEncoding.get_offsets(hf_encoding)

        assert {:ok, iree_decoded} =
                 Tokenizer.decode(iree_tokenizer, Encoding.get_ids(iree_encoding),
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
        assert prefix_ids ++ suffix_ids == Encoding.get_ids(iree_encoding)
      end

      inputs = Enum.map(cases, &elem(&1, 0))

      {:ok, iree_batch} =
        Tokenizer.encode_batch(iree_tokenizer, inputs, add_special_tokens: add_special_tokens)

      {:ok, hf_batch} =
        HFTokenizer.encode_batch(hf_tokenizer, inputs, add_special_tokens: add_special_tokens)

      assert Enum.map(iree_batch, &Encoding.get_ids/1) ==
               Enum.map(hf_batch, &HFEncoding.get_ids/1)
    end
  end

  test "common number and CJK Split sequence keeps default probing semantics" do
    fixture = fixture_path("deepseek_sequence_whitespace_boundary.json")

    root = fixture |> File.read!() |> Jason.decode!()
    pretokenizers = get_in(root, ["pre_tokenizer", "pretokenizers"])

    # Keep the common number/CJK children but remove the exact DeepSeek main
    # Split. These children alone must not opt the Sequence into its special
    # parent-boundary policy.
    generic_pretokenizers = [
      Enum.at(pretokenizers, 0),
      Enum.at(pretokenizers, 1),
      List.last(pretokenizers)
    ]

    json =
      root
      |> put_in(["pre_tokenizer", "pretokenizers"], generic_pretokenizers)
      |> Jason.encode!()

    {:ok, iree_tokenizer} = Tokenizer.from_buffer(json)
    {:ok, hf_tokenizer} = HFTokenizer.from_buffer(json)

    inputs = ["a   1", "a   日", "a\t\t!", "a\n\n!"]

    for add_special_tokens <- [true, false], input <- inputs do
      {:ok, iree_encoding} =
        Tokenizer.encode(iree_tokenizer, input,
          add_special_tokens: add_special_tokens,
          track_offsets: true
        )

      {:ok, hf_encoding} =
        HFTokenizer.encode(hf_tokenizer, input, add_special_tokens: add_special_tokens)

      assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
      assert Encoding.get_tokens(iree_encoding) == HFEncoding.get_tokens(hf_encoding)
      assert Encoding.get_offsets(iree_encoding) == HFEncoding.get_offsets(hf_encoding)
    end

    {:ok, tab_encoding} = Tokenizer.encode(iree_tokenizer, "a\t\t!", add_special_tokens: false)
    assert Encoding.get_tokens(tab_encoding) == ["a", "ĉĉ", "!"]
  end

  test "loads tokenizer.json with LongCat-style Unicode punctuation Split pre_tokenizer" do
    # Reproduces the LongCat-2.0 load failure: its punctuation Split regex has
    # more exact Unicode ranges than the previous native regex char-class cap.
    fixture = fixture_path("longcat_unicode_split_minimal.json")

    assert {:ok, tokenizer} = Tokenizer.from_file(fixture)
    assert {:ok, encoding} = Tokenizer.encode(tokenizer, "hello world", add_special_tokens: false)
    assert Encoding.get_ids(encoding) != []
  end

  test "encode stream preserves one-shot ids for null-pretokenizer bpe tokenizers" do
    fixture = fixture_path("sentencepiece_stream_split_minimal.json")
    {:ok, tokenizer} = Tokenizer.from_file(fixture)

    {:ok, iree_encoding} = Tokenizer.encode(tokenizer, "hello", add_special_tokens: false)

    {:ok, stream} = EncodeStream.new(tokenizer, add_special_tokens: false)

    assert {:ok, []} = EncodeStream.feed(stream, "h")
    assert {:ok, []} = EncodeStream.feed(stream, "ello")
    assert {:ok, suffix_ids} = EncodeStream.finalize(stream)

    assert suffix_ids == Encoding.get_ids(iree_encoding)
  end

  test "Metaspace preserves whitespace unless an explicit normalizer or WhitespaceSplit removes it" do
    root =
      fixture_path("unigram_sequence_normalizer_utf8.json") |> File.read!() |> Jason.decode!()

    metaspace = root["pre_tokenizer"]

    inputs = [
      "日\v本",
      "日\f本",
      "日\t本",
      "日\n本",
      "日\r本",
      "日\v\f本",
      "日  本",
      " 日 本  ",
      "日\u00A0本",
      "日\u2003本"
    ]

    for normalizer <- [nil, root["normalizer"]],
        pre_tokenizer <- [
          metaspace,
          %{"type" => "Sequence", "pretokenizers" => [metaspace]},
          %{
            "type" => "Sequence",
            "pretokenizers" => [%{"type" => "WhitespaceSplit"}, metaspace]
          }
        ] do
      json =
        root
        |> Map.put("normalizer", normalizer)
        |> Map.put("pre_tokenizer", pre_tokenizer)
        |> Jason.encode!()

      {:ok, iree_tokenizer} = Tokenizer.from_buffer(json)
      {:ok, hf_tokenizer} = HFTokenizer.from_buffer(json)

      for add_special_tokens <- [true, false] do
        opts = [add_special_tokens: add_special_tokens]
        {:ok, iree_batch} = Tokenizer.encode_batch(iree_tokenizer, inputs, opts)
        {:ok, hf_batch} = HFTokenizer.encode_batch(hf_tokenizer, inputs, opts)

        for {input, {iree_encoding, hf_encoding}} <-
              Enum.zip(inputs, Enum.zip(iree_batch, hf_batch)) do
          assert {:ok, ^iree_encoding} = Tokenizer.encode(iree_tokenizer, input, opts)
          assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
          assert iree_encoding.tokens == HFEncoding.get_tokens(hf_encoding)
          assert iree_encoding.type_ids == HFEncoding.get_type_ids(hf_encoding)
          assert iree_encoding.attention_mask == HFEncoding.get_attention_mask(hf_encoding)

          assert iree_encoding.special_tokens_mask ==
                   HFEncoding.get_special_tokens_mask(hf_encoding)

          assert iree_encoding.offsets == nil

          assert Tokenizer.decode(iree_tokenizer, iree_encoding.ids, skip_special_tokens: false) ==
                   HFTokenizer.decode(hf_tokenizer, HFEncoding.get_ids(hf_encoding),
                     skip_special_tokens: false
                   )

          {:ok, stream} = EncodeStream.new(iree_tokenizer, opts ++ [max_chunk_bytes: 1])

          prefix_ids =
            for <<byte <- input>>, reduce: [] do
              ids ->
                {:ok, chunk_ids} = EncodeStream.feed(stream, <<byte>>)
                ids ++ chunk_ids
            end

          assert {:ok, suffix_ids} = EncodeStream.finalize(stream)
          assert prefix_ids ++ suffix_ids == iree_encoding.ids

          assert {:error, {:invalid_argument, "stream already finalized"}} =
                   EncodeStream.finalize(stream)
        end
      end
    end
  end

  test "Sequence normalizer encodes long UTF-8 input across tile boundaries" do
    # Regression for the parity-monitor SIGABRT (run 26019404748). The vendored
    # Sequence normalizer tiled its input at a fixed 64-byte boundary that
    # ignored UTF-8 codepoints, so child[0] passed a split multi-byte
    # character through to the NFC child, which asserts on incomplete UTF-8
    # and aborted the BEAM. `fastino/gliguard-LLMGuardrails-300M` (Unigram,
    # Sequence[Replace, NFC, Strip] normalizer) hit this on the long CJK
    # parity cases. The patch in `normalizer/sequence.c` trims the tile to a
    # codepoint boundary.
    fixture = fixture_path("unigram_sequence_normalizer_utf8.json")
    {:ok, iree_tokenizer} = Tokenizer.from_file(fixture)
    {:ok, hf_tokenizer} = HFTokenizer.from_file(fixture)

    # Pure 3-byte codepoints never align with the 64-byte tile, so every tile
    # boundary lands mid-character. Far longer than a single tile.
    text = String.duplicate("日本語のテスト。", 96)

    assert {:ok, iree_encoding} =
             Tokenizer.encode(iree_tokenizer, text, add_special_tokens: false)

    assert Encoding.get_ids(iree_encoding) != []

    {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, text, add_special_tokens: false)
    assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
  end

  describe "encode capacity / silent-truncation regression" do
    # The minimal ByteLevel BPE fixture is the worst case for the IREE NIF's
    # output buffer heuristic: every input byte becomes its own token, so the
    # real token count consistently exceeds `bytes/2 + 16`. Prior to the
    # silent-truncation fix in encode_impl / tokenizer_encode_batch, the
    # native call would stop at the buffer size and return a prefix without
    # raising RESOURCE_EXHAUSTED. These tests pin the fix in place across
    # one-shot encode, batched encode, and the streaming encode path.
    setup do
      fixture = fixture_path("bpe_bytelevel_minimal.json")
      {:ok, iree_tokenizer} = Tokenizer.from_file(fixture)
      {:ok, hf_tokenizer} = HFTokenizer.from_file(fixture)
      {:ok, iree: iree_tokenizer, hf: hf_tokenizer}
    end

    @sample_inputs [
      {"short", "Hello world"},
      {"exactly heuristic boundary", "The tokenizer converts text into tokens."},
      {"4x repeat", String.duplicate("The tokenizer converts text. ", 4)},
      {"64x repeat", String.duplicate("The tokenizer converts text. ", 64)},
      {"256x repeat", String.duplicate("The tokenizer converts text. ", 256)}
    ]

    for {label, text} <- @sample_inputs do
      @text text
      test "one-shot encode matches HFTokenizer / #{label}", %{iree: iree, hf: hf} do
        {:ok, iree_encoding} = Tokenizer.encode(iree, @text, add_special_tokens: false)
        {:ok, hf_encoding} = HFTokenizer.encode(hf, @text, add_special_tokens: false)

        assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
      end

      test "encode_batch matches HFTokenizer / #{label}", %{iree: iree, hf: hf} do
        {:ok, [batch_encoding]} =
          Tokenizer.encode_batch(iree, [@text], add_special_tokens: false)

        {:ok, hf_encoding} = HFTokenizer.encode(hf, @text, add_special_tokens: false)

        assert Encoding.get_ids(batch_encoding) == HFEncoding.get_ids(hf_encoding)
      end

      test "EncodeStream matches one-shot encode / #{label}", %{iree: iree} do
        {:ok, iree_encoding} = Tokenizer.encode(iree, @text, add_special_tokens: false)
        {:ok, stream} = EncodeStream.new(iree, add_special_tokens: false)

        prefix_ids =
          @text
          |> chunk_binary(64)
          |> Enum.flat_map(fn chunk ->
            {:ok, ids} = EncodeStream.feed(stream, chunk)
            ids
          end)

        assert {:ok, suffix_ids} = EncodeStream.finalize(stream)
        assert prefix_ids ++ suffix_ids == Encoding.get_ids(iree_encoding)
      end
    end
  end

  test "matches official tokenizers on shared wordpiece fixture outputs" do
    fixture = fixture_path("minimal_wordpiece.json")
    {:ok, iree_tokenizer} = Tokenizer.from_file(fixture)
    {:ok, hf_tokenizer} = HFTokenizer.from_file(fixture)

    {:ok, iree_encoding} =
      Tokenizer.encode(iree_tokenizer, "hello world", add_special_tokens: false)

    {:ok, hf_encoding} =
      HFTokenizer.encode(hf_tokenizer, "hello world", add_special_tokens: false)

    assert Encoding.get_ids(iree_encoding) == HFEncoding.get_ids(hf_encoding)
    assert Encoding.get_type_ids(iree_encoding) == HFEncoding.get_type_ids(hf_encoding)
    assert Tokenizer.get_vocab_size(iree_tokenizer) == HFTokenizer.get_vocab_size(hf_tokenizer)
  end

  test "encode_batch matches HFTokenizer for tokenizer.json BatchLongest padding defaults" do
    fixture = fixture_path("minimal_wordpiece_batch_longest_padded.json")
    {:ok, iree_tokenizer} = Tokenizer.from_file(fixture)
    {:ok, hf_tokenizer} = HFTokenizer.from_file(fixture)
    inputs = ["hello", "hello world token more text"]

    {:ok, iree_encodings} =
      Tokenizer.encode_batch(iree_tokenizer, inputs, add_special_tokens: false)

    {:ok, hf_encodings} =
      HFTokenizer.encode_batch(hf_tokenizer, inputs, add_special_tokens: false)

    assert Enum.map(iree_encodings, &Encoding.get_ids/1) ==
             Enum.map(hf_encodings, &HFEncoding.get_ids/1)

    assert Enum.map(iree_encodings, &Encoding.get_type_ids/1) ==
             Enum.map(hf_encodings, &HFEncoding.get_type_ids/1)

    assert Enum.map(iree_encodings, &Encoding.get_attention_mask/1) ==
             Enum.map(hf_encodings, &HFEncoding.get_attention_mask/1)

    assert Enum.map(iree_encodings, &Encoding.get_special_tokens_mask/1) ==
             Enum.map(hf_encodings, &HFEncoding.get_special_tokens_mask/1)

    assert Enum.map(iree_encodings, &Encoding.get_tokens/1) ==
             Enum.map(hf_encodings, &HFEncoding.get_tokens/1)
  end

  defp fixture_path(name) do
    Path.join([__DIR__, "..", "fixtures", name])
  end

  defp chunk_binary(binary, chunk_bytes) do
    do_chunk_binary(binary, chunk_bytes, [])
  end

  defp do_chunk_binary(<<>>, _chunk_bytes, acc), do: Enum.reverse(acc)

  defp do_chunk_binary(binary, chunk_bytes, acc) when byte_size(binary) <= chunk_bytes,
    do: Enum.reverse([binary | acc])

  defp do_chunk_binary(binary, chunk_bytes, acc) do
    <<chunk::binary-size(chunk_bytes), rest::binary>> = binary
    do_chunk_binary(rest, chunk_bytes, [chunk | acc])
  end
end
