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
