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
