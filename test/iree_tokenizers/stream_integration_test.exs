defmodule IREETokenizers.StreamIntegrationTest do
  use ExUnit.Case, async: false

  alias IREE.Tokenizers.{EncodeStream, Tokenizer}
  alias Tokenizers.Encoding, as: HFEncoding
  alias Tokenizers.Tokenizer, as: HFTokenizer

  @moduletag integration: true
  @moduletag skip:
               if(System.get_env("RUN_PRETRAINED_STREAM_INTEGRATION") in ["1", "true"],
                 do: false,
                 else:
                   "set RUN_PRETRAINED_STREAM_INTEGRATION=1 to run pretrained stream integration tests"
               )

  @chunk_bytes 16_384

  test "gemma streaming encode matches one-shot encode and tokenizers package" do
    repo = "google/gemma-4-31B-it"
    text = benchmark_corpus(512_000)

    {:ok, iree_tokenizer} = Tokenizer.from_pretrained(repo)
    {:ok, hf_tokenizer} = HFTokenizer.from_pretrained(repo)

    {:ok, iree_encoding} = Tokenizer.encode(iree_tokenizer, text, add_special_tokens: false)
    {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, text, add_special_tokens: false)

    assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)

    {:ok, stream} =
      EncodeStream.new(iree_tokenizer, add_special_tokens: false, max_chunk_bytes: @chunk_bytes)

    prefix_ids =
      text
      |> chunk_binary(@chunk_bytes)
      |> Enum.flat_map(fn chunk ->
        {:ok, ids} = EncodeStream.feed(stream, chunk)
        ids
      end)

    assert {:ok, suffix_ids} = EncodeStream.finalize(stream)
    assert prefix_ids ++ suffix_ids == iree_encoding.ids
  end

  test "llama tokenizer streaming preserves one-shot ids across split chunks" do
    assert_stream_matches_oneshot("hf-internal-testing/llama-tokenizer", "hello", ["h", "ello"])
  end

  test "llama sentencepiece model streaming preserves one-shot ids across split chunks" do
    assert_stream_matches_oneshot(
      "hf-internal-testing/llama-tokenizer",
      "hello",
      ["h", "ello"],
      format: :sentencepiece_model
    )
  end

  test "phi-3 tokenizer streaming preserves one-shot ids across split chunks" do
    assert_stream_matches_oneshot(
      "microsoft/Phi-3-mini-4k-instruct",
      "hello",
      ["h", "ello"]
    )
  end

  defp benchmark_corpus(byte_target) do
    paragraph =
      "Tokenization performance matters for real-time inference, long-context prompting, retrieval pipelines, and interactive developer tooling. "

    do_benchmark_corpus(paragraph, byte_target, [])
    |> IO.iodata_to_binary()
  end

  defp do_benchmark_corpus(paragraph, byte_target, acc) do
    current_size = acc |> Enum.reverse() |> IO.iodata_length()

    if current_size >= byte_target do
      acc
    else
      do_benchmark_corpus(paragraph, byte_target, [paragraph | acc])
    end
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

  defp assert_stream_matches_oneshot(repo, text, chunks, opts \\ []) do
    {:ok, iree_tokenizer} = Tokenizer.from_pretrained(repo, opts)
    {:ok, hf_tokenizer} = HFTokenizer.from_pretrained(repo)

    {:ok, iree_encoding} = Tokenizer.encode(iree_tokenizer, text, add_special_tokens: false)
    {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, text, add_special_tokens: false)

    assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)

    {:ok, stream} =
      EncodeStream.new(iree_tokenizer, add_special_tokens: false, max_chunk_bytes: @chunk_bytes)

    prefix_ids =
      Enum.flat_map(chunks, fn chunk ->
        {:ok, ids} = EncodeStream.feed(stream, chunk)
        ids
      end)

    assert {:ok, suffix_ids} = EncodeStream.finalize(stream)
    assert prefix_ids ++ suffix_ids == iree_encoding.ids
  end
end
