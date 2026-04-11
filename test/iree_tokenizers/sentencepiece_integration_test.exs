defmodule IREETokenizers.SentencePieceIntegrationTest do
  use ExUnit.Case, async: false

  alias IREE.Tokenizers.Tokenizer, as: IREETokenizer
  alias Tokenizers.Encoding, as: HFEncoding
  alias Tokenizers.Tokenizer, as: HFTokenizer

  @moduletag integration: true
  @moduletag skip:
               if(System.get_env("RUN_SENTENCEPIECE_INTEGRATION") in ["1", "true"],
                 do: false,
                 else:
                   "set RUN_SENTENCEPIECE_INTEGRATION=1 to run SentencePiece integration tests"
               )

  test "matches official tokenizers on google-t5/t5-small sentencepiece model" do
    repo = "google-t5/t5-small"

    texts = [
      "translate English to German: The house is wonderful.",
      " translate English to German: The house is wonderful.",
      "translate  English\n to German: The house is wonderful."
    ]

    {:ok, iree_tokenizer} = IREETokenizer.from_pretrained(repo, format: :sentencepiece_model)
    {:ok, hf_tokenizer} = HFTokenizer.from_pretrained(repo)

    Enum.each(texts, fn text ->
      {:ok, iree_encoding} = IREETokenizer.encode(iree_tokenizer, text, add_special_tokens: false)
      {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, text, add_special_tokens: false)

      assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
      assert iree_encoding.type_ids == HFEncoding.get_type_ids(hf_encoding)
      assert iree_encoding.tokens == HFEncoding.get_tokens(hf_encoding)

      assert {:ok, iree_decoded} =
               IREETokenizer.decode(iree_tokenizer, iree_encoding.ids, skip_special_tokens: false)

      assert {:ok, hf_decoded} =
               HFTokenizer.decode(hf_tokenizer, HFEncoding.get_ids(hf_encoding),
                 skip_special_tokens: false
               )

      assert iree_decoded == hf_decoded
    end)
  end

  test "matches official tokenizers on llama sentencepiece bpe model" do
    repo = "hf-internal-testing/llama-tokenizer"

    texts = [
      "The quick brown fox jumps over the lazy dog.",
      " The quick brown fox jumps over the lazy dog.",
      "The  quick brown fox\n jumps over the lazy dog."
    ]

    {:ok, iree_tokenizer_json} = IREETokenizer.from_pretrained(repo)

    {:ok, iree_tokenizer_model} =
      IREETokenizer.from_pretrained(repo, format: :sentencepiece_model)

    {:ok, hf_tokenizer} = HFTokenizer.from_pretrained(repo)

    Enum.each(texts, fn text ->
      {:ok, iree_json_encoding} =
        IREETokenizer.encode(iree_tokenizer_json, text, add_special_tokens: false)

      {:ok, iree_model_encoding} =
        IREETokenizer.encode(iree_tokenizer_model, text, add_special_tokens: false)

      {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, text, add_special_tokens: false)

      assert iree_json_encoding.ids == HFEncoding.get_ids(hf_encoding)
      assert iree_model_encoding.ids == HFEncoding.get_ids(hf_encoding)
      assert iree_json_encoding.ids == iree_model_encoding.ids
      assert iree_json_encoding.tokens == HFEncoding.get_tokens(hf_encoding)
      assert iree_model_encoding.tokens == HFEncoding.get_tokens(hf_encoding)

      assert {:ok, iree_json_decoded} =
               IREETokenizer.decode(iree_tokenizer_json, iree_json_encoding.ids,
                 skip_special_tokens: false
               )

      assert {:ok, iree_model_decoded} =
               IREETokenizer.decode(iree_tokenizer_model, iree_model_encoding.ids,
                 skip_special_tokens: false
               )

      assert {:ok, hf_decoded} =
               HFTokenizer.decode(hf_tokenizer, HFEncoding.get_ids(hf_encoding),
                 skip_special_tokens: false
               )

      assert iree_json_decoded == hf_decoded
      assert iree_model_decoded == hf_decoded
    end)
  end
end
