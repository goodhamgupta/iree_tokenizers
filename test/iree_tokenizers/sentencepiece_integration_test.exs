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
    text = "translate English to German: The house is wonderful."

    {:ok, iree_tokenizer} = IREETokenizer.from_pretrained(repo, format: :sentencepiece_model)
    {:ok, hf_tokenizer} = HFTokenizer.from_pretrained(repo)

    {:ok, iree_encoding} = IREETokenizer.encode(iree_tokenizer, text, add_special_tokens: false)
    {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, text, add_special_tokens: false)

    assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
    assert iree_encoding.type_ids == HFEncoding.get_type_ids(hf_encoding)
    assert iree_encoding.tokens == HFEncoding.get_tokens(hf_encoding)

    assert {:ok, ^text} =
             IREETokenizer.decode(iree_tokenizer, iree_encoding.ids, skip_special_tokens: false)
  end

  test "matches official tokenizers on llama sentencepiece bpe model" do
    repo = "hf-internal-testing/llama-tokenizer"
    text = "The quick brown fox jumps over the lazy dog."

    {:ok, iree_tokenizer} = IREETokenizer.from_pretrained(repo, format: :sentencepiece_model)
    {:ok, hf_tokenizer} = HFTokenizer.from_pretrained(repo)

    {:ok, iree_encoding} = IREETokenizer.encode(iree_tokenizer, text, add_special_tokens: false)
    {:ok, hf_encoding} = HFTokenizer.encode(hf_tokenizer, text, add_special_tokens: false)

    assert iree_encoding.ids == HFEncoding.get_ids(hf_encoding)
    assert iree_encoding.tokens == HFEncoding.get_tokens(hf_encoding)

    assert {:ok, ^text} =
             IREETokenizer.decode(iree_tokenizer, iree_encoding.ids, skip_special_tokens: false)
  end
end
