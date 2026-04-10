defmodule IREETokenizers.ModelTest do
  use ExUnit.Case, async: true

  alias IREE.Tokenizers.Model
  alias IREE.Tokenizers.Model.{BPE, Unigram, WordPiece}
  alias IREE.Tokenizers.Tokenizer

  test "builds a wordpiece model and initializes a tokenizer" do
    assert {:ok, %Model{} = model} =
             WordPiece.init(%{"[UNK]" => 0, "hello" => 1, "world" => 2})

    assert model.info["model_type"] == "WordPiece"

    assert {:ok, tokenizer} = Tokenizer.init(model)
    assert Tokenizer.get_model(tokenizer).info["model_type"] == "WordPiece"
    assert Tokenizer.get_vocab_size(tokenizer) == 3
  end

  test "builds a unigram model and initializes a tokenizer" do
    assert {:ok, %Model{} = model} =
             Unigram.init([{"<unk>", -10.0}, {"▁hello", -1.0}, {"▁world", -1.0}], unk_id: 0)

    assert {:ok, tokenizer} = Tokenizer.init(model)
    assert Tokenizer.model_type(tokenizer) == "Unigram"
  end

  test "builds a bpe model and initializes a tokenizer" do
    assert {:ok, %Model{} = model} =
             BPE.init(%{"a" => 0, "b" => 1, "ab" => 2}, [{"a", "b"}])

    assert {:ok, tokenizer} = Tokenizer.init(model)
    assert Tokenizer.model_type(tokenizer) == "BPE"
  end
end
