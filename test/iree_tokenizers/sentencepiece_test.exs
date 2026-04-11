defmodule IREETokenizers.SentencePieceTest do
  use ExUnit.Case, async: false

  alias IREE.Tokenizers.Tokenizer

  defmodule MockHTTPClient do
    def request(opts) do
      agent = Keyword.fetch!(opts, :agent)
      method = Keyword.fetch!(opts, :method)
      url = Keyword.fetch!(opts, :url)

      Agent.get_and_update(agent, fn state ->
        requests = [{method, url, opts} | state.requests]
        response = Map.get(state.responses, {method, url}, {:error, :unset})
        {response, %{state | requests: requests}}
      end)
    end
  end

  setup do
    {:ok, agent} =
      Agent.start_link(fn ->
        %{
          requests: [],
          responses: %{}
        }
      end)

    {:ok, agent: agent}
  end

  test "loads sentencepiece model from file and buffer" do
    path = fixture_path("test_sentencepiece.model")
    buffer = File.read!(path)

    assert {:ok, tokenizer} = Tokenizer.from_file(path)

    assert {:ok, tokenizer_from_buffer} =
             Tokenizer.from_buffer(buffer, format: :sentencepiece_model)

    assert Tokenizer.model_type(tokenizer) == "Unigram"
    assert Tokenizer.model_type(tokenizer_from_buffer) == "Unigram"
    assert Tokenizer.vocab_size(tokenizer) == 1000
    assert Tokenizer.token_to_id(tokenizer, "<unk>") == 0
    assert Tokenizer.token_to_id(tokenizer, "<s>") == 1
    assert Tokenizer.token_to_id(tokenizer, "</s>") == 2
  end

  test "matches official sentencepiece test fixture behavior" do
    {:ok, tokenizer} = Tokenizer.from_file(fixture_path("test_sentencepiece.model"))

    assert {:ok, encoding} =
             Tokenizer.encode(tokenizer, " hello  world ",
               add_special_tokens: false,
               track_offsets: true
             )

    assert encoding.tokens == ["▁he", "ll", "o", "▁world"]
    assert encoding.ids == [39, 88, 21, 887]
    assert encoding.offsets == [{0, 5}, {5, 7}, {7, 8}, {8, 16}]

    assert {:ok, "hello world"} =
             Tokenizer.decode(tokenizer, encoding.ids, skip_special_tokens: false)

    assert {:ok, roundtrip} =
             Tokenizer.encode(tokenizer, "I saw a girl with a telescope.",
               add_special_tokens: false
             )

    assert roundtrip.ids == [9, 459, 11, 939, 44, 11, 4, 142, 82, 8, 28, 21, 132, 6]

    assert {:ok, "I saw a girl with a telescope."} =
             Tokenizer.decode(tokenizer, roundtrip.ids, skip_special_tokens: false)
  end

  test "from_pretrained falls back from tokenizer.model to spiece.model", %{agent: agent} do
    body = File.read!(fixture_path("test_sentencepiece.model"))

    tokenizer_model_url = "/owner/repo/resolve/main/tokenizer.model"
    spiece_model_url = "/owner/repo/resolve/main/spiece.model"

    Agent.update(agent, fn _ ->
      %{
        requests: [],
        responses: %{
          {:head, tokenizer_model_url} => {:ok, %{status: 404, headers: [], body: ""}},
          {:get, tokenizer_model_url} => {:ok, %{status: 404, headers: [], body: "missing"}},
          {:head, spiece_model_url} =>
            {:ok, %{status: 200, headers: [{"etag", "etag-sp"}], body: ""}},
          {:get, spiece_model_url} => {:ok, %{status: 200, headers: [], body: body}}
        }
      }
    end)

    cache_dir =
      Path.join(System.tmp_dir!(), "iree-tokenizers-#{System.unique_integer([:positive])}")

    File.rm_rf!(cache_dir)

    assert {:ok, tokenizer} =
             Tokenizer.from_pretrained("owner/repo",
               cache_dir: cache_dir,
               http_client: {MockHTTPClient, agent: agent},
               format: :sentencepiece_model
             )

    assert Tokenizer.model_type(tokenizer) == "Unigram"

    requests =
      Agent.get(agent, fn state ->
        Enum.reverse(state.requests)
      end)

    assert Enum.map(requests, fn {method, url, _opts} -> {method, url} end) == [
             {:head, tokenizer_model_url},
             {:get, tokenizer_model_url},
             {:head, spiece_model_url},
             {:get, spiece_model_url}
           ]
  end

  defp fixture_path(name) do
    Path.join([__DIR__, "..", "fixtures", name])
  end
end
