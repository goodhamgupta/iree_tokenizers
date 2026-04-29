defmodule IREETokenizers.TokenizerTest do
  use ExUnit.Case, async: false

  alias IREE.Tokenizers.{DecodeStream, EncodeStream, Encoding, Tokenizer}

  defmodule MockHTTPClient do
    def request(opts) do
      agent = Keyword.fetch!(opts, :agent)

      Agent.get_and_update(agent, fn state ->
        state = Map.update(state, :requests, [opts], &[opts | &1])

        method = Keyword.fetch!(opts, :method)

        response =
          case {method, state[:head_response], state[:get_response]} do
            {:head, {:ok, response}, _} -> {:ok, response}
            {:head, {:error, reason}, _} -> {:error, reason}
            {:get, _, {:ok, response}} -> {:ok, response}
            {:get, _, {:error, reason}} -> {:error, reason}
          end

        {response, state}
      end)
    end
  end

  defmodule RoutedMockHTTPClient do
    @moduledoc false
    # A URL-routed mock that returns a 404 (HEAD + GET) for any path not
    # explicitly registered, and the registered response otherwise. The
    # routing table is a list of `{url_pattern, response}` pairs; the first
    # pattern whose `url` is a suffix match wins, which keeps route strings
    # short (`"/tokenizer/tokenizer.json"`) while still matching the full
    # `/<repo>/resolve/<rev>/<path>` URL the loader sends.
    def request(opts) do
      agent = Keyword.fetch!(opts, :agent)
      url = Keyword.fetch!(opts, :url)
      method = Keyword.fetch!(opts, :method)

      Agent.get_and_update(agent, fn state ->
        state =
          Map.update(state, :requests, [{method, url}], &[{method, url} | &1])

        route =
          Enum.find(state.routes, fn {pattern, _response} ->
            String.ends_with?(url, pattern)
          end)

        response =
          case route do
            {_pattern, response} -> {:ok, response}
            nil -> {:ok, %{status: 404, headers: [], body: "not found"}}
          end

        {response, state}
      end)
    end
  end

  setup do
    {:ok, agent} =
      Agent.start_link(fn ->
        %{
          requests: [],
          head_response: {:error, :unset},
          get_response: {:error, :unset}
        }
      end)

    {:ok, agent: agent}
  end

  test "loads from buffer and file for bpe fixture" do
    buffer = File.read!(fixture_path("bpe_bytelevel_minimal.json"))

    assert {:ok, tokenizer} = Tokenizer.from_buffer(buffer)

    assert {:ok, tokenizer_from_file} =
             Tokenizer.from_file(fixture_path("bpe_bytelevel_minimal.json"))

    assert Tokenizer.vocab_size(tokenizer) == 112
    assert Tokenizer.model_type(tokenizer) == "BPE"
    assert Tokenizer.vocab_size(tokenizer_from_file) == 112
  end

  test "exposes supported tiktoken encodings" do
    assert Tokenizer.supported_tiktoken_encodings() == [
             "cl100k_base",
             "o200k_base",
             "o200k_harmony",
             "r50k_base",
             "gpt2",
             "p50k_base",
             "p50k_edit"
           ]
  end

  test "maps known model names to tiktoken encodings" do
    assert Tokenizer.tiktoken_encoding_for_model("gpt-4o") == "o200k_base"
    assert Tokenizer.tiktoken_encoding_for_model("gpt-4-0613") == "cl100k_base"
    assert Tokenizer.tiktoken_encoding_for_model("gpt-oss-120b") == "o200k_harmony"
    assert Tokenizer.tiktoken_encoding_for_model("gpt2") == "gpt2"
    assert Tokenizer.tiktoken_encoding_for_model("unknown-model") == nil
  end

  test "loads tiktoken from buffer and file and roundtrips ASCII" do
    buffer = minimal_tiktoken_fixture()
    tmp_path = temp_path("gpt2.tiktoken")
    on_exit(fn -> File.rm_rf(tmp_path) end)
    File.write!(tmp_path, buffer)

    assert {:ok, tokenizer} =
             Tokenizer.from_buffer(buffer, format: :tiktoken, tiktoken_encoding: "gpt2")

    assert {:ok, tokenizer_from_file} =
             Tokenizer.from_file(tmp_path, format: :tiktoken, tiktoken_encoding: "gpt2")

    assert {:ok, tokenizer_from_inferred_file} =
             Tokenizer.from_file(tmp_path, format: :tiktoken)

    assert Tokenizer.model_type(tokenizer) == "BPE"
    assert Tokenizer.model_type(tokenizer_from_file) == "BPE"
    assert Tokenizer.model_type(tokenizer_from_inferred_file) == "BPE"
    assert Tokenizer.vocab_size(tokenizer) >= 256

    assert {:ok, %Encoding{ids: [72, 101, 108, 108, 111]}} =
             Tokenizer.encode(tokenizer, "Hello", add_special_tokens: false)

    assert {:ok, "Hello"} = Tokenizer.decode(tokenizer, [72, 101, 108, 108, 111])

    assert {:ok, [%Encoding{ids: [72, 105]}, %Encoding{ids: [33]}]} =
             Tokenizer.encode_batch(tokenizer, ["Hi", "!"], add_special_tokens: false)

    assert Tokenizer.token_to_id(tokenizer, "H") == 72
    assert Tokenizer.id_to_token(tokenizer, 72) == "H"

    assert {:ok, stream} = EncodeStream.new(tokenizer, add_special_tokens: false)
    assert {:ok, chunk1} = EncodeStream.feed(stream, "He")
    assert {:ok, chunk2} = EncodeStream.feed(stream, "llo")
    assert {:ok, final_ids} = EncodeStream.finalize(stream)

    assert chunk1 ++ chunk2 ++ final_ids == [72, 101, 108, 108, 111]
  end

  test "loads tiktoken from pretrained using openai model inference", %{agent: agent} do
    Agent.update(agent, fn _ ->
      %{
        requests: [],
        head_response: {:ok, %{status: 200, headers: [{"etag", "etag-tiktoken"}], body: ""}},
        get_response: {:ok, %{status: 200, headers: [], body: minimal_tiktoken_fixture()}}
      }
    end)

    cache_dir =
      Path.join(System.tmp_dir!(), "iree-tokenizers-#{System.unique_integer([:positive])}")

    File.rm_rf!(cache_dir)

    assert {:ok, tokenizer} =
             Tokenizer.from_pretrained("gpt2",
               cache_dir: cache_dir,
               http_client: {MockHTTPClient, agent: agent},
               format: :tiktoken
             )

    assert Tokenizer.model_type(tokenizer) == "BPE"

    requests =
      Agent.get(agent, fn state ->
        Enum.reverse(state.requests)
      end)

    assert Enum.at(requests, 0)[:base_url] == "https://openaipublic.blob.core.windows.net"
    assert Enum.at(requests, 0)[:url] == "/encodings/r50k_base.tiktoken"
    assert Enum.at(requests, 1)[:base_url] == "https://openaipublic.blob.core.windows.net"
    assert Enum.at(requests, 1)[:url] == "/encodings/r50k_base.tiktoken"
  end

  test "loads tiktoken from pretrained using openai slash encoding shorthand", %{agent: agent} do
    Agent.update(agent, fn _ ->
      %{
        requests: [],
        head_response: {:ok, %{status: 200, headers: [{"etag", "etag-cl100k"}], body: ""}},
        get_response: {:ok, %{status: 200, headers: [], body: minimal_tiktoken_fixture()}}
      }
    end)

    cache_dir =
      Path.join(System.tmp_dir!(), "iree-tokenizers-#{System.unique_integer([:positive])}")

    File.rm_rf!(cache_dir)

    assert {:ok, tokenizer} =
             Tokenizer.from_pretrained("openai/cl100k_base",
               cache_dir: cache_dir,
               http_client: {MockHTTPClient, agent: agent},
               format: :tiktoken
             )

    assert Tokenizer.model_type(tokenizer) == "BPE"

    requests =
      Agent.get(agent, fn state ->
        Enum.reverse(state.requests)
      end)

    assert Enum.at(requests, 0)[:base_url] == "https://openaipublic.blob.core.windows.net"
    assert Enum.at(requests, 0)[:url] == "/encodings/cl100k_base.tiktoken"
    assert Enum.at(requests, 1)[:base_url] == "https://openaipublic.blob.core.windows.net"
    assert Enum.at(requests, 1)[:url] == "/encodings/cl100k_base.tiktoken"
  end

  test "uses hugging face paths for custom tiktoken repos", %{agent: agent} do
    Agent.update(agent, fn _ ->
      %{
        requests: [],
        head_response: {:ok, %{status: 200, headers: [{"etag", "etag-custom"}], body: ""}},
        get_response: {:ok, %{status: 200, headers: [], body: minimal_tiktoken_fixture()}}
      }
    end)

    cache_dir =
      Path.join(System.tmp_dir!(), "iree-tokenizers-#{System.unique_integer([:positive])}")

    File.rm_rf!(cache_dir)

    assert {:ok, tokenizer} =
             Tokenizer.from_pretrained("owner/custom-tiktoken",
               cache_dir: cache_dir,
               http_client: {MockHTTPClient, agent: agent},
               format: :tiktoken,
               filename: "custom.tiktoken",
               tiktoken_encoding: "cl100k_base"
             )

    assert Tokenizer.model_type(tokenizer) == "BPE"

    requests =
      Agent.get(agent, fn state ->
        Enum.reverse(state.requests)
      end)

    assert Enum.at(requests, 0)[:base_url] == "https://huggingface.co"
    assert Enum.at(requests, 0)[:url] == "/owner/custom-tiktoken/resolve/main/custom.tiktoken"
    assert Enum.at(requests, 1)[:base_url] == "https://huggingface.co"
    assert Enum.at(requests, 1)[:url] == "/owner/custom-tiktoken/resolve/main/custom.tiktoken"
  end

  test "validates required tiktoken options" do
    assert {:error, {:invalid_argument, message}} =
             Tokenizer.from_buffer(minimal_tiktoken_fixture(), format: :tiktoken)

    assert message =~ "could not infer a tiktoken encoding"

    assert {:error, {:invalid_argument, message}} =
             Tokenizer.from_buffer(minimal_tiktoken_fixture(),
               format: :tiktoken,
               tiktoken_encoding: "unknown"
             )

    assert message =~ "unsupported tiktoken encoding"

    assert {:error, {:invalid_argument, message}} =
             Tokenizer.from_pretrained("owner/custom", format: :tiktoken)

    assert message =~ "could not infer a tiktoken encoding"
  end

  test "encodes, decodes, batches, and exposes offsets for bpe" do
    tokenizer = load_fixture!("bpe_bytelevel_minimal.json")

    assert {:ok, %Encoding{} = encoding} =
             Tokenizer.encode(tokenizer, "Hello world",
               add_special_tokens: false,
               track_offsets: true
             )

    assert encoding.ids == [39, 68, 105, 110]
    assert encoding.type_ids == [0, 0, 0, 0]
    assert encoding.offsets == [{0, 1}, {1, 2}, {2, 5}, {5, 11}]

    assert {:ok, "Hello world"} =
             Tokenizer.decode(tokenizer, encoding.ids, skip_special_tokens: false)

    assert {:ok, [left, right]} =
             Tokenizer.encode_batch(tokenizer, ["Hello", "world"], add_special_tokens: false)

    assert left.ids == [39, 68, 105]
    assert right.ids == [86, 108]

    assert {:ok, ["Hello", "world"]} =
             Tokenizer.decode_batch(
               tokenizer,
               [left.ids, right.ids],
               skip_special_tokens: false
             )
  end

  test "supports token lookup helpers" do
    tokenizer = load_fixture!("bpe_bytelevel_minimal.json")

    assert Tokenizer.token_to_id(tokenizer, "hello") == 109
    assert Tokenizer.id_to_token(tokenizer, 109) == "hello"
    assert Tokenizer.id_to_token(tokenizer, -1) == nil
  end

  test "loads and roundtrips wordpiece fixture" do
    tokenizer = load_fixture!("minimal_wordpiece.json")

    assert Tokenizer.model_type(tokenizer) == "WordPiece"
    assert Tokenizer.unk_token_id(tokenizer) == 0

    assert {:ok, %Encoding{ids: [1, 2]}} =
             Tokenizer.encode(tokenizer, "hello world", add_special_tokens: false)

    assert {:ok, "hello world"} = Tokenizer.decode(tokenizer, [1, 2], skip_special_tokens: false)
  end

  test "auto-applies tokenizer.json padding and truncation config" do
    tokenizer = load_fixture!("minimal_wordpiece_padded.json")

    assert {:ok, %Encoding{} = encoding} =
             Tokenizer.encode(tokenizer, "hello", add_special_tokens: false)

    assert encoding.ids == [2, 0, 0, 0]
    assert encoding.type_ids == [0, 0, 0, 0]
    assert encoding.offsets == nil
    assert encoding.attention_mask == [1, 0, 0, 0]
    assert encoding.special_tokens_mask == [0, 1, 1, 1]
    assert encoding.tokens == ["hello", "[PAD]", "[PAD]", "[PAD]"]

    assert {:ok, [%Encoding{} = left, %Encoding{} = right]} =
             Tokenizer.encode_batch(tokenizer, ["hello", "hello world token more text"],
               add_special_tokens: false
             )

    assert left.ids == [2, 0, 0, 0]
    assert left.offsets == nil
    assert right.ids == [2, 3, 4, 6]
    assert right.offsets == nil

    assert {:ok, stream} = EncodeStream.new(tokenizer, add_special_tokens: false)
    assert {:ok, []} = EncodeStream.feed(stream, "hello ")
    assert {:ok, []} = EncodeStream.feed(stream, "world token more text")
    assert {:ok, streamed_ids} = EncodeStream.finalize(stream)
    assert streamed_ids == [2, 3, 4, 6]

    assert {:error, {:invalid_argument, "stream already finalized"}} =
             EncodeStream.finalize(stream)

    assert {:error, {:invalid_argument, "stream already finalized"}} =
             EncodeStream.feed(stream, "more")
  end

  test "honors tokenizer.json left truncation direction" do
    tokenizer = load_fixture!("minimal_wordpiece_left_truncation.json")

    assert {:ok, %Encoding{} = encoding} =
             Tokenizer.encode(tokenizer, "hello world token more text", add_special_tokens: false)

    assert encoding.ids == [3, 4, 6, 7]
    assert encoding.offsets == nil
  end

  test "loads unigram fixture and exposes metadata" do
    tokenizer = load_fixture!("minimal_unigram.json")

    assert Tokenizer.model_type(tokenizer) == "Unigram"
    assert Tokenizer.unk_token_id(tokenizer) == 0
    assert Tokenizer.vocab_size(tokenizer) == 3
  end

  test "streaming encode and decode match one-shot encode" do
    tokenizer = load_fixture!("bpe_bytelevel_minimal.json")

    assert {:ok, stream} = EncodeStream.new(tokenizer, add_special_tokens: false)
    assert {:ok, ids1} = EncodeStream.feed(stream, "Hello ")
    assert {:ok, ids2} = EncodeStream.feed(stream, "world")
    assert {:ok, ids3} = EncodeStream.finalize(stream)

    streamed_ids = ids1 ++ ids2 ++ ids3

    assert {:ok, %Encoding{ids: oneshot_ids}} =
             Tokenizer.encode(tokenizer, "Hello world", add_special_tokens: false)

    assert streamed_ids == oneshot_ids

    assert {:ok, decode_stream} = DecodeStream.new(tokenizer, skip_special_tokens: false)
    assert {:ok, text1} = DecodeStream.feed(decode_stream, [39, 68])
    assert {:ok, text2} = DecodeStream.feed(decode_stream, [105, 110])
    assert {:ok, text3} = DecodeStream.finalize(decode_stream)

    assert text1 <> text2 <> text3 == "Hello world"

    assert {:error, {:invalid_argument, "stream already finalized"}} =
             DecodeStream.finalize(decode_stream)
  end

  test "rejects unsupported pair input" do
    tokenizer = load_fixture!("bpe_bytelevel_minimal.json")

    assert {:error, {:invalid_argument, "pair sequence inputs are not supported in v1"}} =
             Tokenizer.encode(tokenizer, {"hello", "world"})
  end

  test "downloads and caches pretrained tokenizers using HEAD etag", %{agent: agent} do
    body = File.read!(fixture_path("bpe_bytelevel_minimal.json"))

    Agent.update(agent, fn _ ->
      %{
        requests: [],
        head_response: {:ok, %{status: 200, headers: [{"etag", "etag-123"}], body: ""}},
        get_response: {:ok, %{status: 200, headers: [], body: body}}
      }
    end)

    cache_dir =
      Path.join(System.tmp_dir!(), "iree-tokenizers-#{System.unique_integer([:positive])}")

    File.rm_rf!(cache_dir)

    assert {:ok, tokenizer} =
             Tokenizer.from_pretrained("owner/repo",
               cache_dir: cache_dir,
               http_client: {MockHTTPClient, agent: agent},
               token: "secret-token"
             )

    assert Tokenizer.vocab_size(tokenizer) == 112

    assert {:ok, tokenizer2} =
             Tokenizer.from_pretrained("owner/repo",
               cache_dir: cache_dir,
               http_client: {MockHTTPClient, agent: agent}
             )

    assert Tokenizer.vocab_size(tokenizer2) == 112

    requests =
      Agent.get(agent, fn state ->
        Enum.reverse(state.requests)
      end)

    assert length(requests) == 3
    assert Enum.at(requests, 0)[:method] == :head
    assert Enum.at(requests, 1)[:method] == :get
    assert Enum.at(requests, 2)[:method] == :head
    assert {"authorization", "Bearer secret-token"} in Enum.at(requests, 1)[:headers]
  end

  test "falls back to get when head fails", %{agent: agent} do
    body = File.read!(fixture_path("minimal_wordpiece.json"))

    Agent.update(agent, fn _ ->
      %{
        requests: [],
        head_response: {:error, :timeout},
        get_response: {:ok, %{status: 200, headers: [], body: body}}
      }
    end)

    cache_dir =
      Path.join(System.tmp_dir!(), "iree-tokenizers-#{System.unique_integer([:positive])}")

    File.rm_rf!(cache_dir)

    assert {:ok, tokenizer} =
             Tokenizer.from_pretrained("owner/repo",
               cache_dir: cache_dir,
               http_client: {MockHTTPClient, agent: agent}
             )

    assert Tokenizer.model_type(tokenizer) == "WordPiece"
  end

  test "maps a 404 download to not_found", %{agent: agent} do
    Agent.update(agent, fn _ ->
      %{
        requests: [],
        head_response: {:ok, %{status: 404, headers: [], body: "missing"}},
        get_response: {:ok, %{status: 404, headers: [], body: "missing"}}
      }
    end)

    cache_dir =
      Path.join(System.tmp_dir!(), "iree-tokenizers-#{System.unique_integer([:positive])}")

    File.rm_rf!(cache_dir)

    assert {:error, {:not_found, "resource not found"}} =
             Tokenizer.from_pretrained("owner/missing",
               cache_dir: cache_dir,
               http_client: {MockHTTPClient, agent: agent}
             )
  end

  test "maps a 401 without auth to not_found-or-auth-needed", %{agent: agent} do
    Agent.update(agent, fn _ ->
      %{
        requests: [],
        head_response: {:ok, %{status: 401, headers: [], body: "missing"}},
        get_response: {:ok, %{status: 401, headers: [], body: "missing"}}
      }
    end)

    cache_dir =
      Path.join(System.tmp_dir!(), "iree-tokenizers-#{System.unique_integer([:positive])}")

    File.rm_rf!(cache_dir)

    assert {:error, {:not_found, "resource not found or requires authentication"}} =
             Tokenizer.from_pretrained("owner/missing",
               cache_dir: cache_dir,
               http_client: {MockHTTPClient, agent: agent}
             )
  end

  test "maps a 403 download to permission_denied", %{agent: agent} do
    Agent.update(agent, fn _ ->
      %{
        requests: [],
        head_response: {:ok, %{status: 403, headers: [], body: "forbidden"}},
        get_response: {:ok, %{status: 403, headers: [], body: "forbidden"}}
      }
    end)

    cache_dir =
      Path.join(System.tmp_dir!(), "iree-tokenizers-#{System.unique_integer([:positive])}")

    File.rm_rf!(cache_dir)

    assert {:error, {:permission_denied, "access denied"}} =
             Tokenizer.from_pretrained("owner/private",
               cache_dir: cache_dir,
               http_client: {MockHTTPClient, agent: agent}
             )
  end

  test "maps a 401 with auth to permission_denied", %{agent: agent} do
    Agent.update(agent, fn _ ->
      %{
        requests: [],
        head_response: {:ok, %{status: 401, headers: [], body: "forbidden"}},
        get_response: {:ok, %{status: 401, headers: [], body: "forbidden"}}
      }
    end)

    cache_dir =
      Path.join(System.tmp_dir!(), "iree-tokenizers-#{System.unique_integer([:positive])}")

    File.rm_rf!(cache_dir)

    assert {:error, {:permission_denied, "access denied"}} =
             Tokenizer.from_pretrained("owner/private",
               cache_dir: cache_dir,
               http_client: {MockHTTPClient, agent: agent},
               token: "secret"
             )
  end

  test "from_pretrained falls back to a diffusers-style tokenizer subfolder" do
    # Simulates `stabilityai/stable-diffusion-xl-base-1.0`, which does not
    # carry a tokenizer.json at the repo root. The loader must probe the
    # root first, fail with 404, then retry under `tokenizer/`.
    body = File.read!(fixture_path("bpe_bytelevel_minimal.json"))

    {:ok, agent} =
      Agent.start_link(fn ->
        %{
          requests: [],
          routes: [
            {"/tokenizer/tokenizer.json",
             %{status: 200, headers: [{"etag", "etag-sub"}], body: body}}
          ]
        }
      end)

    cache_dir = fresh_cache_dir()

    assert {:ok, tokenizer} =
             Tokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
               cache_dir: cache_dir,
               http_client: {RoutedMockHTTPClient, agent: agent}
             )

    assert Tokenizer.vocab_size(tokenizer) == 112

    requests =
      Agent.get(agent, fn state -> Enum.reverse(state.requests) end)

    # Expect at least one probe at the root, then a successful probe/get
    # pair under the tokenizer/ subfolder.
    assert Enum.any?(requests, fn {_m, url} ->
             String.ends_with?(url, "/resolve/main/tokenizer.json")
           end)

    assert Enum.any?(requests, fn {_m, url} ->
             String.ends_with?(url, "/tokenizer/tokenizer.json")
           end)
  end

  test "from_pretrained falls back through the tokenizer_2 subfolder" do
    # Simulates an SDXL text-encoder-2 style repo where only `tokenizer_2/`
    # exists: the root, `tokenizer/`, and `tokenizer_2/` must all be probed
    # and the last one wins.
    body = File.read!(fixture_path("bpe_bytelevel_minimal.json"))

    {:ok, agent} =
      Agent.start_link(fn ->
        %{
          requests: [],
          routes: [
            {"/tokenizer_2/tokenizer.json",
             %{status: 200, headers: [{"etag", "etag-sub2"}], body: body}}
          ]
        }
      end)

    cache_dir = fresh_cache_dir()

    assert {:ok, tokenizer} =
             Tokenizer.from_pretrained("owner/sdxl-like",
               cache_dir: cache_dir,
               http_client: {RoutedMockHTTPClient, agent: agent}
             )

    assert Tokenizer.vocab_size(tokenizer) == 112
  end

  test "from_pretrained honors an explicit :subfolder and skips the fallback walk" do
    body = File.read!(fixture_path("bpe_bytelevel_minimal.json"))

    {:ok, agent} =
      Agent.start_link(fn ->
        %{
          requests: [],
          routes: [
            # Only the explicit subfolder is served; root and tokenizer_2
            # intentionally 404 so we can assert we did not probe them.
            {"/text_encoder/tokenizer.json",
             %{status: 200, headers: [{"etag", "etag-sub"}], body: body}}
          ]
        }
      end)

    cache_dir = fresh_cache_dir()

    assert {:ok, tokenizer} =
             Tokenizer.from_pretrained("owner/explicit-sub",
               cache_dir: cache_dir,
               http_client: {RoutedMockHTTPClient, agent: agent},
               subfolder: "text_encoder"
             )

    assert Tokenizer.vocab_size(tokenizer) == 112

    requests =
      Agent.get(agent, fn state -> Enum.reverse(state.requests) end)

    # With an explicit subfolder we should see exactly the text_encoder
    # URL — no attempts at the bare repo root or at tokenizer/.
    refute Enum.any?(requests, fn {_m, url} ->
             String.ends_with?(url, "/resolve/main/tokenizer.json")
           end)

    refute Enum.any?(requests, fn {_m, url} ->
             String.ends_with?(url, "/tokenizer/tokenizer.json")
           end)

    assert Enum.any?(requests, fn {_m, url} ->
             String.ends_with?(url, "/text_encoder/tokenizer.json")
           end)
  end

  test "from_pretrained still returns not_found when every subfolder 404s" do
    {:ok, agent} =
      Agent.start_link(fn ->
        %{requests: [], routes: []}
      end)

    cache_dir = fresh_cache_dir()

    assert {:error, {:not_found, "resource not found"}} =
             Tokenizer.from_pretrained("owner/nowhere",
               cache_dir: cache_dir,
               http_client: {RoutedMockHTTPClient, agent: agent}
             )
  end

  defp fresh_cache_dir do
    cache_dir =
      Path.join(System.tmp_dir!(), "iree-tokenizers-#{System.unique_integer([:positive])}")

    File.rm_rf!(cache_dir)
    cache_dir
  end

  defp load_fixture!(name) do
    name
    |> fixture_path()
    |> Tokenizer.from_file()
    |> case do
      {:ok, tokenizer} -> tokenizer
      {:error, reason} -> flunk("failed to load fixture #{name}: #{inspect(reason)}")
    end
  end

  defp fixture_path(name) do
    Path.join([__DIR__, "..", "fixtures", name])
  end

  defp minimal_tiktoken_fixture do
    0..255
    |> Enum.map_join("\n", fn id -> "#{Base.encode64(<<id>>)} #{id}" end)
    |> Kernel.<>("\n")
  end

  defp temp_path(name) do
    Path.join(System.tmp_dir!(), "iree-tokenizers-#{System.unique_integer([:positive])}-#{name}")
  end
end
