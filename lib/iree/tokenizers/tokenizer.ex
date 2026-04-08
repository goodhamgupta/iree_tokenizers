defmodule IREE.Tokenizers.Tokenizer do
  @moduledoc """
  Core tokenizer API.
  """

  alias IREE.Tokenizers.{Encoding, HTTPClient}

  defstruct [:resource]

  @type t :: %__MODULE__{resource: reference()}

  @type encode_input :: binary()
  @type result(value) :: {:ok, value} | {:error, {atom(), binary()}}

  @spec from_buffer(binary()) :: result(t())
  def from_buffer(buffer) when is_binary(buffer) do
    IREE.Tokenizers.Native.tokenizer_from_buffer(buffer)
  end

  def from_buffer(_buffer), do: {:error, {:invalid_argument, "expected a binary tokenizer.json buffer"}}

  @spec from_file(Path.t()) :: result(t())
  def from_file(path) when is_binary(path) do
    with {:ok, contents} <- File.read(path) do
      from_buffer(contents)
    else
      {:error, reason} ->
        {:error, {:not_found, "failed to read #{path}: #{Exception.message(File.Error.exception(action: \"read\", path: path, reason: reason))}"}}
    end
  end

  def from_file(_path), do: {:error, {:invalid_argument, "expected a file path"}}

  @spec from_pretrained(binary(), keyword()) :: result(t())
  def from_pretrained(repo_id, opts \\ []) when is_binary(repo_id) do
    opts =
      Keyword.validate!(opts,
        revision: "main",
        use_cache: true,
        cache_dir: :filename.basedir(:user_cache, "iree_tokenizers"),
        http_client: {HTTPClient, []},
        token: nil
      )

    {http_client, http_opts} = opts[:http_client]
    {:ok, app_version} = :application.get_key(:iree_tokenizers, :vsn)
    app_version = List.to_string(app_version)

    url = "https://huggingface.co/#{repo_id}/resolve/#{opts[:revision]}/tokenizer.json"
    headers =
      [{"user-agent", "iree_tokenizers/#{app_version}"}]
      |> maybe_put_auth(opts[:token])

    cache_dir = opts[:cache_dir]
    cache_key = cache_key(url, opts[:revision], repo_id)

    with {:ok, response} <- maybe_cached_get(http_client, http_opts, url, headers, cache_dir, cache_key, opts[:use_cache]) do
      from_buffer(response.body)
    end
  end

  def from_pretrained(_repo_id, _opts), do: {:error, {:invalid_argument, "expected a Hugging Face repo id"}}

  @spec encode(t(), encode_input(), keyword()) :: result(Encoding.t())
  def encode(%__MODULE__{} = tokenizer, input, opts \\ []) when is_binary(input) do
    opts = Keyword.validate!(opts, add_special_tokens: true, track_offsets: false)
    IREE.Tokenizers.Native.tokenizer_encode(tokenizer, input, opts)
  end

  def encode(%__MODULE__{}, {_left, _right}, _opts),
    do: {:error, {:invalid_argument, "pair sequence inputs are not supported in v1"}}

  def encode(%__MODULE__{}, _input, _opts),
    do: {:error, {:invalid_argument, "expected a binary input"}}

  @spec encode_batch(t(), [encode_input()], keyword()) :: result([Encoding.t()])
  def encode_batch(%__MODULE__{} = tokenizer, inputs, opts \\ []) when is_list(inputs) do
    opts = Keyword.validate!(opts, add_special_tokens: true, track_offsets: false)

    case Enum.find(inputs, &(not is_binary(&1))) do
      nil -> IREE.Tokenizers.Native.tokenizer_encode_batch(tokenizer, inputs, opts)
      {_left, _right} -> {:error, {:invalid_argument, "pair sequence inputs are not supported in v1"}}
      _ -> {:error, {:invalid_argument, "expected a list of binary inputs"}}
    end
  end

  def encode_batch(%__MODULE__{}, _inputs, _opts),
    do: {:error, {:invalid_argument, "expected a list of binary inputs"}}

  @spec decode(t(), [integer()], keyword()) :: result(binary())
  def decode(%__MODULE__{} = tokenizer, ids, opts \\ []) when is_list(ids) do
    opts = Keyword.validate!(opts, skip_special_tokens: true)
    IREE.Tokenizers.Native.tokenizer_decode(tokenizer, ids, opts)
  end

  def decode(%__MODULE__{}, _ids, _opts),
    do: {:error, {:invalid_argument, "expected a list of token ids"}}

  @spec decode_batch(t(), [[integer()]], keyword()) :: result([binary()])
  def decode_batch(%__MODULE__{} = tokenizer, batch_ids, opts \\ []) when is_list(batch_ids) do
    opts = Keyword.validate!(opts, skip_special_tokens: true)

    case Enum.find(batch_ids, &(not is_list(&1))) do
      nil -> IREE.Tokenizers.Native.tokenizer_decode_batch(tokenizer, batch_ids, opts)
      _ -> {:error, {:invalid_argument, "expected a list of token id lists"}}
    end
  end

  def decode_batch(%__MODULE__{}, _batch_ids, _opts),
    do: {:error, {:invalid_argument, "expected a list of token id lists"}}

  @spec vocab_size(t()) :: non_neg_integer()
  def vocab_size(%__MODULE__{} = tokenizer), do: IREE.Tokenizers.Native.tokenizer_vocab_size(tokenizer)

  @spec model_type(t()) :: binary()
  def model_type(%__MODULE__{} = tokenizer), do: IREE.Tokenizers.Native.tokenizer_model_type(tokenizer)

  @spec token_to_id(t(), binary()) :: integer() | nil
  def token_to_id(%__MODULE__{} = tokenizer, token) when is_binary(token),
    do: IREE.Tokenizers.Native.tokenizer_token_to_id(tokenizer, token)

  def token_to_id(%__MODULE__{}, _token), do: nil

  @spec id_to_token(t(), integer()) :: binary() | nil
  def id_to_token(%__MODULE__{} = tokenizer, id) when is_integer(id),
    do: IREE.Tokenizers.Native.tokenizer_id_to_token(tokenizer, id)

  def id_to_token(%__MODULE__{}, _id), do: nil

  for {fun, native} <- [
        {:bos_token_id, :tokenizer_bos_token_id},
        {:eos_token_id, :tokenizer_eos_token_id},
        {:unk_token_id, :tokenizer_unk_token_id},
        {:pad_token_id, :tokenizer_pad_token_id},
        {:sep_token_id, :tokenizer_sep_token_id},
        {:cls_token_id, :tokenizer_cls_token_id},
        {:mask_token_id, :tokenizer_mask_token_id}
      ] do
    @spec unquote(fun)(t()) :: integer() | nil
    def unquote(fun)(%__MODULE__{} = tokenizer), do: apply(IREE.Tokenizers.Native, unquote(native), [tokenizer])
  end

  defp maybe_put_auth(headers, nil), do: headers
  defp maybe_put_auth(headers, token), do: [{"authorization", "Bearer #{token}"} | headers]

  defp maybe_cached_get(http_client, http_opts, url, headers, cache_dir, cache_key, use_cache) do
    cache_path = Path.join(cache_dir, cache_key)

    if use_cache and File.exists?(cache_path) do
      {:ok, %{status: 200, headers: [], body: File.read!(cache_path)}}
    else
      File.mkdir_p!(cache_dir)

      request_opts =
        http_opts
        |> Keyword.put(:url, url)
        |> Keyword.put(:headers, headers)
        |> Keyword.put(:method, :get)

      with {:ok, %{status: status} = response} when status in 200..299 <- http_client.request(request_opts) do
        File.write!(cache_path, response.body)
        {:ok, response}
      else
        {:ok, %{status: status, body: body}} ->
          {:error, {:invalid_argument, "download failed with status #{status}: #{inspect(body)}"}}

        {:error, reason} ->
          {:error, {:internal, "download failed: #{inspect(reason)}"}}
      end
    end
  end

  defp cache_key(url, revision, repo_id) do
    [repo_id, revision, url]
    |> Enum.join(":")
    |> :erlang.md5()
    |> Base.encode16(case: :lower)
  end
end

defimpl Inspect, for: IREE.Tokenizers.Tokenizer do
  import Inspect.Algebra

  def inspect(tokenizer, opts) do
    attrs = [
      vocab_size: IREE.Tokenizers.Tokenizer.vocab_size(tokenizer),
      model_type: IREE.Tokenizers.Tokenizer.model_type(tokenizer)
    ]

    concat(["#IREE.Tokenizers.Tokenizer<", to_doc(attrs, opts), ">"])
  end
end
