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

  def from_buffer(_buffer),
    do: {:error, {:invalid_argument, "expected a binary tokenizer.json buffer"}}

  @spec from_file(Path.t()) :: result(t())
  def from_file(path) when is_binary(path) do
    with {:ok, contents} <- File.read(path) do
      from_buffer(contents)
    else
      {:error, reason} ->
        error = File.Error.exception(action: "read", path: path, reason: reason)
        {:error, {:not_found, "failed to read #{path}: #{Exception.message(error)}"}}
    end
  end

  def from_file(_path), do: {:error, {:invalid_argument, "expected a file path"}}

  @spec from_pretrained(binary(), keyword()) :: result(t())
  def from_pretrained(repo_id, opts \\ [])

  def from_pretrained(repo_id, opts) when is_binary(repo_id) do
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

    headers =
      [{"user-agent", "iree_tokenizers/#{app_version}"}]
      |> maybe_put_auth(opts[:token])

    url = "/#{repo_id}/resolve/#{opts[:revision]}/tokenizer.json"

    http_opts =
      http_opts
      |> Keyword.put_new(:base_url, "https://huggingface.co")
      |> Keyword.put(:url, url)
      |> Keyword.put(:method, :get)
      |> Keyword.update(:headers, headers, fn existing -> existing ++ headers end)

    file_path_fun = fn etag ->
      Path.join(opts[:cache_dir], entry_filename(url, etag))
    end

    if opts[:use_cache] do
      case request(http_client, Keyword.put(http_opts, :method, :head)) do
        {:ok, response} ->
          etag = fetch_etag(response.headers)
          file_path = file_path_fun.(etag)

          if File.exists?(file_path) do
            from_file(file_path)
          else
            with {:ok, response} <- request(http_client, http_opts) do
              File.mkdir_p!(opts[:cache_dir])
              File.write!(file_path, response.body)
              from_file(file_path)
            end
          end

        {:error, _reason} ->
          with {:ok, response} <- request(http_client, http_opts) do
            etag = fetch_etag(response.headers)
            file_path = file_path_fun.(etag)

            File.mkdir_p!(opts[:cache_dir])
            File.write!(file_path, response.body)
            from_file(file_path)
          end
      end
    else
      with {:ok, response} <- request(http_client, http_opts) do
        etag = fetch_etag(response.headers)
        file_path = file_path_fun.(etag)

        File.mkdir_p!(opts[:cache_dir])
        File.write!(file_path, response.body)
        from_file(file_path)
      end
    end
  end

  def from_pretrained(_repo_id, _opts),
    do: {:error, {:invalid_argument, "expected a Hugging Face repo id"}}

  @spec encode(t(), encode_input(), keyword()) :: result(Encoding.t())
  def encode(tokenizer, input, opts \\ [])

  def encode(%__MODULE__{} = tokenizer, input, opts) when is_binary(input) do
    opts = Keyword.validate!(opts, add_special_tokens: true, track_offsets: false)
    IREE.Tokenizers.Native.tokenizer_encode(tokenizer, input, opts)
  end

  def encode(%__MODULE__{}, {_left, _right}, _opts),
    do: {:error, {:invalid_argument, "pair sequence inputs are not supported in v1"}}

  def encode(%__MODULE__{}, _input, _opts),
    do: {:error, {:invalid_argument, "expected a binary input"}}

  @spec encode_batch(t(), [encode_input()], keyword()) :: result([Encoding.t()])
  def encode_batch(tokenizer, inputs, opts \\ [])

  def encode_batch(%__MODULE__{} = tokenizer, inputs, opts) when is_list(inputs) do
    opts = Keyword.validate!(opts, add_special_tokens: true, track_offsets: false)

    case Enum.find(inputs, &(not is_binary(&1))) do
      nil ->
        IREE.Tokenizers.Native.tokenizer_encode_batch(tokenizer, inputs, opts)

      {_left, _right} ->
        {:error, {:invalid_argument, "pair sequence inputs are not supported in v1"}}

      _ ->
        {:error, {:invalid_argument, "expected a list of binary inputs"}}
    end
  end

  def encode_batch(%__MODULE__{}, _inputs, _opts),
    do: {:error, {:invalid_argument, "expected a list of binary inputs"}}

  @spec decode(t(), [integer()], keyword()) :: result(binary())
  def decode(tokenizer, ids, opts \\ [])

  def decode(%__MODULE__{} = tokenizer, ids, opts) when is_list(ids) do
    opts = Keyword.validate!(opts, skip_special_tokens: true)
    IREE.Tokenizers.Native.tokenizer_decode(tokenizer, ids, opts)
  end

  def decode(%__MODULE__{}, _ids, _opts),
    do: {:error, {:invalid_argument, "expected a list of token ids"}}

  @spec decode_batch(t(), [[integer()]], keyword()) :: result([binary()])
  def decode_batch(tokenizer, batch_ids, opts \\ [])

  def decode_batch(%__MODULE__{} = tokenizer, batch_ids, opts) when is_list(batch_ids) do
    opts = Keyword.validate!(opts, skip_special_tokens: true)

    case Enum.find(batch_ids, &(not is_list(&1))) do
      nil -> IREE.Tokenizers.Native.tokenizer_decode_batch(tokenizer, batch_ids, opts)
      _ -> {:error, {:invalid_argument, "expected a list of token id lists"}}
    end
  end

  def decode_batch(%__MODULE__{}, _batch_ids, _opts),
    do: {:error, {:invalid_argument, "expected a list of token id lists"}}

  @spec vocab_size(t()) :: non_neg_integer()
  def vocab_size(%__MODULE__{} = tokenizer),
    do: IREE.Tokenizers.Native.tokenizer_vocab_size(tokenizer)

  @spec model_type(t()) :: binary()
  def model_type(%__MODULE__{} = tokenizer),
    do: IREE.Tokenizers.Native.tokenizer_model_type(tokenizer)

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
    def unquote(fun)(%__MODULE__{} = tokenizer),
      do: apply(IREE.Tokenizers.Native, unquote(native), [tokenizer])
  end

  defp maybe_put_auth(headers, nil), do: headers
  defp maybe_put_auth(headers, token), do: [{"authorization", "Bearer #{token}"} | headers]

  defp fetch_etag(headers) do
    case List.keyfind(headers, "etag", 0) do
      {_, etag} -> etag
      nil -> "no-etag"
    end
  end

  defp request(http_client, http_opts) do
    case http_client.request(http_opts) do
      {:ok, response} ->
        case response.status do
          status when status in 200..299 ->
            {:ok, response}

          404 ->
            {:error, {:not_found, "resource not found"}}

          status when status in [401, 403] ->
            {:error, {:permission_denied, "access denied"}}

          other ->
            {:error,
             {:invalid_argument,
              "download of pretrained file failed with status #{other}. Response: #{inspect(response.body)}"}}
        end

      {:error, reason} ->
        {:error, {:internal, "download failed: #{inspect(reason)}"}}
    end
  end

  defp entry_filename(url, etag) do
    encode_url(url) <> "." <> encode_etag(etag)
  end

  defp encode_url(url) do
    url
    |> :erlang.md5()
    |> Base.encode32(case: :lower, padding: false)
  end

  defp encode_etag(etag) do
    Base.encode32(etag, case: :lower, padding: false)
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
