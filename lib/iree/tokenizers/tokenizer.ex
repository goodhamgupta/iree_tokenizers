defmodule IREE.Tokenizers.Tokenizer do
  @moduledoc """
  Core tokenizer API.

  This module is the main entrypoint for loading tokenizers and running
  inference-time encode/decode operations.

  Supported load paths:

  - local or in-memory Hugging Face `tokenizer.json`
  - local or in-memory OpenAI `.tiktoken`
  - local or in-memory SentencePiece `.model`
  - remote Hugging Face repositories via `from_pretrained/2`

  Supported model families:

  - BPE
  - WordPiece
  - Unigram

  The API is intentionally inference-focused. It mirrors a useful subset of
  `elixir-nx/tokenizers` while keeping IREE as the underlying runtime.
  """

  alias IREE.Tokenizers.{ComponentRegistry, Encoding, HTTPClient, Model}

  defstruct [:resource]

  @typedoc """
  A loaded tokenizer handle.
  """
  @type t :: %__MODULE__{resource: reference()}
  @typedoc """
  Supported serialized tokenizer formats accepted by the constructor family.
  """
  @type load_format :: :huggingface_json | :tiktoken | :sentencepiece_model
  @typedoc """
  Input accepted by encode operations.

  The current implementation supports only single binary sequences.
  """
  @type encode_input :: binary()
  @typedoc """
  Common `{:ok, value} | {:error, {kind, message}}` result shape used by the
  public API.
  """
  @type result(value) :: {:ok, value} | {:error, {atom(), binary()}}
  @openai_tiktoken_base_url "https://openaipublic.blob.core.windows.net"
  @huggingface_base_url "https://huggingface.co"
  @tiktoken_encodings [
    "cl100k_base",
    "o200k_base",
    "o200k_harmony",
    "r50k_base",
    "gpt2",
    "p50k_base",
    "p50k_edit"
  ]
  @tiktoken_model_prefix_to_encoding %{
    "o1-" => "o200k_base",
    "o3-" => "o200k_base",
    "o4-mini-" => "o200k_base",
    "gpt-5-" => "o200k_base",
    "gpt-4.5-" => "o200k_base",
    "gpt-4.1-" => "o200k_base",
    "chatgpt-4o-" => "o200k_base",
    "gpt-4o-" => "o200k_base",
    "gpt-4-" => "cl100k_base",
    "gpt-3.5-turbo-" => "cl100k_base",
    "gpt-35-turbo-" => "cl100k_base",
    "gpt-oss-" => "o200k_harmony",
    "ft:gpt-4o" => "o200k_base",
    "ft:gpt-4" => "cl100k_base",
    "ft:gpt-3.5-turbo" => "cl100k_base",
    "ft:davinci-002" => "cl100k_base",
    "ft:babbage-002" => "cl100k_base"
  }
  @tiktoken_model_to_encoding %{
    "o1" => "o200k_base",
    "o3" => "o200k_base",
    "o4-mini" => "o200k_base",
    "gpt-5" => "o200k_base",
    "gpt-4.1" => "o200k_base",
    "gpt-4o" => "o200k_base",
    "gpt-4" => "cl100k_base",
    "gpt-3.5-turbo" => "cl100k_base",
    "gpt-3.5" => "cl100k_base",
    "gpt-35-turbo" => "cl100k_base",
    "davinci-002" => "cl100k_base",
    "babbage-002" => "cl100k_base",
    "text-embedding-ada-002" => "cl100k_base",
    "text-embedding-3-small" => "cl100k_base",
    "text-embedding-3-large" => "cl100k_base",
    "text-davinci-003" => "p50k_base",
    "text-davinci-002" => "p50k_base",
    "text-davinci-001" => "r50k_base",
    "text-curie-001" => "r50k_base",
    "text-babbage-001" => "r50k_base",
    "text-ada-001" => "r50k_base",
    "davinci" => "r50k_base",
    "curie" => "r50k_base",
    "babbage" => "r50k_base",
    "ada" => "r50k_base",
    "code-davinci-002" => "p50k_base",
    "code-davinci-001" => "p50k_base",
    "code-cushman-002" => "p50k_base",
    "code-cushman-001" => "p50k_base",
    "davinci-codex" => "p50k_base",
    "cushman-codex" => "p50k_base",
    "text-davinci-edit-001" => "p50k_edit",
    "code-davinci-edit-001" => "p50k_edit",
    "text-similarity-davinci-001" => "r50k_base",
    "text-similarity-curie-001" => "r50k_base",
    "text-similarity-babbage-001" => "r50k_base",
    "text-similarity-ada-001" => "r50k_base",
    "text-search-davinci-doc-001" => "r50k_base",
    "text-search-curie-doc-001" => "r50k_base",
    "text-search-babbage-doc-001" => "r50k_base",
    "text-search-ada-doc-001" => "r50k_base",
    "code-search-babbage-code-001" => "r50k_base",
    "code-search-ada-code-001" => "r50k_base",
    "gpt2" => "gpt2",
    "gpt-2" => "gpt2"
  }
  @openai_public_tiktoken_files %{
    "cl100k_base" => "cl100k_base.tiktoken",
    "o200k_base" => "o200k_base.tiktoken",
    "o200k_harmony" => "o200k_base.tiktoken",
    "r50k_base" => "r50k_base.tiktoken",
    "gpt2" => "r50k_base.tiktoken",
    "p50k_base" => "p50k_base.tiktoken",
    "p50k_edit" => "p50k_base.tiktoken"
  }

  @doc """
  Loads a tokenizer from an in-memory buffer.

  Supported options:

  - `:format` - one of `:huggingface_json`, `:tiktoken`, or
    `:sentencepiece_model`
  - `:tiktoken_encoding` - required for raw `.tiktoken` buffers when the
    encoding cannot be inferred from a filename or model name
  """
  @spec from_buffer(binary(), keyword()) :: result(t())
  def from_buffer(buffer, opts \\ [])

  def from_buffer(buffer, opts) when is_binary(buffer) do
    with {:ok, opts} <- normalize_load_options(opts, :buffer) do
      case opts[:format] do
        :huggingface_json ->
          with {:ok, tokenizer} <- IREE.Tokenizers.Native.tokenizer_from_buffer(buffer) do
            {:ok, register_components(tokenizer, %{})}
          end

        :tiktoken ->
          with {:ok, tokenizer} <-
                 IREE.Tokenizers.Native.tokenizer_from_tiktoken_buffer(
                   buffer,
                   opts[:tiktoken_encoding]
                 ) do
            {:ok, register_components(tokenizer, %{})}
          end

        :sentencepiece_model ->
          with {:ok, tokenizer} <-
                 IREE.Tokenizers.Native.tokenizer_from_sentencepiece_model(buffer) do
            {:ok, register_components(tokenizer, %{source_format: :sentencepiece_model})}
          end
      end
    end
  end

  def from_buffer(_buffer, _opts),
    do: {:error, {:invalid_argument, "expected a binary tokenizer buffer"}}

  @doc """
  Loads a tokenizer from a local file.

  Format can be inferred from the file extension:

  - `.json` -> Hugging Face tokenizer JSON
  - `.tiktoken` -> OpenAI tiktoken
  - `.model` -> SentencePiece model

  You can also override the inferred format with `:format`.
  """
  @spec from_file(Path.t(), keyword()) :: result(t())
  def from_file(path, opts \\ [])

  def from_file(path, opts) when is_binary(path) do
    with {:ok, opts} <- normalize_load_options(opts, path),
         {:ok, contents} <- File.read(path) do
      from_buffer(contents, opts)
    else
      {:error, {_kind, _message}} = error ->
        error

      {:error, reason} ->
        error = File.Error.exception(action: "read", path: path, reason: reason)
        {:error, {:not_found, "failed to read #{path}: #{Exception.message(error)}"}}
    end
  end

  def from_file(_path, _opts), do: {:error, {:invalid_argument, "expected a file path"}}

  @doc """
  Downloads, caches, and loads a tokenizer from a remote repository.

  By default this expects a Hugging Face repository containing
  `tokenizer.json`. For `.tiktoken` and SentencePiece `.model` loads, pass
  `:format`.

  Common options:

  - `:revision` - revision or branch name, defaults to `"main"`
  - `:use_cache` - whether to reuse an existing cached file, defaults to `true`
  - `:cache_dir` - cache directory, defaults to a per-user application cache
  - `:http_client` - `{module, opts}` tuple implementing `request/1`
  - `:token` - optional Hugging Face token for gated/private repos
  - `:filename` - optional explicit remote filename override
  - `:format` - serialized tokenizer format
  - `:tiktoken_encoding` - optional explicit tiktoken encoding override
  """
  @spec from_pretrained(binary(), keyword()) :: result(t())
  def from_pretrained(repo_id, opts \\ [])

  def from_pretrained(repo_id, opts) when is_binary(repo_id) do
    with {:ok, opts} <- validate_pretrained_options(opts, repo_id),
         {:ok, sources} <- pretrained_sources(repo_id, opts) do
      fetch_pretrained_from_sources(sources, opts)
    end
  end

  def from_pretrained(_repo_id, _opts),
    do: {:error, {:invalid_argument, "expected a Hugging Face repo id"}}

  @doc """
  Returns the predefined IREE tiktoken encoding names supported by the loader.
  """
  @spec supported_tiktoken_encodings() :: [binary()]
  def supported_tiktoken_encodings, do: @tiktoken_encodings

  @doc """
  Builds a tokenizer from a pure Elixir model specification.

  See `IREE.Tokenizers.Model.BPE`, `IREE.Tokenizers.Model.WordPiece`, and
  `IREE.Tokenizers.Model.Unigram`.
  """
  @spec init(Model.t()) :: result(t())
  def init(%Model{} = model) do
    with {:ok, buffer} <- tokenizer_json_from_model(model),
         {:ok, tokenizer} <- from_buffer(buffer) do
      {:ok, register_components(tokenizer, %{model: model})}
    end
  end

  def init(_model), do: {:error, {:invalid_argument, "expected a model"}}

  @doc """
  Infers a tiktoken encoding name from a known model or deployment name.

  Returns `nil` when the model name is not recognized.
  """
  @spec tiktoken_encoding_for_model(binary()) :: binary() | nil
  def tiktoken_encoding_for_model(model) when is_binary(model) do
    infer_tiktoken_encoding(model)
  end

  def tiktoken_encoding_for_model(_model), do: nil

  @doc """
  Encodes a single binary input into an `IREE.Tokenizers.Encoding`.

  Supported options:

  - `:add_special_tokens` - include tokenizer post-processing special tokens,
    defaults to `true`
  - `:track_offsets` - track byte offsets, defaults to `false`
  - `:encoding_transformations` - list of
    `IREE.Tokenizers.Encoding.Transformation` values applied after encoding
  """
  @spec encode(t(), encode_input(), keyword()) :: result(Encoding.t())
  def encode(tokenizer, input, opts \\ [])

  def encode(%__MODULE__{} = tokenizer, input, opts) when is_binary(input) do
    opts =
      Keyword.validate!(opts,
        add_special_tokens: true,
        track_offsets: false,
        encoding_transformations: []
      )

    with {:ok, encoding} <-
           IREE.Tokenizers.Native.tokenizer_encode(
             tokenizer,
             input,
             Keyword.take(opts, [:add_special_tokens, :track_offsets])
           ) do
      {:ok, Encoding.transform(encoding, opts[:encoding_transformations])}
    end
  end

  def encode(%__MODULE__{}, {_left, _right}, _opts),
    do: {:error, {:invalid_argument, "pair sequence inputs are not supported in v1"}}

  def encode(%__MODULE__{}, _input, _opts),
    do: {:error, {:invalid_argument, "expected a binary input"}}

  @doc """
  Encodes multiple binary inputs in one batch call.

  Uses the same options as `encode/3`.
  """
  @spec encode_batch(t(), [encode_input()], keyword()) :: result([Encoding.t()])
  def encode_batch(tokenizer, inputs, opts \\ [])

  def encode_batch(%__MODULE__{} = tokenizer, inputs, opts) when is_list(inputs) do
    opts =
      Keyword.validate!(opts,
        add_special_tokens: true,
        track_offsets: false,
        encoding_transformations: []
      )

    case Enum.find(inputs, &(not is_binary(&1))) do
      nil ->
        with {:ok, encodings} <-
               IREE.Tokenizers.Native.tokenizer_encode_batch(
                 tokenizer,
                 inputs,
                 Keyword.take(opts, [:add_special_tokens, :track_offsets])
               ) do
          {:ok, Enum.map(encodings, &Encoding.transform(&1, opts[:encoding_transformations]))}
        end

      {_left, _right} ->
        {:error, {:invalid_argument, "pair sequence inputs are not supported in v1"}}

      _ ->
        {:error, {:invalid_argument, "expected a list of binary inputs"}}
    end
  end

  def encode_batch(%__MODULE__{}, _inputs, _opts),
    do: {:error, {:invalid_argument, "expected a list of binary inputs"}}

  @doc """
  Decodes a list of token IDs back into text.

  Supported options:

  - `:skip_special_tokens` - suppress special tokens in the output text,
    defaults to `true`
  """
  @spec decode(t(), [integer()], keyword()) :: result(binary())
  def decode(tokenizer, ids, opts \\ [])

  def decode(%__MODULE__{} = tokenizer, ids, opts) when is_list(ids) do
    opts = Keyword.validate!(opts, skip_special_tokens: true)

    with {:ok, text} <- IREE.Tokenizers.Native.tokenizer_decode(tokenizer, ids, opts) do
      {:ok, maybe_fix_sentencepiece_bpe_decode(tokenizer, text)}
    end
  end

  def decode(%__MODULE__{}, _ids, _opts),
    do: {:error, {:invalid_argument, "expected a list of token ids"}}

  @doc """
  Decodes multiple token ID lists in one batch call.
  """
  @spec decode_batch(t(), [[integer()]], keyword()) :: result([binary()])
  def decode_batch(tokenizer, batch_ids, opts \\ [])

  def decode_batch(%__MODULE__{} = tokenizer, batch_ids, opts) when is_list(batch_ids) do
    opts = Keyword.validate!(opts, skip_special_tokens: true)

    case Enum.find(batch_ids, &(not is_list(&1))) do
      nil ->
        with {:ok, texts} <-
               IREE.Tokenizers.Native.tokenizer_decode_batch(tokenizer, batch_ids, opts) do
          {:ok, Enum.map(texts, &maybe_fix_sentencepiece_bpe_decode(tokenizer, &1))}
        end

      _ ->
        {:error, {:invalid_argument, "expected a list of token id lists"}}
    end
  end

  def decode_batch(%__MODULE__{}, _batch_ids, _opts),
    do: {:error, {:invalid_argument, "expected a list of token id lists"}}

  @doc """
  Returns the number of active vocabulary entries.
  """
  @spec vocab_size(t()) :: non_neg_integer()
  def vocab_size(%__MODULE__{} = tokenizer),
    do: IREE.Tokenizers.Native.tokenizer_vocab_size(tokenizer)

  @doc """
  Returns the tokenizer vocabulary as a `%{token => id}` map.

  The `:with_added_tokens` option is accepted for compatibility and currently
  defaults to `true`.
  """
  @spec get_vocab(t(), keyword()) :: %{binary() => integer()}
  def get_vocab(%__MODULE__{} = tokenizer, opts \\ []) do
    _opts = Keyword.validate!(opts, with_added_tokens: true)

    0..max(IREE.Tokenizers.Native.tokenizer_vocab_capacity(tokenizer) - 1, 0)
    |> Enum.reduce(%{}, fn id, acc ->
      case id_to_token(tokenizer, id) do
        nil -> acc
        token -> Map.put(acc, token, id)
      end
    end)
  end

  @doc """
  Returns the size of the tokenizer vocabulary.

  The `:with_added_tokens` option is accepted for compatibility and currently
  defaults to `true`.
  """
  @spec get_vocab_size(t(), keyword()) :: non_neg_integer()
  def get_vocab_size(%__MODULE__{} = tokenizer, opts \\ []) do
    _opts = Keyword.validate!(opts, with_added_tokens: true)
    vocab_size(tokenizer)
  end

  @doc """
  Returns the model specification used to build this tokenizer when available.

  For tokenizers loaded from serialized files, this returns a minimal
  `%IREE.Tokenizers.Model{}` containing only the model type metadata.
  """
  @spec get_model(t()) :: Model.t()
  def get_model(%__MODULE__{} = tokenizer) do
    case ComponentRegistry.get(tokenizer.resource)[:model] do
      %Model{} = model ->
        model

      nil ->
        %Model{type: model_type(tokenizer), info: %{"model_type" => model_type(tokenizer)}}
    end
  end

  @doc """
  Replaces the tokenizer model with the given model specification.

  This currently rebuilds a new tokenizer from the provided model and returns
  that tokenizer.
  """
  @spec set_model(t(), Model.t()) :: t()
  def set_model(%__MODULE__{} = _tokenizer, %Model{} = model) do
    case init(model) do
      {:ok, tokenizer} -> tokenizer
      {:error, {kind, message}} -> raise RuntimeError, "#{kind}: #{message}"
    end
  end

  @doc """
  Returns the tokenizer model type name, such as `"BPE"`, `"WordPiece"`, or
  `"Unigram"`.
  """
  @spec model_type(t()) :: binary()
  def model_type(%__MODULE__{} = tokenizer),
    do: IREE.Tokenizers.Native.tokenizer_model_type(tokenizer)

  @doc """
  Looks up the token ID for a token string.
  """
  @spec token_to_id(t(), binary()) :: integer() | nil
  def token_to_id(%__MODULE__{} = tokenizer, token) when is_binary(token),
    do: IREE.Tokenizers.Native.tokenizer_token_to_id(tokenizer, token)

  def token_to_id(%__MODULE__{}, _token), do: nil

  @doc """
  Looks up the token string for a token ID.
  """
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
    @doc """
    Returns the token ID for the requested special token, or `nil` when that
    token is not defined.
    """
    @spec unquote(fun)(t()) :: integer() | nil
    def unquote(fun)(%__MODULE__{} = tokenizer),
      do: apply(IREE.Tokenizers.Native, unquote(native), [tokenizer])
  end

  defp maybe_put_auth(headers, nil), do: headers
  defp maybe_put_auth(headers, token), do: [{"authorization", "Bearer #{token}"} | headers]

  defp maybe_fix_sentencepiece_bpe_decode(%__MODULE__{} = tokenizer, text) do
    components = ComponentRegistry.get(tokenizer.resource)

    if components[:source_format] == :sentencepiece_model and model_type(tokenizer) == "BPE" do
      String.replace_prefix(text, " ", "")
    else
      text
    end
  end

  defp register_components(tokenizer, components) do
    ComponentRegistry.put(tokenizer.resource, components)
    tokenizer
  end

  defp tokenizer_json_from_model(%Model{type: "BPE", spec: spec}) do
    {:ok,
     Jason.encode!(%{
       "version" => "1.0",
       "model" => %{
         "type" => "BPE",
         "vocab" => spec["vocab"],
         "merges" => Enum.map(spec["merges"], &Enum.join(&1, " ")),
         "unk_token" => spec["unk_token"],
         "continuing_subword_prefix" => spec["continuing_subword_prefix"],
         "end_of_word_suffix" => spec["end_of_word_suffix"],
         "byte_fallback" => spec["byte_fallback"],
         "fuse_unk" => spec["fuse_unk"]
       }
     })}
  end

  defp tokenizer_json_from_model(%Model{type: "WordPiece", spec: spec}) do
    {:ok,
     Jason.encode!(%{
       "version" => "1.0",
       "model" => %{
         "type" => "WordPiece",
         "unk_token" => spec["unk_token"],
         "continuing_subword_prefix" => spec["continuing_subword_prefix"],
         "max_input_chars_per_word" => spec["max_input_chars_per_word"],
         "vocab" => spec["vocab"]
       },
       "pre_tokenizer" => %{"type" => "Whitespace"},
       "decoder" => %{
         "type" => "WordPiece",
         "prefix" => spec["continuing_subword_prefix"],
         "cleanup" => false
       }
     })}
  end

  defp tokenizer_json_from_model(%Model{type: "Unigram", spec: spec}) do
    {:ok,
     Jason.encode!(%{
       "version" => "1.0",
       "model" => %{
         "type" => "Unigram",
         "vocab" => spec["vocab"],
         "unk_id" => spec["unk_id"],
         "byte_fallback" => spec["byte_fallback"]
       },
       "pre_tokenizer" => %{
         "type" => "Metaspace",
         "replacement" => "▁",
         "prepend_scheme" => "always",
         "split" => true
       },
       "decoder" => %{
         "type" => "Metaspace",
         "replacement" => "▁",
         "prepend_scheme" => "always",
         "split" => true
       }
     })}
  end

  defp tokenizer_json_from_model(%Model{} = model) do
    {:error, {:unimplemented, "unsupported model type #{inspect(model.type)}"}}
  end

  defp validate_load_options(opts) do
    opts =
      Keyword.validate!(opts,
        format: nil,
        tiktoken_encoding: nil
      )

    opts =
      Keyword.put_new_lazy(opts, :format, fn ->
        infer_load_format(:buffer) || :huggingface_json
      end)

    case opts[:format] do
      :huggingface_json ->
        {:ok, opts}

      :tiktoken ->
        case opts[:tiktoken_encoding] do
          nil ->
            {:ok, opts}

          encoding ->
            case normalize_tiktoken_encoding(encoding) do
              {:ok, normalized} -> {:ok, Keyword.put(opts, :tiktoken_encoding, normalized)}
              {:error, _reason} = error -> error
            end
        end

      :sentencepiece_model ->
        {:ok, opts}

      other ->
        {:error, {:invalid_argument, "unsupported tokenizer format: #{inspect(other)}"}}
    end
  end

  defp normalize_load_options(opts, source_hint) do
    opts =
      if Keyword.has_key?(opts, :format) do
        opts
      else
        Keyword.put(opts, :format, infer_load_format(source_hint) || :huggingface_json)
      end

    with {:ok, opts} <- validate_load_options(opts),
         {:ok, encoding} <- ensure_tiktoken_encoding(opts, source_hint) do
      {:ok, Keyword.put(opts, :tiktoken_encoding, encoding)}
    end
  end

  defp validate_pretrained_options(opts, repo_id) do
    opts =
      Keyword.validate!(opts,
        revision: "main",
        use_cache: true,
        cache_dir: :filename.basedir(:user_cache, "iree_tokenizers"),
        http_client: {HTTPClient, []},
        token: nil,
        format: :huggingface_json,
        filename: nil,
        tiktoken_encoding: nil
      )

    with {:ok, load_opts} <-
           normalize_load_options(Keyword.take(opts, [:format, :tiktoken_encoding]), repo_id),
         {:ok, filename} <-
           normalize_filename(
             load_opts[:format],
             opts[:filename],
             load_opts[:tiktoken_encoding]
           ) do
      {:ok,
       opts
       |> Keyword.put(:format, load_opts[:format])
       |> Keyword.put(:tiktoken_encoding, load_opts[:tiktoken_encoding])
       |> Keyword.put(:filename, filename)}
    end
  end

  defp normalize_filename(:huggingface_json, nil, _encoding), do: {:ok, "tokenizer.json"}

  defp normalize_filename(:huggingface_json, filename, _encoding) when is_binary(filename),
    do: {:ok, filename}

  defp normalize_filename(:tiktoken, nil, encoding) do
    with {:ok, encoding} <- normalize_tiktoken_encoding(encoding) do
      {:ok, "#{encoding}.tiktoken"}
    end
  end

  defp normalize_filename(:tiktoken, filename, _encoding) when is_binary(filename),
    do: {:ok, filename}

  defp normalize_filename(:sentencepiece_model, nil, _encoding), do: {:ok, nil}

  defp normalize_filename(:sentencepiece_model, filename, _encoding) when is_binary(filename),
    do: {:ok, filename}

  defp normalize_filename(_format, _filename, _encoding),
    do: {:error, {:invalid_argument, "expected :filename to be a binary path when provided"}}

  defp normalize_tiktoken_encoding(encoding) when encoding in @tiktoken_encodings,
    do: {:ok, encoding}

  defp normalize_tiktoken_encoding(nil) do
    {:error,
     {:invalid_argument,
      "expected :tiktoken_encoding when format is :tiktoken; supported encodings: #{Enum.join(@tiktoken_encodings, ", ")}"}}
  end

  defp normalize_tiktoken_encoding(encoding) when is_binary(encoding) do
    {:error,
     {:invalid_argument,
      "unsupported tiktoken encoding #{inspect(encoding)}; supported encodings: #{Enum.join(@tiktoken_encodings, ", ")}"}}
  end

  defp normalize_tiktoken_encoding(_encoding) do
    {:error,
     {:invalid_argument,
      "expected :tiktoken_encoding to be a binary; supported encodings: #{Enum.join(@tiktoken_encodings, ", ")}"}}
  end

  defp ensure_tiktoken_encoding(opts, source_hint) do
    if opts[:format] != :tiktoken do
      {:ok, opts[:tiktoken_encoding]}
    else
      case opts[:tiktoken_encoding] || infer_tiktoken_encoding(source_hint) do
        nil ->
          {:error,
           {:invalid_argument,
            "could not infer a tiktoken encoding from #{inspect(source_hint)}; pass :tiktoken_encoding explicitly. Supported encodings: #{Enum.join(@tiktoken_encodings, ", ")}"}}

        encoding ->
          normalize_tiktoken_encoding(encoding)
      end
    end
  end

  defp infer_tiktoken_encoding(source_hint) when is_binary(source_hint) do
    candidate =
      source_hint
      |> Path.basename()
      |> String.replace_suffix(".tiktoken", "")

    cond do
      candidate in @tiktoken_encodings ->
        candidate

      encoding = Enum.find(@tiktoken_encodings, &String.ends_with?(candidate, &1)) ->
        encoding

      Map.has_key?(@tiktoken_model_to_encoding, source_hint) ->
        @tiktoken_model_to_encoding[source_hint]

      Map.has_key?(@tiktoken_model_to_encoding, candidate) ->
        @tiktoken_model_to_encoding[candidate]

      model = Enum.find(Map.keys(@tiktoken_model_to_encoding), &String.ends_with?(candidate, &1)) ->
        @tiktoken_model_to_encoding[model]

      repo_leaf(source_hint) in @tiktoken_encodings ->
        repo_leaf(source_hint)

      Map.has_key?(@tiktoken_model_to_encoding, repo_leaf(source_hint)) ->
        @tiktoken_model_to_encoding[repo_leaf(source_hint)]

      true ->
        Enum.find_value(@tiktoken_model_prefix_to_encoding, fn {prefix, encoding} ->
          if String.starts_with?(source_hint, prefix) or String.starts_with?(candidate, prefix) or
               String.starts_with?(repo_leaf(source_hint), prefix) do
            encoding
          end
        end)
    end
  end

  defp infer_tiktoken_encoding(_source_hint), do: nil

  defp infer_load_format(source_hint) when source_hint in [:buffer, nil], do: nil

  defp infer_load_format(source_hint) when is_binary(source_hint) do
    cond do
      String.ends_with?(source_hint, ".tiktoken") -> :tiktoken
      String.ends_with?(source_hint, ".model") -> :sentencepiece_model
      true -> nil
    end
  end

  defp pretrained_sources(repo_id, opts) do
    load_opts = Keyword.take(opts, [:format, :tiktoken_encoding])

    case opts[:format] do
      :huggingface_json ->
        url = "/#{repo_id}/resolve/#{opts[:revision]}/#{opts[:filename]}"

        {:ok,
         [
           %{
             base_url: @huggingface_base_url,
             url: url,
             cache_key: @huggingface_base_url <> url,
             load_opts: load_opts
           }
         ]}

      :tiktoken ->
        if builtin_openai_tiktoken_source?(repo_id, opts[:filename]) do
          filename = Map.fetch!(@openai_public_tiktoken_files, opts[:tiktoken_encoding])
          url = "/encodings/#{filename}"

          {:ok,
           [
             %{
               base_url: @openai_tiktoken_base_url,
               url: url,
               cache_key: @openai_tiktoken_base_url <> url,
               load_opts: load_opts
             }
           ]}
        else
          url = "/#{repo_id}/resolve/#{opts[:revision]}/#{opts[:filename]}"

          {:ok,
           [
             %{
               base_url: @huggingface_base_url,
               url: url,
               cache_key: @huggingface_base_url <> url,
               load_opts: load_opts
             }
           ]}
        end

      :sentencepiece_model ->
        filenames =
          case opts[:filename] do
            nil -> ["tokenizer.model", "spiece.model"]
            filename -> [filename]
          end

        {:ok,
         Enum.map(filenames, fn filename ->
           url = "/#{repo_id}/resolve/#{opts[:revision]}/#{filename}"

           %{
             base_url: @huggingface_base_url,
             url: url,
             cache_key: @huggingface_base_url <> url,
             load_opts: load_opts
           }
         end)}
    end
  end

  defp fetch_pretrained_from_sources(sources, opts) do
    do_fetch_pretrained_from_sources(sources, opts, nil)
  end

  defp do_fetch_pretrained_from_sources([], _opts, nil) do
    {:error, {:not_found, "resource not found"}}
  end

  defp do_fetch_pretrained_from_sources([], _opts, last_error), do: last_error

  defp do_fetch_pretrained_from_sources([source | rest], opts, _last_error) do
    case fetch_pretrained_from_source(source, opts) do
      {:error, {:not_found, _}} = error when rest != [] ->
        do_fetch_pretrained_from_sources(rest, opts, error)

      other ->
        other
    end
  end

  defp fetch_pretrained_from_source(source, opts) do
    {http_client, http_opts} = opts[:http_client]
    {:ok, app_version} = :application.get_key(:iree_tokenizers, :vsn)
    app_version = List.to_string(app_version)

    headers =
      [{"user-agent", "iree_tokenizers/#{app_version}"}]
      |> maybe_put_auth(opts[:token])

    http_opts =
      http_opts
      |> Keyword.put_new(:base_url, source.base_url)
      |> Keyword.put(:url, source.url)
      |> Keyword.put(:method, :get)
      |> Keyword.update(:headers, headers, fn existing -> existing ++ headers end)

    file_path_fun = fn etag ->
      Path.join(opts[:cache_dir], entry_filename(source.cache_key, etag))
    end

    if opts[:use_cache] do
      case request(http_client, Keyword.put(http_opts, :method, :head)) do
        {:ok, response} ->
          etag = fetch_etag(response.headers)
          file_path = file_path_fun.(etag)

          if File.exists?(file_path) do
            from_file(file_path, source.load_opts)
          else
            with {:ok, response} <- request(http_client, http_opts) do
              File.mkdir_p!(opts[:cache_dir])
              File.write!(file_path, response.body)
              from_file(file_path, source.load_opts)
            end
          end

        {:error, _reason} ->
          with {:ok, response} <- request(http_client, http_opts) do
            etag = fetch_etag(response.headers)
            file_path = file_path_fun.(etag)

            File.mkdir_p!(opts[:cache_dir])
            File.write!(file_path, response.body)
            from_file(file_path, source.load_opts)
          end
      end
    else
      with {:ok, response} <- request(http_client, http_opts) do
        etag = fetch_etag(response.headers)
        file_path = file_path_fun.(etag)

        File.mkdir_p!(opts[:cache_dir])
        File.write!(file_path, response.body)
        from_file(file_path, source.load_opts)
      end
    end
  end

  defp builtin_openai_tiktoken_source?(repo_id, filename) do
    repo_id == repo_leaf(repo_id) or
      String.starts_with?(repo_id, "openai/") or
      is_nil(filename)
  end

  defp repo_leaf(repo_id) when is_binary(repo_id) do
    repo_id
    |> String.split("/")
    |> List.last()
  end

  defp fetch_etag(headers) do
    case List.keyfind(headers, "etag", 0) do
      {_, etag} -> etag
      nil -> "no-etag"
    end
  end

  defp request(http_client, http_opts) do
    has_auth? =
      http_opts
      |> Keyword.get(:headers, [])
      |> Enum.any?(fn {key, _value} -> String.downcase(to_string(key)) == "authorization" end)

    case http_client.request(http_opts) do
      {:ok, response} ->
        case response.status do
          status when status in 200..299 ->
            {:ok, response}

          404 ->
            {:error, {:not_found, "resource not found"}}

          401 ->
            if has_auth? do
              {:error, {:permission_denied, "access denied"}}
            else
              {:error, {:not_found, "resource not found or requires authentication"}}
            end

          403 ->
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
