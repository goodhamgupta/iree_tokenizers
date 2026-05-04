Mix.Task.run("app.start")

defmodule TrendingModels do
  @moduledoc """
  Fetches the top trending text models on Hugging Face and emits a JSON
  list consumable by `validate_parity.exs` via the `MODELS_JSON` env var.

  Env vars:

  - `HF_TOKEN`     — used for authenticated HEAD probes (counts gated repos).
  - `LIMIT`        — max number of models to keep (default 10).
  - `FETCH_LIMIT`  — how many trending entries to query before filtering
                     (default 50). Increase if too many candidates get
                     filtered out and you want a deeper pool.
  - `OUTPUT_PATH`  — destination JSON file (default
                     `bench/results/trending_models.json`).
  """

  @hf_api "https://huggingface.co/api/models"
  @hf_base "https://huggingface.co"

  # Pipeline tags we treat as "text" for parity testing. ASR / TTS / image
  # pipelines are excluded — even if they ship a tokenizer it isn't the
  # contract we're validating here.
  @text_pipeline_tags MapSet.new([
                        "text-generation",
                        "text2text-generation",
                        "fill-mask",
                        "text-classification",
                        "token-classification",
                        "feature-extraction",
                        "sentence-similarity",
                        "summarization",
                        "translation",
                        "question-answering",
                        "zero-shot-classification",
                        "table-question-answering",
                        "text-ranking"
                      ])

  # Hard exclusions on library_name. We never want to load diffusers
  # checkpoints — even if they technically embed a CLIP tokenizer, parity
  # for those is out of scope.
  @excluded_libraries MapSet.new(["diffusers"])

  # Tokenizer file candidates we know how to load. Order matters: we try
  # `tokenizer.json` first because it covers the broadest set of models
  # (including modern GPT-2-style BPE repos). SentencePiece `.model` is the
  # fallback for T5/Llama/Mistral families. Legacy GPT-2 `vocab.json +
  # merges.txt` is not directly loadable by the IREE runtime, so any repo
  # that only ships those is dropped.
  @tokenizer_candidates [
    %{filename: "tokenizer.json", format: :huggingface_json},
    %{filename: "tokenizer.model", format: :sentencepiece_model},
    %{filename: "spiece.model", format: :sentencepiece_model}
  ]

  def run do
    limit = parse_int(System.get_env("LIMIT"), 10)
    fetch_limit = parse_int(System.get_env("FETCH_LIMIT"), 50)
    token = System.get_env("HF_TOKEN")

    output_path =
      System.get_env("OUTPUT_PATH") ||
        Path.expand("results/trending_models.json", __DIR__)

    File.mkdir_p!(Path.dirname(output_path))

    IO.puts("Fetching #{fetch_limit} trending models from Hugging Face...")
    candidates = fetch_trending(fetch_limit, token)
    IO.puts("Got #{length(candidates)} raw candidates.")

    text_only =
      candidates
      |> Enum.reject(&excluded?/1)
      |> Enum.filter(&text_pipeline?/1)

    IO.puts("#{length(text_only)} candidates after text/library filter.")

    selected =
      text_only
      |> Enum.reduce_while([], fn entry, acc ->
        if length(acc) >= limit do
          {:halt, acc}
        else
          case probe_tokenizer(entry, token) do
            nil ->
              IO.puts("  skip #{entry["id"]}: no supported tokenizer file")
              {:cont, acc}

            resolved ->
              IO.puts(
                "  keep #{entry["id"]} (#{resolved.format}, gated=#{resolved.gated})"
              )

              {:cont, acc ++ [resolved]}
          end
        end
      end)

    IO.puts("\nSelected #{length(selected)} models:")

    Enum.each(selected, fn m ->
      IO.puts("  - #{m.label} (#{m.format})")
    end)

    File.write!(output_path, Jason.encode!(selected, pretty: true) <> "\n")
    IO.puts("\nWrote #{output_path}")
  end

  defp fetch_trending(limit, token) do
    url =
      "#{@hf_api}?sort=trendingScore&direction=-1&limit=#{limit}&full=true"

    headers = auth_headers(token)

    case http_get(url, headers) do
      {:ok, %{status: 200, body: body}} ->
        case Jason.decode(body) do
          {:ok, list} when is_list(list) -> list
          {:ok, other} -> raise "unexpected HF response shape: #{inspect(other)}"
          {:error, err} -> raise "failed to decode HF response: #{inspect(err)}"
        end

      {:ok, %{status: status, body: body}} ->
        raise "HF API returned #{status}: #{inspect(body)}"

      {:error, reason} ->
        raise "HF API request failed: #{inspect(reason)}"
    end
  end

  defp excluded?(entry) do
    library = entry["library_name"]
    library != nil and MapSet.member?(@excluded_libraries, library)
  end

  defp text_pipeline?(entry) do
    case entry["pipeline_tag"] do
      nil ->
        # No pipeline tag: only keep if `library_name` is a clearly textual
        # one. Otherwise we have no idea what this is.
        entry["library_name"] in ["transformers", "sentence-transformers"]

      tag when is_binary(tag) ->
        MapSet.member?(@text_pipeline_tags, tag)
    end
  end

  defp probe_tokenizer(entry, token) do
    repo_id = entry["id"]
    siblings = entry["siblings"] || []
    available = MapSet.new(Enum.map(siblings, & &1["rfilename"]))

    Enum.find_value(@tokenizer_candidates, fn candidate ->
      filename = candidate.filename

      cond do
        # If the API gave us the file listing, trust it (one fewer HEAD
        # request per repo).
        MapSet.size(available) > 0 and MapSet.member?(available, filename) ->
          build_entry(entry, repo_id, candidate, gated?(entry))

        MapSet.size(available) > 0 ->
          nil

        true ->
          # Fall back to a HEAD probe when siblings weren't in the response.
          if file_exists?(repo_id, filename, token) do
            build_entry(entry, repo_id, candidate, gated?(entry))
          end
      end
    end)
  end

  defp build_entry(entry, repo_id, candidate, gated) do
    %{
      label: repo_id,
      repo: repo_id,
      format: format_to_string(candidate.format),
      filename: candidate.filename,
      gated: gated,
      pipeline_tag: entry["pipeline_tag"],
      library_name: entry["library_name"],
      trending_score: entry["trendingScore"] || entry["trending_score"]
    }
  end

  defp format_to_string(:huggingface_json), do: "huggingface_json"
  defp format_to_string(:sentencepiece_model), do: "sentencepiece_model"

  defp gated?(entry) do
    case entry["gated"] do
      false -> false
      nil -> false
      _ -> true
    end
  end

  defp file_exists?(repo_id, filename, token) do
    url = "#{@hf_base}/#{repo_id}/resolve/main/#{filename}"

    case http_head(url, auth_headers(token)) do
      {:ok, %{status: status}} when status in 200..299 -> true
      _ -> false
    end
  end

  defp auth_headers(nil), do: []
  defp auth_headers(token), do: [{"authorization", "Bearer #{token}"}]

  defp http_get(url, headers), do: http_request(:get, url, headers)
  defp http_head(url, headers), do: http_request(:head, url, headers)

  defp http_request(method, url, headers) do
    host = URI.parse(url).host |> to_charlist()

    request_headers =
      [{"user-agent", "iree_tokenizers/parity-monitor"} | headers]
      |> Enum.map(fn {k, v} -> {to_charlist(k), to_charlist(v)} end)

    http_opts = [
      ssl: [
        verify: :verify_peer,
        cacertfile: String.to_charlist(CAStore.file_path()),
        server_name_indication: host,
        customize_hostname_check: [
          match_fun: :public_key.pkix_verify_hostname_match_fun(:https)
        ]
      ],
      autoredirect: true
    ]

    case :httpc.request(method, {to_charlist(url), request_headers}, http_opts,
           body_format: :binary
         ) do
      {:ok, {{_, status, _}, raw_headers, body}} ->
        normalized =
          Enum.map(raw_headers, fn {k, v} ->
            {String.downcase(to_string(k)), to_string(v)}
          end)

        {:ok, %{status: status, headers: normalized, body: body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp parse_int(nil, default), do: default

  defp parse_int(value, default) do
    case Integer.parse(value) do
      {n, _} -> n
      :error -> default
    end
  end
end

TrendingModels.run()
