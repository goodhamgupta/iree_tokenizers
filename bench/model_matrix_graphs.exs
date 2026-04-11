Mix.Task.run("app.start")

alias IREETokenizersBench.Support
alias IREE.Tokenizers.Tokenizer, as: IREETokenizer
alias Tokenizers.Tokenizer, as: ElixirTokenizers

defmodule IREETokenizersBench.ModelMatrix do
  alias IREETokenizersBench.Support

  @models [
    %{
      label: "LiquidAI/LFM2.5-1.2B-Instruct",
      requested_repo: "LiquidAI/LFM2.5-1.2B-Instruct",
      repos: ["LiquidAI/LFM2.5-1.2B-Instruct"]
    },
    %{
      label: "Qwen/Qwen3.5-9B",
      requested_repo: "Qwen/Qwen3.5-9B",
      repos: ["Qwen/Qwen3.5-9B"]
    },
    %{
      label: "zai-org/GLM-5.1",
      requested_repo: "zai-org/GLM-5.1",
      repos: ["zai-org/GLM-5.1", "zai-org/GLM-5"]
    },
    %{
      label: "mistralai/Ministral-3-3B-Reasoning-2512",
      requested_repo: "mistralai/Ministral-3-3B-Reasoning-2512",
      repos: ["mistralai/Ministral-3-3B-Reasoning-2512"]
    },
    %{
      label: "BAAI/bge-m3",
      requested_repo: "BAAI/bge-m3",
      repos: ["BAAI/bge-m3"],
      skip_reason: "embedding model excluded from the latency matrix"
    },
    %{
      label: "arcee-ai/Trinity-Large-Preview",
      requested_repo: "arcee-ai/Trinity-Large-Preview",
      repos: ["arcee-ai/Trinity-Large-Preview"]
    },
    %{
      label: "google/gemma-4-31B-it",
      requested_repo: "google/gemma-4-31B-it",
      repos: ["google/gemma-4-31B-it"]
    }
  ]

  @chunk_bytes 16_384

  def run do
    results_dir = Path.expand("results", __DIR__)
    File.mkdir_p!(results_dir)

    models = filter_models(@models, System.get_env("MODEL_FILTER"))
    corpus = Support.benchmark_corpus(512_000)

    {rows, skipped} =
      Enum.reduce(models, {[], []}, fn model, {rows, skipped} ->
        IO.puts("Benchmarking #{model.label}...")

        cond do
          model[:skip_reason] ->
            {rows, [%{label: model.label, reason: model[:skip_reason]} | skipped]}

          true ->
            case load_model(model) do
              {:ok, actual_repo, iree_tokenizer, tokenizers_tokenizer} ->
                case benchmark_model(
                       model.label,
                       actual_repo,
                       iree_tokenizer,
                       tokenizers_tokenizer,
                       corpus
                     ) do
                  {:ok, row} ->
                    {[row | rows], skipped}

                  {:skip, reason} ->
                    {rows, [%{label: model.label, reason: reason} | skipped]}
                end

              {:error, reason} ->
                {rows, [%{label: model.label, reason: reason} | skipped]}
            end
        end
      end)

    rows = Enum.reverse(rows)
    skipped = Enum.reverse(skipped)

    Support.render_latency_svg(
      Path.join(results_dir, "model_matrix_latency.svg"),
      "Tokenizer latency comparison",
      rows
    )

    Support.render_speedup_svg(
      Path.join(results_dir, "model_matrix_speedup.svg"),
      "IREE speedup vs elixir-nx/tokenizers",
      rows
    )

    File.write!(
      Path.join(results_dir, "model_matrix.md"),
      render_summary(rows, skipped)
    )

    IO.puts("Generated model matrix artifacts in #{results_dir}")
  end

  defp load_model(model) do
    Enum.find_value(model.repos, {:error, "no usable tokenizer.json found"}, fn repo ->
      case Support.load_tokenizers(repo, Support.pretrained_opts()) do
        {:ok, iree_tokenizer, tokenizers_tokenizer} ->
          {:ok, repo, iree_tokenizer, tokenizers_tokenizer}

        {:error, _reason} ->
          nil
      end
    end)
  end

  defp benchmark_model(label, actual_repo, iree_tokenizer, tokenizers_tokenizer, corpus) do
    {:ok, comparison} =
      Support.encode_comparison(iree_tokenizer, tokenizers_tokenizer, corpus,
        add_special_tokens: false
      )

    if not Support.equivalent_outputs?(comparison) do
      {:skip,
       "encode outputs diverged on benchmark corpus (ids_equal=#{comparison.ids_equal}, decoded_equal=#{comparison.decoded_equal})"}
    else
      tokenizers_ms =
        Support.time_ms(fn ->
          ElixirTokenizers.encode(tokenizers_tokenizer, corpus, add_special_tokens: false)
        end)

      iree_oneshot_ms =
        Support.time_ms(fn ->
          IREETokenizer.encode(iree_tokenizer, corpus, add_special_tokens: false)
        end)

      {iree_stream_ms, iree_stream_speedup, stream_note} =
        case Support.stream_encode_ids(iree_tokenizer, corpus, @chunk_bytes) do
          {:ok, stream_ids} when stream_ids == comparison.iree_ids ->
            stream_ms = Support.iree_stream_encode_ms(iree_tokenizer, corpus, @chunk_bytes)
            {stream_ms, tokenizers_ms / stream_ms, nil}

          {:ok, stream_ids} ->
            {nil, nil,
             "stream output diverged from IREE one-shot encode (#{length(stream_ids)} ids vs #{length(comparison.iree_ids)})"}
        end

      {:ok,
       %{
         label: label,
         actual_repo: actual_repo,
         hf_ms: tokenizers_ms,
         iree_oneshot_ms: iree_oneshot_ms,
         iree_stream_ms: iree_stream_ms,
         iree_oneshot_speedup: tokenizers_ms / iree_oneshot_ms,
         iree_stream_speedup: iree_stream_speedup,
         stream_note: stream_note
       }}
    end
  end

  defp render_summary(rows, skipped) do
    table_header = """
    | Model | Repo used | Tokenizers package (ms) | IREE oneshot / stream (ms) | Speedup |
    | --- | --- | ---: | ---: | --- |
    """

    table_rows =
      rows
      |> Enum.map(fn row ->
        stream_ms = if row.iree_stream_ms, do: Support.format_ms(row.iree_stream_ms), else: "n/a"

        stream_speedup =
          if row.iree_stream_speedup,
            do: "#{Float.round(row.iree_stream_speedup, 1)}x",
            else: "n/a"

        "| #{row.label} | #{row.actual_repo} | #{Support.format_ms(row.hf_ms)} | " <>
          "#{Support.format_ms(row.iree_oneshot_ms)} / #{stream_ms} | " <>
          "#{Float.round(row.iree_oneshot_speedup, 1)}x / #{stream_speedup} |"
      end)
      |> Enum.join("\n")

    notes =
      rows
      |> Enum.filter(& &1.stream_note)
      |> Enum.map_join("\n", fn row -> "- #{row.label}: #{row.stream_note}" end)

    notes_section =
      if notes == "" do
        ""
      else
        """

        ## Notes

        #{notes}
        """
      end

    skipped_section =
      case skipped do
        [] ->
          ""

        entries ->
          """

          ## Skipped

          #{Enum.map_join(entries, "\n", fn entry -> "- #{entry.label}: #{entry.reason}" end)}
          """
      end

    """
    # Tokenizer latency comparison

    #{table_header}
    #{table_rows}
    #{notes_section}
    #{skipped_section}
    """
  end

  defp filter_models(models, nil), do: models

  defp filter_models(models, filter) do
    wanted =
      filter
      |> String.split(",", trim: true)
      |> MapSet.new()

    Enum.filter(models, fn model ->
      MapSet.member?(wanted, model.label) or MapSet.member?(wanted, model.requested_repo)
    end)
  end
end

IREETokenizersBench.ModelMatrix.run()
