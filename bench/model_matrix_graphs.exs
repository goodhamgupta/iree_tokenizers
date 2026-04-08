Mix.Task.run("app.start")

alias IREETokenizersBench.Support
alias IREE.Tokenizers.Tokenizer, as: IREETokenizer
alias Tokenizers.Tokenizer, as: HFTokenizer

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
      label: "bartowski/arcee-ai_Trinity-Large-Thinking-GGUF",
      requested_repo: "bartowski/arcee-ai_Trinity-Large-Thinking-GGUF",
      repos: [
        "bartowski/arcee-ai_Trinity-Large-Thinking-GGUF",
        "arcee-ai/Trinity-Large-Thinking"
      ]
    },
    %{
      label: "google/gemma-4-31B-it",
      requested_repo: "google/gemma-4-31B-it",
      repos: ["google/gemma-4-31B-it"]
    },
    %{
      label: "google/gemma-4-31B",
      requested_repo: "google/gemma-4-31B",
      repos: ["google/gemma-4-31B"]
    },
    %{
      label: "google/gemma-4-26B-A4B-it",
      requested_repo: "google/gemma-4-26B-A4B-it",
      repos: ["google/gemma-4-26B-A4B-it"]
    },
    %{
      label: "google/gemma-4-26B-A4B",
      requested_repo: "google/gemma-4-26B-A4B",
      repos: ["google/gemma-4-26B-A4B"]
    },
    %{
      label: "google/gemma-4-E4B-it",
      requested_repo: "google/gemma-4-E4B-it",
      repos: ["google/gemma-4-E4B-it"]
    },
    %{
      label: "google/gemma-4-E4B",
      requested_repo: "google/gemma-4-E4B",
      repos: ["google/gemma-4-E4B"]
    },
    %{
      label: "google/gemma-4-E2B-it",
      requested_repo: "google/gemma-4-E2B-it",
      repos: ["google/gemma-4-E2B-it"]
    },
    %{
      label: "google/gemma-4-E2B",
      requested_repo: "google/gemma-4-E2B",
      repos: ["google/gemma-4-E2B"]
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

        case load_model(model) do
          {:ok, actual_repo, iree_tokenizer, hf_tokenizer} ->
            row = benchmark_model(model.label, actual_repo, iree_tokenizer, hf_tokenizer, corpus)
            {[row | rows], skipped}

          {:error, reason} ->
            {rows, [%{label: model.label, reason: reason} | skipped]}
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
      "IREE speedup vs Hugging Face",
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
        {:ok, iree_tokenizer, hf_tokenizer} ->
          {:ok, repo, iree_tokenizer, hf_tokenizer}

        {:error, _reason} ->
          nil
      end
    end)
  end

  defp benchmark_model(label, actual_repo, iree_tokenizer, hf_tokenizer, corpus) do
    hf_ms =
      Support.time_ms(fn ->
        HFTokenizer.encode(hf_tokenizer, corpus, add_special_tokens: false)
      end)

    iree_oneshot_ms =
      Support.time_ms(fn ->
        IREETokenizer.encode(iree_tokenizer, corpus, add_special_tokens: false)
      end)

    iree_stream_ms = Support.iree_stream_encode_ms(iree_tokenizer, corpus, @chunk_bytes)

    %{
      label: label,
      actual_repo: actual_repo,
      hf_ms: hf_ms,
      iree_oneshot_ms: iree_oneshot_ms,
      iree_stream_ms: iree_stream_ms,
      iree_oneshot_speedup: hf_ms / iree_oneshot_ms,
      iree_stream_speedup: hf_ms / iree_stream_ms
    }
  end

  defp render_summary(rows, skipped) do
    table_header = """
    | Model | Repo used | Hugging Face (ms) | IREE oneshot / stream (ms) | Speedup |
    | --- | --- | ---: | ---: | --- |
    """

    table_rows =
      rows
      |> Enum.map(fn row ->
        "| #{row.label} | #{row.actual_repo} | #{Support.format_ms(row.hf_ms)} | " <>
          "#{Support.format_ms(row.iree_oneshot_ms)} / #{Support.format_ms(row.iree_stream_ms)} | " <>
          "#{Float.round(row.iree_oneshot_speedup, 1)}x / #{Float.round(row.iree_stream_speedup, 1)}x |"
      end)
      |> Enum.join("\n")

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
