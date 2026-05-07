Mix.Task.run("app.start")

alias IREE.Tokenizers.Tokenizer, as: IREETokenizer

defmodule IREETokenizersBench.LongBenchBatch do
  @moduledoc """
  Batch encode + decode runner for IREE.Tokenizers.

  Reads the manifest produced by ``run_fastokens_batch.py`` (rows with
  ``kind`` ∈ {"single_synth", "single_decode", "batch"}), shares the same
  ``texts/<sha>.txt`` corpus, and times the corresponding IREE workloads.
  """

  @warmup 3
  @iters 10

  def run do
    base = Path.expand("results/longbench", __DIR__)
    texts_dir = Path.join(base, "texts")
    inputs_path = Path.join(base, "inputs_batch.jsonl")
    output_path = Path.join(base, "iree_batch.json")

    unless File.exists?(inputs_path) do
      IO.puts(:stderr, "Missing #{inputs_path}. Run bench/longbench/run_fastokens_batch.py first.")
      System.halt(1)
    end

    rows =
      inputs_path
      |> File.stream!()
      |> Stream.map(&String.trim/1)
      |> Stream.reject(&(&1 == ""))
      |> Stream.map(&Jason.decode!/1)
      |> Enum.to_list()

    IO.puts(:stderr, "Loaded #{length(rows)} manifest rows from #{inputs_path}")

    grouped = Enum.group_by(rows, & &1["model"])
    token = hf_token()

    results =
      Enum.reduce(grouped, %{}, fn {model, model_rows}, acc ->
        IO.puts(:stderr, "\n=== #{model} ===")

        case load_tokenizer(model, token) do
          {:ok, tokenizer} ->
            Map.put(acc, model, bench_model(tokenizer, model_rows, texts_dir))

          {:error, reason} ->
            IO.puts(:stderr, "  load failed: #{inspect(reason)}")
            Map.put(acc, model, %{"error" => "load failed: #{inspect(reason)}"})
        end
      end)

    File.write!(output_path, Jason.encode!(results, pretty: true))
    IO.puts(:stderr, "\nWrote #{output_path}")
  end

  defp hf_token do
    case System.get_env("HF_TOKEN") do
      nil ->
        path = Path.expand("~/.cache/huggingface/token")
        if File.exists?(path), do: path |> File.read!() |> String.trim(), else: nil

      token ->
        token
    end
  end

  defp load_tokenizer(model, token) do
    opts = if token, do: [token: token], else: []
    IREETokenizer.from_pretrained(model, opts)
  end

  defp bench_model(tokenizer, rows, texts_dir) do
    %{
      "single_synth" => bench_single(tokenizer, rows, texts_dir, "single_synth", :encode),
      "single_decode" => bench_single(tokenizer, rows, texts_dir, "single_decode", :decode),
      "batch_encode" => bench_batch(tokenizer, rows, texts_dir, :encode),
      "batch_decode" => bench_batch(tokenizer, rows, texts_dir, :decode)
    }
  end

  defp bench_single(tokenizer, rows, texts_dir, kind, op) do
    rows
    |> Enum.filter(&(&1["kind"] == kind))
    |> Enum.group_by(& &1["bucket"])
    |> Enum.into(%{}, fn {bucket, bucket_rows} ->
      samples =
        bucket_rows
        |> Enum.sort_by(& &1["sample_idx"])
        |> Enum.map(fn row ->
          text = File.read!(Path.join(texts_dir, row["sha1"] <> ".txt"))

          fun =
            case op do
              :encode ->
                fn ->
                  {:ok, _} = IREETokenizer.encode(tokenizer, text, add_special_tokens: false)
                end

              :decode ->
                {:ok, encoding} =
                  IREETokenizer.encode(tokenizer, text, add_special_tokens: false)

                ids = encoding.ids

                fn ->
                  {:ok, _} = IREETokenizer.decode(tokenizer, ids, skip_special_tokens: false)
                end
            end

          timing = time_fun(fun)

          tokens =
            case op do
              :decode ->
                {:ok, encoding} =
                  IREETokenizer.encode(tokenizer, text, add_special_tokens: false)

                length(encoding.ids)

              :encode ->
                {:ok, encoding} =
                  IREETokenizer.encode(tokenizer, text, add_special_tokens: false)

                length(encoding.ids)
            end

          IO.puts(
            :stderr,
            "  #{kind} bucket=#{bucket} sample=#{row["sample_idx"]}: tokens=#{tokens} " <>
              "median=#{Float.round(timing.median_ms, 3)}ms"
          )

          %{
            "sample_idx" => row["sample_idx"],
            "sha1" => row["sha1"],
            "tokens" => tokens,
            "samples_ms" => timing.samples_ms,
            "median_ms" => timing.median_ms,
            "p95_ms" => timing.p95_ms
          }
        end)

      {to_string(bucket), samples}
    end)
  end

  defp bench_batch(tokenizer, rows, texts_dir, op) do
    rows
    |> Enum.filter(&(&1["kind"] == "batch"))
    |> Enum.into(%{}, fn row ->
      bucket = row["bucket"]
      batch_size = row["batch_size"]
      texts = Enum.map(row["sha1s"], fn sha -> File.read!(Path.join(texts_dir, sha <> ".txt")) end)

      {fun, total_tokens} =
        case op do
          :encode ->
            {:ok, encs} = IREETokenizer.encode_batch(tokenizer, texts, add_special_tokens: false)
            tot = Enum.reduce(encs, 0, fn e, acc -> acc + length(e.ids) end)

            {fn ->
               {:ok, _} =
                 IREETokenizer.encode_batch(tokenizer, texts, add_special_tokens: false)
             end, tot}

          :decode ->
            {:ok, encs} = IREETokenizer.encode_batch(tokenizer, texts, add_special_tokens: false)
            id_lists = Enum.map(encs, & &1.ids)
            tot = Enum.reduce(id_lists, 0, fn ids, acc -> acc + length(ids) end)

            {fn ->
               {:ok, _} =
                 IREETokenizer.decode_batch(tokenizer, id_lists, skip_special_tokens: false)
             end, tot}
        end

      timing = time_fun(fun)

      IO.puts(
        :stderr,
        "  batch_#{op} bucket=#{bucket} batch_size=#{batch_size}: " <>
          "total_tokens=#{total_tokens} median=#{Float.round(timing.median_ms, 3)}ms"
      )

      key = "#{bucket}|#{batch_size}"

      {key,
       %{
         "bucket" => bucket,
         "batch_size" => batch_size,
         "total_tokens" => total_tokens,
         "samples_ms" => timing.samples_ms,
         "median_ms" => timing.median_ms,
         "p95_ms" => timing.p95_ms
       }}
    end)
  end

  defp time_fun(fun) do
    Enum.each(1..@warmup, fn _ -> fun.() end)

    samples_ms =
      Enum.map(1..@iters, fn _ ->
        t0 = System.monotonic_time(:nanosecond)
        fun.()
        (System.monotonic_time(:nanosecond) - t0) / 1_000_000
      end)

    %{
      samples_ms: samples_ms,
      median_ms: median(samples_ms),
      p95_ms: percentile(samples_ms, 0.95)
    }
  end

  defp median(values) do
    sorted = Enum.sort(values)
    Enum.at(sorted, div(length(sorted), 2))
  end

  defp percentile(values, p) do
    sorted = Enum.sort(values)
    n = length(sorted)
    Enum.at(sorted, max(0, round(p * (n - 1))))
  end
end

IREETokenizersBench.LongBenchBatch.run()
