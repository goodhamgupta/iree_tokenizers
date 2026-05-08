Mix.Task.run("app.start")

alias IREE.Tokenizers.Tokenizer, as: IREETokenizer

defmodule IREETokenizersBench.LongBench do
  @moduledoc """
  Long-context fastokens-vs-IREE.Tokenizers comparison runner.

  Reads the inputs.jsonl + texts/ directory produced by
  `bench/longbench/run_fastokens.py` and times `IREE.Tokenizers.encode/3`
  on the exact same byte-identical strings, with `WARMUP` warmups +
  `ITERS` timed iterations. Output is written to
  `bench/results/longbench/iree.json` for the plotting script to join.
  """

  @warmup 3
  @iters 10

  def run do
    base = Path.expand("results/longbench", __DIR__)
    texts_dir = Path.join(base, "texts")
    inputs_path = Path.join(base, "inputs.jsonl")
    output_path = Path.join(base, "iree.json")

    unless File.exists?(inputs_path) do
      IO.puts(:stderr, "Missing #{inputs_path}. Run bench/longbench/run_fastokens.py first.")
      System.halt(1)
    end

    rows =
      inputs_path
      |> File.stream!()
      |> Stream.map(&String.trim/1)
      |> Stream.reject(&(&1 == ""))
      |> Stream.map(&Jason.decode!/1)
      |> Enum.to_list()

    IO.puts(:stderr, "Loaded #{length(rows)} input rows from #{inputs_path}")

    grouped = Enum.group_by(rows, & &1["model"])
    token = hf_token()

    results =
      grouped
      |> Enum.reduce(%{}, fn {model, model_rows}, acc ->
        IO.puts(:stderr, "\n=== #{model} ===")

        case load_tokenizer(model, token) do
          {:ok, tokenizer} ->
            Map.put(acc, model, %{"buckets" => bench_model(tokenizer, model_rows, texts_dir)})

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

        if File.exists?(path) do
          path |> File.read!() |> String.trim()
        else
          nil
        end

      token ->
        token
    end
  end

  defp load_tokenizer(model, token) do
    opts = if token, do: [token: token], else: []
    IREETokenizer.from_pretrained(model, opts)
  end

  defp bench_model(tokenizer, rows, texts_dir) do
    rows
    |> Enum.group_by(& &1["bucket"])
    |> Enum.into(%{}, fn {bucket, bucket_rows} ->
      IO.puts(:stderr, "  bucket #{bucket}: #{length(bucket_rows)} sample(s)")

      samples =
        bucket_rows
        |> Enum.sort_by(& &1["sample_idx"])
        |> Enum.map(fn row ->
          text_path = Path.join(texts_dir, row["sha1"] <> ".txt")
          text = File.read!(text_path)
          timing = time_encode(tokenizer, text)

          IO.puts(
            :stderr,
            "    sample #{row["sample_idx"]}: tokens=#{timing.tokens} " <>
              "median=#{Float.round(timing.median_ms, 2)}ms " <>
              "p95=#{Float.round(timing.p95_ms, 2)}ms"
          )

          %{
            "sample_idx" => row["sample_idx"],
            "sha1" => row["sha1"],
            "natural_tokens" => row["natural_tokens"],
            "samples_ms" => timing.samples_ms,
            "median_ms" => timing.median_ms,
            "p95_ms" => timing.p95_ms,
            "tokens" => timing.tokens,
            "ids_sha" => timing.ids_sha
          }
        end)

      {to_string(bucket), samples}
    end)
  end

  defp time_encode(tokenizer, text) do
    Enum.each(1..@warmup, fn _ ->
      {:ok, _} = IREETokenizer.encode(tokenizer, text, add_special_tokens: false)
    end)

    {samples_ms, last_ids} =
      Enum.reduce(1..@iters, {[], []}, fn _, {acc, _} ->
        t0 = System.monotonic_time(:nanosecond)
        {:ok, encoding} = IREETokenizer.encode(tokenizer, text, add_special_tokens: false)
        elapsed_ns = System.monotonic_time(:nanosecond) - t0
        {[elapsed_ns / 1_000_000 | acc], encoding.ids}
      end)

    samples_ms = Enum.reverse(samples_ms)

    %{
      samples_ms: samples_ms,
      median_ms: median(samples_ms),
      p95_ms: percentile(samples_ms, 0.95),
      tokens: length(last_ids),
      ids_sha: ids_sha(last_ids)
    }
  end

  defp ids_sha(ids) do
    payload = ids |> Enum.map_join(",", &Integer.to_string/1)
    :crypto.hash(:sha256, payload) |> Base.encode16(case: :lower)
  end

  defp median(values) do
    sorted = Enum.sort(values)
    n = length(sorted)
    Enum.at(sorted, div(n, 2))
  end

  defp percentile(values, p) do
    sorted = Enum.sort(values)
    n = length(sorted)
    k = max(0, round(p * (n - 1)))
    Enum.at(sorted, k)
  end
end

IREETokenizersBench.LongBench.run()
