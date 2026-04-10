defmodule IREETokenizersBench.Support do
  alias IREE.Tokenizers.EncodeStream
  alias IREE.Tokenizers.Tokenizer, as: IREETokenizer
  alias Tokenizers.Tokenizer, as: ElixirTokenizers

  @default_chunk_bytes 16_384

  def benchmark_corpus(byte_target \\ 512_000) do
    paragraph =
      "Tokenization performance matters for real-time inference, long-context prompting, retrieval pipelines, and interactive developer tooling. "

    do_benchmark_corpus(paragraph, byte_target, [])
    |> IO.iodata_to_binary()
  end

  def time_ms(fun, repeats \\ 5) do
    _ = timed_call(fun)

    1..repeats
    |> Enum.map(fn _ -> timed_call(fun) end)
    |> Enum.sort()
    |> median()
  end

  def iree_stream_encode_ms(tokenizer, corpus, chunk_bytes \\ @default_chunk_bytes) do
    time_ms(fn ->
      {:ok, stream} = EncodeStream.new(tokenizer, add_special_tokens: false)

      corpus
      |> chunk_binary(chunk_bytes)
      |> Enum.each(fn chunk ->
        {:ok, _ids} = EncodeStream.feed(stream, chunk)
      end)

      {:ok, _final_ids} = EncodeStream.finalize(stream)
      :ok
    end)
  end

  def benchmark_throughput(label, tokens_per_run, fun, warmup_seconds, measure_seconds) do
    _ = run_for(fun, tokens_per_run, warmup_seconds)
    measured = run_for(fun, tokens_per_run, measure_seconds)

    %{
      label: label,
      tokens_per_second: measured.tokens / measured.seconds,
      tokens: measured.tokens,
      runs: measured.runs,
      seconds: measured.seconds
    }
  end

  def count_iree_ids(encodings),
    do: Enum.reduce(encodings, 0, fn encoding, acc -> acc + length(encoding.ids) end)

  def count_hf_ids(encodings),
    do:
      Enum.reduce(encodings, 0, fn encoding, acc ->
        acc + length(Tokenizers.Encoding.get_ids(encoding))
      end)

  def count_id_lists(id_lists), do: Enum.reduce(id_lists, 0, fn ids, acc -> acc + length(ids) end)

  def format_tokens_per_second(value) when value >= 1_000_000,
    do: "#{Float.round(value / 1_000_000, 1)}M tokens/sec"

  def format_tokens_per_second(value) when value >= 1_000,
    do: "#{Float.round(value / 1_000, 1)}K tokens/sec"

  def format_tokens_per_second(value), do: "#{Float.round(value, 1)} tokens/sec"

  def format_ms(value) do
    cond do
      value < 1 ->
        "#{Float.round(value * 1_000, 1)} μs"

      value >= 100 ->
        "#{Float.round(value, 0)} ms"

      value >= 10 ->
        "#{Float.round(value, 1)} ms"

      true ->
        "#{Float.round(value, 2)} ms"
    end
  end

  def render_throughput_svg(path, subtitle, bars) do
    width = 800
    height = 120 + length(bars) * 56
    chart_left = 170
    chart_top = 24
    row_gap = 56
    bar_height = 32
    chart_width = 590
    max_value = Enum.max_by(bars, & &1.value).value

    rows =
      bars
      |> Enum.sort_by(& &1.value, :desc)
      |> Enum.with_index()
      |> Enum.map(fn {bar, index} ->
        y = chart_top + index * row_gap
        bar_width = chart_width * (bar.value / max_value)
        label = format_tokens_per_second(bar.value)
        inside = bar_width > 190
        value_x = if inside, do: chart_left + bar_width - 155, else: chart_left + bar_width + 14
        value_fill = if inside, do: "#F7FAFF", else: "#A7B0C3"

        """
        <text x="120" y="#{y + 22}" fill="#D9E1F2" font-family="system-ui, sans-serif" font-size="18" text-anchor="end">#{bar.label}</text>
        <rect x="#{chart_left}" y="#{y}" width="#{Float.round(bar_width, 2)}" height="#{bar_height}" rx="5" fill="#{bar.color}" />
        <text x="#{value_x}" y="#{y + 22}" fill="#{value_fill}" font-family="system-ui, sans-serif" font-size="16">#{label}</text>
        """
      end)
      |> Enum.join("\n")

    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="#{width}" height="#{height}" viewBox="0 0 #{width} #{height}">
      <rect width="#{width}" height="#{height}" rx="10" fill="#0E1118" />
      #{rows}
      <text x="#{chart_left}" y="#{height - 24}" fill="#5D6472" font-family="system-ui, sans-serif" font-size="14">#{subtitle}</text>
    </svg>
    """

    File.write!(path, svg)
  end

  def render_latency_svg(path, title, rows) do
    width = 980
    row_height = 68
    header_height = 78
    height = header_height + row_height * length(rows) + 32
    chart_left = 330
    chart_width = 610

    max_value =
      rows
      |> Enum.flat_map(fn row -> [row.hf_ms, row.iree_oneshot_ms, row.iree_stream_ms] end)
      |> Enum.max()

    bars =
      rows
      |> Enum.with_index()
      |> Enum.map(fn {row, index} ->
        y = header_height + index * row_height

        [
          metric_bar(chart_left, chart_width, y, "HF", row.hf_ms, max_value, "#5D6472", 0),
          metric_bar(
            chart_left,
            chart_width,
            y,
            "IREE oneshot",
            row.iree_oneshot_ms,
            max_value,
            "#5A9BF6",
            18
          ),
          metric_bar(
            chart_left,
            chart_width,
            y,
            "IREE stream",
            row.iree_stream_ms,
            max_value,
            "#35C296",
            36
          )
        ]
        |> Enum.join("\n")
        |> then(
          &{"<text x=\"18\" y=\"#{y + 25}\" fill=\"#D9E1F2\" font-family=\"system-ui, sans-serif\" font-size=\"14\">#{row.label}</text>\n" <>
             &1}
        )
      end)
      |> Enum.map_join("\n", &elem(&1, 0))

    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="#{width}" height="#{height}" viewBox="0 0 #{width} #{height}">
      <rect width="#{width}" height="#{height}" rx="10" fill="#0E1118" />
      <text x="18" y="34" fill="#F7FAFF" font-family="system-ui, sans-serif" font-size="22">#{title}</text>
      <text x="18" y="58" fill="#7F8796" font-family="system-ui, sans-serif" font-size="14">Latency in milliseconds, lower is better</text>
      #{bars}
    </svg>
    """

    File.write!(path, svg)
  end

  def render_speedup_svg(path, title, rows) do
    width = 980
    row_height = 58
    header_height = 78
    height = header_height + row_height * length(rows) + 32
    chart_left = 330
    chart_width = 610

    max_value =
      rows
      |> Enum.flat_map(fn row -> [row.iree_oneshot_speedup, row.iree_stream_speedup] end)
      |> Enum.max()

    bars =
      rows
      |> Enum.with_index()
      |> Enum.map(fn {row, index} ->
        y = header_height + index * row_height
        oneshot_width = chart_width * (row.iree_oneshot_speedup / max_value)
        stream_width = chart_width * (row.iree_stream_speedup / max_value)

        """
        <text x="18" y="#{y + 22}" fill="#D9E1F2" font-family="system-ui, sans-serif" font-size="14">#{row.label}</text>
        <rect x="#{chart_left}" y="#{y}" width="#{Float.round(oneshot_width, 2)}" height="16" rx="4" fill="#5A9BF6" />
        <text x="#{chart_left + oneshot_width + 10}" y="#{y + 13}" fill="#A7B0C3" font-family="system-ui, sans-serif" font-size="12">#{Float.round(row.iree_oneshot_speedup, 1)}x oneshot</text>
        <rect x="#{chart_left}" y="#{y + 22}" width="#{Float.round(stream_width, 2)}" height="16" rx="4" fill="#35C296" />
        <text x="#{chart_left + stream_width + 10}" y="#{y + 35}" fill="#A7B0C3" font-family="system-ui, sans-serif" font-size="12">#{Float.round(row.iree_stream_speedup, 1)}x stream</text>
        """
      end)
      |> Enum.join("\n")

    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="#{width}" height="#{height}" viewBox="0 0 #{width} #{height}">
      <rect width="#{width}" height="#{height}" rx="10" fill="#0E1118" />
      <text x="18" y="34" fill="#F7FAFF" font-family="system-ui, sans-serif" font-size="22">#{title}</text>
      <text x="18" y="58" fill="#7F8796" font-family="system-ui, sans-serif" font-size="14">Speedup relative to elixir-nx/tokenizers, higher is better</text>
      #{bars}
    </svg>
    """

    File.write!(path, svg)
  end

  def render_dual_series_svg(path, title, subtitle, rows, left_series, right_series) do
    width = 1280
    header_height = 116
    row_height = 78
    height = header_height + row_height * length(rows) + 36
    chart_left = 320
    chart_width = 720
    speedup_x = width - 40
    speedup_label_x = width - 120
    speedup_guard_x = width - 220

    max_value =
      rows
      |> Enum.flat_map(fn row -> [row[left_series.key], row[right_series.key]] end)
      |> Enum.max(fn -> 1.0 end)

    legend = """
    <rect x="18" y="76" width="14" height="14" rx="3" fill="#{left_series.color}" />
    <text x="40" y="88" fill="#{Map.get(left_series, :text_fill, "#D9E1F2")}" font-family="system-ui, sans-serif" font-size="13">#{left_series.label}</text>
    <rect x="188" y="76" width="14" height="14" rx="3" fill="#{right_series.color}" />
    <text x="210" y="88" fill="#{Map.get(right_series, :text_fill, "#D9E1F2")}" font-family="system-ui, sans-serif" font-size="13">#{right_series.label}</text>
    """

    body =
      rows
      |> Enum.with_index()
      |> Enum.map_join("\n", fn {row, index} ->
        y = header_height + index * row_height
        left_value = row[left_series.key]
        right_value = row[right_series.key]
        left_width = chart_width * (left_value / max_value)
        right_width = chart_width * (right_value / max_value)
        left_label = left_series.formatter.(left_value)
        right_label = right_series.formatter.(right_value)

        {left_value_x, left_anchor, left_fill} =
          value_label_position(chart_left, left_width, left_label, speedup_guard_x)

        {right_value_x, right_anchor, right_fill} =
          value_label_position(chart_left, right_width, right_label, speedup_guard_x)

        """
        <text x="18" y="#{y + 24}" fill="#D9E1F2" font-family="system-ui, sans-serif" font-size="15">#{row.label}</text>
        <text x="18" y="#{y + 44}" fill="#7F8796" font-family="system-ui, sans-serif" font-size="12">#{row.subtitle}</text>
        <rect x="#{chart_left}" y="#{y}" width="#{Float.round(left_width, 2)}" height="16" rx="4" fill="#{left_series.color}" />
        <text x="#{left_value_x}" y="#{y + 13}" text-anchor="#{left_anchor}" fill="#{left_fill}" font-family="system-ui, sans-serif" font-size="12">#{left_label}</text>
        <rect x="#{chart_left}" y="#{y + 26}" width="#{Float.round(right_width, 2)}" height="16" rx="4" fill="#{right_series.color}" />
        <text x="#{right_value_x}" y="#{y + 39}" text-anchor="#{right_anchor}" fill="#{right_fill}" font-family="system-ui, sans-serif" font-size="12">#{right_label}</text>
        <text x="#{speedup_x}" y="#{y + 24}" fill="#D9E1F2" font-family="system-ui, sans-serif" font-size="13" text-anchor="end">#{Float.round(row.speedup, 2)}x</text>
        <text x="#{speedup_label_x}" y="#{y + 42}" fill="#7F8796" font-family="system-ui, sans-serif" font-size="11" text-anchor="end">IREE speedup</text>
        """
      end)

    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="#{width}" height="#{height}" viewBox="0 0 #{width} #{height}">
      <rect width="#{width}" height="#{height}" rx="10" fill="#0E1118" />
      <text x="18" y="36" fill="#F7FAFF" font-family="system-ui, sans-serif" font-size="22">#{title}</text>
      <text x="18" y="62" fill="#7F8796" font-family="system-ui, sans-serif" font-size="14">#{subtitle}</text>
      #{legend}
      #{body}
    </svg>
    """

    File.write!(path, svg)
  end

  def load_tokenizers(repo, opts \\ []) do
    with {:ok, iree_tokenizer} <- IREETokenizer.from_pretrained(repo, iree_pretrained_opts(opts)),
         {:ok, tokenizers_tokenizer} <-
           ElixirTokenizers.from_pretrained(repo, hf_pretrained_opts(opts)) do
      {:ok, iree_tokenizer, tokenizers_tokenizer}
    end
  end

  def pretrained_opts do
    case System.get_env("HF_TOKEN") do
      nil -> []
      token -> [token: token]
    end
  end

  defp iree_pretrained_opts(opts), do: opts

  defp hf_pretrained_opts(opts) do
    token = Keyword.get(opts, :token)
    opts = Keyword.drop(opts, [:token])

    if token do
      Keyword.put(opts, :http_client, {__MODULE__.HFHTTPClient, token: token})
    else
      opts
    end
  end

  defmodule HFHTTPClient do
    def request(opts) do
      token = opts[:token]
      url = build_url(opts)
      method = opts |> Keyword.get(:method, :get) |> to_method()

      headers =
        opts
        |> Keyword.get(:headers, [])
        |> maybe_put_auth(token)
        |> Enum.map(fn {key, value} -> {to_charlist(key), to_charlist(value)} end)

      http_opts = [
        ssl: [
          verify: :verify_peer,
          cacertfile: String.to_charlist(CAStore.file_path())
        ]
      ]

      request = {to_charlist(url), headers}

      case :httpc.request(method, request, http_opts, body_format: :binary) do
        {:ok, {{_, status, _}, raw_headers, body}} ->
          response_headers =
            Enum.map(raw_headers, fn {key, value} ->
              {String.downcase(to_string(key)), to_string(value)}
            end)

          {:ok, %{status: status, headers: response_headers, body: body}}

        {:error, reason} ->
          {:error, reason}
      end
    end

    defp maybe_put_auth(headers, nil), do: headers
    defp maybe_put_auth(headers, token), do: [{"authorization", "Bearer #{token}"} | headers]

    defp to_method(:get), do: :get
    defp to_method(:head), do: :head

    defp build_url(opts) do
      url = Keyword.fetch!(opts, :url)

      case Keyword.get(opts, :base_url) do
        nil -> url
        base_url -> URI.merge(base_url, url) |> to_string()
      end
    end
  end

  def chunk_binary(binary, chunk_bytes) do
    do_chunk_binary(binary, chunk_bytes, [])
  end

  defp do_chunk_binary(<<>>, _chunk_bytes, acc), do: Enum.reverse(acc)

  defp do_chunk_binary(binary, chunk_bytes, acc) when byte_size(binary) <= chunk_bytes,
    do: Enum.reverse([binary | acc])

  defp do_chunk_binary(binary, chunk_bytes, acc) do
    <<chunk::binary-size(chunk_bytes), rest::binary>> = binary
    do_chunk_binary(rest, chunk_bytes, [chunk | acc])
  end

  defp metric_bar(chart_left, chart_width, y, prefix, value, max_value, color, offset) do
    width = chart_width * (value / max_value)

    """
    <rect x="#{chart_left}" y="#{y + offset}" width="#{Float.round(width, 2)}" height="14" rx="4" fill="#{color}" />
    <text x="#{chart_left + width + 10}" y="#{y + offset + 11}" fill="#A7B0C3" font-family="system-ui, sans-serif" font-size="12">#{prefix}: #{format_ms(value)}</text>
    """
  end

  defp value_label_position(chart_left, bar_width, label, guard_x) do
    estimated_label_width = max(String.length(label) * 7, 28)
    outside_x = chart_left + bar_width + 10

    if bar_width > estimated_label_width + 18 and outside_x + estimated_label_width > guard_x do
      {chart_left + bar_width - 10, "end", "#F7FAFF"}
    else
      {outside_x, "start", "#A7B0C3"}
    end
  end

  defp timed_call(fun) do
    started_at = System.monotonic_time(:microsecond)
    _ = fun.()
    (System.monotonic_time(:microsecond) - started_at) / 1_000
  end

  defp median(values) do
    index = div(length(values), 2)
    Enum.at(values, index)
  end

  defp run_for(fun, tokens_per_run, seconds) do
    started_at = System.monotonic_time(:microsecond)
    deadline = started_at + trunc(seconds * 1_000_000)
    do_run_for(fun, tokens_per_run, deadline, started_at, 0, 0)
  end

  defp do_run_for(fun, tokens_per_run, deadline, started_at, tokens, runs) do
    now = System.monotonic_time(:microsecond)

    if now >= deadline and runs > 0 do
      %{
        tokens: tokens,
        runs: runs,
        seconds: max((now - started_at) / 1_000_000, 0.000001)
      }
    else
      _ = fun.()
      do_run_for(fun, tokens_per_run, deadline, started_at, tokens + tokens_per_run, runs + 1)
    end
  end

  defp do_benchmark_corpus(paragraph, byte_target, acc) do
    current_size = acc |> Enum.reverse() |> IO.iodata_length()

    if current_size >= byte_target do
      acc
    else
      do_benchmark_corpus(paragraph, byte_target, [paragraph | acc])
    end
  end
end
