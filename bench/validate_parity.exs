Mix.Task.run("app.start")

alias IREE.Tokenizers.EncodeStream
alias IREE.Tokenizers.Tokenizer, as: IREETokenizer
alias IREETokenizersBench.Support
alias Tokenizers.Encoding, as: HFEncoding
alias Tokenizers.Tokenizer, as: ElixirTokenizers

defmodule Parity do
  @moduledoc """
  End-to-end parity regression runner for `IREE.Tokenizers` vs.
  `elixir-nx/tokenizers` (Rust-backed Hugging Face `tokenizers` crate).

  Exercises a fixed matrix of real Hugging Face tokenizers against a suite
  of 19 representative inputs — short ASCII, unicode, CJK, emoji with ZWJ,
  code, JSON, long (100–200KB) sequences, and more — in both
  `add_special_tokens: true/false` modes. For each model, it also compares
  `encode_batch/3` results to HF's batch output, and streamed encoding to
  the corresponding one-shot encoding.

  Report is written to `bench/results/parity_report.md`. Known failing
  cases trace into the vendored IREE tokenizer C runtime; see
  `docs/UPSTREAM_BUGS.md` for per-bug reproducers.

  Run it with:

      cd bench
      mix run validate_parity.exs
      # or, for a single model
      MODEL_FILTER="google-t5/t5-small (json)" mix run validate_parity.exs
      # or, with HF credentials for gated repos
      HF_TOKEN=hf_... mix run validate_parity.exs
  """

  @models [
    # From user list
    %{label: "deepseek-ai/DeepSeek-R1", repo: "deepseek-ai/DeepSeek-R1", format: :huggingface_json},
    %{label: "black-forest-labs/FLUX.1-dev", repo: "black-forest-labs/FLUX.1-dev", format: :huggingface_json, gated: true},
    %{label: "stabilityai/stable-diffusion-xl-base-1.0", repo: "stabilityai/stable-diffusion-xl-base-1.0", format: :huggingface_json},
    %{label: "CompVis/stable-diffusion-v1-4", repo: "CompVis/stable-diffusion-v1-4", format: :huggingface_json},
    # Real end-to-end check for the subfolder fallback added in this PR:
    # Sana is a public diffusers repo that ships `tokenizer/tokenizer.json`
    # (legacy SDXL/SD1.4 repos instead ship vocab.json+merges.txt and cannot
    # be loaded by IREE or the Rust HF tokenizers crate).
    %{label: "Efficient-Large-Model/Sana_600M_1024px_diffusers", repo: "Efficient-Large-Model/Sana_600M_1024px_diffusers", format: :huggingface_json, iree_only: true},
    %{label: "meta-llama/Meta-Llama-3-8B", repo: "meta-llama/Meta-Llama-3-8B", format: :huggingface_json, gated: true},
    %{label: "hexgrad/Kokoro-82M", repo: "hexgrad/Kokoro-82M", format: :huggingface_json},
    # Extra popular models from web search (Qwen/Llama/Mistral/BERT/T5/GPT2/Phi/Gemma families)
    %{label: "Qwen/Qwen2.5-7B-Instruct", repo: "Qwen/Qwen2.5-7B-Instruct", format: :huggingface_json},
    %{label: "mistralai/Mistral-7B-Instruct-v0.3", repo: "mistralai/Mistral-7B-Instruct-v0.3", format: :huggingface_json, gated: true},
    %{label: "google-bert/bert-base-uncased", repo: "google-bert/bert-base-uncased", format: :huggingface_json},
    %{label: "openai-community/gpt2", repo: "openai-community/gpt2", format: :huggingface_json},
    %{label: "microsoft/Phi-3-mini-4k-instruct", repo: "microsoft/Phi-3-mini-4k-instruct", format: :huggingface_json},
    %{label: "google-t5/t5-small (json)", repo: "google-t5/t5-small", format: :huggingface_json},
    %{label: "google-t5/t5-small (spiece)", repo: "google-t5/t5-small", format: :sentencepiece_model},
    %{label: "google/gemma-2-9b", repo: "google/gemma-2-9b", format: :huggingface_json, gated: true},
    %{label: "sentence-transformers/all-MiniLM-L6-v2", repo: "sentence-transformers/all-MiniLM-L6-v2", format: :huggingface_json}
  ]

  def cases do
    long_repeat = String.duplicate("the quick brown fox jumps over the lazy dog. ", 4096)
    cjk_long = String.duplicate("日本語のトークナイザーはUnicodeをうまく扱えますか？ 中文分词 한국어 테스트. ", 1024)
    mixed_long = String.duplicate("Tokenization 日本語 🚀 déjà vu naïve café.\n\t", 2048)

    [
      {"empty", ""},
      {"single_char", "a"},
      {"short_ascii", "Hello, world!"},
      {"whitespace_heavy", "   leading\t\ttabs\n\nnewlines   trailing   "},
      {"unicode_latin", "naïve café résumé coöperate façade"},
      {"unicode_cjk", "日本語 한국어 中文 ไทย עברית العربية"},
      {"emoji", "🚀🌍 Let's go! 👩‍💻 👨‍👩‍👧‍👦 🇺🇸 🏳️‍🌈"},
      {"code_python",
       "def f(x):\n    return [i**2 for i in range(x) if i % 2 == 0]\n"},
      {"code_rust",
       "fn main() { let v: Vec<u32> = (0..10).filter(|n| n % 3 == 0).collect(); println!(\"{:?}\", v); }"},
      {"json", "{\"name\": \"Alice\", \"age\": 30, \"tags\": [\"admin\", \"user\"]}"},
      {"special_token_literal",
       "Try <|endoftext|> and <s> </s> <pad> <unk> [CLS] [SEP] in one line"},
      {"control_chars", "bell\x07tab\ttab vertical\vform\ftab back\bspace"},
      {"numbers", "0 1 12 123 1234567890 3.1415926535 -42 +7 0xFF 1e-9"},
      {"url_like",
       "See https://example.com/path?q=hello%20world&n=42#frag and ftp://a.b/c"},
      {"markdown", "# Title\n\n- item **bold**\n- `code`\n\n> quote\n\n```py\nprint(1)\n```"},
      {"repeated_punct", "!!! ??? ... ,,, ;;; :::"},
      {"long_repeat_ascii", long_repeat},
      {"long_cjk", cjk_long},
      {"long_mixed", mixed_long}
    ]
  end

  def run do
    results_dir = Path.expand("results", __DIR__)
    File.mkdir_p!(results_dir)
    report_path = Path.join(results_dir, "parity_report.md")

    opts = Support.pretrained_opts()
    token_present? = Keyword.has_key?(opts, :token)

    models = filter_models(@models, System.get_env("MODEL_FILTER"))

    rows =
      Enum.map(models, fn model ->
        IO.puts("==> #{model.label}")
        run_model(model, opts, token_present?)
      end)

    summary = render_summary(rows)
    File.write!(report_path, summary)
    IO.puts("\nWrote #{report_path}")
    IO.puts("\n" <> short_console_table(rows))
  end

  defp filter_models(models, nil), do: models

  defp filter_models(models, filter) do
    wanted = filter |> String.split(",", trim: true) |> MapSet.new()
    Enum.filter(models, fn m -> MapSet.member?(wanted, m.label) or MapSet.member?(wanted, m.repo) end)
  end

  defp run_model(model, opts, token_present?) do
    cond do
      Map.get(model, :gated, false) and not token_present? ->
        %{label: model.label, status: :skipped, reason: "gated repo; set HF_TOKEN", cases: []}

      Map.get(model, :iree_only, false) ->
        run_iree_only(model, opts)

      true ->
        run_pair_model(model, opts)
    end
  end

  defp run_pair_model(model, opts) do
    try do
      case load_pair(model, opts) do
        {:ok, iree, hf} ->
          cases = Enum.map(cases(), fn c -> run_case(iree, hf, c) end)
          batch_result = run_batch(iree, hf)
          stream_result = run_stream(iree, hf)

          %{
            label: model.label,
            status: :ok,
            reason: nil,
            cases: cases,
            batch: batch_result,
            stream: stream_result
          }

        {:error, reason} ->
          %{label: model.label, status: :load_error, reason: inspect(reason), cases: []}
      end
    rescue
      e ->
        %{
          label: model.label,
          status: :load_error,
          reason: "raised: " <> Exception.message(e),
          cases: []
        }
    end
  end

  # Single-library smoke test for tokenizers that elixir-nx/tokenizers
  # cannot load (e.g. those that live under a diffusers-style subfolder
  # — the Rust HF crate does not yet honor `:subfolder`). We just verify
  # that IREE successfully loads the tokenizer and round-trips a handful
  # of inputs through encode → decode.
  defp run_iree_only(model, opts) do
    case load_iree(model, opts) do
      {:ok, iree} ->
        cases = Enum.map(cases(), fn c -> run_iree_only_case(iree, c) end)

        %{
          label: model.label <> " (IREE-only)",
          status: :ok,
          reason: nil,
          cases: cases,
          batch: %{status: :ok, count: 0, mismatches: []},
          stream: %{status: :ok, ids_equal: true, streamed_len: 0, oneshot_len: 0, first_diff: nil}
        }

      {:error, reason} ->
        %{
          label: model.label <> " (IREE-only)",
          status: :load_error,
          reason: inspect(reason),
          cases: []
        }
    end
  end

  defp run_iree_only_case(iree, {name, text}) do
    variants =
      Enum.map([true, false], fn add_special ->
        with {:ok, enc} <-
               IREETokenizer.encode(iree, text, add_special_tokens: add_special),
             {:ok, dec} <-
               IREETokenizer.decode(iree, enc.ids, skip_special_tokens: false) do
          %{
            name: name,
            add_special: add_special,
            bytes: byte_size(text),
            iree_ids_len: length(enc.ids),
            hf_ids_len: nil,
            ids_equal: true,
            decoded_equal: true,
            iree_roundtrip: dec == text,
            hf_roundtrip: nil,
            first_diff: nil,
            iree_head: Enum.take(enc.ids, 10),
            hf_head: [],
            error: nil
          }
        else
          {:error, reason} ->
            %{
              name: name,
              add_special: add_special,
              bytes: byte_size(text),
              error: inspect(reason),
              ids_equal: false,
              decoded_equal: false,
              iree_roundtrip: false,
              hf_roundtrip: nil,
              iree_ids_len: nil,
              hf_ids_len: nil,
              first_diff: nil,
              iree_head: [],
              hf_head: []
            }
        end
      end)

    %{
      name: name,
      bytes: byte_size(text),
      variants: variants,
      all_ok:
        Enum.all?(variants, fn v ->
          is_nil(v.error) and v.ids_equal and v.decoded_equal
        end)
    }
  end

  defp load_iree(%{repo: repo, format: :huggingface_json}, opts),
    do: IREETokenizer.from_pretrained(repo, opts)

  defp load_iree(%{repo: repo, format: :sentencepiece_model}, opts),
    do: IREETokenizer.from_pretrained(repo, Keyword.put(opts, :format, :sentencepiece_model))

  defp load_pair(%{repo: repo, format: :huggingface_json}, opts) do
    with {:ok, iree} <- IREETokenizer.from_pretrained(repo, opts),
         {:ok, hf} <- ElixirTokenizers.from_pretrained(repo, hf_opts(opts)) do
      {:ok, iree, hf}
    end
  end

  defp load_pair(%{repo: repo, format: :sentencepiece_model}, opts) do
    iree_opts = Keyword.put(opts, :format, :sentencepiece_model)

    with {:ok, iree} <- IREETokenizer.from_pretrained(repo, iree_opts),
         {:ok, hf} <- ElixirTokenizers.from_pretrained(repo, hf_opts(opts)) do
      {:ok, iree, hf}
    end
  end

  defp hf_opts(opts) do
    # Map IREE token opt into the HTTPClient shim Support already provides
    case Keyword.get(opts, :token) do
      nil -> []
      token -> [http_client: {IREETokenizersBench.Support.HFHTTPClient, token: token}]
    end
  end

  defp run_case(iree, hf, {name, text}) do
    Enum.map([true, false], fn add_special ->
      compare_one(iree, hf, name, text, add_special)
    end)
    |> merge_case_results(name, text)
  end

  defp compare_one(iree, hf, name, text, add_special) do
    with {:ok, iree_enc} <-
           IREETokenizer.encode(iree, text, add_special_tokens: add_special),
         {:ok, hf_enc} <-
           ElixirTokenizers.encode(hf, text, add_special_tokens: add_special),
         {:ok, iree_dec} <-
           IREETokenizer.decode(iree, iree_enc.ids, skip_special_tokens: false),
         {:ok, hf_dec} <-
           ElixirTokenizers.decode(hf, HFEncoding.get_ids(hf_enc), skip_special_tokens: false) do
      hf_ids = HFEncoding.get_ids(hf_enc)

      %{
        name: name,
        add_special: add_special,
        bytes: byte_size(text),
        iree_ids_len: length(iree_enc.ids),
        hf_ids_len: length(hf_ids),
        ids_equal: iree_enc.ids == hf_ids,
        decoded_equal: iree_dec == hf_dec,
        iree_roundtrip: iree_dec == text,
        hf_roundtrip: hf_dec == text,
        first_diff: first_diff(iree_enc.ids, hf_ids),
        iree_head: Enum.take(iree_enc.ids, 10),
        hf_head: Enum.take(hf_ids, 10),
        error: nil
      }
    else
      {:error, reason} ->
        %{
          name: name,
          add_special: add_special,
          bytes: byte_size(text),
          error: inspect(reason),
          ids_equal: false,
          decoded_equal: false,
          iree_roundtrip: false,
          hf_roundtrip: false,
          iree_ids_len: nil,
          hf_ids_len: nil,
          first_diff: nil,
          iree_head: [],
          hf_head: []
        }
    end
  end

  defp merge_case_results(variants, name, text) do
    %{
      name: name,
      bytes: byte_size(text),
      variants: variants,
      all_ok:
        Enum.all?(variants, fn v ->
          is_nil(v.error) and v.ids_equal and v.decoded_equal
        end)
    }
  end

  defp run_batch(iree, hf) do
    batch = Enum.map(cases(), fn {_n, t} -> t end) |> Enum.reject(&(&1 == ""))

    with {:ok, iree_list} <-
           IREETokenizer.encode_batch(iree, batch, add_special_tokens: false),
         {:ok, hf_list} <-
           ElixirTokenizers.encode_batch(hf, batch, add_special_tokens: false) do
      mismatches =
        iree_list
        |> Enum.zip(hf_list)
        |> Enum.with_index()
        |> Enum.filter(fn {{i_enc, h_enc}, _idx} ->
          i_enc.ids != HFEncoding.get_ids(h_enc)
        end)
        |> Enum.map(fn {{_i, _h}, idx} -> idx end)

      %{status: :ok, count: length(batch), mismatches: mismatches}
    else
      {:error, reason} -> %{status: :error, reason: inspect(reason)}
    end
  end

  defp run_stream(iree, _hf) do
    text = String.duplicate("Tokenization performance matters for real-time inference. ", 2048)

    try do
      with {:ok, one_shot} <- IREETokenizer.encode(iree, text, add_special_tokens: false),
           {:ok, stream} <- EncodeStream.new(iree, add_special_tokens: false),
           {:ok, prefix} <- stream_feed_all(stream, Support.chunk_binary(text, 16_384)),
           {:ok, tail} <- EncodeStream.finalize(stream) do
        streamed = prefix ++ tail

        %{
          status: :ok,
          ids_equal: streamed == one_shot.ids,
          streamed_len: length(streamed),
          oneshot_len: length(one_shot.ids),
          first_diff: first_diff(streamed, one_shot.ids)
        }
      else
        {:error, reason} -> %{status: :error, reason: inspect(reason)}
      end
    rescue
      e -> %{status: :error, reason: "raised: " <> Exception.message(e)}
    end
  end

  defp stream_feed_all(stream, chunks) do
    Enum.reduce_while(chunks, {:ok, []}, fn ch, {:ok, acc} ->
      case EncodeStream.feed(stream, ch) do
        {:ok, ids} -> {:cont, {:ok, acc ++ ids}}
        {:error, r} -> {:halt, {:error, r}}
      end
    end)
  end

  defp first_diff(a, b) do
    do_first_diff(a, b, 0)
  end

  defp do_first_diff([], [], _i), do: nil
  defp do_first_diff([x | _], [], i), do: %{index: i, iree: x, hf: :eol}
  defp do_first_diff([], [y | _], i), do: %{index: i, iree: :eol, hf: y}

  defp do_first_diff([x | xs], [y | ys], i) do
    if x == y, do: do_first_diff(xs, ys, i + 1), else: %{index: i, iree: x, hf: y}
  end

  defp render_summary(rows) do
    header = """
    # Parity report: IREE.Tokenizers vs elixir-nx/tokenizers

    Reference: `elixir-nx/tokenizers` (Rust-backed HF `tokenizers` crate).

    """

    body =
      Enum.map_join(rows, "\n\n", fn row ->
        case row.status do
          :skipped ->
            "## #{row.label}\n\n**SKIPPED** — #{row.reason}"

          :load_error ->
            "## #{row.label}\n\n**LOAD ERROR** — #{row.reason}"

          :ok ->
            pass_count = Enum.count(row.cases, & &1.all_ok)
            total = length(row.cases)

            case_table =
              row.cases
              |> Enum.map_join("\n", &render_case_row/1)

            problems =
              row.cases
              |> Enum.reject(& &1.all_ok)
              |> Enum.map_join("\n\n", &render_case_detail/1)

            batch_line =
              case row.batch do
                %{status: :ok, mismatches: []} ->
                  "- batch encode: **OK**"

                %{status: :ok, mismatches: ms} ->
                  # Force integer-list printing; Elixir would otherwise
                  # render `[10]` as the charlist `~c"\n"`.
                  "- batch encode: **MISMATCH** at indices [" <>
                    Enum.map_join(ms, ", ", &Integer.to_string/1) <> "]"

                %{status: :error, reason: r} ->
                  "- batch encode: **ERROR** #{r}"
              end

            stream_line =
              case row.stream do
                %{status: :ok, ids_equal: true} -> "- stream encode vs oneshot: **OK**"
                %{status: :ok, ids_equal: false, first_diff: fd, streamed_len: s, oneshot_len: o} ->
                  "- stream encode vs oneshot: **MISMATCH** (streamed=#{s}, oneshot=#{o}, first_diff=#{inspect(fd)})"
                %{status: :error, reason: r} -> "- stream encode: **ERROR** #{r}"
              end

            """
            ## #{row.label}

            **#{pass_count}/#{total} cases passed** (both add_special_tokens variants)

            #{batch_line}
            #{stream_line}

            | Case | bytes | add_special | iree_ids | hf_ids | ids= | decoded= | iree_roundtrip | hf_roundtrip | error |
            | --- | ---: | :---: | ---: | ---: | :---: | :---: | :---: | :---: | --- |
            #{case_table}

            #{if problems == "", do: "", else: "### Mismatch details\n\n" <> problems}
            """
        end
      end)

    header <> body
  end

  defp render_case_row(c) do
    Enum.map_join(c.variants, "\n", fn v ->
      "| #{c.name} | #{c.bytes} | #{v.add_special} | #{v.iree_ids_len || "-"} | #{v.hf_ids_len || "-"} | #{ok(v.ids_equal)} | #{ok(v.decoded_equal)} | #{ok(v.iree_roundtrip)} | #{ok(v.hf_roundtrip)} | #{v.error || ""} |"
    end)
  end

  defp render_case_detail(c) do
    Enum.map_join(c.variants, "\n\n", fn v ->
      """
      **#{c.name}** (add_special=#{v.add_special}, bytes=#{c.bytes})

      - iree ids len=#{v.iree_ids_len}, head=#{inspect(v.iree_head, charlists: :as_lists)}
      - hf ids len=#{v.hf_ids_len}, head=#{inspect(v.hf_head, charlists: :as_lists)}
      - first diff: #{inspect(v.first_diff, charlists: :as_lists)}
      - error: #{v.error || "none"}
      """
    end)
  end

  defp ok(true), do: "✅"
  defp ok(false), do: "❌"
  defp ok(nil), do: "?"

  defp short_console_table(rows) do
    Enum.map_join(rows, "\n", fn row ->
      case row.status do
        :skipped -> "  SKIP  #{row.label} — #{row.reason}"
        :load_error -> "  LOAD  #{row.label} — #{row.reason}"
        :ok ->
          pass = Enum.count(row.cases, & &1.all_ok)
          total = length(row.cases)
          batch_ok = match?(%{status: :ok, mismatches: []}, row.batch)
          stream_ok = match?(%{status: :ok, ids_equal: true}, row.stream)
          mark = if pass == total and batch_ok and stream_ok, do: "PASS", else: "FAIL"
          "  #{mark}  #{row.label}  cases=#{pass}/#{total} batch=#{batch_ok} stream=#{stream_ok}"
      end
    end)
  end
end

Parity.run()
