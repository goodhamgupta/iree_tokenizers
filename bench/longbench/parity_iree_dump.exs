Mix.Task.run("app.start")

alias IREE.Tokenizers.Tokenizer, as: IREETokenizer

defmodule ParityDump do
  @models [
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V3",
    "MiniMaxAI/MiniMax-M2.1",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "zai-org/GLM-4.7"
  ]

  def run do
    base = Path.expand("../results/longbench", __DIR__)
    inputs_path = Path.join(base, "inputs.jsonl")
    texts_dir = Path.join(base, "texts")

    rows =
      inputs_path
      |> File.stream!()
      |> Stream.map(&String.trim/1)
      |> Stream.reject(&(&1 == ""))
      |> Stream.map(&Jason.decode!/1)
      |> Enum.to_list()

    by_model = Enum.group_by(rows, & &1["model"])
    token = hf_token()
    out_dir = Path.join(base, "parity")
    File.mkdir_p!(out_dir)

    Enum.each(@models, fn model ->
      sha =
        case Enum.find(by_model[model] || [], &(&1["bucket"] == 16384 and &1["sample_idx"] == 0)) do
          nil -> nil
          row -> row["sha1"]
        end

      if sha do
        text = File.read!(Path.join(texts_dir, sha <> ".txt"))

        case IREETokenizer.from_pretrained(model, token: token) do
          {:ok, tokenizer} ->
            {:ok, encoding} =
              IREETokenizer.encode(tokenizer, text, add_special_tokens: false)

            ids = encoding.ids
            slug = String.replace(model, "/", "__")
            File.write!(Path.join(out_dir, "#{slug}__iree.json"), Jason.encode!(%{model: model, sha1: sha, ids: ids}))
            IO.puts("#{model}: wrote #{length(ids)} ids to parity/#{slug}__iree.json")

          {:error, reason} ->
            IO.puts(:stderr, "#{model}: load failed #{inspect(reason)}")
        end
      else
        IO.puts(:stderr, "#{model}: no 16K sample 0 in manifest")
      end
    end)
  end

  defp hf_token do
    case System.get_env("HF_TOKEN") do
      nil ->
        path = Path.expand("~/.cache/huggingface/token")
        if File.exists?(path), do: path |> File.read!() |> String.trim(), else: nil

      t ->
        t
    end
  end
end

ParityDump.run()
