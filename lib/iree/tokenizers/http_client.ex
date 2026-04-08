defmodule IREE.Tokenizers.HTTPClient do
  @moduledoc """
  Minimal HTTP client used by `IREE.Tokenizers.Tokenizer.from_pretrained/2`.
  """

  @type response :: %{status: non_neg_integer(), headers: [{binary(), binary()}], body: binary()}

  @spec request(keyword()) :: {:ok, response()} | {:error, term()}
  def request(opts) do
    url = Keyword.fetch!(opts, :url)
    method = opts |> Keyword.get(:method, :get) |> to_method()

    headers =
      opts
      |> Keyword.get(:headers, [])
      |> Enum.map(fn {key, value} -> {to_charlist(key), to_charlist(value)} end)

    http_opts = [
      ssl: [
        verify: :verify_peer,
        cacerts:
          CAStore.file_path()
          |> File.read!()
          |> :public_key.pem_decode()
          |> Enum.map(&:public_key.pem_entry_decode/1)
      ]
    ]

    request = {to_charlist(url), headers}

    case :httpc.request(method, request, http_opts, body_format: :binary) do
      {:ok, {{_, status, _}, raw_headers, body}} ->
        headers =
          Enum.map(raw_headers, fn {key, value} ->
            {String.downcase(to_string(key)), to_string(value)}
          end)

        {:ok, %{status: status, headers: headers, body: body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp to_method(:get), do: :get
  defp to_method(:head), do: :head
end
