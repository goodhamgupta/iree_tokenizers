defmodule IREE.Tokenizers.HTTPClient do
  @moduledoc """
  Minimal HTTP client used by `IREE.Tokenizers.Tokenizer.from_pretrained/2`.

  This module follows the same lightweight callback shape used by
  `elixir-nx/tokenizers`:

      {:ok, %{status: integer(), headers: [{binary(), binary()}], body: binary()}}
      {:error, term()}

  It is public so callers can provide a compatible replacement through the
  `:http_client` option.
  """

  @type response :: %{status: non_neg_integer(), headers: [{binary(), binary()}], body: binary()}

  @doc """
  Performs a single HTTP request.

  Expected options:

  - `:url` - absolute URL or path
  - `:method` - `:get` or `:head`
  - `:base_url` - optional base URL for relative paths
  - `:headers` - optional request headers as `{binary(), binary()}` tuples
  """
  @spec request(keyword()) :: {:ok, response()} | {:error, term()}
  def request(opts) do
    url = build_url(opts)
    method = opts |> Keyword.get(:method, :get) |> to_method()
    host = URI.parse(url).host |> to_charlist()

    headers =
      opts
      |> Keyword.get(:headers, [])
      |> Enum.map(fn {key, value} -> {to_charlist(key), to_charlist(value)} end)

    http_opts = [
      ssl: [
        verify: :verify_peer,
        cacertfile: String.to_charlist(CAStore.file_path()),
        server_name_indication: host,
        customize_hostname_check: [
          match_fun: :public_key.pkix_verify_hostname_match_fun(:https)
        ]
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

  defp build_url(opts) do
    url = Keyword.fetch!(opts, :url)

    case Keyword.get(opts, :base_url) do
      nil -> url
      base_url -> URI.merge(base_url, url) |> to_string()
    end
  end
end
