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

    headers =
      opts
      |> Keyword.get(:headers, [])
      |> Enum.map(fn {key, value} -> {to_string(key), to_string(value)} end)

    request(url, method, headers, Keyword.get(opts, :max_redirects, 5))
  end

  defp request(_url, _method, _headers, 0), do: {:error, :too_many_redirects}

  defp request(url, method, headers, redirects_left) do
    http_opts = [
      autoredirect: false,
      ssl: [
        verify: :verify_peer,
        cacertfile: String.to_charlist(CAStore.file_path()),
        server_name_indication: url |> URI.parse() |> Map.fetch!(:host) |> to_charlist(),
        customize_hostname_check: [
          match_fun: :public_key.pkix_verify_hostname_match_fun(:https)
        ]
      ]
    ]

    request = {to_charlist(url), charlist_headers(headers)}

    case :httpc.request(method, request, http_opts, body_format: :binary) do
      {:ok, {{_, status, _}, raw_headers, body}} ->
        response_headers =
          Enum.map(raw_headers, fn {key, value} ->
            {String.downcase(to_string(key)), to_string(value)}
          end)

        if redirect_status?(status) do
          case redirect_url(url, response_headers) do
            {:ok, next_url} ->
              next_method = redirect_method(status, method)
              next_headers = redirect_headers(url, next_url, headers)
              request(next_url, next_method, next_headers, redirects_left - 1)

            :error ->
              {:ok, %{status: status, headers: response_headers, body: body}}
          end
        else
          {:ok, %{status: status, headers: response_headers, body: body}}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp charlist_headers(headers) do
    Enum.map(headers, fn {key, value} -> {to_charlist(key), to_charlist(value)} end)
  end

  defp redirect_status?(status), do: status in 300..399

  defp redirect_url(current_url, headers) do
    case List.keyfind(headers, "location", 0) do
      {_, location} -> {:ok, URI.merge(current_url, location) |> to_string()}
      nil -> :error
    end
  end

  defp redirect_method(303, :head), do: :head
  defp redirect_method(303, _method), do: :get
  defp redirect_method(_status, method), do: method

  defp redirect_headers(current_url, next_url, headers) do
    current_host = URI.parse(current_url).host
    next_host = URI.parse(next_url).host

    if current_host == next_host do
      headers
    else
      Enum.reject(headers, fn {key, _value} -> String.downcase(key) == "authorization" end)
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
