defmodule IREETokenizers.HTTPClientTest do
  use ExUnit.Case, async: false

  alias IREE.Tokenizers.HTTPClient

  test "follows redirects explicitly and strips auth across hosts" do
    {:ok, listen_socket} =
      :gen_tcp.listen(0, [:binary, packet: :raw, active: false, reuseaddr: true])

    {:ok, {_address, port}} = :inet.sockname(listen_socket)
    parent = self()

    server =
      spawn_link(fn ->
        serve_redirect_pair(listen_socket, port, parent)
      end)

    on_exit(fn ->
      Process.exit(server, :kill)
      :gen_tcp.close(listen_socket)
    end)

    assert {:ok, %{status: 200, body: "ok"}} =
             HTTPClient.request(
               method: :get,
               url: "http://127.0.0.1:#{port}/redirect",
               headers: [{"authorization", "Bearer secret"}]
             )

    assert_receive {:request, "/redirect", redirect_headers}, 1_000
    assert {"authorization", "Bearer secret"} in redirect_headers

    assert_receive {:request, "/target", target_headers}, 1_000
    refute Enum.any?(target_headers, fn {key, _value} -> key == "authorization" end)
  end

  defp serve_redirect_pair(listen_socket, port, parent) do
    {:ok, socket} = :gen_tcp.accept(listen_socket)
    {path, headers} = read_request(socket)
    send(parent, {:request, path, headers})

    :ok =
      :gen_tcp.send(socket, [
        "HTTP/1.1 302 Found\r\n",
        "Location: http://localhost:#{port}/target\r\n",
        "Content-Length: 0\r\n",
        "Connection: close\r\n\r\n"
      ])

    :gen_tcp.close(socket)

    {:ok, socket} = :gen_tcp.accept(listen_socket)
    {path, headers} = read_request(socket)
    send(parent, {:request, path, headers})
    :ok = :gen_tcp.send(socket, "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
    :gen_tcp.close(socket)
  end

  defp read_request(socket) do
    {:ok, request} = :gen_tcp.recv(socket, 0, 1_000)
    [request_line | header_lines] = String.split(request, "\r\n")
    [_method, path, _version] = String.split(request_line, " ", parts: 3)

    headers =
      header_lines
      |> Enum.take_while(&(&1 != ""))
      |> Enum.map(fn line ->
        [key, value] = String.split(line, ":", parts: 2)
        {String.downcase(key), String.trim(value)}
      end)

    {path, headers}
  end
end
