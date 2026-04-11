defmodule IREE.Tokenizers.ComponentRegistry do
  @moduledoc false

  @name __MODULE__

  def put(tokenizer_resource, components) do
    ensure_server()

    Agent.update(@name, &Map.put(&1, tokenizer_resource, components))
  end

  def get(tokenizer_resource) do
    ensure_server()

    Agent.get(@name, &Map.get(&1, tokenizer_resource, %{}))
  end

  defp ensure_server do
    case Process.whereis(@name) do
      nil ->
        case Agent.start_link(fn -> %{} end, name: @name) do
          {:ok, _pid} -> :ok
          {:error, {:already_started, _pid}} -> :ok
        end

      _pid ->
        :ok
    end
  end
end
