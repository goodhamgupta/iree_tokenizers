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

  # Agent.start (not start_link): we don't want the registry's lifetime tied
  # to whichever caller happens to bootstrap it first. Under `async: true`
  # tests that caller is a transient test process, and when it exits the
  # registry shuts down with reason :shutdown — concurrent tests still
  # holding tokenizer resources then crash with `EXIT shutdown` on the next
  # ComponentRegistry call.
  defp ensure_server do
    case Process.whereis(@name) do
      nil ->
        case Agent.start(fn -> %{} end, name: @name) do
          {:ok, _pid} -> :ok
          {:error, {:already_started, _pid}} -> :ok
        end

      _pid ->
        :ok
    end
  end
end
