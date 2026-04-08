defmodule IREETokenizersBench.MixProject do
  use Mix.Project

  def project do
    [
      app: :iree_tokenizers_bench,
      version: "0.1.0",
      elixir: "~> 1.19",
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:benchee, "~> 1.3"},
      {:iree_tokenizers, path: ".."},
      {:rustler, "~> 0.37.3"},
      {:tokenizers, "~> 0.5.1"}
    ]
  end
end
