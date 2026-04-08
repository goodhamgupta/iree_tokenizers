defmodule IREETokenizers.MixProject do
  use Mix.Project

  @source_url "https://github.com/shubhamgupta/iree_tokenizers"
  @version "0.1.0-dev"

  def project do
    [
      app: :iree_tokenizers,
      version: @version,
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      docs: docs(),
      description: "Fast Hugging Face tokenizer.json bindings for Elixir via the IREE tokenizer runtime",
      preferred_cli_env: [
        docs: :docs,
        "hex.publish": :docs
      ]
    ]
  end

  def application do
    [
      extra_applications: [:logger, :inets, :ssl, :public_key]
    ]
  end

  defp deps do
    [
      {:castore, "~> 0.1 or ~> 1.0"},
      {:ex_doc, "~> 0.38", only: :docs, runtime: false},
      {:rustler, "~> 0.37.3", optional: true, runtime: false},
      {:rustler_precompiled, "~> 0.8"}
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      maintainers: ["Shubham Gupta"],
      links: %{"GitHub" => @source_url},
      files: [
        "lib",
        "native",
        "scripts",
        "test/fixtures",
        ".formatter.exs",
        "mix.exs",
        "README.md",
        "LICENSE",
        "checksum-*.exs"
      ]
    ]
  end

  defp docs do
    [
      main: "IREE.Tokenizers",
      source_ref: "v#{@version}",
      source_url: @source_url,
      groups_for_modules: [
        Tokenization: [
          IREE.Tokenizers.Tokenizer,
          IREE.Tokenizers.Encoding,
          IREE.Tokenizers.EncodeStream,
          IREE.Tokenizers.DecodeStream
        ],
        Other: [
          IREE.Tokenizers.HTTPClient
        ]
      ]
    ]
  end
end
