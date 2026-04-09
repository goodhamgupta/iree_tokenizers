defmodule IREETokenizers.MixProject do
  use Mix.Project

  @source_url "https://github.com/goodhamgupta/iree_tokenizers"
  @version "0.1.0"

  def project do
    [
      app: :iree_tokenizers,
      version: @version,
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      docs: docs(),
      description:
        "Fast Hugging Face tokenizer.json and OpenAI tiktoken bindings for Elixir via the IREE tokenizer runtime"
    ]
  end

  def application do
    [
      extra_applications: [:logger, :inets, :ssl, :public_key]
    ]
  end

  def cli do
    [
      preferred_envs: [
        docs: :docs,
        "hex.publish": :docs
      ]
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
      files: package_files()
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

  defp package_files do
    [
      ".formatter.exs",
      ".tool-versions",
      "LICENSE",
      "README.md",
      "mix.exs",
      "mix.lock",
      "scripts/update_iree_bundle.sh",
      "lib/iree/tokenizers.ex",
      "lib/iree/tokenizers/decode_stream.ex",
      "lib/iree/tokenizers/encode_stream.ex",
      "lib/iree/tokenizers/encoding.ex",
      "lib/iree/tokenizers/http_client.ex",
      "lib/iree/tokenizers/native.ex",
      "lib/iree/tokenizers/tokenizer.ex",
      "native/iree_tokenizers_native/Cargo.lock",
      "native/iree_tokenizers_native/Cargo.toml",
      "native/iree_tokenizers_native/build.rs",
      "native/iree_tokenizers_native/sources/base_sources.txt",
      "native/iree_tokenizers_native/sources/tokenizer_sources.txt",
      "native/iree_tokenizers_native/src/error.rs",
      "native/iree_tokenizers_native/src/ffi.rs",
      "native/iree_tokenizers_native/src/lib.rs",
      "native/iree_tokenizers_native/src/stream.rs",
      "native/iree_tokenizers_native/src/tokenizer.rs",
      "native/iree_tokenizers_native/vendor/IREE_COMMIT",
      "native/iree_tokenizers_native/vendor/iree_tokenizer_src/IREE-LICENSE"
    ] ++
      Path.wildcard("native/iree_tokenizers_native/vendor/iree_tokenizer_src/iree/**/*.{c,h}") ++
      Path.wildcard("test/fixtures/*")
  end
end
