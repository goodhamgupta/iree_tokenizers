defmodule IREE.Tokenizers.Native do
  @moduledoc false

  mix_config = Mix.Project.config()
  version = mix_config[:version]
  github_url = mix_config[:package][:links]["GitHub"]

  use RustlerPrecompiled,
    otp_app: :iree_tokenizers,
    crate: "iree_tokenizers_native",
    version: version,
    base_url: "#{github_url}/releases/download/v#{version}",
    targets: [
      "aarch64-apple-darwin",
      "x86_64-apple-darwin",
      "x86_64-unknown-linux-gnu"
    ],
    force_build:
      Mix.env() in [:dev, :test] or System.get_env("IREE_TOKENIZERS_BUILD") in ["1", "true"]

  def tokenizer_from_buffer(_buffer), do: err()
  def tokenizer_encode(_tokenizer, _text, _opts), do: err()
  def tokenizer_encode_batch(_tokenizer, _texts, _opts), do: err()
  def tokenizer_decode(_tokenizer, _ids, _opts), do: err()
  def tokenizer_decode_batch(_tokenizer, _batch_ids, _opts), do: err()
  def tokenizer_vocab_size(_tokenizer), do: err()
  def tokenizer_model_type(_tokenizer), do: err()
  def tokenizer_token_to_id(_tokenizer, _token), do: err()
  def tokenizer_id_to_token(_tokenizer, _id), do: err()
  def tokenizer_bos_token_id(_tokenizer), do: err()
  def tokenizer_eos_token_id(_tokenizer), do: err()
  def tokenizer_unk_token_id(_tokenizer), do: err()
  def tokenizer_pad_token_id(_tokenizer), do: err()
  def tokenizer_sep_token_id(_tokenizer), do: err()
  def tokenizer_cls_token_id(_tokenizer), do: err()
  def tokenizer_mask_token_id(_tokenizer), do: err()
  def encode_stream_new(_tokenizer, _opts), do: err()
  def encode_stream_feed(_stream, _chunk), do: err()
  def encode_stream_finalize(_stream), do: err()
  def decode_stream_new(_tokenizer, _opts), do: err()
  def decode_stream_feed(_stream, _ids), do: err()
  def decode_stream_finalize(_stream), do: err()

  defp err, do: :erlang.nif_error(:nif_not_loaded)
end
