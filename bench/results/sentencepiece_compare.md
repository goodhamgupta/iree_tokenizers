# SentencePiece `.model` comparison against elixir-nx/tokenizers

## Encode latency

| Model | Repo | Input bytes | Output ids | IREE `.model` | `tokenizers` | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T5-small (SentencePiece Unigram) | google-t5/t5-small | 52 | 10 | 12.0 μs | 35.0 μs | 2.92x |
| LLaMA tokenizer (SentencePiece BPE) | hf-internal-testing/llama-tokenizer | 44 | 12 | 15.0 μs | 16.0 μs | 1.07x |

## Decode latency

| Model | Repo | Input bytes | Output ids | IREE `.model` | `tokenizers` | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T5-small (SentencePiece Unigram) | google-t5/t5-small | 52 | 10 | 4.0 μs | 3.0 μs | 0.75x |
| LLaMA tokenizer (SentencePiece BPE) | hf-internal-testing/llama-tokenizer | 44 | 12 | 9.0 μs | 12.0 μs | 1.33x |
