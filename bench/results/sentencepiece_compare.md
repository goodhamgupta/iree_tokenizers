# SentencePiece `.model` comparison against elixir-nx/tokenizers

## Encode latency

| Model | Repo | Input bytes | Output ids | IREE `.model` | `tokenizers` | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T5-small (SentencePiece Unigram) | google-t5/t5-small | 52 | 10 | 10.0 μs | 32.0 μs | 3.2x |
| LLaMA tokenizer (SentencePiece BPE) | hf-internal-testing/llama-tokenizer | 44 | 12 | 18.0 μs | 7.0 μs | 0.39x |

## Decode latency

| Model | Repo | Input bytes | Output ids | IREE `.model` | `tokenizers` | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T5-small (SentencePiece Unigram) | google-t5/t5-small | 52 | 10 | 3.0 μs | 2.0 μs | 0.67x |
| LLaMA tokenizer (SentencePiece BPE) | hf-internal-testing/llama-tokenizer | 44 | 12 | 4.0 μs | 7.0 μs | 1.75x |
