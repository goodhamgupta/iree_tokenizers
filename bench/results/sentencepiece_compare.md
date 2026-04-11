# SentencePiece `.model` comparison against elixir-nx/tokenizers

## Encode latency

| Model | Repo | Input bytes | Output ids | IREE `.model` | `tokenizers` | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T5-small (SentencePiece Unigram) | google-t5/t5-small | 52 | 10 | 7.5 μs | 14.7 μs | 1.97x |
| LLaMA tokenizer (SentencePiece BPE) | hf-internal-testing/llama-tokenizer | 44 | 12 | 8.3 μs | 9.8 μs | 1.18x |

## Decode latency

| Model | Repo | Input bytes | Output ids | IREE `.model` | `tokenizers` | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T5-small (SentencePiece Unigram) | google-t5/t5-small | 52 | 10 | 4.3 μs | 4.1 μs | 0.95x |
| LLaMA tokenizer (SentencePiece BPE) | hf-internal-testing/llama-tokenizer | 44 | 12 | 4.0 μs | 7.3 μs | 1.81x |


