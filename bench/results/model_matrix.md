# Tokenizer latency comparison

| Model | Repo used | Tokenizers package (ms) | IREE oneshot / stream (ms) | Speedup |
| --- | --- | ---: | ---: | --- |

| LiquidAI/LFM2.5-1.2B-Instruct | LiquidAI/LFM2.5-1.2B-Instruct | 60.7 ms | 4.56 ms / 4.62 ms | 13.3x / 13.2x |
| Qwen/Qwen3.5-9B | Qwen/Qwen3.5-9B | 70.6 ms | 5.29 ms / 11.0 ms | 13.4x / 6.4x |
| zai-org/GLM-5.1 | zai-org/GLM-5.1 | 63.5 ms | 5.05 ms / 5.83 ms | 12.6x / 10.9x |
| mistralai/Ministral-3-3B-Reasoning-2512 | mistralai/Ministral-3-3B-Reasoning-2512 | 63.8 ms | 4.53 ms / 5.64 ms | 14.1x / 11.3x |
| google/gemma-4-31B-it | google/gemma-4-31B-it | 19.4 ms | 3.27 ms / 3.63 ms | 5.9x / 5.3x |
| google/gemma-4-31B | google/gemma-4-31B | 20.6 ms | 3.35 ms / 3.61 ms | 6.2x / 5.7x |
| google/gemma-4-26B-A4B-it | google/gemma-4-26B-A4B-it | 18.1 ms | 3.3 ms / 3.61 ms | 5.5x / 5.0x |
| google/gemma-4-26B-A4B | google/gemma-4-26B-A4B | 21.2 ms | 3.6 ms / 3.51 ms | 5.9x / 6.0x |
| google/gemma-4-E4B-it | google/gemma-4-E4B-it | 16.8 ms | 3.53 ms / 3.55 ms | 4.7x / 4.7x |
| google/gemma-4-E4B | google/gemma-4-E4B | 19.7 ms | 3.56 ms / 3.78 ms | 5.5x / 5.2x |
| google/gemma-4-E2B-it | google/gemma-4-E2B-it | 19.9 ms | 3.61 ms / 3.56 ms | 5.5x / 5.6x |
| google/gemma-4-E2B | google/gemma-4-E2B | 20.1 ms | 3.41 ms / 3.59 ms | 5.9x / 5.6x |

## Skipped

- bartowski/arcee-ai_Trinity-Large-Thinking-GGUF: no usable tokenizer.json found

