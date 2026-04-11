# Tokenizer latency comparison

| Model | Repo used | Tokenizers package (ms) | IREE oneshot / stream (ms) | Speedup |
| --- | --- | ---: | ---: | --- |

| LiquidAI/LFM2.5-1.2B-Instruct | LiquidAI/LFM2.5-1.2B-Instruct | 61.4 ms | 15.8 ms / 5.03 ms | 3.9x / 12.2x |
| Qwen/Qwen3.5-9B | Qwen/Qwen3.5-9B | 69.5 ms | 10.9 ms / 10.7 ms | 6.4x / 6.5x |
| zai-org/GLM-5.1 | zai-org/GLM-5.1 | 59.2 ms | 10.7 ms / 5.51 ms | 5.5x / 10.7x |
| mistralai/Ministral-3-3B-Reasoning-2512 | mistralai/Ministral-3-3B-Reasoning-2512 | 79.0 ms | 10.8 ms / 5.89 ms | 7.3x / 13.4x |
| BAAI/bge-m3 | BAAI/bge-m3 | 46.7 ms | 23.1 ms / 14.3 ms | 2.0x / 3.3x |
| google/gemma-4-31B-it | google/gemma-4-31B-it | 20.4 ms | 10.3 ms / 3.78 ms | 2.0x / 5.4x |

## Skipped

- arcee-ai/Trinity-Large-Preview: no usable tokenizer.json found

