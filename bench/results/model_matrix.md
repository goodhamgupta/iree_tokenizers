# Tokenizer latency comparison

| Model | Repo used | Tokenizers package (ms) | IREE oneshot / stream (ms) | Speedup |
| --- | --- | ---: | ---: | --- |

| LiquidAI/LFM2.5-1.2B-Instruct | LiquidAI/LFM2.5-1.2B-Instruct | 64.0 ms | 4.68 ms / 4.77 ms | 13.7x / 13.4x |
| Qwen/Qwen3.5-9B | Qwen/Qwen3.5-9B | 70.2 ms | 4.93 ms / 11.3 ms | 14.2x / 6.2x |
| zai-org/GLM-5.1 | zai-org/GLM-5.1 | 63.1 ms | 4.74 ms / 5.59 ms | 13.3x / 11.3x |
| mistralai/Ministral-3-3B-Reasoning-2512 | mistralai/Ministral-3-3B-Reasoning-2512 | 63.0 ms | 4.69 ms / 5.66 ms | 13.4x / 11.1x |
| google/gemma-4-31B-it | google/gemma-4-31B-it | 20.1 ms | 3.39 ms / 3.81 ms | 5.9x / 5.3x |

## Skipped

- arcee-ai/Trinity-Large-Preview: no usable tokenizer.json found

