# Tokenizer latency comparison

| Model | Repo used | Tokenizers package (ms) | IREE oneshot / stream (ms) | Speedup |
| --- | --- | ---: | ---: | --- |

| LiquidAI/LFM2.5-1.2B-Instruct | LiquidAI/LFM2.5-1.2B-Instruct | 66.6 ms | 14.0 ms / 4.76 ms | 4.8x / 14.0x |
| Qwen/Qwen3.5-9B | Qwen/Qwen3.5-9B | 72.0 ms | 12.8 ms / 11.4 ms | 5.6x / 6.3x |
| zai-org/GLM-5.1 | zai-org/GLM-5.1 | 64.1 ms | 12.6 ms / 6.51 ms | 5.1x / 9.8x |
| mistralai/Ministral-3-3B-Reasoning-2512 | mistralai/Ministral-3-3B-Reasoning-2512 | 62.5 ms | 12.5 ms / 6.57 ms | 5.0x / 9.5x |
| google/gemma-4-31B-it | google/gemma-4-31B-it | 19.8 ms | 12.4 ms / 3.68 ms | 1.6x / 5.4x |


## Skipped

- BAAI/bge-m3: embedding model excluded from the latency matrix
- arcee-ai/Trinity-Large-Preview: no usable tokenizer.json found

