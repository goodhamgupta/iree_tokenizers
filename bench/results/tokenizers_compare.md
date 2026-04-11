# Fixture comparison against elixir-nx/tokenizers

Local fixture: `test/fixtures/bpe_bytelevel_minimal.json`

## Encode latency

| Workload | Input bytes | IREE.Tokenizers | elixir-nx/tokenizers | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Short | 11 | 4.0 μs | 4.0 μs | 1.0x |
| Medium | 1808 | 93.0 μs | 256.0 μs | 2.75x |
| Long | 32000 | 1.23 ms | 4.35 ms | 3.54x |

## Decode latency

| Workload | IREE / tokenizers ids | IREE.Tokenizers | elixir-nx/tokenizers | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Short | 4 / 4 | 2.0 μs | 4.0 μs | 2.0x |
| Medium | 920 / 1680 | 11.0 μs | 137.0 μs | 12.45x |
| Long | 16016 / 30720 | 104.0 μs | 2.2 ms | 21.18x |

## Notes

- Encode latency compares the same input text for both libraries.
- Decode latency compares each library decoding its own encoded ID sequence for the same input text.
- The minimal local BPE fixture diverges in output token counts on longer inputs, so latency is a more faithful cross-library comparison than tokens/sec for this specific fixture.
