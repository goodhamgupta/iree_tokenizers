# Fixture comparison against elixir-nx/tokenizers

Local fixture: `test/fixtures/bpe_bytelevel_minimal.json`

## Encode latency

| Workload | Input bytes | IREE.Tokenizers | elixir-nx/tokenizers | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Short | 11 | 4.0 μs | 4.0 μs | 1.0x |
| Medium | 1808 | 104.0 μs | 258.0 μs | 2.48x |
| Long | 32000 | 1.19 ms | 4.25 ms | 3.57x |

## Decode latency

| Workload | IREE / tokenizers ids | IREE.Tokenizers | elixir-nx/tokenizers | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Short | 4 / 4 | 4.0 μs | 2.0 μs | 0.5x |
| Medium | 920 / 1680 | 8.0 μs | 129.0 μs | 16.13x |
| Long | 16016 / 30720 | 140.0 μs | 2.06 ms | 14.69x |

## Notes

- Encode latency compares the same input text for both libraries.
- Decode latency compares each library decoding its own encoded ID sequence for the same input text.
- The minimal local BPE fixture diverges in output token counts on longer inputs, so latency is a more faithful cross-library comparison than tokens/sec for this specific fixture.
