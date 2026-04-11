# Fixture comparison against elixir-nx/tokenizers

Local fixture: `test/fixtures/bpe_bytelevel_minimal.json`

## Encode latency

| Workload | Input bytes | IREE.Tokenizers | elixir-nx/tokenizers | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Short | 11 | 5.2 μs | 5.3 μs | 1.01x |
| Medium | 1808 | 114.3 μs | 283.5 μs | 2.48x |
| Long | 32000 | 1.59 ms | 4.57 ms | 2.87x |

## Decode latency (shared ID sequences only)

| Workload | Shared ids | IREE.Tokenizers | elixir-nx/tokenizers | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Short | 4 | 2.0 μs | 2.5 μs | 1.28x |

## Skipped decode workloads

- Medium: encode outputs diverged (ids_equal=false, decoded_equal=false)
- Long: encode outputs diverged (ids_equal=false, decoded_equal=false)


## Notes

- Encode latency compares the same input text for both libraries.
- Decode latency is reported only for workloads where both libraries produced equivalent token sequences for the shared input.
- Workloads with divergent encode outputs are omitted from the decode comparison instead of being benchmarked as separate workloads.
