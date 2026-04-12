# Fixture comparison against elixir-nx/tokenizers

Local fixture: `test/fixtures/bpe_bytelevel_minimal.json`

## Encode latency (shared token sequences only)

| Workload | Input bytes | IREE.Tokenizers | elixir-nx/tokenizers | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Short | 11 | 15.4 μs | 9.2 μs | 0.59x |
| Medium | 1808 | 307.4 μs | 405.6 μs | 1.32x |
| Long | 32000 | 5.25 ms | 6.73 ms | 1.28x |

## Decode latency (shared ID sequences only)

| Workload | Shared ids | IREE.Tokenizers | elixir-nx/tokenizers | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Short | 4 | 3.8 μs | 4.1 μs | 1.09x |
| Medium | 1680 | 19.6 μs | 214.5 μs | 10.94x |
| Long | 30720 | 263.7 μs | 2.64 ms | 10.02x |



## Notes

- Latency is only reported for workloads where IREE.Tokenizers and
  elixir-nx/tokenizers produced the same token ids and decoded strings for
  the shared input. Divergent workloads are listed under "Skipped workloads"
  rather than being benchmarked as separate rows.
