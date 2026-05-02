# Repo-local skills

This folder contains lightweight, repo-local skills for agents working on
`IREE.Tokenizers`. They are ordinary markdown files with `SKILL.md` frontmatter
so future agents can read or load them before doing recurring work.

Use them together with the root `AGENTS.md`:

- `iree-tokenizers-parity` - debugging tokenizer correctness, parity failures,
  and new model verification.
- `iree-tokenizers-benchmarking` - running benchmark scripts and updating
  performance claims/charts without overclaiming.
- `iree-tokenizers-vendor-refresh` - refreshing the vendored IREE tokenizer C
  runtime or native dependencies without losing local parity patches.

These skills are intentionally project-specific. They do not replace tests; they
point agents at the right tests, files, and pitfalls.
