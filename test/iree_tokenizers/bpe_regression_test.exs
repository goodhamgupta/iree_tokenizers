defmodule IREETokenizers.BpeRegressionTest do
  @moduledoc """
  Regression tests for BPE backtracking bugs that previously caused
  `IREE.Tokenizers` output to diverge from HuggingFace's reference
  tokenization on real Hugging Face tokenizers.

  Each case is a single short word that was mis-tokenized by IREE prior to
  the corresponding fix in `native/iree_tokenizers_native/vendor/.../model/bpe_backtrack.c`.
  Running these tests guards against regressions: if a fix to one model
  breaks another, the corresponding `iree.ids == hf.ids` assertion fails.

  Skipped by default (requires HF cache + the `tokenizers` Hex package).
  Run with:

      RUN_BPE_REGRESSION=1 mix test test/iree_tokenizers/bpe_regression_test.exs
  """
  use ExUnit.Case, async: false

  alias IREE.Tokenizers.Tokenizer, as: IREETokenizer
  alias Tokenizers.Encoding, as: HFEncoding
  alias Tokenizers.Tokenizer, as: HFTokenizer

  @moduletag integration: true
  @moduletag skip:
               if(System.get_env("RUN_BPE_REGRESSION") in ["1", "true"],
                 do: false,
                 else: "set RUN_BPE_REGRESSION=1 to run BPE parity-vs-HF regression tests"
               )

  @cases [
    # Original branch-tracking commit fix lives in regex/exec.c, but the BPE
    # backtrack issues below are independent.
    #
    # Fix 1: trie-walk preemption in is_suffix_merge_preempted — canonical
    # BPE forms `ion` via i+on (rank 37) where on=o+n (rank 6), but the
    # left-to-right compound builder only sees `i+o → io` and gives up
    # because `io n` is not a merge. Trie lookup at the prefix end finds
    # `ion` directly and detects ct+ion → ction (rank 318) preempts ire+ct.
    {"deepseek-ai/DeepSeek-V3", "directions", ["dire", "ctions"],
     "trie-walk preemption (ire+ct false-blocking dire)"},

    # Fix 2: right-spine consumption check in is_suffix_blocked — the
    # previous check only inspected the rightmost *base byte*, so it missed
    # cases where an *intermediate* node on the spine got consumed. For
    # prefix=ution (= ut+ion rank 1073), the rightmost base `n` is fine
    # (n+s is high-rank), but `ion+s → ions` rank 426 fires before
    # ut+ion can fire — preventing ution from forming.
    {"deepseek-ai/DeepSeek-V3", "contributions", ["contrib", "utions"],
     "right-spine consumption (rib+ution false-blocking contrib)"},
    {"deepseek-ai/DeepSeek-V3", "documents", ["doc", "uments"],
     "right-spine consumption (similar false-blocking)"},

    # The remaining cases are baselines — never broken in this branch but
    # listed here so future fixes can't regress them.
    {"deepseek-ai/DeepSeek-V3", "reductions", ["re", "du", "ctions"],
     "baseline (no merge issue)"},
    {"deepseek-ai/DeepSeek-V3", "abductions", ["ab", "du", "ctions"],
     "baseline (no merge issue)"},
    {"deepseek-ai/DeepSeek-V3", "constructions", ["const", "ructions"],
     "baseline (no merge issue)"},
    {"deepseek-ai/DeepSeek-V3", "directly", ["direct", "ly"], "baseline (no merge issue)"},
    {"deepseek-ai/DeepSeek-V3", "direction", ["direction"], "baseline whole-word match"},
    {"deepseek-ai/DeepSeek-V3", "creativity", ["cre", "ativity"], "baseline DeepSeek (matching)"}

    # NOTE: The following cases still fail and are NOT included as
    # passing tests yet. They need a more principled "context-aware
    # reachability" of alternative BPE decompositions:
    #
    # - Nemotron / `induces` → IREE produces [indu, ces] but HF produces
    #   [ind, uces]. `indu` has two producers: (in, du) rank 214975 and
    #   (ind, u) rank 214976. split_table picks (in, du), with suffix `u`
    #   consumed at rank 634 — but in `induces` context, (in, du) is
    #   preempted by (in, d) → ind rank 563. The (ind, u) decomp would
    #   keep `u` available until rank 214977, allowing u+c rank 1984 to
    #   block `indu`. A naive BFS over all decompositions over-blocks
    #   tokens like Nemotron `there` whose alternative decomps don't fire
    #   in canonical BPE either.
    # - gpt-oss-120b / `downsampling` → IREE [downs, ampling], HF [down, sampling].
    # - GLM-4.7 / `creativity` → IREE [creat, ivity], HF [cre, ativity].
    # - Qwen3-235B / `pacaeval` → IREE [p, ac, ae, val], HF [pac, ae, val].
    #
    # Documented in docs/UPSTREAM_BUGS.md for follow-up.
  ]

  for {repo, text, expected_tokens, description} <- @cases do
    test "BPE parity vs HF — #{repo} / #{inspect(text)} (#{description})" do
      {:ok, iree_tok} = IREETokenizer.from_pretrained(unquote(repo))
      {:ok, hf_tok} = HFTokenizer.from_pretrained(unquote(repo))

      {:ok, iree_enc} =
        IREETokenizer.encode(iree_tok, unquote(text), add_special_tokens: false)

      {:ok, hf_enc} =
        HFTokenizer.encode(hf_tok, unquote(text), add_special_tokens: false)

      hf_ids = HFEncoding.get_ids(hf_enc)
      hf_tokens = HFEncoding.get_tokens(hf_enc)

      assert iree_enc.ids == hf_ids,
             "BPE parity broken on #{inspect(unquote(text))}:\n" <>
               "  IREE tokens   : #{inspect(iree_enc.tokens)}\n" <>
               "  HF tokens     : #{inspect(hf_tokens)}\n" <>
               "  IREE ids      : #{inspect(iree_enc.ids)}\n" <>
               "  HF ids        : #{inspect(hf_ids)}"

      assert hf_tokens == unquote(expected_tokens),
             "test fixture out of date: HF on #{inspect(unquote(text))} produced " <>
               "#{inspect(hf_tokens)}, expected #{inspect(unquote(expected_tokens))}"
    end
  end
end
