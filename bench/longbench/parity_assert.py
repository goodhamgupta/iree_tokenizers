#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Assert IREE.Tokenizers and fastokens produce byte-identical token IDs.

Reads the per-sample timings + ids_sha from
- bench/results/longbench/fastokens.json
- bench/results/longbench/iree.json

For every (model, bucket, sample) row, verifies:
  1. token COUNT matches between fastokens and IREE
  2. ids_sha (sha256 of comma-joined IDs) matches

Exits 0 when all parity checks pass. Exits 1 with a per-failure summary
otherwise. Written to be invoked at the end of the benchmark pipeline so
parity-broken models fail loudly rather than producing misleading timings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results" / "longbench"
FAST_PATH = RESULTS_DIR / "fastokens.json"
IREE_PATH = RESULTS_DIR / "iree.json"


def main() -> int:
    if not FAST_PATH.exists() or not IREE_PATH.exists():
        sys.exit(f"missing {FAST_PATH} or {IREE_PATH}; run the bench first")

    fast = json.loads(FAST_PATH.read_text())
    iree = json.loads(IREE_PATH.read_text())

    failures: list[str] = []
    matched = 0
    total = 0

    for model in sorted(set(fast) | set(iree)):
        if "error" in fast.get(model, {}) or "error" in iree.get(model, {}):
            continue
        fb = fast.get(model, {}).get("buckets", {})
        ib = iree.get(model, {}).get("buckets", {})
        buckets = sorted(set(fb) & set(ib), key=int)
        for bucket in buckets:
            f_samples = {s["sha1"]: s for s in fb[bucket]}
            i_samples = {s["sha1"]: s for s in ib[bucket]}
            for sha in sorted(set(f_samples) & set(i_samples)):
                total += 1
                fs = f_samples[sha]
                is_ = i_samples[sha]
                fast_count = fs["tokens"]
                iree_count = is_["tokens"]
                fast_sha = fs.get("ids_sha")
                iree_sha = is_.get("ids_sha")

                ok_count = fast_count == iree_count
                ok_sha = (
                    fast_sha is not None
                    and iree_sha is not None
                    and fast_sha == iree_sha
                )

                if ok_count and ok_sha:
                    matched += 1
                    continue

                why = []
                if not ok_count:
                    why.append(f"tokens={iree_count} (fastokens={fast_count})")
                if not ok_sha:
                    if fast_sha is None or iree_sha is None:
                        why.append("ids_sha missing in one runner")
                    else:
                        why.append(
                            f"ids_sha={iree_sha[:8]} vs fastokens={fast_sha[:8]}"
                        )
                failures.append(
                    f"  {model:55s} bucket={bucket:>6} "
                    f"sample={is_.get('sample_idx', '?')} sha={sha[:12]}: "
                    + "; ".join(why)
                )

    print(f"\nparity check: {matched}/{total} samples match exactly")
    if failures:
        print(f"{len(failures)} parity failure(s):")
        for f in failures:
            print(f)
        return 1
    print("ALL PARITY CHECKS PASSED ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
