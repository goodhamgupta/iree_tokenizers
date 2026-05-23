#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Reads bench/results/parity_report.json (and the trending model list) and
opens / refreshes a GitHub issue per failing model.

Invoked from the parity-monitor workflow. Uses the preinstalled `gh` CLI for
all GitHub API operations so no extra Python deps are required — `GH_TOKEN`
must be set in the environment (the workflow wires it from `GITHUB_TOKEN`).

Env vars:
  REPORT_PATH         default: bench/results/parity_report.json
  TRENDING_PATH       default: bench/results/trending_models.json
  UPSTREAM_BUGS_PATH  default: docs/UPSTREAM_BUGS.md
  PARITY_LABEL        default: parity-failure
  WORKFLOW_RUN_URL    provided by the workflow
  GH_TOKEN            required (from secrets.GITHUB_TOKEN)
  GH_REPO             owner/repo (provided by GitHub Actions as GITHUB_REPOSITORY)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

HASH_MARKER_PREFIX = "<!-- parity-content-hash: "
HASH_MARKER_SUFFIX = " -->"
HASH_MARKER_RE = re.compile(r"<!--\s*parity-content-hash:\s*([0-9a-f]+)\s*-->")

# Title of the single tracking issue filed when the parity matrix crashes
# before producing a report. Kept stable so repeated crashes dedupe onto one
# issue instead of opening a new one every scheduled run.
RUN_FAILURE_TITLE = "parity-monitor: run crashed before producing a report"


def main() -> int:
    report_path = Path(os.environ.get("REPORT_PATH", "bench/results/parity_report.json"))
    trending_path = Path(
        os.environ.get("TRENDING_PATH", "bench/results/trending_models.json")
    )
    upstream_path = Path(os.environ.get("UPSTREAM_BUGS_PATH", "docs/UPSTREAM_BUGS.md"))
    label = os.environ.get("PARITY_LABEL", "parity-failure")
    run_url = os.environ.get("WORKFLOW_RUN_URL", "(no run URL)")
    repo = os.environ.get("GH_REPO") or os.environ.get("GITHUB_REPOSITORY")

    if not repo:
        print("error: GH_REPO / GITHUB_REPOSITORY is required", file=sys.stderr)
        return 1

    report = load_report(report_path)
    if report is None:
        # The parity matrix produced no readable report — `validate_parity.exs`
        # crashed or was killed before writing one (e.g. a tokenizer model
        # aborted the BEAM with a native assertion). File a single tracking
        # issue so the failure stays visible, then exit 0: the monitor still
        # did its job, and the parity matrix step's own red status already
        # records that something went wrong.
        print(
            f"warning: no readable parity report at {report_path} — "
            "treating as a crashed parity-monitor run",
            file=sys.stderr,
        )
        ensure_label(repo, label)
        file_run_failure_issue(repo, label, run_url, report_path)
        return 0

    trending = (
        json.loads(trending_path.read_text()) if trending_path.exists() else []
    )
    upstream = upstream_path.read_text() if upstream_path.exists() else ""
    trending_by_label = {m["label"]: m for m in trending}

    ensure_label(repo, label)

    failures = [m for m in report.get("models", []) if is_failure(m)]
    print(f"Found {len(failures)} failing models out of {len(report.get('models', []))}.")

    summary_lines: list[str] = []

    for model in failures:
        repo_meta = trending_by_label.get(model["label"])
        known_bug = is_known_upstream_bug(model["label"], upstream)
        title = f"parity: {model['label']}"
        body = render_issue_body(
            model=model, repo_meta=repo_meta, known_bug=known_bug, run_url=run_url
        )
        summary_lines.append(
            publish_deduped_issue(
                repo,
                title,
                body,
                label,
                tag=model["label"],
                create_skip_reason="known upstream bug" if known_bug else None,
            )
        )

    write_step_summary(summary_lines)
    return 0


def load_report(path: Path) -> dict | None:
    """Loads the parity report JSON.

    Returns None when the file is missing or unparseable, which means the
    parity matrix crashed (or was killed) before writing a complete report.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"warning: {path} exists but is not valid JSON: {e}", file=sys.stderr)
        return None


def file_run_failure_issue(
    repo: str, label: str, run_url: str, report_path: Path
) -> None:
    """Files (or refreshes) the tracking issue for a crashed parity-monitor run."""
    body = render_run_failure_body(run_url, report_path)
    summary = publish_deduped_issue(
        repo, RUN_FAILURE_TITLE, body, label, tag="run failure"
    )
    write_step_summary([summary])


def render_run_failure_body(run_url: str, report_path: Path) -> str:
    parity_outcome = os.environ.get("PARITY_OUTCOME", "").strip()
    lines = [
        "Automated alert filed by `parity-monitor`.",
        "",
        f"The parity matrix step produced no readable `{report_path}`, which "
        "means `validate_parity.exs` crashed or was killed before it finished "
        "— for example, a tokenizer model aborted the BEAM with a native "
        "assertion. No per-model parity issues could be filed for this run.",
        "",
        f"- Workflow run: {run_url}",
    ]
    if parity_outcome:
        lines.append(f"- Parity matrix step outcome: `{parity_outcome}`")
    lines += [
        "",
        "### What to do",
        "",
        "1. Open the workflow run above and read the **Run parity matrix** "
        "step log; its final lines identify the offending model and the "
        "native failure.",
        "2. Reproduce locally and fix the root cause.",
        "3. Per-model parity issues resume automatically once a run produces "
        "a report again.",
        "",
        "This issue is deduplicated: while the run keeps crashing the same "
        "way no new comments are added.",
    ]
    return "\n".join(lines)


def publish_deduped_issue(
    repo: str,
    title: str,
    body: str,
    label: str,
    *,
    tag: str,
    create_skip_reason: str | None = None,
) -> str:
    """Files or refreshes a parity-monitor issue, deduplicating across runs.

    - If an issue with this title and label already exists, comment only when
      the content hash has changed; otherwise leave it untouched.
    - If no issue exists and ``create_skip_reason`` is provided, skip creation
      and return a ``skipped`` summary (used to honor the known-upstream-bug
      list without suppressing follow-up comments on an already-open issue).
    - Otherwise open a new issue.

    Returns the step-summary line describing the action taken. ``tag`` labels
    the summary so per-model and run-failure flows produce distinct entries.
    """
    content_hash = compute_content_hash(body)
    body_with_marker = (
        f"{body}\n\n{HASH_MARKER_PREFIX}{content_hash}{HASH_MARKER_SUFFIX}\n"
    )

    existing = find_existing_issue(repo, title, label)
    if existing:
        state = existing["state"].lower()
        number = existing["number"]
        print(f"  exists: #{number} for {tag} (state={state})")
        if latest_content_hash(repo, number) == content_hash:
            print(
                f"    skip comment on #{number}: content unchanged "
                f"(hash={content_hash[:12]})"
            )
            return f"unchanged {state} #{number} ({tag})"
        comment_on_issue(repo, number, body_with_marker)
        return f"commented on {state} #{number} ({tag})"

    if create_skip_reason is not None:
        print(f"  {tag}: {create_skip_reason}, skipping issue creation")
        return f"skipped {tag} ({create_skip_reason})"

    number = create_issue(repo, title, body_with_marker, [label])
    print(f"  opened #{number} for {tag}")
    return f"opened #{number} ({tag})"


def is_failure(model: dict) -> bool:
    if model.get("status") == "skipped":
        return False
    if model.get("status") == "load_error":
        return True
    return model.get("all_ok") is False


def run_gh(args: list[str], *, input: str | None = None) -> subprocess.CompletedProcess:
    """Run a `gh` command, surfacing stderr on failure for fast CI debugging."""
    try:
        return subprocess.run(
            args, check=True, capture_output=True, text=True, input=input
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            f"\n`gh` command failed (exit {e.returncode}):\n"
            f"  args:   {args}\n"
            f"  stdout: {e.stdout!r}\n"
            f"  stderr: {e.stderr!r}\n"
        )
        raise


def ensure_label(repo: str, label: str) -> None:
    result = run_gh(
        ["gh", "label", "list", "--repo", repo, "--json", "name", "--limit", "200"]
    )
    existing = {entry["name"] for entry in json.loads(result.stdout)}
    if label in existing:
        return

    run_gh(
        [
            "gh",
            "label",
            "create",
            label,
            "--repo",
            repo,
            "--color",
            "d73a4a",
            "--description",
            "IREE.Tokenizers vs elixir-nx/tokenizers parity regression",
        ]
    )


def find_existing_issue(repo: str, title: str, label: str) -> dict | None:
    # `gh issue list --search` does a title substring match. We filter to an
    # exact title match locally so we never collide with similarly-titled
    # issues (e.g. `parity: foo/bar` vs `parity: foo/bar-2`).
    result = run_gh(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--label",
            label,
            "--state",
            "all",
            "--search",
            f'"{title}" in:title',
            "--json",
            "number,title,state",
            "--limit",
            "50",
        ]
    )
    items = json.loads(result.stdout)
    for item in items:
        if item["title"] == title:
            return item
    return None


def compute_content_hash(body: str) -> str:
    # Strip lines that change every run (workflow URL) so identical failures
    # produce the same hash across runs.
    normalized_lines = [
        line for line in body.splitlines() if not line.startswith("- Workflow run:")
    ]
    normalized = "\n".join(normalized_lines).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def latest_content_hash(repo: str, number: int) -> str | None:
    """Return the parity-content-hash marker on the newest comment, or the
    issue body if no comments exist. Returns None if no marker is found."""
    result = run_gh(
        [
            "gh",
            "issue",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "body,comments",
        ]
    )
    data = json.loads(result.stdout)
    comments = data.get("comments") or []
    if comments:
        # `gh issue view` returns comments in ascending order.
        candidate = comments[-1].get("body") or ""
    else:
        candidate = data.get("body") or ""
    match = HASH_MARKER_RE.search(candidate)
    return match.group(1) if match else None


def comment_on_issue(repo: str, number: int, body: str) -> None:
    run_gh(
        ["gh", "issue", "comment", str(number), "--repo", repo, "--body-file", "-"],
        input=body,
    )


def create_issue(repo: str, title: str, body: str, labels: list[str]) -> int:
    result = run_gh(
        [
            "gh",
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            title,
            "--label",
            ",".join(labels),
            "--body-file",
            "-",
        ],
        input=body,
    )
    # `gh issue create` prints the issue URL on stdout.
    url = result.stdout.strip().splitlines()[-1]
    return int(url.rsplit("/", 1)[-1])


def is_known_upstream_bug(label: str, upstream_md: str) -> bool:
    if not upstream_md:
        return False
    pattern = re.compile(rf"`{re.escape(label)}`")
    return bool(pattern.search(upstream_md))


def render_issue_body(
    *,
    model: dict,
    repo_meta: dict | None,
    known_bug: bool,
    run_url: str,
) -> str:
    lines: list[str] = []
    lines.append("Automated parity regression filed by `parity-monitor`.")
    lines.append("")
    lines.append(f"- Workflow run: {run_url}")
    if repo_meta:
        lines.append(f"- Repo: `{repo_meta['repo']}`")
        lines.append(f"- Format: `{repo_meta['format']}`")
        if repo_meta.get("pipeline_tag"):
            lines.append(f"- Pipeline tag: `{repo_meta['pipeline_tag']}`")
        score = repo_meta.get("trending_score")
        if isinstance(score, (int, float)):
            lines.append(f"- Trending score: {score}")
        if repo_meta.get("gated"):
            lines.append("- Gated: yes (HF_TOKEN required)")
    if known_bug:
        lines.append("- Known upstream bug: yes (matched `docs/UPSTREAM_BUGS.md`)")
    lines.append("")

    repo_id = (repo_meta or {}).get("repo") or model["label"]

    if model.get("status") == "load_error":
        lines.append("### Load error")
        lines.append("")
        lines.append("```")
        lines.append(model.get("reason") or "(no reason)")
        lines.append("```")
        lines.append("")
        lines.append("### Reproduction")
        lines.append("")
        lines.append("```elixir")
        lines.append(f'{{:ok, t}} = IREE.Tokenizers.Tokenizer.from_pretrained("{repo_id}")')
        lines.append("```")
        return "\n".join(lines)

    lines.append("### Summary")
    lines.append("")
    lines.append(f"- Cases passed: {model['passed']}/{model['total']}")

    batch = model.get("batch") or {}
    if batch.get("status") == "error":
        lines.append(f"- Batch encode: **ERROR** {batch.get('reason')}")
    elif batch.get("mismatches"):
        indices = ", ".join(str(i) for i in batch["mismatches"])
        lines.append(f"- Batch encode: **MISMATCH** at indices [{indices}]")
    elif batch:
        lines.append("- Batch encode: ok")

    stream = model.get("stream") or {}
    if stream.get("status") == "error":
        lines.append(f"- Stream encode: **ERROR** {stream.get('reason')}")
    elif stream.get("ids_equal") is False:
        lines.append(
            "- Stream encode: **MISMATCH** "
            f"(streamed={stream.get('streamed_len')}, oneshot={stream.get('oneshot_len')}, "
            f"first_diff={json.dumps(stream.get('first_diff'))})"
        )
    elif stream:
        lines.append("- Stream encode: ok")
    lines.append("")

    failing_cases = model.get("failing_cases") or []
    if failing_cases:
        lines.append("### Failing cases")
        lines.append("")
        lines.append(
            "| case | bytes | add_special | iree_ids | hf_ids | ids= | decoded= | first_diff | error |"
        )
        lines.append("| --- | ---: | :---: | ---: | ---: | :---: | :---: | --- | --- |")
        for c in failing_cases:
            for v in c.get("variants", []):
                if not v.get("error") and v.get("ids_equal") and v.get("decoded_equal"):
                    continue
                fd = v.get("first_diff")
                fd_str = (
                    f"idx={fd['index']}, iree={fd['iree']}, hf={fd['hf']}"
                    if fd
                    else "-"
                )
                lines.append(
                    "| {name} | {bytes} | {add_special} | {iree} | {hf} | {ids} | {decoded} | {fd} | {err} |".format(
                        name=c.get("name", ""),
                        bytes=c.get("bytes", ""),
                        add_special=v.get("add_special"),
                        iree=v.get("iree_ids_len") if v.get("iree_ids_len") is not None else "-",
                        hf=v.get("hf_ids_len") if v.get("hf_ids_len") is not None else "-",
                        ids="✅" if v.get("ids_equal") else "❌",
                        decoded="✅" if v.get("decoded_equal") else "❌",
                        fd=fd_str,
                        err=v.get("error") or "",
                    )
                )
        lines.append("")

    lines.append("### Reproduction")
    lines.append("")
    lines.append("```elixir")
    lines.append("# cd bench && HF_TOKEN=hf_... mix run -e '")
    lines.append('Mix.Task.run("app.start")')
    lines.append("alias IREE.Tokenizers.Tokenizer, as: IREE")
    lines.append("alias Tokenizers.Tokenizer, as: HF")
    lines.append("alias Tokenizers.Encoding, as: HFEnc")
    lines.append("")
    sp_suffix = (
        ", format: :sentencepiece_model"
        if (repo_meta or {}).get("format") == "sentencepiece_model"
        else ""
    )
    lines.append(f'{{:ok, iree}} = IREE.from_pretrained("{repo_id}"{sp_suffix})')
    lines.append(f'{{:ok, hf}}   = HF.from_pretrained("{repo_id}")')

    first_case = failing_cases[0] if failing_cases else None
    first_failure = None
    if first_case:
        for v in first_case.get("variants", []):
            if v.get("error") or not v.get("ids_equal") or not v.get("decoded_equal"):
                first_failure = v
                break

    if first_case and first_failure:
        add_special = str(first_failure["add_special"]).lower()
        lines.append("")
        lines.append(
            f"# input: {first_case['name']}, "
            f"add_special_tokens: {add_special}, "
            f"bytes: {first_case.get('bytes')}"
        )
        preview = first_case.get("text_preview") or ""
        # ensure_ascii=False keeps CJK / emoji literals readable; the JSON
        # quoting rules align with Elixir's own string literals.
        lines.append(f"text = {json.dumps(preview, ensure_ascii=False)}")
        lines.append(
            f"{{:ok, ie}} = IREE.encode(iree, text, add_special_tokens: {add_special})"
        )
        lines.append(
            f"{{:ok, he}} = HF.encode(hf, text, add_special_tokens: {add_special})"
        )
        lines.append("ie.ids == HFEnc.get_ids(he)")
    else:
        lines.append("")
        lines.append("# Failure occurred in batch / stream path rather than a single case.")
        lines.append("# Re-run the parity matrix for this model to reproduce.")
    lines.append("# '")
    lines.append("```")
    lines.append("")
    lines.append("Or, run the full parity matrix for this single model:")
    lines.append("")
    lines.append("```bash")
    lines.append("cd bench")
    lines.append(
        f'HF_TOKEN=hf_... MODEL_FILTER={shlex.quote(model["label"])} mix run validate_parity.exs'
    )
    lines.append("```")

    return "\n".join(lines)


def write_step_summary(lines: list[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    with open(summary_path, "a", encoding="utf-8") as f:
        if lines:
            f.write("## Parity monitor — issue actions\n\n")
            for line in lines:
                f.write(f"- {line}\n")
        else:
            f.write("## Parity monitor\n\nAll monitored models passed.\n")


if __name__ == "__main__":
    sys.exit(main())
