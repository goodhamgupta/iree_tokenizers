from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("file_parity_issues.py")
SPEC = importlib.util.spec_from_file_location("file_parity_issues", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
tracker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tracker)


def passing_model(label: str = "org/model") -> dict:
    return {
        "label": label,
        "status": "ok",
        "all_ok": True,
        "passed": 19,
        "total": 19,
        "batch": {"status": "ok", "mismatches": []},
        "stream": {"status": "ok", "ids_equal": True},
    }


def failing_model(label: str = "org/model") -> dict:
    return {
        "label": label,
        "status": "ok",
        "all_ok": False,
        "passed": 18,
        "total": 19,
        "batch": {"status": "ok", "mismatches": []},
        "stream": {"status": "ok", "ids_equal": True},
        "failing_cases": [],
    }


class ModelLifecycleTests(unittest.TestCase):
    def test_passing_model_comments_and_closes_open_issue_as_completed(self) -> None:
        model = passing_model()
        with (
            mock.patch.object(
                tracker,
                "find_existing_issue",
                return_value={"number": 12, "state": "OPEN"},
            ),
            mock.patch.object(tracker, "comment_on_issue") as comment,
            mock.patch.object(tracker, "close_issue_as_completed") as close,
        ):
            summary = tracker.process_model_result(
                repo="owner/repo",
                model=model,
                repo_meta={"gated": False},
                upstream="",
                label="parity-failure",
                run_url="https://example.test/runs/1",
            )

        self.assertEqual(summary, "closed #12 (org/model: parity restored)")
        comment.assert_called_once()
        evidence = comment.call_args.args[2]
        self.assertIn("https://example.test/runs/1", evidence)
        self.assertIn("all_ok=true", evidence)
        self.assertIn("Cases passed: 19/19", evidence)
        close.assert_called_once_with("owner/repo", 12)

    def test_skipped_model_does_not_touch_issue_state(self) -> None:
        model = {"label": "org/model", "status": "skipped", "all_ok": True}
        with (
            mock.patch.object(tracker, "close_matching_open_issue") as close,
            mock.patch.object(tracker, "publish_deduped_issue") as publish,
        ):
            summary = tracker.process_model_result(
                repo="owner/repo",
                model=model,
                repo_meta={"gated": True},
                upstream="",
                label="parity-failure",
                run_url="https://example.test/runs/1",
            )

        self.assertIsNone(summary)
        close.assert_not_called()
        publish.assert_not_called()

    def test_gated_access_denied_closes_false_positive_without_filing(self) -> None:
        model = {
            "label": "meta/model",
            "status": "load_error",
            "reason": '{:permission_denied, "access denied"}',
        }
        with (
            mock.patch.object(
                tracker,
                "close_matching_open_issue",
                return_value="closed #44",
            ) as close,
            mock.patch.object(tracker, "publish_deduped_issue") as publish,
        ):
            summary = tracker.process_model_result(
                repo="owner/repo",
                model=model,
                repo_meta={"gated": True},
                upstream="",
                label="parity-failure",
                run_url="https://example.test/runs/2",
            )

        self.assertEqual(summary, "closed #44")
        publish.assert_not_called()
        close.assert_called_once()
        evidence = close.call_args.args[3]
        self.assertIn("permission/access-denied", evidence)
        self.assertIn("false-positive", evidence)

    def test_other_gated_load_error_remains_actionable(self) -> None:
        model = {
            "label": "meta/model",
            "status": "load_error",
            "reason": "TLS handshake failed",
        }
        with (
            mock.patch.object(
                tracker, "publish_deduped_issue", return_value="opened #8"
            ) as publish,
            mock.patch.object(tracker, "close_matching_open_issue") as close,
        ):
            summary = tracker.process_model_result(
                repo="owner/repo",
                model=model,
                repo_meta={
                    "gated": True,
                    "repo": "meta/model",
                    "format": "huggingface_json",
                },
                upstream="",
                label="parity-failure",
                run_url="https://example.test/runs/3",
            )

        self.assertEqual(summary, "opened #8")
        close.assert_not_called()
        publish.assert_called_once()
        self.assertTrue(publish.call_args.kwargs["reopen_closed"])

    def test_access_denied_for_public_repo_remains_actionable(self) -> None:
        model = {
            "label": "public/model",
            "status": "load_error",
            "reason": "access denied",
        }
        with mock.patch.object(
            tracker, "publish_deduped_issue", return_value="opened #9"
        ) as publish:
            summary = tracker.process_model_result(
                repo="owner/repo",
                model=model,
                repo_meta={
                    "gated": False,
                    "repo": "public/model",
                    "format": "huggingface_json",
                },
                upstream="",
                label="parity-failure",
                run_url="https://example.test/runs/4",
            )

        self.assertEqual(summary, "opened #9")
        publish.assert_called_once()

    def test_closed_failure_reopens_even_when_content_hash_is_unchanged(self) -> None:
        body = "same recurring failure"
        content_hash = tracker.compute_content_hash(body)
        with (
            mock.patch.object(
                tracker,
                "find_existing_issue",
                return_value={"number": 21, "state": "CLOSED"},
            ),
            mock.patch.object(
                tracker, "latest_content_hash", return_value=content_hash
            ),
            mock.patch.object(tracker, "reopen_issue") as reopen,
            mock.patch.object(tracker, "comment_on_issue") as comment,
            mock.patch.object(tracker, "create_issue") as create,
        ):
            summary = tracker.publish_deduped_issue(
                "owner/repo",
                "parity: org/model",
                body,
                "parity-failure",
                tag="org/model",
                reopen_closed=True,
            )

        self.assertEqual(summary, "reopened #21 (org/model)")
        reopen.assert_called_once_with("owner/repo", 21)
        comment.assert_not_called()
        create.assert_not_called()

    def test_run_failure_dedupe_does_not_reopen_closed_issue(self) -> None:
        body = "same run failure"
        content_hash = tracker.compute_content_hash(body)
        with (
            mock.patch.object(
                tracker,
                "find_existing_issue",
                return_value={"number": 22, "state": "CLOSED"},
            ),
            mock.patch.object(
                tracker, "latest_content_hash", return_value=content_hash
            ),
            mock.patch.object(tracker, "reopen_issue") as reopen,
            mock.patch.object(tracker, "comment_on_issue") as comment,
        ):
            summary = tracker.publish_deduped_issue(
                "owner/repo",
                tracker.RUN_FAILURE_TITLE,
                body,
                "parity-failure",
                tag="run failure",
            )

        self.assertEqual(summary, "unchanged closed #22 (run failure)")
        reopen.assert_not_called()
        comment.assert_not_called()

    def test_completed_close_uses_github_completed_reason(self) -> None:
        with mock.patch.object(tracker, "run_gh") as run_gh:
            tracker.close_issue_as_completed("owner/repo", 33)

        run_gh.assert_called_once_with(
            [
                "gh",
                "issue",
                "close",
                "33",
                "--repo",
                "owner/repo",
                "--reason",
                "completed",
            ]
        )


class MainFlowTests(unittest.TestCase):
    def test_main_processes_passing_rows_but_never_closes_skipped_rows(self) -> None:
        report = {
            "models": [
                passing_model("org/passing"),
                {"label": "org/skipped", "status": "skipped"},
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            env = {
                "GH_REPO": "owner/repo",
                "REPORT_PATH": str(Path(temp_dir) / "report.json"),
                "TRENDING_PATH": str(Path(temp_dir) / "missing-trending.json"),
                "UPSTREAM_BUGS_PATH": str(Path(temp_dir) / "missing-upstream.md"),
                "WORKFLOW_RUN_URL": "https://example.test/runs/5",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(tracker, "load_report", return_value=report),
                mock.patch.object(tracker, "ensure_label"),
                mock.patch.object(
                    tracker,
                    "close_matching_open_issue",
                    return_value="closed #40",
                ) as close,
                mock.patch.object(tracker, "publish_deduped_issue") as publish,
                mock.patch.object(tracker, "write_step_summary") as summary,
            ):
                result = tracker.main()

        self.assertEqual(result, 0)
        close.assert_called_once()
        self.assertEqual(close.call_args.kwargs["tag"], "org/passing")
        publish.assert_not_called()
        summary.assert_called_once_with(["closed #40"])


if __name__ == "__main__":
    unittest.main()
