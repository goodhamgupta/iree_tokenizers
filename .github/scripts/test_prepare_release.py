import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "prepare_release", Path(__file__).with_name("prepare_release.py")
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleasePlanTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        previous = os.getcwd()
        os.chdir(self.directory.name)
        self.addCleanup(os.chdir, previous)
        self.git("init", "-q")
        self.git("config", "user.name", "Release test")
        self.git("config", "user.email", "release@example.invalid")
        self.before = self.commit("0.8.17")

    def git(self, *args):
        return subprocess.check_output(["git", *args], text=True).strip()

    def commit(self, version, extra=""):
        Path("mix.exs").write_text(f'  @version "{version}"\n{extra}')
        self.git("add", "mix.exs")
        self.git("commit", "-qm", "Change package")
        return self.git("rev-parse", "HEAD")

    def event(self, after):
        return {"ref": "refs/heads/main", "before": self.before, "after": after}

    def test_version_bump_releases_exact_push_commit(self):
        after = self.commit("0.8.18")
        self.assertEqual(release.plan_release("push", self.event(after)),
                         {"tag": "v0.8.18", "version": "0.8.18", "sha": after})

    def test_other_mix_changes_do_not_release(self):
        after = self.commit("0.8.17", "# dependency change\n")
        self.assertIsNone(release.plan_release("push", self.event(after)))

    def test_uses_before_push_not_last_commit(self):
        self.commit("0.8.18")
        after = self.commit("0.8.18", "# later commit in same push\n")
        self.assertEqual(release.plan_release("push", self.event(after))["sha"], after)

    def test_existing_tag_cannot_be_retargeted(self):
        self.git("tag", "v0.8.18", self.before)
        after = self.commit("0.8.18")
        with self.assertRaisesRegex(ValueError, "another commit"):
            release.plan_release("push", self.event(after))

    def test_matching_tag_allows_retry(self):
        after = self.commit("0.8.18")
        self.git("tag", "v0.8.18", after)
        self.assertEqual(release.plan_release("push", self.event(after))["sha"], after)

    def test_manual_and_release_events_build_tag_not_current_main(self):
        self.git("tag", "-a", "v0.8.17", "-m", "release", self.before)
        self.commit("0.8.18")
        for name, event in [
            ("workflow_dispatch", {"inputs": {"release_tag": "v0.8.17"}}),
            ("release", {"release": {"tag_name": "v0.8.17"}}),
        ]:
            self.assertEqual(release.plan_release(name, event)["sha"], self.before)

    def test_manual_tag_must_match_package_version(self):
        self.git("tag", "v0.9.0")
        with self.assertRaisesRegex(ValueError, "does not match"):
            release.plan_release("workflow_dispatch", {"inputs": {"release_tag": "v0.9.0"}})

    def test_invalid_version_is_rejected(self):
        after = self.commit("0.8.018")
        with self.assertRaisesRegex(ValueError, "Invalid SemVer"):
            release.plan_release("push", self.event(after))

    def test_prerelease_version(self):
        after = self.commit("0.9.0-rc.1")
        self.assertEqual(release.plan_release("push", self.event(after))["tag"], "v0.9.0-rc.1")

    def test_new_branch_and_non_main_push_do_not_release(self):
        event = self.event(self.before)
        event["before"] = "0" * 40
        self.assertIsNone(release.plan_release("push", event))
        event = self.event(self.before)
        event["ref"] = "refs/heads/feature"
        self.assertIsNone(release.plan_release("push", event))

    def test_unchanged_version_writes_false_output(self):
        after = self.commit("0.8.17", "# no release\n")
        Path("event.json").write_text(json.dumps(self.event(after)))
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push",
                                     "GITHUB_EVENT_PATH": "event.json",
                                     "GITHUB_OUTPUT": "output"}):
            release.main()
        self.assertEqual(Path("output").read_text(), "release=false\n")


if __name__ == "__main__":
    unittest.main()
