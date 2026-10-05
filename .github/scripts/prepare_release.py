"""Plan a release without evaluating mix.exs or changing tags/releases."""

import json
import os
import re
import subprocess
from pathlib import Path


SEMVER = re.compile(
    r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def version_at(sha):
    source = git("show", f"{sha}:mix.exs")
    versions = re.findall(r'^\s*@version\s+"([^"\n]+)"\s*$', source, re.MULTILINE)
    if len(versions) != 1:
        raise ValueError("mix.exs must define exactly one literal @version")
    version = versions[0]
    match = SEMVER.fullmatch(version)
    if not match or any(
        part.isdigit() and len(part) > 1 and part.startswith("0")
        for part in (match.group(4) or "").split(".")
    ):
        raise ValueError(f"Invalid SemVer version: {version!r}")
    return version


def commit(ref):
    return git("rev-parse", "--verify", f"{ref}^{{commit}}")


def plan_release(event_name, event):
    if event_name == "push":
        if event["ref"] != "refs/heads/main":
            return None
        before = event["before"]
        if event.get("deleted") or before == "0" * 40:
            # A new branch has no previous version to compare.
            return None
        sha = commit(event["after"])
        version = version_at(sha)
        if version_at(commit(before)) == version:
            return None
        tag = f"v{version}"
        existing = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
            text=True, capture_output=True,
        )
        if existing.returncode == 0 and existing.stdout.strip() != sha:
            raise ValueError(f"Tag {tag} already points to another commit")
    elif event_name in ("release", "workflow_dispatch"):
        tag = (
            event["release"]["tag_name"] if event_name == "release"
            else event["inputs"]["release_tag"]
        )
        if not tag.startswith("v") or not SEMVER.fullmatch(tag[1:]):
            raise ValueError(f"Expected a v-prefixed SemVer tag, got {tag!r}")
        sha = commit(f"refs/tags/{tag}")
        version = version_at(sha)
        if tag != f"v{version}":
            raise ValueError(f"Tag {tag} does not match mix.exs version {version}")
    else:
        raise ValueError(f"Unsupported release event: {event_name}")
    return {"tag": tag, "sha": sha, "version": version}


def main():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    plan = plan_release(os.environ["GITHUB_EVENT_NAME"], event)
    outputs = {"release": "true" if plan else "false", **(plan or {})}
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for key, value in outputs.items():
            output.write(f"{key}={value}\n")
    print(f"Release plan: {plan}" if plan else "Version unchanged; no release.")


if __name__ == "__main__":
    main()
