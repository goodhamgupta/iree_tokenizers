"""Narrow recovery for four checksum-less archives; never rebuild native assets."""

import hashlib
import io
import json
import os
import re
import tarfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from verify_release_package import MANIFEST, TARGETS


def main():
    version = os.environ["VERSION"]
    if version not in {"0.8.14", "0.8.15", "0.8.16", "0.8.17"}:
        raise ValueError("This recovery is limited to the four affected releases")
    package = Path("package")
    manifest = (package / MANIFEST).read_bytes()
    hashes = dict(re.findall(rb'"([^"]+)"\s*=>\s*"sha256:([0-9a-f]{64})"', manifest))
    expected = {f"libiree_tokenizers_native-v{version}-nif-2.15-{t}.so.tar.gz" for t in TARGETS}
    if set(k.decode() for k in hashes) != expected:
        raise ValueError("Unexpected checksum entries")
    for name in expected:
        digest = hashlib.sha256((package / name).read_bytes()).hexdigest().encode()
        if hashes[name.encode()] != digest:
            raise ValueError(f"Native asset checksum mismatch: {name}")

    with urllib.request.urlopen(f"https://hex.pm/api/packages/iree_tokenizers/releases/{version}") as response:
        metadata = json.load(response)
    with urllib.request.urlopen(f"https://repo.hex.pm/tarballs/iree_tokenizers-{version}.tar") as response:
        tarball = response.read()
    with tarfile.open(fileobj=io.BytesIO(tarball)) as archive:
        contents = archive.extractfile("contents.tar.gz").read()
    with tarfile.open(fileobj=io.BytesIO(contents), mode="r:gz") as archive:
        if MANIFEST in archive.getnames():
            if archive.extractfile(MANIFEST).read() != manifest:
                raise ValueError("Existing package has a different manifest; refusing replacement")
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write("repair=false\n")
            print("Package already includes the verified checksum manifest")
            return
        # The downloaded archive must match the tag; only packaging metadata may change.
        for member in archive.getmembers():
            if member.isfile() and (package / member.name).read_bytes() != archive.extractfile(member).read():
                raise ValueError(f"Published source differs from release tag: {member.name}")

    inserted = datetime.fromisoformat(metadata["inserted_at"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) >= inserted + timedelta(hours=1):
        raise ValueError("Hex replacement window closed; publish a new patch version instead")
    mix = package / "mix.exs"
    source = mix.read_text()
    marker = '      "mix.lock",\n'
    if source.count(marker) != 1 or MANIFEST in source:
        raise ValueError("Unexpected package file list; refusing automatic edit")
    mix.write_text(source.replace(marker, marker + f'      "{MANIFEST}",\n'))
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("repair=true\n")
    print(f"Verified original sources and native assets for {version}; added checksum to package files")


if __name__ == "__main__":
    main()
