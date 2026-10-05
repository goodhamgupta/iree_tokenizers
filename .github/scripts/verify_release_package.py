"""Verify the Hex archive contains the expected native checksums before publishing."""

import argparse
import io
import re
import tarfile
from pathlib import Path

MANIFEST = "checksum-Elixir.IREE.Tokenizers.Native.exs"
TARGETS = ("aarch64-apple-darwin", "x86_64-apple-darwin", "x86_64-unknown-linux-gnu")


def verify(package, checksum, version):
    expected = Path(checksum).read_bytes()
    entries = dict(re.findall(rb'"([^"]+)"\s*=>\s*"sha256:([0-9a-f]{64})"', expected))
    names = {f"libiree_tokenizers_native-v{version}-nif-2.15-{t}.so.tar.gz".encode() for t in TARGETS}
    if set(entries) != names:
        raise ValueError("Checksum manifest does not cover exactly the three release targets")
    with tarfile.open(package) as archive:
        contents = archive.extractfile("contents.tar.gz").read()
    with tarfile.open(fileobj=io.BytesIO(contents), mode="r:gz") as contents:
        if MANIFEST not in contents.getnames():
            raise ValueError("Hex archive is missing the native checksum manifest")
        if contents.extractfile(MANIFEST).read() != expected:
            raise ValueError("Hex archive checksum manifest differs from the release")
        mix = contents.extractfile("mix.exs").read().decode()
        if re.findall(r'^\s*@version\s+"([^"]+)"', mix, re.MULTILINE) != [version]:
            raise ValueError("Hex archive version does not match the release")
    print(f"Verified {package}: version {version}, all three native checksums included")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--checksum", default=MANIFEST)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    verify(args.package, args.checksum, args.version)
