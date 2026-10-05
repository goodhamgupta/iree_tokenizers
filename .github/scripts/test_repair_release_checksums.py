import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import repair_release_checksums as repair


class RepairTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        previous = os.getcwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, previous)
        Path("package").mkdir()
        self.source = '@version "0.8.17"\n      "mix.lock",\n'
        Path("package/mix.exs").write_text(self.source)
        entries = []
        for target in repair.TARGETS:
            name = f"libiree_tokenizers_native-v0.8.17-nif-2.15-{target}.so.tar.gz"
            body = target.encode()
            Path("package", name).write_bytes(body)
            entries.append(f'"{name}" => "sha256:{hashlib.sha256(body).hexdigest()}",')
        self.manifest = ("%{\n" + "\n".join(entries) + "\n}").encode()
        Path("package", repair.MANIFEST).write_bytes(self.manifest)
        self.old = False
        self.packaged_manifest = None
        self.published_source = self.source
        self.env = patch.dict(os.environ, {"VERSION": "0.8.17", "GITHUB_OUTPUT": "output"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def response(self, url):
        if "api/packages" in url:
            inserted = datetime.now(timezone.utc) - timedelta(minutes=120 if self.old else 10)
            return io.BytesIO(json.dumps({"inserted_at": inserted.isoformat()}).encode())
        inner = io.BytesIO()
        files = {"mix.exs": self.published_source.encode()}
        if self.packaged_manifest is not None:
            files[repair.MANIFEST] = self.packaged_manifest
        with tarfile.open(fileobj=inner, mode="w:gz") as tar:
            for name, body in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(body)
                tar.addfile(info, io.BytesIO(body))
        outer = io.BytesIO()
        with tarfile.open(fileobj=outer, mode="w") as tar:
            info = tarfile.TarInfo("contents.tar.gz")
            info.size = len(inner.getvalue())
            tar.addfile(info, io.BytesIO(inner.getvalue()))
        return io.BytesIO(outer.getvalue())

    def run_repair(self):
        with patch.object(repair.urllib.request, "urlopen", side_effect=self.response):
            repair.main()

    def test_adds_only_checksum_file_to_packaging(self):
        self.run_repair()
        self.assertEqual(Path("package/mix.exs").read_text(), self.source + f'      "{repair.MANIFEST}",\n')
        self.assertEqual(Path("output").read_text(), "repair=true\n")

    def test_already_corrected_package_skips(self):
        self.packaged_manifest = self.manifest
        self.run_repair()
        self.assertEqual(Path("output").read_text(), "repair=false\n")

    def test_closed_replacement_window_fails(self):
        self.old = True
        with self.assertRaisesRegex(ValueError, "window closed"):
            self.run_repair()

    def test_changed_native_asset_fails(self):
        next(Path("package").glob("*.tar.gz")).write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "asset checksum mismatch"):
            self.run_repair()

    def test_changed_source_fails(self):
        self.published_source += "# drift"
        with self.assertRaisesRegex(ValueError, "source differs"):
            self.run_repair()
