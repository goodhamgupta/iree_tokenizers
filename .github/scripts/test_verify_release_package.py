import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("verify_release_package", Path(__file__).with_name("verify_release_package.py"))
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class PackageVerificationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checksum = self.root / verifier.MANIFEST
        self.data = ('%{\n' + ''.join(
            f'"libiree_tokenizers_native-v0.8.18-nif-2.15-{t}.so.tar.gz" => "sha256:{"a" * 64}",\n'
            for t in verifier.TARGETS) + '}\n').encode()
        self.checksum.write_bytes(self.data)

    def package(self, manifest=True, data=None, version="0.8.18"):
        inner = io.BytesIO()
        with tarfile.open(fileobj=inner, mode="w:gz") as archive:
            files = {"mix.exs": f'@version "{version}"\n'.encode()}
            if manifest:
                files[verifier.MANIFEST] = self.data if data is None else data
            for name, body in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))
        path = self.root / "package.tar"
        with tarfile.open(path, mode="w") as archive:
            body = inner.getvalue()
            info = tarfile.TarInfo("contents.tar.gz")
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
        return path

    def test_complete_package(self):
        verifier.verify(self.package(), self.checksum, "0.8.18")

    def test_missing_manifest_fails(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            verifier.verify(self.package(manifest=False), self.checksum, "0.8.18")

    def test_different_manifest_fails(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            verifier.verify(self.package(data=b"%{}"), self.checksum, "0.8.18")

    def test_wrong_version_fails(self):
        with self.assertRaisesRegex(ValueError, "version does not match"):
            verifier.verify(self.package(version="0.8.17"), self.checksum, "0.8.18")

    def test_incomplete_target_coverage_fails(self):
        self.checksum.write_bytes(b"%{}")
        with self.assertRaisesRegex(ValueError, "three release targets"):
            verifier.verify(self.package(), self.checksum, "0.8.18")
