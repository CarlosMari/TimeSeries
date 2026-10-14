"""Check archive preservation without executing any archived source."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("check_archive", ROOT / "scripts/check_archive.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class ArchiveIntegrityTests(unittest.TestCase):
    def test_repository_archive_matches_preservation_manifest(self):
        manifest = json.loads((ROOT / "archive/manifest.json").read_text())
        self.assertGreater(len(manifest["relocations"]), 50)
        self.assertEqual(checker.verify_archive(ROOT, manifest), [])

    def test_fixture_reports_missing_changed_duplicate_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "archive").mkdir()
            content = b"preserved bytes\n"
            entry = {"archived_path": "archive/example.txt", "bytes": len(content),
                     "sha256": hashlib.sha256(content).hexdigest()}
            manifest = {"relocations": [entry]}
            self.assertIn("Missing", checker.verify_archive(root, manifest)[0])
            (root / "archive/example.txt").write_bytes(content)
            self.assertEqual(checker.verify_archive(root, manifest), [])
            manifest["relocations"].append(entry)
            self.assertIn("Duplicate", checker.verify_archive(root, manifest)[0])
            manifest["relocations"] = [entry]
            (root / "archive/example.txt").write_bytes(b"changed")
            self.assertIn("Changed", checker.verify_archive(root, manifest)[0])
            manifest["relocations"] = [{**entry, "archived_path": "../outside.txt"}]
            self.assertIn("Unsafe", checker.verify_archive(root, manifest)[0])


if __name__ == "__main__":
    unittest.main()
