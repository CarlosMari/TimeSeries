"""Safe stdlib-only fixtures; no original data or scientific packages needed."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "catalog_artifacts", Path(__file__).resolve().parents[1] / "scripts/catalog_artifacts.py")
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


class ArtifactCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def put(self, name, value=b"opaque bytes, not a pickle or checkpoint"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return path

    def cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = catalog.main(["--root", str(self.root), *args])
        return result, stdout.getvalue() + stderr.getvalue()

    def test_scoped_discovery_and_metadata_only(self):
        self.put("RESULTS.json", {"model_config": {"latent_dim": 3}})
        self.put("results/legacy/old.json", {"seed": 42})
        dataset = self.put("data/nested/example.pkl")
        checkpoint = self.put("model_ckpts/example.pth")
        for ignored in ("env/RESULTS.json", "wandb/RESULTS.json", ".git/RESULTS.json",
                        "results/manifest.json", "model_ckpts/nested/hidden.pth",
                        "data/.venv/hidden.pkl", "data/wandb/hidden.pkl"):
            self.put(ignored, {})
        original_open = Path.open

        def safe_open(path, *args, **kwargs):
            if path in (dataset, checkpoint):
                self.fail("Large artifact contents opened without --hash-large-files")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", safe_open):
            manifest = catalog.build_manifest(self.root)
        self.assertEqual(manifest["counts"], {
            "result": 2, "dataset": 1, "checkpoint": 1, "total": 4, "invalid_json": 0})
        for entry in manifest["artifacts"]:
            self.assertNotIn(str(self.root), json.dumps(entry))
            if entry["kind"] != "result":
                self.assertIsNone(entry["sha256"])
            else:
                self.assertEqual(entry["sha256"], hashlib.sha256(
                    (self.root / entry["path"]).read_bytes()).hexdigest())

    def test_large_hashes_are_opt_in(self):
        path = self.put("data/x.pkl")
        self.put("model_ckpts/x.pth")
        manifest = catalog.build_manifest(self.root, hash_large_files=True)
        for entry in manifest["artifacts"]:
            self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(self.cli("--hash-large-files")[0], 0)
        self.assertEqual(self.cli("--hash-large-files", "--check")[0], 0)
        self.assertEqual(self.cli("--check")[0], 1)

    def test_relocation_preserves_result_identity_and_removes_root_entry(self):
        content = b'{"model_type": "example", "seed": 42}'
        source = self.put("RESULTS_EXAMPLE.json", content)
        before = catalog.build_manifest(self.root)["artifacts"][0]
        destination = self.root / "results/legacy/RESULTS_EXAMPLE.json"
        destination.parent.mkdir(parents=True)
        source.rename(destination)
        after = catalog.build_manifest(self.root)["artifacts"][0]
        self.assertEqual(after["path"], "results/legacy/RESULTS_EXAMPLE.json")
        self.assertEqual(after["sha256"], before["sha256"])
        self.assertEqual(after["bytes"], before["bytes"])
        self.assertEqual(after["explicit_metadata"], before["explicit_metadata"])
        self.assertFalse(source.exists())

    def test_explicit_metadata_and_unverified_inferences(self):
        self.put("RESULTS_M8_OOD_Exp5.json", {
            "cvae-heteroscedastic:m8_seed42": {
                "model_type": "cvae-heteroscedastic", "seed": 42,
                "checkpoint": "model_ckpts/actual.pth",
                "test_path": "data/custom.pkl", "reconstruction": {"N_recon": 50}}})
        self.put("data/TEST_OOD_Exp5.pkl")
        entry = catalog.build_manifest(self.root)["artifacts"][0]
        self.assertEqual(entry["model_keys"], ["cvae-heteroscedastic:m8_seed42"])
        explicit = {m["source"]: m["value"] for m in entry["explicit_metadata"]}
        self.assertEqual(explicit["/cvae-heteroscedastic:m8_seed42/seed"], 42)
        refs = entry["references"]
        self.assertTrue(any(r["path"] == "model_ckpts/m8_seed42.pth"
                            and r["provenance"] == "inferred" for r in refs))
        self.assertTrue(any(r["path"] == "data/custom.pkl"
                            and r["provenance"] == "explicit" for r in refs))
        self.assertTrue(any(r["path"] == "data/TEST_OOD_Exp5.pkl"
                            and r["candidate_present"] for r in refs))
        self.assertTrue(all(r["verified"] is False for r in refs))
        self.assertTrue({"ood_compound_shift", "m8_mean_vs_sampled"}.issubset(
            {c["category"] for c in entry["caveats"]}))

    def test_caveats_do_not_claim_unknown_sorting(self):
        self.put("RESULTS_UNKNOWN.json", {"score": 1})
        self.put("RESULTS_FINAL_NOSORT.json", {})
        entries = {e["path"]: e for e in catalog.build_manifest(self.root)["artifacts"]}
        unknown = entries["RESULTS_UNKNOWN.json"]
        self.assertEqual(unknown["references"], [])
        self.assertEqual(unknown["model_keys"], [])
        self.assertIn("not established", unknown["caveats"][0]["message"])
        self.assertEqual(entries["RESULTS_FINAL_NOSORT.json"]["caveats"][0]["category"],
                         "no_sort_scale_head")

    def test_alternate_schemas_and_bad_json(self):
        self.put("RESULTS_noise.json", {"models": {"baseline": {"seed": 4}}})
        self.put("RESULTS_list.json", [{"N_test": 5}])
        self.put("RESULTS_bad.json", b"{broken")
        self.put("results/legacy/nonutf8.json", b"\xff")
        manifest = catalog.build_manifest(self.root)
        self.assertEqual(manifest["counts"]["invalid_json"], 2)
        noise = next(e for e in manifest["artifacts"] if e["path"] == "RESULTS_noise.json")
        self.assertEqual(noise["model_keys"], ["baseline"])
        self.assertEqual(len(manifest["artifacts"]), 4)

    def test_redaction_and_nonfinite_metadata(self):
        self.put("RESULTS.json", {"checkpoint": "/private/machine/model.pth",
                                  "metadata": {"path": "C:\\private\\file", "value": float("nan"),
                                               "/private/machine/key": {"seed": 5}}})
        rendered = catalog.render_manifest(catalog.build_manifest(self.root))
        self.assertNotIn("/private/machine", rendered)
        self.assertNotIn("C:", rendered)
        self.assertIn("redacted-path", rendered)
        json.loads(rendered)

    def test_stable_across_roots_and_creation_order(self):
        self.put("RESULTS_Z.json", {"seed": 9})
        self.put("RESULTS_A.json", {"seed": 1})
        with tempfile.TemporaryDirectory() as other:
            root = Path(other)
            for name in ("RESULTS_A.json", "RESULTS_Z.json"):
                (root / name).write_bytes((self.root / name).read_bytes())
            self.assertEqual(catalog.render_manifest(catalog.build_manifest(self.root)),
                             catalog.render_manifest(catalog.build_manifest(root)))

    def test_check_missing_current_stale_never_writes(self):
        self.put("RESULTS.json", {})
        code, message = self.cli("--check")
        self.assertEqual(code, 1)
        self.assertIn("Missing", message)
        output = self.root / catalog.DEFAULT_OUTPUT
        self.assertFalse(output.parent.exists())
        self.assertEqual(self.cli()[0], 0)
        saved = output.read_bytes()
        self.assertEqual(self.cli("--check")[0], 0)
        self.put("RESULTS.json", {"seed": 10})
        code, message = self.cli("--check")
        self.assertEqual(code, 1)
        self.assertIn("Stale", message)
        self.assertEqual(output.read_bytes(), saved)
        self.assertEqual(self.cli()[0], 0)
        self.assertEqual(self.cli("--check")[0], 0)

    def test_refuses_input_collisions_and_outside_output(self):
        source = self.put("RESULTS.json", {"seed": 1})
        before = source.read_bytes()
        for output in ("RESULTS.json", "RESULTS_new.json", "results/legacy/new.json",
                       "data/new.pkl", "model_ckpts/new.pth", "../outside.json", "env/out.json"):
            with self.subTest(output=output):
                self.assertEqual(self.cli("--output", output)[0], 2)
        self.assertEqual(source.read_bytes(), before)
        alias = self.root / "alias.json"
        alias.hardlink_to(source)
        self.assertEqual(self.cli("--output", "alias.json")[0], 2)

    def test_protected_noninventory_files_are_never_overwritten(self):
        fixtures = {
            "figures/final.pdf": b"%PDF-protected figure",
            "archive/source.py": b"print('protected source')\n",
            "RESULTS_COMPARISON.md": b"# Protected comparison\n",
            "README.md": b"# Protected readme\n",
            "ignored/private.json": b'{"research": "protected"}',
            "archive/source.json": b'{"source": [1, 2, 3]}',
            catalog.DEFAULT_OUTPUT: b'{"not": "a catalog manifest"}',
        }
        paths = {name: self.put(name, data) for name, data in fixtures.items()}
        for name in fixtures:
            for check in ([], ["--check"]):
                with self.subTest(output=name, check=bool(check)):
                    self.assertEqual(self.cli("--output", name, *check)[0], 2)
                    for protected, path in paths.items():
                        self.assertEqual(path.read_bytes(), fixtures[protected])

    def test_custom_new_json_output_and_recognized_replacement(self):
        self.put("RESULTS.json", {"seed": 1})
        name = "custom/catalog.json"
        output = self.root / name
        self.assertEqual(self.cli("--output", name, "--check")[0], 1)
        self.assertFalse(output.parent.exists())
        self.assertEqual(self.cli("--output", name)[0], 0)
        self.assertTrue(catalog._recognized_manifest(json.loads(output.read_bytes())))
        saved = output.read_bytes()
        self.put("RESULTS.json", {"seed": 2})
        self.assertEqual(self.cli("--output", name, "--check")[0], 1)
        self.assertEqual(output.read_bytes(), saved)
        self.assertEqual(self.cli("--output", name)[0], 0)
        self.assertNotEqual(output.read_bytes(), saved)
        self.assertEqual(self.cli("--output", name, "--check")[0], 0)
        self.assertEqual(self.cli("--output", "custom/new.md")[0], 2)
        self.assertFalse((self.root / "custom/new.md").exists())

    def test_manifest_name_or_keys_alone_do_not_allow_replacement(self):
        invalid = [b"not json", b"\xff", [], {"schema_version": 1},
                   {"schema_version": 1, "artifacts": [], "counts": {}, "limitations": []},
                   {"schema_version": 1, "artifacts": "not a list",
                    "counts": {}, "limitations": []}]
        for value in invalid:
            with self.subTest(value=value):
                path = self.put(catalog.DEFAULT_OUTPUT, value)
                saved = path.read_bytes()
                self.assertEqual(self.cli()[0], 2)
                self.assertEqual(self.cli("--check")[0], 2)
                self.assertEqual(path.read_bytes(), saved)

    def test_symlinked_output_ancestors_are_rejected(self):
        destination = self.root / "real"
        destination.mkdir()
        (self.root / "alias").symlink_to(destination, target_is_directory=True)
        for check in ([], ["--check"]):
            self.assertEqual(self.cli("--output", "alias/new.json", *check)[0], 2)
            self.assertFalse((destination / "new.json").exists())
        existing = self.put("real/existing.json", catalog.build_manifest(self.root))
        saved = existing.read_bytes()
        self.assertEqual(self.cli("--output", "alias/existing.json")[0], 2)
        self.assertEqual(existing.read_bytes(), saved)
        with tempfile.TemporaryDirectory() as external:
            (self.root / "external").symlink_to(external, target_is_directory=True)
            self.assertEqual(self.cli("--output", "external/new.json")[0], 2)
            self.assertFalse((Path(external) / "new.json").exists())

    def test_symlinks_are_not_followed(self):
        self.put("RESULTS.json", {})
        with tempfile.TemporaryDirectory() as external:
            outside = Path(external)
            (outside / "hidden.pkl").write_bytes(b"not loaded")
            (self.root / "data").symlink_to(outside, target_is_directory=True)
            (self.root / "RESULTS_link.json").symlink_to(self.root / "RESULTS.json")
            (self.root / "out.json").symlink_to(self.root / "RESULTS.json")
            self.assertEqual(catalog.build_manifest(self.root)["counts"]["total"], 1)
            self.assertEqual(self.cli("--output", "out.json")[0], 2)


if __name__ == "__main__":
    unittest.main()
