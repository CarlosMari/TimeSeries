"""Data-free checks for the maintained repository layout and source syntax."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent


class RepositoryLayoutTests(unittest.TestCase):
    def test_root_python_entry_points_are_only_trainers(self):
        self.assertEqual(
            {p.name for p in ROOT.glob("*.py")},
            {"train_cvae.py", "train_pivot.py", "train_baseline.py", "train_glv_regression.py"},
        )

    def test_required_local_training_configuration_exists(self):
        for name in ["src/utils/__init__.py", "src/utils/config.py", "src/utils/loss.py"]:
            with self.subTest(path=name):
                self.assertTrue((ROOT / name).is_file())

    def test_maintained_and_archived_python_sources_parse_without_importing(self):
        paths = list(ROOT.glob("train_*.py"))
        for directory in ["analysis", "src", "data_generation", "scripts", "archive/scripts"]:
            paths.extend((ROOT / directory).rglob("*.py"))
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_historical_narratives_and_environment_are_preserved(self):
        for name in [
            "archive/docs/PROJECT_2026-05.md", "archive/docs/PLAN_2026-05.md",
            "archive/environment/requirements-legacy.txt", "archive/README.md",
            "docs/SCIENTIFIC_AUDIT.md", "docs/REPRODUCING.md", "results/README.md",
        ]:
            with self.subTest(path=name):
                self.assertTrue((ROOT / name).is_file())

    def test_minimal_environment_declares_eagerly_imported_latent_ode_dependency(self):
        dependencies = (ROOT / "requirements.txt").read_text().splitlines()
        self.assertIn("torchdiffeq==0.2.5", dependencies)
        self.assertFalse(any(line.startswith("appnope") for line in dependencies))


if __name__ == "__main__":
    unittest.main()
