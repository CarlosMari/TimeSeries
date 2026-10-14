"""Launcher contract tests. Only tiny temporary fixtures are ever executed."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("run_legacy", ROOT / "scripts/run_legacy.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LegacyLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "archive/scripts"
        self.directory.mkdir(parents=True)

    def test_listing_does_not_execute_scripts(self):
        (self.directory / "z.py").write_text("raise RuntimeError('do not import')\n")
        (self.directory / "a.py").write_text("syntax deliberately invalid here\n")
        (self.directory / "old.sh").write_text("exit 99\n")
        (self.directory / "directory.py").mkdir()
        self.assertEqual(launcher.discover_scripts(self.root), ["a.py", "z.py"])

    def test_arbitrary_paths_and_shell_scripts_rejected(self):
        (self.directory / "safe.py").write_text("pass\n")
        for name in ["../safe.py", "archive/scripts/safe.py", "/tmp/safe.py", "old.sh", "missing.py"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                launcher.execution_spec(self.root, name, [])

    def test_symlink_scripts_are_not_executable(self):
        outside = self.root / "outside.py"
        outside.write_text("pass\n")
        (self.directory / "link.py").symlink_to(outside)
        self.assertEqual(launcher.discover_scripts(self.root), [])
        with self.assertRaises(ValueError):
            launcher.execution_spec(self.root, "link.py", [])

    def test_execution_preserves_root_cwd_imports_arguments_and_environment(self):
        (self.root / "root_helper.py").write_text("VALUE = 'root'\n")
        (self.directory / "sibling.py").write_text("VALUE = 'sibling'\n")
        (self.directory / "probe.py").write_text(
            "import json, os, sys, root_helper, sibling\n"
            "print(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:], "
            "'root': root_helper.VALUE, 'sibling': sibling.VALUE, "
            "'pythonpath': os.environ['PYTHONPATH']}))\n"
        )
        original_env = {"PYTHONPATH": "existing-path", "KEEP_ME": "untouched"}
        command, cwd, env = launcher.execution_spec(
            self.root, "probe.py", ["--value", "with spaces", ";not-a-shell"],
            environ=original_env, interpreter=sys.executable,
        )
        completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=True)
        result = json.loads(completed.stdout)
        self.assertEqual(result["cwd"], str(self.root.resolve()))
        self.assertEqual(result["args"], ["--value", "with spaces", ";not-a-shell"])
        self.assertEqual((result["root"], result["sibling"]), ("root", "sibling"))
        self.assertEqual(result["pythonpath"].split(os.pathsep)[-1], "existing-path")
        self.assertEqual(original_env, {"PYTHONPATH": "existing-path", "KEEP_ME": "untouched"})

    def test_command_line_requires_opt_in(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_legacy.py"), "analyze_failure_cases.py"],
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--execute", completed.stderr)

    def test_cli_listing_is_available_outside_repo(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_legacy.py"), "--list"],
            cwd=self.root, capture_output=True, text=True, check=True,
        )
        self.assertIn("analyze_failure_cases.py", completed.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()
