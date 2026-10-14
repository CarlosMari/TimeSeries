"""Prevent legacy metrics from leaking back into the repository root.

Inspect path expressions and shell syntax without importing numerical libraries,
loading research data, running analyses, or executing process-control queues.
"""

import ast
from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parent.parent
LEGACY = ROOT / "results/legacy"


def result_bindings(module):
    namespace = {"__builtins__": {}, "REPO_ROOT": ROOT, "Path": Path, "str": str}
    result_names = {"OUT_JSON", "OUT_MD", "OUTPUT_JSON", "OUTPUT_MD", "RESULTS_JSON"}
    for node in module.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in result_names):
            name = node.targets[0].id
            namespace[name] = eval(compile(ast.Expression(node.value), "<result-path>", "eval"), namespace)
    return namespace


class ResultPathTests(unittest.TestCase):
    def test_no_result_files_remain_at_repository_root(self):
        self.assertEqual([p.name for p in ROOT.glob("RESULTS*") if p.is_file()], [])

    def test_historical_summaries_and_metrics_have_a_home(self):
        for name in ["README.md", "RESULTS.json", "RESULTS.md", "RESULTS_COMPARISON.md"]:
            with self.subTest(name=name):
                self.assertTrue((LEGACY / name).is_file())

    def test_analysis_result_constants_target_legacy_directory(self):
        checked = 0
        for path in (ROOT / "analysis").glob("*.py"):
            module = ast.parse(path.read_text())
            for name, value in result_bindings(module).items():
                if name not in {"OUT_JSON", "OUT_MD", "OUTPUT_JSON", "OUTPUT_MD", "RESULTS_JSON"}:
                    continue
                with self.subTest(module=path.name, constant=name):
                    self.assertEqual(Path(value).parent, LEGACY)
                    self.assertTrue(Path(value).name.startswith("RESULTS"))
                checked += 1
        self.assertGreaterEqual(checked, 10)

    def test_result_cli_defaults_target_legacy_directory(self):
        for filename, flag in [
            ("evaluate_all_models.py", "--out"),
            ("noise_addition_sweep.py", "--out"),
            ("make_comparative_figures.py", "--results"),
        ]:
            module = ast.parse((ROOT / "analysis" / filename).read_text())
            defaults = [keyword.value for node in ast.walk(module)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                        and node.args and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == flag
                        for keyword in node.keywords if keyword.arg == "default"]
            self.assertEqual(len(defaults), 1, filename)
            value = eval(compile(ast.Expression(defaults[0]), "<cli-default>", "eval"), result_bindings(module))
            self.assertEqual(Path(value).parent, LEGACY, filename)

    def test_shell_result_names_all_have_relocated_prefix(self):
        for path in (ROOT / "scripts").glob("*.sh"):
            text = path.read_text()
            for match in re.finditer(r"\bRESULTS[A-Za-z0-9_${}.-]*\.(?:json|md)\b", text):
                with self.subTest(script=path.name, name=match.group()):
                    self.assertTrue(text[:match.start()].endswith("results/legacy/"))

    def test_comparative_figure_sanity_fallback_targets_legacy_directory(self):
        module = ast.parse((ROOT / "analysis/make_comparative_figures.py").read_text())
        filename = "RESULTS_COMPARATIVE_v1_sanity.json"
        fallbacks = [node.value for node in ast.walk(module)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(value, ast.Constant) and value.value == filename
                             for value in ast.walk(node.value))]
        self.assertEqual(len(fallbacks), 1)
        value = eval(compile(ast.Expression(fallbacks[0]), "<sanity-fallback>", "eval"), result_bindings(module))
        self.assertEqual(Path(value), LEGACY / filename)

    @unittest.skipUnless(shutil.which("bash"), "bash is needed for shell syntax validation")
    def test_shell_queues_parse_without_execution(self):
        for path in (ROOT / "scripts").glob("*.sh"):
            with self.subTest(script=path.name):
                subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=True)
                for embedded in re.findall(r"<<'PYEOF'\n(.*?)\nPYEOF", path.read_text(), re.DOTALL):
                    ast.parse(embedded, filename=path.name + ":embedded-python")


if __name__ == "__main__":
    unittest.main()
