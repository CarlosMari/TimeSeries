"""List preserved exploratory scripts, or explicitly opt in to running one.

No historical script is imported during discovery. Execution uses a child process
with the selected interpreter, repository-root CWD, and historical import paths.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent


def discover_scripts(root: Path) -> list[str]:
    """Return only regular, nonsymlink Python files in the historical directory."""
    directory = root / "archive" / "scripts"
    if directory.is_symlink():
        return []
    return sorted(
        path.name for path in directory.glob("*.py")
        if path.is_file() and not path.is_symlink()
    )


def execution_spec(
    root: Path, name: str, arguments: list[str], *, environ=None, interpreter=None,
) -> tuple[list[str], Path, dict[str, str]]:
    """Resolve an allowlisted basename without interpreting shell commands."""
    if name not in discover_scripts(root):
        raise ValueError(
            f"Unknown historical Python script: {name!r}. Use --list for choices."
        )
    root = root.resolve()
    directory = root / "archive" / "scripts"
    env = dict(os.environ if environ is None else environ)
    search_path = [str(root), str(directory)]
    if env.get("PYTHONPATH"):
        search_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(search_path)
    command = [interpreter or sys.executable, str(directory / name), *arguments]
    return command, root, env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="list without executing")
    mode.add_argument(
        "--execute", action="store_true",
        help="opt in to executing a historical script; outputs may be overwritten",
    )
    parser.add_argument("script", nargs="?", help="an exact basename from --list")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.list:
        if args.script or args.arguments:
            parser.error("--list does not accept a script or script arguments")
        for name in discover_scripts(REPO_ROOT):
            print(name)
        return 0

    if not args.script:
        parser.error("--execute requires a historical Python script basename")
    try:
        command, cwd, env = execution_spec(REPO_ROOT, args.script, args.arguments)
    except ValueError as error:
        parser.error(str(error))

    print(
        "Running historical exploratory code. It may overwrite outputs, require "
        "local data/CUDA, or use synthetic demonstrations; see archive/README.md.",
        file=sys.stderr,
    )
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    return completed.returncode if completed.returncode >= 0 else 128 - completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
