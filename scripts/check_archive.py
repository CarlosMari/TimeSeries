"""Verify byte identity of files relocated during repository cleanup.

The relocation manifest records original paths and hashes from the pre-cleanup
inventory. This verifier never rewrites that baseline or loads research artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent


def verify_archive(root: Path, manifest: dict) -> list[str]:
    root = root.resolve()
    problems = []
    seen = set()
    for entry in manifest["relocations"]:
        name = entry["archived_path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("archive",):
            problems.append(f"Unsafe archived path: {name}")
            continue
        path = root / relative
        if name in seen:
            problems.append(f"Duplicate archived path: {name}")
            continue
        seen.add(name)
        if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
            problems.append(f"Symlink in archived path: {name}")
            continue
        if not path.is_file():
            problems.append(f"Missing archived file: {name}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            problems.append(f"Changed archived file: {name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads((args.root / "archive/manifest.json").read_text(encoding="utf-8"))
        problems = verify_archive(args.root, manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cannot verify archive: {type(error).__name__}", file=sys.stderr)
        return 2
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"Archive verified: {len(manifest['relocations'])} byte-identical relocated files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
