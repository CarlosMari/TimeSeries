#!/usr/bin/env python3
"""Build a deterministic, metadata-only research artifact inventory.

Usage: python3 scripts/catalog_artifacts.py --root . [--check]
No pickle or checkpoint is deserialized. Only the narrowly scoped artifact
locations below are visited; scientific results are never modified.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile


DEFAULT_OUTPUT = "results/manifest.json"
EXCLUDED = {".git", "wandb", "env", "venv", ".env", ".venv", "__pycache__"}
METADATA_KEYS = {
    "metadata", "model_type", "model_config", "config", "seed", "seeds",
    "checkpoint", "checkpoint_path", "checkpoints", "test_path", "train_path",
    "dataset", "dataset_path", "dataset_name", "n_samples", "n_samples_per_group",
    "n_real", "n_gen", "n_chaos", "N_recon", "N_test", "N_per_group",
    "generation_mode", "sampling_mode", "sort_species", "scale_prediction_mode",
}
LIMITATIONS = [
    "Inventory only: no scientific results, provenance, or reproducibility claims are validated.",
    "Dataset and checkpoint contents are never deserialized; SHA256 is not proof of provenance.",
    "Filename/key-derived references are inferred and unverified, even when a candidate exists.",
    "Explicit metadata is reported as supplied, not independently verified; absent metadata remains unknown.",
    "Legacy species sorting, no-sort scale-head behavior, OOD compound shifts, and M8 mean-versus-sampled outputs require protocol review; labels alone cannot establish them.",
    "Only root RESULTS*.json, results/legacy/*.json, data/**/*.pkl, and model_ckpts/*.pth are inventoried; symlinks and excluded directories are skipped.",
    "Absolute paths in explicit metadata are redacted to basenames; no timestamps or machine paths are recorded.",
]


def _safe_value(value):
    """Keep supplied metadata JSON-safe without leaking absolute machine paths."""
    if isinstance(value, dict):
        return {_safe_text(k): _safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_value(v) for v in value]
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _safe_text(text):
    if text.startswith(("/", "~", "\\")) or re.match(r"^[A-Za-z]:[\\/]", text):
        return "<redacted-path>/" + text.replace("\\", "/").rstrip("/").split("/")[-1]
    return text


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_path(root, path):
    """Reject symlinks in every component, including discovery directories."""
    return not any(part.is_symlink() for part in [path, *path.parents] if part != root.parent)


def discover(root):
    """Return only allowlisted regular files, sorted by repository-relative path."""
    found = {}
    for directory, pattern, kind in [
        (root, "RESULTS*.json", "result"),
        (root / "results/legacy", "*.json", "result"),
        (root / "model_ckpts", "*.pth", "checkpoint"),
    ]:
        if not directory.is_dir() or not _plain_path(root, directory):
            continue
        for path in directory.glob(pattern):
            if path.is_file() and _plain_path(root, path):
                found[path.relative_to(root).as_posix()] = (path, kind)
    data = root / "data"
    if data.is_dir() and _plain_path(root, data):
        for directory, dirs, files in os.walk(data, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED
                             and not (Path(directory) / d).is_symlink())
            for name in files:
                path = Path(directory) / name
                if name.endswith(".pkl") and path.is_file() and not path.is_symlink():
                    found[path.relative_to(root).as_posix()] = (path, "dataset")
    return [(name, *found[name]) for name in sorted(found)]


def _json_pointer(parts):
    return "/" + "/".join(_safe_text(str(p)).replace("~", "~0").replace("/", "~1")
                           for p in parts)


def _content_details(content, filename, available):
    metadata, models, references = [], set(), []
    hints = {filename}

    def add_reference(kind, value, source, inferred):
        references.append({
            "kind": kind, "path": _safe_text(value), "source": source,
            "provenance": "inferred" if inferred else "explicit",
            "verified": False, "candidate_present": value in available,
        })

    def walk(value, parts=()):
        if isinstance(value, dict):
            for key in sorted(value):
                child = value[key]
                pointer = _json_pointer(parts + (key,))
                hints.add(key)
                if key in METADATA_KEYS:
                    metadata.append({"source": pointer, "value": _safe_value(child)})
                if isinstance(child, dict) and (
                    ":" in key or "model_type" in child or (parts and parts[-1] == "models")
                ):
                    models.add(_safe_text(key))
                    if ":" in key:
                        stem = key.split(":", 1)[1]
                        if re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
                            add_reference("checkpoint", "model_ckpts/" + stem + ".pth",
                                          "model key " + pointer, True)
                walk(child, parts + (key,))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, parts + (index,))
        elif isinstance(value, str):
            if value.lower().endswith((".pkl", ".pth")):
                add_reference("dataset" if value.lower().endswith(".pkl") else "checkpoint",
                              value, _json_pointer(parts), False)

    walk(content)
    for hint in sorted(hints):
        for suffix in re.findall(r"(?:TEST_)?(FINAL_NOSORT|OOD_Exp\d+)(?![A-Za-z0-9])", hint):
            add_reference("dataset", "data/TEST_" + suffix + ".pkl",
                          "filename/key " + _safe_text(hint), True)
        if re.fullmatch(r"(?:TRAIN|TEST)_[A-Za-z0-9_]+", hint):
            add_reference("dataset", "data/" + hint + ".pkl", "key " + hint, True)
    unique = {json.dumps(ref, sort_keys=True): ref for ref in references}
    return sorted(models), metadata, [unique[key] for key in sorted(unique)]


def _caveats(name, models, references):
    labels = " ".join([name, *models, *(ref["path"] for ref in references)]).upper()
    caveats = []

    def add(category, message):
        caveats.append({"category": category, "status": "review_required", "message": message})

    if "NOSORT" in labels:
        add("no_sort_scale_head", "A no-sort label does not establish whether the scale head preserves species order; inspect the evaluation and checkpoint configuration.")
    else:
        add("legacy_sorted", "Species ordering is not established here. Some legacy protocols sorted species; do not assume comparability with no-sort results.")
    if "OOD" in labels:
        add("ood_compound_shift", "An OOD label alone does not isolate a single intervention; parameter, noise, and preprocessing shifts may be compounded.")
    if "M8" in labels or "HETEROSCEDASTIC" in labels:
        add("m8_mean_vs_sampled", "Decoder means and sampled heteroscedastic outputs are different estimands; verify which output each metric evaluates before comparing them.")
    return caveats


def build_manifest(root, hash_large_files=False):
    """Read JSON and file metadata only; never write to the repository."""
    root = Path(root).resolve()
    artifacts = discover(root)
    available = {name for name, _, _ in artifacts}
    entries = []
    for name, path, kind in artifacts:
        entry = {"path": name, "kind": kind, "bytes": path.stat().st_size,
                 "sha256": _sha256(path) if kind == "result" or hash_large_files else None}
        if kind == "result":
            try:
                content = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as error:
                entry.update(json_status="invalid", model_keys=[], explicit_metadata=[], references=[])
                entry["parse_error"] = {"type": type(error).__name__}
                if isinstance(error, json.JSONDecodeError):
                    entry["parse_error"].update(line=error.lineno, column=error.colno)
            else:
                models, metadata, references = _content_details(content, path.name, available)
                entry.update(json_status="valid", json_type=type(content).__name__,
                             model_keys=models, explicit_metadata=metadata, references=references)
            entry["caveats"] = _caveats(name, entry["model_keys"], entry["references"])
        entries.append(entry)
    counts = {kind: sum(entry["kind"] == kind for entry in entries)
              for kind in ("result", "dataset", "checkpoint")}
    counts.update(total=len(entries), invalid_json=sum(
        entry.get("json_status") == "invalid" for entry in entries))
    return {"schema_version": 1, "hash_large_files": bool(hash_large_files),
            "counts": counts, "artifacts": entries, "limitations": LIMITATIONS}


def render_manifest(manifest):
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def _recognized_manifest(value):
    """Recognize this catalog's schema, not just a file named manifest.json."""
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        return False
    if value["schema_version"] != 1:
        return False
    artifacts = value.get("artifacts")
    counts = value.get("counts")
    limitations = value.get("limitations")
    if (not isinstance(artifacts, list) or not isinstance(counts, dict)
            or not isinstance(limitations, list)
            or not all(isinstance(item, str) for item in limitations)):
        return False
    for entry in artifacts:
        if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                or entry.get("kind") not in ("result", "dataset", "checkpoint")
                or type(entry.get("bytes")) is not int or entry["bytes"] < 0
                or "sha256" not in entry):
            return False
        digest = entry["sha256"]
        if digest is not None and (not isinstance(digest, str)
                                   or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            return False
    expected_counts = {kind: sum(entry["kind"] == kind for entry in artifacts)
                       for kind in ("result", "dataset", "checkpoint")}
    expected_counts.update(total=len(artifacts), invalid_json=sum(
        entry.get("json_status") == "invalid" for entry in artifacts))
    return all(type(counts.get(key)) is int and counts[key] == count
               for key, count in expected_counts.items())


def _output_path(root, requested):
    output = Path(requested)
    if not output.is_absolute():
        output = root / output
    if output.suffix != ".json":
        raise ValueError("output must have a .json suffix")
    if not _plain_path(root, output):
        raise ValueError("output and its ancestors must not be symlinks")
    output = output.resolve()
    try:
        relative = output.relative_to(root)
    except ValueError:
        raise ValueError("output must be inside --root") from None
    parts = relative.parts
    if any(part in EXCLUDED for part in parts):
        raise ValueError("output must not be in an excluded directory")
    artifact_name = PurePosixPath(relative.as_posix())
    if ((len(parts) == 1 and artifact_name.match("RESULTS*.json"))
            or (parts[:2] == ("results", "legacy") and output.suffix == ".json")
            or (parts[:1] == ("data",) and output.suffix == ".pkl")
            or (parts[:1] == ("model_ckpts",) and output.suffix == ".pth")):
        raise ValueError("output must not occupy an input artifact location")
    for _, path, _ in discover(root):
        if output.exists() and os.path.samefile(output, path):
            raise ValueError("output must not overwrite an input artifact")
    if output.exists():
        if not output.is_file():
            raise ValueError("existing output must be a regular catalog manifest")
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            raise ValueError("existing output is not a recognized catalog manifest") from None
        if not _recognized_manifest(existing):
            raise ValueError("existing output is not a recognized catalog manifest")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="new .json path or existing catalog manifest, relative to --root")
    parser.add_argument("--hash-large-files", action="store_true", help="also SHA256 datasets/checkpoints")
    parser.add_argument("--check", action="store_true", help="exit 1 if missing/stale; do not write")
    args = parser.parse_args(argv)
    temporary = None
    try:
        root = args.root.resolve()
        if not root.is_dir():
            raise ValueError("--root must be a directory")
        output = _output_path(root, args.output)
        expected = render_manifest(build_manifest(root, args.hash_large_files))
        label = output.relative_to(root).as_posix()
        if args.check:
            if not output.exists():
                print("Missing manifest: " + label, file=sys.stderr)
                return 1
            if output.read_bytes() != expected.encode("utf-8"):
                print("Stale manifest: " + label, file=sys.stderr)
                return 1
            print("Manifest is current: " + label)
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".manifest-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(expected)
        os.replace(temporary, output)
        temporary = None
        print("Wrote manifest: " + label)
        return 0
    except (OSError, ValueError) as error:
        # Do not emit OS exceptions containing absolute machine paths.
        print("Catalog error: " + (str(error) if isinstance(error, ValueError)
                                   else type(error).__name__), file=sys.stderr)
        return 2
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
