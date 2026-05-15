"""3-stage preprocessor for GLV trajectories.

Stages:
    1. Family normalization: divide each sample by the max over (species, time).
    2. (OPTIONAL) Sort curves by peak (descending) — the "sort step" that the
       2026-05-15 pivot identified as breaking species identifiability. Toggle
       with --sort / --no-sort.
    3. Per-curve normalization: divide each curve by its own peak.

Usage (CLI):
    python -m data_generation.preprocessor --input data/TRAIN_FINAL_FIXED.pkl \\
                                           --output data/TRAIN_FINAL_NOSORT.pkl \\
                                           --no-sort

For the v1 with-sort reproducibility, omit --no-sort (default is --sort).
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def preprocess(raw_data: np.ndarray, sort_curves: bool = True) -> dict:
    """Run the 3-stage preprocessing on a (N, 7, T) array.

    Returns a dict with keys: data, reconstruction_max_values, family_max_values.
    Adds ``sort_permutations`` when sort_curves=True, so the operation is fully
    reversible.
    """
    if raw_data.ndim != 3:
        raise ValueError(f"expected (N, 7, T), got shape {raw_data.shape}")

    # Stage 1 — family normalization
    family_max_values = np.max(raw_data, axis=(1, 2), keepdims=True)
    family_max_values[family_max_values == 0] = 1e-8
    family_normalized_data = raw_data / family_max_values

    # Stage 2 — optional sort
    if sort_curves:
        max_values_for_sorting = np.max(family_normalized_data, axis=2)
        sort_permutations = np.argsort(-max_values_for_sorting, axis=1)
        data_after_sort = np.take_along_axis(
            family_normalized_data, sort_permutations[:, :, np.newaxis], axis=1
        )
    else:
        data_after_sort = family_normalized_data
        sort_permutations = None

    # Stage 3 — per-curve normalization
    max_values_per_curve = np.max(data_after_sort, axis=2, keepdims=True)
    max_values_per_curve[max_values_per_curve == 0] = 1e-8
    final_normalized_curves = data_after_sort / max_values_per_curve

    package = {
        "data": final_normalized_curves,
        "reconstruction_max_values": np.squeeze(max_values_per_curve, axis=2),
        "family_max_values": np.squeeze(family_max_values, axis=(1, 2)),
        "sort_applied": bool(sort_curves),
    }
    if sort_permutations is not None:
        package["sort_permutations"] = sort_permutations
    return package


def round_trip_check(raw_data: np.ndarray, package: dict, atol: float = 1e-6) -> None:
    """Sanity-check: reconstruct the raw data from the package and compare."""
    d = package["data"]
    rmv = package["reconstruction_max_values"]
    fmv = package["family_max_values"]
    reconstructed = d * rmv[:, :, None] * fmv[:, None, None]
    if package.get("sort_applied", False):
        perms = package["sort_permutations"]
        # Invert the sort to get back to original curve order
        inverse = np.argsort(perms, axis=1)
        reconstructed = np.take_along_axis(
            reconstructed, inverse[:, :, np.newaxis], axis=1
        )
    err = np.max(np.abs(reconstructed - raw_data))
    print(f"Round-trip max abs error: {err:.3e}")
    if err > atol:
        print(f"WARNING: round-trip error {err:.3e} exceeds tolerance {atol:.3e}",
              file=sys.stderr)
    else:
        print("Round-trip OK.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, help="path to raw .pkl")
    ap.add_argument("--output", "-o", required=True, help="path to save processed .pkl")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sort", dest="sort", action="store_true", default=True,
                   help="apply sort-by-peak step (v1 default)")
    g.add_argument("--no-sort", dest="sort", action="store_false",
                   help="skip sort-by-peak step (2026-05-15 pivot default)")
    args = ap.parse_args()

    print(f"--- Preprocessing {args.input}  (sort={args.sort}) ---")

    with open(args.input, "rb") as f:
        raw_data = pickle.load(f)
    print(f"Loaded raw data with shape: {raw_data.shape}")

    package = preprocess(raw_data, sort_curves=args.sort)
    print(f"Stage 1: family normalized.")
    print(f"Stage 2: sort {'APPLIED' if args.sort else 'SKIPPED'}.")
    print(f"Stage 3: per-curve normalized.")

    round_trip_check(raw_data, package)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(package, f)

    print(f"\nSaved → {out_path}")
    print(f"  data shape: {package['data'].shape}")
    print(f"  recon_max_values shape: {package['reconstruction_max_values'].shape}")
    print(f"  family_max_values shape: {package['family_max_values'].shape}")
    if "sort_permutations" in package:
        print(f"  sort_permutations shape: {package['sort_permutations'].shape}")


if __name__ == "__main__":
    main()
