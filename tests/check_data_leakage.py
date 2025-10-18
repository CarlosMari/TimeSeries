"""
DATA LEAKAGE DETECTION SCRIPT

This script performs comprehensive checks to verify there's no data leakage
between training and test sets for the TimeSeries VAE project.

Author: Claude Code
Date: 2025
"""

import numpy as np
import pickle
from tqdm import tqdm
import hashlib
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
TRAIN_FILE = 'data/TRAIN_FINAL_PROCESSED.pkl'
TEST_FILE = 'data/TEST_FINAL_PROCESSED.pkl'

# Seeds from generate_family.py
TRAIN_SEED = 74
TEST_SEED = 42


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_seed_overlap(train_seed, test_seed, max_samples=1000000):
    """
    Check if there's any mathematical overlap in seeds.

    For train: myseed = train_seed * i  (i = 0, 1, 2, ...)
    For test:  myseed = test_seed * j   (j = 0, 1, 2, ...)

    Overlap occurs if train_seed * i == test_seed * j
    => i / j = test_seed / train_seed

    If gcd(train_seed, test_seed) > 1, there could be overlap.
    """
    print_section("CHECK 1: Mathematical Seed Overlap Analysis")

    from math import gcd

    g = gcd(train_seed, test_seed)
    print(f"Train seed: {train_seed}")
    print(f"Test seed:  {test_seed}")
    print(f"GCD(train_seed, test_seed) = {g}")

    if g == 1:
        print("\n✓ PASS: GCD = 1, seeds are coprime")
        print("   No mathematical seed overlap possible.")
        return True, []
    else:
        print(f"\n⚠ WARNING: GCD = {g} > 1")
        print("   Potential seed overlaps exist!")

        # Find first few overlaps
        lcm = (train_seed * test_seed) // g
        print(f"   LCM(train_seed, test_seed) = {lcm}")

        # First overlap occurs at:
        # train_seed * i = test_seed * j = LCM
        i_overlap = lcm // train_seed
        j_overlap = lcm // test_seed

        overlaps = []
        print(f"\n   First overlap: train[{i_overlap}] == test[{j_overlap}]")
        print(f"   (myseed = {train_seed * i_overlap})")
        overlaps.append((i_overlap, j_overlap, train_seed * i_overlap))

        # Check if this falls within actual dataset ranges
        # Assume train has ~500k samples, test has ~100k samples
        if i_overlap < max_samples and j_overlap < max_samples:
            print(f"   ⚠ This overlap is within typical dataset size!")
        else:
            print(f"   ✓ This overlap is beyond typical dataset size.")

        return False, overlaps


def load_datasets():
    """Load train and test datasets."""
    print_section("CHECK 2: Loading Datasets")

    print(f"Loading train data from: {TRAIN_FILE}")
    with open(TRAIN_FILE, 'rb') as f:
        train_data = pickle.load(f)

    print(f"Loading test data from: {TEST_FILE}")
    with open(TEST_FILE, 'rb') as f:
        test_data = pickle.load(f)

    train_curves = train_data['data']  # Shape: (N_train, 7, 65)
    test_curves = test_data['data']    # Shape: (N_test, 7, 65)

    print(f"\nTrain dataset shape: {train_curves.shape}")
    print(f"Test dataset shape:  {test_curves.shape}")
    print(f"Total samples: {train_curves.shape[0] + test_curves.shape[0]}")

    return train_curves, test_curves, train_data, test_data


def compute_sample_hashes(curves, name="dataset"):
    """
    Compute unique hash for each sample.
    Uses SHA256 hash of the flattened numpy array.
    """
    print(f"\nComputing hashes for {name}...")
    hashes = []

    for i in tqdm(range(len(curves)), desc=f"Hashing {name}"):
        # Flatten the curve and convert to bytes
        curve_bytes = curves[i].tobytes()
        # Compute hash
        hash_val = hashlib.sha256(curve_bytes).hexdigest()
        hashes.append(hash_val)

    return hashes


def check_exact_duplicates(train_hashes, test_hashes):
    """Check for exact duplicate samples between train and test."""
    print_section("CHECK 3: Exact Duplicate Detection (Hash-based)")

    print(f"Train samples: {len(train_hashes)}")
    print(f"Test samples:  {len(test_hashes)}")

    # Convert to sets for fast lookup
    train_set = set(train_hashes)
    test_set = set(test_hashes)

    # Find duplicates
    duplicates = train_set.intersection(test_set)

    print(f"\nUnique train samples: {len(train_set)}")
    print(f"Unique test samples:  {len(test_set)}")
    print(f"Exact duplicates:     {len(duplicates)}")

    if len(duplicates) == 0:
        print("\n✓ PASS: No exact duplicates found between train and test!")
        return True, []
    else:
        print(f"\n✗ FAIL: Found {len(duplicates)} exact duplicates!")

        # Find indices of duplicates
        dup_indices = []
        for i, h in enumerate(train_hashes):
            if h in duplicates:
                j = test_hashes.index(h)
                dup_indices.append((i, j, h))
                if len(dup_indices) <= 5:  # Show first 5
                    print(f"   Duplicate: train[{i}] == test[{j}]")

        if len(dup_indices) > 5:
            print(f"   ... and {len(dup_indices) - 5} more")

        return False, dup_indices


def check_internal_duplicates(hashes, name):
    """Check for duplicates within a single dataset."""
    print(f"\nChecking internal duplicates in {name}...")

    hash_counts = defaultdict(int)
    for h in hashes:
        hash_counts[h] += 1

    duplicates = {h: count for h, count in hash_counts.items() if count > 1}

    if len(duplicates) == 0:
        print(f"✓ No internal duplicates in {name}")
    else:
        print(f"⚠ Found {len(duplicates)} duplicated samples in {name}")
        total_dup_samples = sum(count - 1 for count in duplicates.values())
        print(f"  Total duplicate samples: {total_dup_samples}")

    return duplicates


def check_statistical_similarity(train_curves, test_curves, n_samples=1000):
    """
    Check statistical similarity between train and test distributions.
    Computes minimum distance between each test sample and train set.
    """
    print_section("CHECK 4: Statistical Similarity Analysis")

    print(f"Sampling {n_samples} test samples for similarity check...")

    # Randomly sample test indices
    test_indices = np.random.choice(len(test_curves),
                                   size=min(n_samples, len(test_curves)),
                                   replace=False)

    # Sample train set (to keep computation reasonable)
    train_sample_size = min(10000, len(train_curves))
    train_indices = np.random.choice(len(train_curves),
                                    size=train_sample_size,
                                    replace=False)

    min_distances = []

    print(f"Computing distances (test_samples={len(test_indices)}, train_samples={len(train_indices)})...")

    for test_idx in tqdm(test_indices, desc="Computing min distances"):
        test_sample = train_curves[test_idx].flatten()

        # Compute distances to all train samples
        distances = []
        for train_idx in train_indices:
            train_sample = train_curves[train_idx].flatten()
            dist = np.linalg.norm(test_sample - train_sample)
            distances.append(dist)

        min_dist = np.min(distances)
        min_distances.append(min_dist)

    min_distances = np.array(min_distances)

    print(f"\nMinimum distance statistics:")
    print(f"  Mean:   {min_distances.mean():.6f}")
    print(f"  Median: {np.median(min_distances):.6f}")
    print(f"  Min:    {min_distances.min():.6f}")
    print(f"  Max:    {min_distances.max():.6f}")
    print(f"  Std:    {min_distances.std():.6f}")

    # Flag suspiciously close samples
    threshold = 1e-6  # Very small threshold for near-duplicates
    near_duplicates = np.sum(min_distances < threshold)

    print(f"\nSamples with min_distance < {threshold}: {near_duplicates}")

    if near_duplicates > 0:
        print(f"⚠ WARNING: Found {near_duplicates} near-duplicate samples!")
        print("  These samples are suspiciously close to training data.")
        return False, min_distances
    else:
        print(f"✓ PASS: No near-duplicates found (all distances > {threshold})")
        return True, min_distances


def plot_distance_distribution(min_distances, output_file='data_leakage_distances.png'):
    """Plot distribution of minimum distances."""
    print(f"\nPlotting distance distribution to {output_file}...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(min_distances, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Minimum Distance to Train Set')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Minimum Distances (Test → Train)')
    axes[0].axvline(np.median(min_distances), color='r', linestyle='--',
                   label=f'Median = {np.median(min_distances):.4f}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Log-scale histogram (to see near-zero values)
    axes[1].hist(min_distances, bins=50, edgecolor='black', alpha=0.7, log=True)
    axes[1].set_xlabel('Minimum Distance to Train Set')
    axes[1].set_ylabel('Frequency (log scale)')
    axes[1].set_title('Distribution of Minimum Distances (Log Scale)')
    axes[1].axvline(1e-6, color='r', linestyle='--',
                   label='Near-duplicate threshold (1e-6)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved plot to {output_file}")


def generate_summary_report(results):
    """Generate a comprehensive summary report."""
    print_section("DATA LEAKAGE DETECTION SUMMARY")

    print("\n📊 TEST RESULTS:")
    print("-" * 80)

    for test_name, (passed, details) in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}  {test_name}")
        if details and not passed:
            print(f"         Details: {details}")

    print("-" * 80)

    all_passed = all(passed for passed, _ in results.values())

    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("   No data leakage detected between train and test sets.")
        print("   The model evaluation is valid.")
    else:
        print("\n⚠ SOME CHECKS FAILED!")
        print("   Potential data leakage detected.")
        print("   Review the detailed results above.")

    return all_passed


def main():
    """Main execution function."""
    print("=" * 80)
    print("  DATA LEAKAGE DETECTION FOR TIMESERIES VAE PROJECT")
    print("=" * 80)
    print(f"\nTrain file: {TRAIN_FILE}")
    print(f"Test file:  {TEST_FILE}")

    results = {}

    # CHECK 1: Seed overlap
    passed, overlaps = check_seed_overlap(TRAIN_SEED, TEST_SEED)
    results["1. Mathematical Seed Overlap"] = (passed,
        f"{len(overlaps)} overlaps found" if not passed else "No overlaps")

    # CHECK 2 & 3: Load data and check exact duplicates
    train_curves, test_curves, train_data, test_data = load_datasets()

    train_hashes = compute_sample_hashes(train_curves, "train")
    test_hashes = compute_sample_hashes(test_curves, "test")

    # Check internal duplicates
    print_section("CHECK 2.5: Internal Duplicate Detection")
    train_internal_dups = check_internal_duplicates(train_hashes, "train")
    test_internal_dups = check_internal_duplicates(test_hashes, "test")

    results["2a. Train Internal Duplicates"] = (len(train_internal_dups) == 0,
        f"{len(train_internal_dups)} duplicates" if train_internal_dups else "None")
    results["2b. Test Internal Duplicates"] = (len(test_internal_dups) == 0,
        f"{len(test_internal_dups)} duplicates" if test_internal_dups else "None")

    # Check train-test duplicates
    passed, duplicates = check_exact_duplicates(train_hashes, test_hashes)
    results["3. Exact Duplicates (Train ↔ Test)"] = (passed,
        f"{len(duplicates)} duplicates found" if not passed else "No duplicates")

    # CHECK 4: Statistical similarity
    passed, min_distances = check_statistical_similarity(train_curves, test_curves,
                                                         n_samples=1000)
    results["4. Statistical Similarity"] = (passed,
        f"Median distance = {np.median(min_distances):.6f}")

    # Plot distance distribution
    plot_distance_distribution(min_distances)

    # Generate summary
    all_passed = generate_summary_report(results)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
