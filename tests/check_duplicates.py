"""
Check for duplicate data in generated GLV datasets.
"""
import numpy as np
import pickle
import sys

def check_duplicates_in_file(filepath):
    """Check for duplicate time series in a pickle file."""
    print(f"\n{'='*60}")
    print(f"Checking: {filepath}")
    print('='*60)

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            if 'data' in data:
                curves = data['data']
            elif 'curves' in data:
                curves = data['curves']
            else:
                curves = data
        else:
            curves = data

        print(f"Data shape: {curves.shape}")
        print(f"Data type: {curves.dtype}")

        # Check for exact duplicates (only first 100 samples for speed)
        n_samples = len(curves)
        duplicates = []

        print(f"\nChecking first 100 samples for exact duplicates...")
        check_n = min(n_samples, 100)
        for i in range(check_n):
            for j in range(i+1, check_n):
                if np.allclose(curves[i], curves[j], atol=1e-10):
                    duplicates.append((i, j))
                    if len(duplicates) <= 10:  # Print first 10
                        print(f"  Duplicate found: sample {i} == sample {j}")

        print(f"\nTotal exact duplicates found (in first {check_n}): {len(duplicates)}")

        # Check for near-duplicates (similar within tolerance)
        print(f"\nChecking first 50 samples for near-duplicates (atol=1e-6)...")
        near_duplicates = []
        check_n = min(n_samples, 50)
        for i in range(check_n):
            for j in range(i+1, check_n):
                if np.allclose(curves[i], curves[j], atol=1e-6):
                    near_duplicates.append((i, j))
                    if len(near_duplicates) <= 10:
                        print(f"  Near-duplicate: sample {i} ≈ sample {j}")

        print(f"\nTotal near-duplicates found (in first {check_n}): {len(near_duplicates)}")

        # Check statistics
        print("\n" + "="*60)
        print("Data Statistics:")
        print("="*60)
        print(f"Min value: {np.min(curves):.6f}")
        print(f"Max value: {np.max(curves):.6f}")
        print(f"Mean value: {np.mean(curves):.6f}")
        print(f"Std value: {np.std(curves):.6f}")

        # Check if all samples are identical
        if n_samples > 1:
            all_same = np.all([np.allclose(curves[0], curves[i]) for i in range(1, min(10, n_samples))])
            print(f"\nFirst 10 samples all identical: {all_same}")

        # Check diversity - variance across samples
        sample_means = np.mean(curves, axis=(1, 2))
        sample_stds = np.std(curves, axis=(1, 2))

        print(f"\nDiversity metrics:")
        print(f"  Variance of sample means: {np.var(sample_means):.6f}")
        print(f"  Variance of sample stds: {np.var(sample_stds):.6f}")

        if np.var(sample_means) < 1e-6:
            print("  ⚠️  WARNING: Very low diversity - samples might be duplicates!")

        return duplicates, near_duplicates

    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def test_seed_generation():
    """Test if the seed generation logic produces duplicates."""
    print("\n" + "="*60)
    print("Testing seed generation logic")
    print("="*60)

    TRAIN_SEED = 74
    TEST_SEED = 42

    # Check for seed collisions in the current approach
    train_seeds = set()
    test_seeds = set()
    collisions = []

    print("\nChecking for seed collisions in seed*i approach...")
    print(f"TRAIN_SEED={TRAIN_SEED}, TEST_SEED={TEST_SEED}")

    for i in range(1000):
        ts = TRAIN_SEED * i
        train_seeds.add(ts)

    for i in range(1000):
        ts = TEST_SEED * i
        if ts in train_seeds:
            collisions.append((TRAIN_SEED, i, TEST_SEED, ts))
        test_seeds.add(ts)

    print(f"\nFirst 20 train seeds: {sorted(list(train_seeds))[:20]}")
    print(f"First 20 test seeds: {sorted(list(test_seeds))[:20]}")
    print(f"\nCollisions found: {len(collisions)}")
    if collisions:
        print("First 10 collisions:")
        for col in collisions[:10]:
            print(f"  {col}")

    # The big issue: i=0 always gives seed=0!
    print("\n⚠️  CRITICAL ISSUE: When i=0, seed*i = 0 regardless of seed!")
    print(f"  TRAIN_SEED * 0 = {TRAIN_SEED * 0}")
    print(f"  TEST_SEED * 0 = {TEST_SEED * 0}")
    print("  This means the FIRST sample is ALWAYS generated with seed=0!")


if __name__ == "__main__":
    # Test seed logic
    test_seed_generation()

    # Check actual data files
    files_to_check = [
        'data/TRAIN_FINAL_FIXED.pkl',
        'data/TEST_FINAL_FIXED.pkl',
    ]

    for filepath in files_to_check:
        check_duplicates_in_file(filepath)
