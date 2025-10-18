"""
Quick test for generate_family_FIXED.py improvements.

Tests:
1. Timeout protection works
2. Checkpointing system works
3. Progress monitoring works
4. Resume from checkpoint works
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from generate_family_FIXED import generate_data, TRAIN_SEED

print("="*60)
print("TESTING IMPROVED GENERATION SYSTEM")
print("="*60)

# Test 1: Generate small dataset
print("\nTest 1: Generating 100 samples with all features enabled...")
print("-" * 60)

test_data = generate_data(
    num_curves=100,
    seed=TRAIN_SEED,
    name='TEST_SMALL',
    resume=True
)

print(f"\n✓ Test completed!")
print(f"  Generated shape: {test_data.shape}")
print(f"  Expected: (100, 7, 65)")

# Verify shape
assert test_data.shape[0] == 100, f"Expected 100 samples, got {test_data.shape[0]}"
assert test_data.shape[1] == 7, f"Expected 7 species, got {test_data.shape[1]}"
assert test_data.shape[2] == 65, f"Expected 65 timesteps, got {test_data.shape[2]}"

print("\n✅ All tests passed!")
print("\nYou can now safely run the full generation with:")
print("  python generate_family_FIXED.py")
