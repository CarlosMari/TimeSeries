"""
FIXED VERSION: Addresses seed collision and random state issues.

Key fixes:
1. Changed seed*i to seed + large_offset + i to avoid collisions
2. Use separate RNG for noise to maintain consistency
3. Avoid seed=0 issue
4. Better train/test separation
5. Timeout protection for hanging seeds
6. Checkpointing for resumability
7. Verbose logging and progress monitoring
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm
import pickle
import signal
import time
import json
from pathlib import Path
from custom_glv_FIXED import generate_curves_Mario

LOG = False
VERBOSE = True  # Enable detailed logging
DEBUG_SLOW_SEEDS = True  # Track seeds that take >10 seconds
CHECKPOINT_INTERVAL = 10000  # Save every N valid samples
TIMEOUT_PER_SEED = 30  # Max seconds per seed generation

# Use large, distinct seeds to avoid collisions
TRAIN_SEED = 123456789
TEST_SEED = 987654321

N = 7  # How many curves are there in a family
SIGMA = 0.01  # Lognormal noise variance


# ============================================================================
# TIMEOUT HANDLER
# ============================================================================

class TimeoutError(Exception):
    """Custom exception for timeouts."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")


def generate_with_timeout(sample_seed, timeout=TIMEOUT_PER_SEED):
    """
    Generate curves with timeout protection.

    Returns:
        sol (np.ndarray or None): Generated curves or None if failed/timeout
        elapsed (float): Time taken in seconds
    """
    start_time = time.time()

    # Set up timeout (Unix-only - uses signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        sol = generate_curves_Mario(
            myseed=sample_seed,
            noise_level=0.01,
            species=7,
            tmax=20,
            n_points=65,
            plot_this=False
        )
        signal.alarm(0)  # Cancel the alarm
        elapsed = time.time() - start_time
        return sol, elapsed

    except TimeoutError:
        signal.alarm(0)
        elapsed = time.time() - start_time
        if VERBOSE:
            print(f"\n⏱️  Timeout after {elapsed:.1f}s for seed {sample_seed}")
        return None, elapsed

    except Exception as e:
        signal.alarm(0)
        elapsed = time.time() - start_time
        if VERBOSE:
            print(f"\n❌ Error for seed {sample_seed}: {e}")
        return None, elapsed


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(name, num_sols, last_iteration, sim_lists, checkpoint_dir='./checkpoints'):
    """Save checkpoint for resumability."""
    Path(checkpoint_dir).mkdir(exist_ok=True)
    checkpoint_path = Path(checkpoint_dir) / f"{name}_checkpoint.pkl"

    checkpoint_data = {
        'num_sols': num_sols,
        'last_iteration': last_iteration,
        'sim_lists': sim_lists,
        'timestamp': time.time()
    }

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    if VERBOSE:
        print(f"\n💾 Checkpoint saved: {checkpoint_path} ({num_sols} samples)")


def load_checkpoint(name, checkpoint_dir='./checkpoints'):
    """Load checkpoint if exists."""
    checkpoint_path = Path(checkpoint_dir) / f"{name}_checkpoint.pkl"

    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        if VERBOSE:
            print(f"\n📂 Loaded checkpoint: {checkpoint_data['num_sols']} samples from iteration {checkpoint_data['last_iteration']}")

        return checkpoint_data
    return None


def generate_data(num_curves, seed, name='TRAIN', resume=True):
    """
    Generate GLV time series data with timeout protection and checkpointing.

    Args:
        num_curves: Number of valid curves to generate
        seed: Base random seed for this dataset
        name: Dataset name for saving
        resume: Whether to resume from checkpoint if available
    """
    # Try to load checkpoint
    start_iteration = 0
    sim_lists = []
    num_sols = 0

    if resume:
        checkpoint = load_checkpoint(name)
        if checkpoint:
            sim_lists = checkpoint['sim_lists']
            num_sols = checkpoint['num_sols']
            start_iteration = checkpoint['last_iteration'] + 1
            print(f"✓ Resuming from {num_sols} samples, iteration {start_iteration}")

    # Create independent RNG for noise generation
    noise_rng = np.random.RandomState(seed)

    # Generate more than needed since some will be rejected
    # Use higher multiplier for small datasets to ensure completion
    if num_curves < 1000:
        max_attempts = num_curves * 10  # 10x for small datasets
    else:
        max_attempts = num_curves * 5  # 5x for large datasets

    scaling_factor = float(np.exp(SIGMA**2 / 2))

    # Tracking for debugging
    slow_seeds = []
    total_time = 0
    timeouts = 0
    rejections = 0
    last_report_time = time.time()
    last_report_sols = num_sols

    pbar = tqdm(range(start_iteration, max_attempts), desc='Finding Correct Solutions', initial=start_iteration, total=max_attempts)

    total_iterations = start_iteration
    for i in pbar:
        total_iterations = i + 1
        if num_sols >= num_curves:
            break

        # FIXED: Use seed + large offset + i instead of seed*i
        sample_seed = seed + 1000000 + i

        # Generate with timeout protection
        sol, elapsed = generate_with_timeout(sample_seed)
        total_time += elapsed

        # Track slow seeds
        if DEBUG_SLOW_SEEDS and elapsed > 10.0:
            slow_seeds.append((sample_seed, elapsed))
            if VERBOSE and len(slow_seeds) % 10 == 0:
                print(f"\n🐌 {len(slow_seeds)} slow seeds detected (>{10}s)")

        if sol is None:
            timeouts += 1
            continue

        shape = sol.shape

        # FIXED: Use independent RNG for noise instead of global np.random
        noise = noise_rng.lognormal(mean=0, sigma=SIGMA, size=shape)

        # Center the noise around 1
        centered_noise = noise / scaling_factor
        sol = sol * centered_noise

        # Quality checks
        steady_states = np.mean(sol[:, -10:], axis=1)
        overshoot_flags = np.max(sol, axis=1) > 1.2 * steady_states
        overshoot_count = np.sum(overshoot_flags)

        # Check for NaN or extreme values
        if (np.isnan(sol).any() or
            sol[sol > 3.0].any() or
            np.any(np.max(sol, axis=1) < 0.1) or
            overshoot_count < 3):
            rejections += 1
            continue

        if LOG:
            print('=================================')
            print(f'Correct Family {num_sols}')
            for curve in sol:
                print(f'Max {np.max(curve)}')
            print('=================================')

        num_sols += 1
        sim_lists.append(sol)

        # Update progress bar with detailed stats
        acceptance_rate = num_sols / (i + 1) * 100 if i > 0 else 0
        avg_time = total_time / (i - start_iteration + 1)
        pbar.set_postfix({
            'Valid': num_sols,
            'Accept%': f'{acceptance_rate:.1f}',
            'AvgTime': f'{avg_time:.2f}s',
            'Timeouts': timeouts
        })

        # Periodic checkpointing
        if num_sols % CHECKPOINT_INTERVAL == 0 and num_sols > 0:
            save_checkpoint(name, num_sols, i, sim_lists)

        # Periodic stats report
        if VERBOSE and time.time() - last_report_time > 300:  # Every 5 minutes
            elapsed_mins = (time.time() - last_report_time) / 60
            rate = (num_sols - last_report_sols) / elapsed_mins
            print(f"\n📊 Progress: {num_sols}/{num_curves} samples ({rate:.1f}/min)")
            last_report_time = time.time()
            last_report_sols = num_sols

    if num_sols < num_curves:
        print(f"\n⚠️  WARNING: Only generated {num_sols}/{num_curves} valid samples")

    sols = np.array(sim_lists)

    # Visualization
    fig, axs = plt.subplots(5, 5, figsize=(15, 15), sharex=True, sharey=True)
    axs = axs.flatten()

    # Choose 25 random families and plot them
    if len(sols) >= 25:
        indices = noise_rng.choice(len(sols), size=25, replace=False)
        A = sols[indices]
    else:
        A = sols[:min(25, len(sols))]

    for i in range(min(25, len(A))):
        for z in range(N):
            axs[i].plot(A[i][z])
        axs[i].axis("off")

    plt.tight_layout()

    # Ensure output directory exists
    Path('./generation_comparison').mkdir(exist_ok=True)
    plt.savefig(f'./generation_comparison/{name}_FIXED.png', dpi=150)
    plt.close()

    print(f'\nGenerated data shape: {sols.shape}')

    # Save the filtered data
    Path('./data').mkdir(exist_ok=True)
    output_path = f'./data/{name}_FIXED.pkl'
    with open(output_path, 'wb') as output:
        pickle.dump(sols, output)
    print(f'Saved to: {output_path}')

    # Final checkpoint
    save_checkpoint(name, num_sols, i, sim_lists)

    # Print summary statistics
    print("\n" + "="*60)
    print(f"GENERATION SUMMARY: {name}")
    print("="*60)
    print(f"Valid samples: {num_sols}/{num_curves}")
    print(f"Total iterations: {total_iterations}")
    print(f"Acceptance rate: {num_sols / total_iterations * 100:.2f}%")
    print(f"Timeouts: {timeouts}")
    print(f"Quality rejections: {rejections}")
    print(f"Average time/seed: {total_time / total_iterations:.3f}s")
    print(f"Total time: {total_time / 3600:.2f} hours")

    # Report slow seeds
    if DEBUG_SLOW_SEEDS and len(slow_seeds) > 0:
        print(f"\n⚠️  {len(slow_seeds)} slow seeds detected (>10s)")
        # Save slow seed report
        slow_seed_path = f'./data/{name}_slow_seeds.json'
        with open(slow_seed_path, 'w') as f:
            json.dump({
                'count': len(slow_seeds),
                'seeds': [(int(s), float(t)) for s, t in slow_seeds[:100]]  # Save top 100
            }, f, indent=2)
        print(f"Slow seed report saved to: {slow_seed_path}")

        # Print slowest seeds
        slow_seeds_sorted = sorted(slow_seeds, key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 slowest seeds:")
        for seed, elapsed in slow_seeds_sorted[:10]:
            print(f"  Seed {seed}: {elapsed:.1f}s")

    print("="*60)

    return sols


def verify_no_seed_collisions(train_seed, test_seed, n_samples=10000):
    """Verify that train and test seeds don't collide."""
    print("\n" + "="*60)
    print("Verifying seed separation...")
    print("="*60)

    train_seeds = set()
    test_seeds = set()

    for i in range(n_samples):
        train_seeds.add(train_seed + 1000000 + i)
        test_seeds.add(test_seed + 1000000 + i)

    collisions = train_seeds & test_seeds

    print(f"Checked {n_samples} samples per dataset")
    print(f"Train seeds: {len(train_seeds)} unique")
    print(f"Test seeds: {len(test_seeds)} unique")
    print(f"Collisions: {len(collisions)}")

    if len(collisions) > 0:
        print(f"⚠️  WARNING: Found {len(collisions)} collisions!")
        print(f"First 10: {list(collisions)[:10]}")
    else:
        print("✓ No collisions found!")

    return len(collisions) == 0


if __name__ == "__main__":
    # Verify seeds don't collide
    verify_no_seed_collisions(TRAIN_SEED, TEST_SEED, n_samples=1000000)

    print("\n" + "="*60)
    print("Generating datasets...")
    print("="*60)

    # Generate datasets
    train_data = generate_data(150000, TRAIN_SEED, 'TRAIN_FINAL')
    test_data = generate_data(50000, TEST_SEED, 'TEST_FINAL')

    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)
    print(f"Train: {train_data.shape}")
    print(f"Test: {test_data.shape}")
