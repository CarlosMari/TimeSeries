"""Generate out-of-distribution (F4) test sets for the pivoted paper.

Two held-out family distributions:
  - `r ~ Exp(1)` (faster growth than training distribution `r ~ Exp(2)`)
  - `r ~ Exp(5)` (slower growth)

Otherwise the GLV parameter distribution matches training: A_{ii} ~ -Exp(2),
off-diagonals ~ N(0,1), stability check, positive fixed point, 65 timesteps.

Saves both raw trajectories and the no-sort preprocessed package.
"""

from __future__ import annotations

import argparse
import pickle
import signal
import sys
import time
from pathlib import Path

import numpy as np
from scipy import integrate
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data_generation.preprocessor import preprocess  # noqa: E402

TIMEOUT_PER_SEED = 30
SPECIES = 7
T_POINTS = 65


def _gLV(t, x, r, A):
    return x * (r + A @ x)


def _generate_one(myseed: int, r_scale: float, max_attempts: int = 500):
    rng = np.random.RandomState(myseed)
    for c in range(max_attempts):
        r0 = rng.exponential(scale=r_scale, size=SPECIES)
        a0 = rng.randn(SPECIES, SPECIES)
        for i in range(SPECIES):
            a0[i, i] = -rng.exponential(scale=2.0)

        # Stability check
        try:
            xss = np.linalg.solve(a0, -r0)
            d0 = np.diag(xss)
            eigenvalues = np.real(np.linalg.eigvals(d0 @ a0))
            ok = np.all(eigenvalues <= 0) and np.all(xss > 0)
        except np.linalg.LinAlgError:
            ok = False
        if ok:
            break
    else:
        return None
    x0 = rng.exponential(scale=0.1, size=SPECIES)
    tmax = -20.0 / float(np.min(eigenvalues))
    t_eval = np.linspace(0, tmax, T_POINTS)

    # solve_ivp with timeout
    class _TO(Exception):
        pass

    def _h(signum, frame): raise _TO()

    signal.signal(signal.SIGALRM, _h)
    signal.alarm(TIMEOUT_PER_SEED)
    try:
        sol = integrate.solve_ivp(
            fun=lambda t, y: _gLV(t, y, r0, a0),
            t_span=(0, tmax), y0=x0, t_eval=t_eval, method="RK45", max_step=0.5,
        )
        signal.alarm(0)
    except _TO:
        signal.alarm(0)
        return None
    if not sol.success:
        return None
    return sol.y, r0, a0, x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-scale", type=float, required=True,
                    help="exponential scale for growth-rate distribution (training uses 2.0)")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed-base", type=int, default=777_000_001,
                    help="base seed (different from train (~123M) and test (~987M) and matched (555M))")
    ap.add_argument("--out", required=True, help="output .pkl path (raw + preprocessed package combined)")
    args = ap.parse_args()

    print(f"Generating OOD trajectories with r ~ Exp({args.r_scale}); N={args.n}")
    raw, R, A, X0, seeds = [], [], [], [], []
    seed = args.seed_base
    pbar = tqdm(total=args.n, desc="OOD samples")
    attempts = 0
    while len(raw) < args.n:
        attempts += 1
        result = _generate_one(seed, args.r_scale)
        seed += 1
        if result is None:
            continue
        sol, r, A_mat, x0 = result
        if not np.all(np.isfinite(sol)):
            continue
        raw.append(sol)
        R.append(r)
        A.append(A_mat)
        X0.append(x0)
        seeds.append(seed - 1)
        pbar.update(1)
    pbar.close()
    print(f"  Attempts: {attempts}, kept: {len(raw)} ({100*len(raw)/attempts:.1f}%)")

    raw_arr = np.stack(raw, axis=0)  # (N, 7, 65)
    print(f"  Raw shape: {raw_arr.shape}")

    # Preprocess no-sort (the pivot pipeline)
    pkg = preprocess(raw_arr, sort_curves=False)
    pkg.update({
        "raw_trajectories": raw_arr,
        "r": np.stack(R, axis=0),
        "A": np.stack(A, axis=0),
        "x0": np.stack(X0, axis=0),
        "seeds": np.array(seeds),
        "r_scale_used": args.r_scale,
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pkg, f)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
