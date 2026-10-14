"""Run small regression tests with GPUs hidden and CPU thread counts capped.

Use the numerical environment's interpreter to include scientific tests. With a
stdlib-only interpreter, optional numerical checks are reported as skipped.
This script never starts a training/evaluation queue or regenerates artifacts.
"""

import os
from pathlib import Path
import subprocess
import sys


def main():
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MPLBACKEND": "Agg", "WANDB_MODE": "disabled",
    })
    root = Path(__file__).resolve().parent.parent
    print("CPU-only checks: CUDA hidden; numerical thread limits set to 1; no production runs.", flush=True)
    return subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        cwd=root, env=env, check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
