"""Controlled, CPU-only GLV foundation; no dataset orchestration or file I/O.

The reference law is ``dx/dt = x * (r + A @ x)`` from
custom_glv_FIXED.generate_curves_Mario. Unlike generate_family_FIXED, selection
is on *clean* trajectories, never observations. All draws use local PCG64 RNGs.
Replay is deterministic within the same NumPy/SciPy numerical environment;
bitwise equivalence across library versions/platforms is not promised.

Parameter attempts and RHS evaluations have deterministic caps, not wall-clock
timeouts. An evaluation cap bounds solver work, not elapsed execution time.
For an OOD scale intervention use dataclasses.replace(config, r_scale=...).
"""

from dataclasses import dataclass
import hashlib
from typing import Iterable, Literal

import numpy as np
from scipy.integrate import solve_ivp


PARAMETER_DOMAIN = 0x50415241
NOISE_DOMAIN = 0x4E4F4953
RNG_SCHEME = "pcg64-seedsequence-domain-v1"


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class GLVConfig:
    species: int = 7
    tmax: float = 20.0
    n_points: int = 65
    r_scale: float = 2.0  # Exponential scale, NOT rate.
    diagonal_scale: float = 2.0  # A_ii = -Exponential(scale).
    offdiag_mean: float = 0.0
    offdiag_std: float = 1.0
    x0_scale: float = 0.1
    solver_rtol: float = 1e-6
    solver_atol: float = 1e-9
    solver_max_step: float = 0.5
    observation_lognormal_sigma: float = 0.01  # Log-space standard deviation.
    max_parameter_attempts: int = 500
    max_rhs_evaluations: int = 100000
    selection: Literal["clean", "none"] = "clean"
    max_abundance: float = 3.0
    min_species_peak: float = 0.1
    min_overshoot_species: int = 3
    overshoot_factor: float = 1.2
    steady_state_points: int = 10

    def __post_init__(self):
        for name, minimum in (("species", 1), ("n_points", 2),
                              ("max_parameter_attempts", 1), ("max_rhs_evaluations", 1),
                              ("steady_state_points", 1), ("min_overshoot_species", 0)):
            _integer(getattr(self, name), name, minimum)
        for name in ("tmax", "r_scale", "diagonal_scale", "x0_scale", "solver_rtol",
                     "solver_atol", "solver_max_step", "max_abundance", "overshoot_factor"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("offdiag_std", "observation_lognormal_sigma", "min_species_peak"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.offdiag_mean):
            raise ValueError("offdiag_mean must be finite")
        if self.selection not in ("clean", "none"):
            raise ValueError("selection must be 'clean' or 'none'")


@dataclass(frozen=True)
class GLVParameters:
    r: np.ndarray
    A: np.ndarray
    x0: np.ndarray


@dataclass(frozen=True)
class GLVRecord:
    config: GLVConfig
    parameter_seed: int  # Underlying system identity; also the split grouping key.
    noise_seed: int
    parameters: GLVParameters
    parameter_source: Literal["sampled", "provided"]
    parameter_attempts: int
    parameter_rejections: tuple[str, ...]
    equilibrium: np.ndarray
    times: np.ndarray
    clean: np.ndarray  # (species, n_points), original physical coordinates.
    observed: np.ndarray  # Same system/grid, independent observation-only noise.
    rhs_evaluations: int
    solver_method: str = "RK45"
    observation_law: str = "exp(N(-sigma**2/2, sigma**2))"
    rng_scheme: str = RNG_SCHEME


@dataclass(frozen=True)
class GenerationFailure:
    config: GLVConfig
    parameter_seed: int
    noise_seed: int
    reason: str
    parameter_attempts: int
    parameter_rejections: tuple[str, ...] = ()
    detail: str = ""


def draw_parameters(config: GLVConfig, rng: np.random.Generator) -> GLVParameters:
    """One candidate, with the reference independent exponential/normal laws."""
    r = rng.exponential(config.r_scale, config.species)
    A = rng.normal(config.offdiag_mean, config.offdiag_std,
                   (config.species, config.species))
    np.fill_diagonal(A, -rng.exponential(config.diagonal_scale, config.species))
    x0 = rng.exponential(config.x0_scale, config.species)
    return GLVParameters(r, A, x0)


def _check_parameters(config, parameters):
    r, A, x0 = (np.array(value, dtype=float, copy=True)
                for value in (parameters.r, parameters.A, parameters.x0))
    n = config.species
    if r.shape != (n,) or A.shape != (n, n) or x0.shape != (n,):
        return None, None, "parameter_shape"
    if not all(np.isfinite(value).all() for value in (r, A, x0)):
        return None, None, "parameter_nonfinite"
    if np.any(x0 <= 0):
        return None, None, "initial_condition_nonpositive"
    try:
        equilibrium = np.linalg.solve(A, -r)
        if not np.isfinite(equilibrium).all() or np.any(equilibrium <= 0):
            return None, None, "equilibrium_infeasible"
        eigenvalues = np.linalg.eigvals(equilibrium[:, None] * A)
        # Strict stability: exclude marginal modes rather than calling them stable.
        if not np.isfinite(eigenvalues).all() or np.any(eigenvalues.real >= 0):
            return None, None, "equilibrium_unstable"
    except np.linalg.LinAlgError:
        return None, None, "equilibrium_singular"
    return GLVParameters(r, A, x0), equilibrium, None


def clean_selection_reason(clean: np.ndarray, config: GLVConfig) -> str | None:
    """Legacy thresholds, explicitly applied only to clean grid samples.

    The tail mean uses the last min(steady_state_points, n_points) samples;
    it is a selection heuristic, not an assertion of reached equilibrium.
    Nonfinite/negative trajectories are invalid even when selection is disabled.
    """
    if clean.shape != (config.species, config.n_points):
        return "trajectory_shape"
    if not np.isfinite(clean).all():
        return "trajectory_nonfinite"
    if np.any(clean < 0):
        return "trajectory_negative"
    if config.selection == "none":
        return None
    peaks = clean.max(axis=1)
    if np.any(peaks > config.max_abundance):
        return "selection_max_abundance"
    if np.any(peaks < config.min_species_peak):
        return "selection_min_species_peak"
    tail = clean[:, -config.steady_state_points:].mean(axis=1)
    if np.count_nonzero(peaks > config.overshoot_factor * tail) < config.min_overshoot_species:
        return "selection_overshoot"
    return None


def observe(clean: np.ndarray, sigma: float, noise_seed: int) -> np.ndarray:
    """Mean-one multiplicative lognormal noise, without empirical recentering."""
    _integer(noise_seed, "noise_seed")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([noise_seed, NOISE_DOMAIN])))
    with np.errstate(over="ignore", invalid="ignore"):
        return clean * rng.lognormal(-0.5 * sigma * sigma, sigma, size=clean.shape)


class _EvaluationLimit(Exception):
    pass


def generate_one(config: GLVConfig, *, parameter_seed: int, noise_seed: int,
                 parameters: GLVParameters | None = None) -> GLVRecord | GenerationFailure:
    """Pure single-system operation: a paired record or an explicit failure.

    Provided parameters undergo the same feasibility/stability checks; their seed
    is caller-assigned provenance/group identity, not a claim they were sampled.
    Invalid API/configuration inputs raise ValueError; scientific rejections and
    solver failures are returned. No retries after integration/selection failure.
    """
    _integer(parameter_seed, "parameter_seed")
    _integer(noise_seed, "noise_seed")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([parameter_seed, PARAMETER_DOMAIN])))
    rejections = []
    source = "provided" if parameters is not None else "sampled"
    limit = 1 if parameters is not None else config.max_parameter_attempts
    for attempt in range(1, limit + 1):
        candidate = parameters if parameters is not None else draw_parameters(config, rng)
        accepted, equilibrium, reason = _check_parameters(config, candidate)
        if reason is None:
            break
        rejections.append(reason)
    else:
        return GenerationFailure(config, parameter_seed, noise_seed,
                                 "provided_parameters_rejected" if parameters is not None
                                 else "parameter_attempts_exhausted", limit, tuple(rejections))

    def failure(reason, detail=""):
        return GenerationFailure(config, parameter_seed, noise_seed, reason,
                                 attempt, tuple(rejections), detail)

    times = np.linspace(0.0, config.tmax, config.n_points)
    evaluations = 0

    def rhs(t, x):
        nonlocal evaluations
        if evaluations >= config.max_rhs_evaluations:
            raise _EvaluationLimit
        evaluations += 1
        return x * (accepted.r + accepted.A @ x)

    try:
        sol = solve_ivp(rhs, (0.0, config.tmax), accepted.x0, t_eval=times,
                        method="RK45", rtol=config.solver_rtol, atol=config.solver_atol,
                        max_step=config.solver_max_step)
    except _EvaluationLimit:
        return failure("rhs_evaluation_limit")
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
        return failure("solver_exception", str(error))
    if not sol.success:
        return failure("solver_failed", sol.message)
    if not np.array_equal(sol.t, times):
        return failure("solver_incomplete_grid")
    reason = clean_selection_reason(sol.y, config)
    if reason is not None:
        return failure(reason)
    observed = observe(sol.y, config.observation_lognormal_sigma, noise_seed)
    # Numerical validity is not an abundance/overshoot selection on observations.
    if not np.isfinite(observed).all():
        return failure("observation_nonfinite")
    return GLVRecord(config, parameter_seed, noise_seed, accepted, source, attempt,
                     tuple(rejections), equilibrium, times, sol.y, observed, evaluations)


@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def __post_init__(self):
        values = (self.train, self.validation, self.test)
        if not all(np.isfinite(x) and 0 <= x <= 1 for x in values):
            raise ValueError("split fractions must be finite and in [0, 1]")
        if not np.isclose(sum(values), 1.0, rtol=0, atol=1e-12):
            raise ValueError("split fractions must sum to one")


def assign_split(system_seed: int, fractions: SplitFractions = SplitFractions(), *,
                 split_seed: int = 0) -> Literal["train", "validation", "test"]:
    """Stable hash assignment, independent of order, noise seed and dataset size.

    Fractions are probabilities, not exact quotas. Reuse split_seed and fractions
    across ID/OOD/noise variants; all copies sharing a system seed stay together.
    """
    _integer(system_seed, "system_seed")
    _integer(split_seed, "split_seed")
    digest = hashlib.sha256(f"controlled-glv-split-v1:{split_seed}:{system_seed}".encode()).digest()
    # 53 bits so conversion to float never rounds a draw up to 1.0.
    u = (int.from_bytes(digest[:8], "big") >> 11) / 2**53
    if u < fractions.train:
        return "train"
    if u < fractions.train + fractions.validation:
        return "validation"
    return "test"


def split_report(system_seeds: Iterable[int], fractions: SplitFractions = SplitFractions(), *,
                 split_seed: int = 0) -> dict:
    """Report target and realized fractions for unique systems and noisy copies.

    Pass one system seed per observation. Empty collections report zero realized
    fractions. This does not silently rebalance groups to meet target quotas.
    """
    seeds = list(system_seeds)
    names = ("train", "validation", "test")
    observations = dict.fromkeys(names, 0)
    systems = dict.fromkeys(names, 0)
    for seed in seeds:
        observations[assign_split(seed, fractions, split_seed=split_seed)] += 1
    unique_seeds = set(seeds)
    for seed in unique_seeds:
        systems[assign_split(seed, fractions, split_seed=split_seed)] += 1
    return {
        "target_fractions": {name: getattr(fractions, name) for name in names},
        "system_counts": systems,
        "observation_counts": observations,
        "system_fractions": {name: count / len(unique_seeds) if seeds else 0.0
                             for name, count in systems.items()},
        "observation_fractions": {name: count / len(seeds) if seeds else 0.0
                                  for name, count in observations.items()},
    }
