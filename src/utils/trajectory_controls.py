"""NumPy-only controls in declared relative-family units, not physical LV tests."""

from numbers import Integral, Real

import numpy as np


def summarize_trajectory_controls(
    trajectories,
    *,
    tail_points=10,
    min_terminal_fraction=0.02,
    stationary_drift_tolerance=0.05,
    stationary_cv_tolerance=0.05,
):
    """Summarize a nonnegative ``(samples, species, time)`` batch on CPU.

    Community abundance is the species mean. ``terminal_level`` is its final
    ``tail_points`` mean in the input's units, *not* divided by the peak.
    ``peak_excess_ratio`` is ``max(peak - terminal_level, 0) / terminal_level``.
    ``overshoot_ratio`` counts a peak after the initial sample only when the
    community starts below its terminal level; simple decay is not overshoot.
    Ratios require finite, nonnegative input, peak > 1e-8, and terminal level
    strictly greater than ``min_terminal_fraction * peak``. Invalid samples
    remain present with ``valid=False`` and NaN ratios; malformed input rows
    also have NaN terminal levels. A finite near-zero terminal level is kept.

    ``tail_cv`` uses population standard deviation divided by terminal level.
    ``tail_drift`` is the absolute difference of second- and first-half tail
    means divided by terminal level (an odd tail gives the second half one
    extra point). ``stationary`` requires validity and both ratios <= their
    tolerances. This is only a late-terminal proxy, not true equilibrium or
    evidence of physical Lotka--Volterra adherence. Callers must retain invalid
    counts rather than silently removing failed samples.

    ``tail_points`` must be an integer >= 4 and <= the time dimension.
    The terminal fraction must be finite and in (0, 1); stationarity tolerances
    must be finite and positive. All outputs are arrays of shape (samples,).
    """
    if isinstance(tail_points, (bool, np.bool_)) or not isinstance(tail_points, Integral):
        raise ValueError("tail_points must be an integer >= 4")
    if tail_points < 4:
        raise ValueError("tail_points must be an integer >= 4")
    thresholds = {
        "min_terminal_fraction": min_terminal_fraction,
        "stationary_drift_tolerance": stationary_drift_tolerance,
        "stationary_cv_tolerance": stationary_cv_tolerance,
    }
    for name, value in thresholds.items():
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if min_terminal_fraction >= 1:
        raise ValueError("min_terminal_fraction must be less than 1")

    values = np.asarray(trajectories)
    if (values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1
            or values.shape[2] < tail_points):
        raise ValueError("trajectories must have shape (N, S, T), N,S >= 1, T >= tail_points")
    if np.iscomplexobj(values):
        raise ValueError("trajectories must contain real numeric values")
    values = np.asarray(values, dtype=np.float64)
    count = values.shape[0]
    result = {
        name: np.full(count, np.nan)
        for name in ("overshoot_ratio", "peak_excess_ratio", "terminal_level", "tail_cv", "tail_drift")
    }
    result["valid"] = np.zeros(count, dtype=bool)
    result["stationary"] = np.zeros(count, dtype=bool)
    result["starts_below_terminal"] = np.zeros(count, dtype=bool)
    input_valid = np.isfinite(values).all(axis=(1, 2)) & (values >= 0).all(axis=(1, 2))
    indices = np.flatnonzero(input_valid)
    if indices.size == 0:
        return result

    # Work only on valid rows; normalize internally to avoid overflow in means
    # and squared deviations. Restore terminal units, never peak-normalize it.
    selected = values[indices]
    scale = selected.max(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        normalized = np.divide(
            selected, scale[:, None, None], out=np.zeros_like(selected),
            where=scale[:, None, None] > 0,
        )
        community = normalized.mean(axis=1)
        peak = community.max(axis=1)
        tail = community[:, -tail_points:]
        level = tail.mean(axis=1)
        result["terminal_level"][indices] = level * scale
        eligible = (peak * scale > 1e-8) & (level > min_terminal_fraction * peak)
        valid_indices = indices[eligible]
        valid_level = level[eligible]
        valid_tail = tail[eligible]
        excess = np.maximum(peak[eligible] - valid_level, 0) / valid_level
        starts_below = community[eligible, 0] < valid_level
        after_initial_peak = community[eligible, 1:].max(axis=1)
        overshoot = np.where(starts_below,
                             np.maximum(after_initial_peak - valid_level, 0) / valid_level, 0.)
        cv = valid_tail.std(axis=1) / valid_level
        split = tail_points // 2
        drift = np.abs(
            valid_tail[:, split:].mean(axis=1) - valid_tail[:, :split].mean(axis=1)
        ) / valid_level

    result["overshoot_ratio"][valid_indices] = overshoot
    result["peak_excess_ratio"][valid_indices] = excess
    result["starts_below_terminal"][valid_indices] = starts_below
    result["tail_cv"][valid_indices] = cv
    result["tail_drift"][valid_indices] = drift
    result["valid"][valid_indices] = True
    result["stationary"][valid_indices] = (
        (cv <= stationary_cv_tolerance) & (drift <= stationary_drift_tolerance)
    )
    return result
