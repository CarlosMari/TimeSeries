"""Relative-amplitude metrics with explicit species inclusion, independent of GPU."""

import numpy as np


def relative_scale_metrics(targets, predictions, contract, *, log_floor=1e-8):
    # Keep this module NumPy-only so metric tests never initialize PyTorch/CUDA.
    if contract not in {"legacy_sorted_v1", "species_v2"}:
        raise ValueError(f"Unknown scale contract: {contract!r}")
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if targets.ndim != 2 or predictions.shape != targets.shape or targets.shape[0] == 0:
        raise ValueError("Relative scales must have equal, nonempty (batch, species) shapes")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Non-finite relative scales must be reported as failures, not scored")
    indices = np.arange(targets.shape[1]) if contract == "species_v2" else np.arange(1, targets.shape[1])
    if indices.size == 0:
        raise ValueError("At least two species are required")
    true, pred = targets[:, indices], predictions[:, indices]

    def r2(a, b, *, undefined_constant=False):
        variance = float(((a - a.mean()) ** 2).sum())
        if undefined_constant and variance <= np.finfo(float).eps * max(1.0, a.size * float(np.mean(a ** 2))):
            return None
        return float(1.0 - ((a - b) ** 2).sum() / max(variance, 1e-12))

    result = {
        "scale_contract": contract,
        "max_val_species_indices": indices.tolist(),
        "max_val_R2_pooled": r2(true, pred, undefined_constant=contract == "species_v2"),
        "max_val_MSE_pooled": float(np.mean((true - pred) ** 2)),
        "max_val_R2_per_species": {str(i): r2(targets[:, i], predictions[:, i], undefined_constant=True) for i in indices},
    }
    if contract == "species_v2":
        if not np.isfinite(log_floor) or not 0 < log_floor < 1:
            raise ValueError("log_floor must be between 0 and 1")
        if np.any(targets <= 0) or np.any(predictions <= 0):
            raise ValueError("species_v2 requires positive relative scales")
        tlog, plog = np.log(np.maximum(targets, log_floor)), np.log(np.maximum(predictions, log_floor))
        result["scale_centered_log_ratio_MSE"] = float(np.mean(
            ((tlog - tlog.mean(axis=1, keepdims=True)) - (plog - plog.mean(axis=1, keepdims=True))) ** 2))
        predicted_dominant = predictions.argmax(axis=1)
        target_dominants = np.isclose(targets, targets.max(axis=1, keepdims=True))
        result["scale_dominant_species_accuracy"] = float(target_dominants[np.arange(len(targets)), predicted_dominant].mean())
        result["scale_target_at_or_below_log_floor_count"] = int(np.count_nonzero(targets <= log_floor * (1 + 1e-6)))
    return result
