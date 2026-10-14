"""Explicit relative-amplitude contracts for legacy and unsorted-species models.

Legacy models predict C-1 transformed factors and insert a fixed 1 for species 0.
The corrected contract predicts C log factors and removes their shared additive
offset. Any species can therefore define the family maximum. These factors do
not replace the separate absolute family-scale component.
"""

import math

import torch


LEGACY_SCALE_CONTRACT = "legacy_sorted_v1"
SPECIES_SCALE_CONTRACT = "species_v2"
SCALE_CONTRACTS = (LEGACY_SCALE_CONTRACT, SPECIES_SCALE_CONTRACT)


def scale_contract_from_config(config):
    # Missing metadata means legacy behavior, never an implicit checkpoint upgrade.
    contract = config.get("scale_contract", LEGACY_SCALE_CONTRACT)
    if contract not in SCALE_CONTRACTS:
        raise ValueError(f"Unknown scale_contract: {contract!r}")
    if contract == SPECIES_SCALE_CONTRACT and config.get("scale_prediction_mode", "linear") != "log":
        raise ValueError("species_v2 requires scale_prediction_mode='log'")
    floor = config.get("scale_log_floor", 1e-8)
    if not isinstance(floor, (int, float)) or not math.isfinite(floor) or not 0 < floor < 1:
        raise ValueError("scale_log_floor must be finite and between 0 and 1")
    return contract


def scale_output_dim(n_curves, contract):
    if contract not in SCALE_CONTRACTS or n_curves < 2:
        raise ValueError("A known scale contract and at least two species are required")
    return n_curves if contract == SPECIES_SCALE_CONTRACT else n_curves - 1


def scale_target_slice(contract):
    if contract not in SCALE_CONTRACTS:
        raise ValueError(f"Unknown scale_contract: {contract!r}")
    return slice(None) if contract == SPECIES_SCALE_CONTRACT else slice(1, None)


def decode_scale_predictions(raw, contract, mode):
    """Return (relative factors, loss coordinates) without choosing an index.

    For species_v2, normalize in log space before exponentiating. A numerical
    floor at the floating-point minimum prevents underflow to zero. Corrected loss
    coordinates are centered logits, not max-normalized log factors, so a wrongly
    dominant species can receive a useful gradient. The contract
    models positive species; exactly absent species require a separate model.
    The legacy branch intentionally retains its original transformations/dtypes.
    """
    if raw.ndim != 2 or not raw.is_floating_point():
        raise ValueError("Scale predictions must be a floating (batch, head_dim) tensor")
    if contract == SPECIES_SCALE_CONTRACT:
        if mode != "log":
            raise ValueError("species_v2 requires scale_prediction_mode='log'")
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("Non-finite scale predictions")
        stable = raw if raw.dtype == torch.float64 else raw.float()
        centered = stable - stable.mean(dim=1, keepdim=True)
        log_factors = stable - stable.amax(dim=1, keepdim=True)
        log_factors = log_factors.clamp(min=math.log(torch.finfo(stable.dtype).tiny))
        return log_factors.exp(), centered
    if contract != LEGACY_SCALE_CONTRACT:
        raise ValueError(f"Unknown scale_contract: {contract!r}")

    ones = torch.ones(raw.shape[0], 1, device=raw.device)
    if mode == "log":
        partial = raw.exp()
        leading = torch.log(ones + 1e-8)
    elif mode == "exp":
        partial = torch.log(raw.clamp(min=1e-8))
        leading = ones.exp()
    else:
        partial, leading = raw, ones
    return torch.cat([ones, partial], dim=1), torch.cat([leading, raw], dim=1)


def validate_scale_targets(values, contract, n_curves=None):
    """Check a corrected target's geometry, without sorting or relabeling it."""
    scale_target_slice(contract)  # Validate the contract even for legacy input.
    if (values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2
            or (n_curves is not None and values.shape[1] != n_curves)):
        raise ValueError("Scale targets must have shape (batch, n_curves)")
    if contract == LEGACY_SCALE_CONTRACT:
        return  # Historical unsorted experiments must remain explicitly reproducible.
    if not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
        raise ValueError("species_v2 requires finite, strictly positive relative scales")
    maxima = values.amax(dim=1)
    if bool((values > 1 + 1e-6).any()) or not torch.allclose(maxima, torch.ones_like(maxima), atol=1e-6, rtol=1e-6):
        raise ValueError("species_v2 relative scales must have a family maximum of 1")


def validate_dataset_scale_contract(values, contract, *, sort_applied=None,
                                    allow_legacy_on_unsorted=False, n_curves=None,
                                    normalized_curves=None):
    validate_scale_targets(values, contract, n_curves)
    if contract == SPECIES_SCALE_CONTRACT and normalized_curves is not None:
        if (normalized_curves.ndim != 3 or normalized_curves.shape[:2] != values.shape
                or normalized_curves.shape[-1] == 0):
            raise ValueError("Normalized curves must have shape (batch, species, time)")
        for start in range(0, normalized_curves.shape[0], 512):
            block = normalized_curves[start:start + 512]
            if not bool(torch.isfinite(block).all()):
                raise ValueError("Non-finite normalized curves")
            if not bool((block != 0).any(dim=-1).all()):
                raise ValueError("Exactly absent species are unsupported by species_v2; preprocessing epsilon is not a real peak")
    if contract == LEGACY_SCALE_CONTRACT and not allow_legacy_on_unsorted:
        first_is_max = torch.allclose(values[:, 0], torch.ones_like(values[:, 0]), atol=1e-6, rtol=1e-6)
        if sort_applied is False or not first_is_max:
            raise ValueError(
                "Legacy six-output scaling is invalid for unsorted species. Use species_v2; "
                "--allow-legacy-on-unsorted is only for explicitly labeled historical reproduction."
            )


def scale_reconstruction_loss(predicted, targets, contract, *, transformed=None, log_floor=1e-8):
    """Legacy linear MSE, or all-species centered log-ratio MSE in fp32.

    The floor is numerical stabilization, not a biological detection threshold.
    Its value is part of the model configuration. Corrected targets must remain
    positive: exactly absent species and all-zero families are not silently fixed.
    """
    selected = scale_target_slice(contract)
    if predicted.shape != targets.shape:
        raise ValueError("Predicted and target relative scales must have equal shapes")
    if contract == LEGACY_SCALE_CONTRACT:
        return torch.nn.functional.mse_loss(predicted[:, selected], targets[:, selected], reduction="mean")
    validate_scale_targets(targets, contract, predicted.shape[1])
    if transformed is None or transformed.shape != targets.shape:
        raise ValueError("species_v2 loss requires the all-species centered logits")
    if not math.isfinite(log_floor) or not 0 < log_floor < 1:
        raise ValueError("log_floor must be between 0 and 1")
    log_targets = targets.float().clamp_min(log_floor).log()
    centered_targets = log_targets - log_targets.mean(dim=1, keepdim=True)
    centered_predictions = transformed.float()
    centered_predictions = centered_predictions - centered_predictions.mean(dim=1, keepdim=True)
    return torch.nn.functional.mse_loss(centered_predictions, centered_targets)
