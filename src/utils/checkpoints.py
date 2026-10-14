"""Versioned corrected-model checkpoints, with an explicit legacy loading path."""

from collections.abc import Mapping
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import torch

from src.models.scale_contract import (
    LEGACY_SCALE_CONTRACT, SPECIES_SCALE_CONTRACT, scale_contract_from_config,
)


CHECKPOINT_FORMAT_VERSION = 2


def make_checkpoint(model, training_metadata=None):
    if getattr(model, "scale_contract", None) != SPECIES_SCALE_CONTRACT:
        raise ValueError("Versioned corrected checkpoints require species_v2")
    config = deepcopy(model.config)
    config["scale_contract"] = SPECIES_SCALE_CONTRACT
    for name in ["n_curves", "seq_len", "latent_dim", "rnn_hidden_size", "rnn_num_layers",
                 "scale_prediction_mode", "scale_log_floor", "use_scale_conditioning",
                 "d_model", "n_heads", "n_enc_layers", "n_dec_layers", "ff_dim", "dropout"]:
        if hasattr(model, name):
            config[name] = getattr(model, name)
    # Refuse arbitrary Python objects so weights_only loading remains supported.
    config = json.loads(json.dumps(config, allow_nan=False))
    metadata = json.loads(json.dumps(training_metadata or {}, allow_nan=False))
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": type(model).__name__,
        "model_config": config,
        "state_dict": model.state_dict(),
        "training_metadata": metadata,
    }


def unpack_checkpoint(payload, *, expected_architecture=None, legacy_config=None):
    """Return (model config, state dict); never infer corrected config from weights.

    Plain historical state dictionaries require an explicit legacy configuration.
    Passing that fallback cannot bypass validation of a malformed versioned file.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must be a mapping")
    if "checkpoint_format_version" in payload or "state_dict" in payload:
        if (type(payload.get("checkpoint_format_version")) is not int
                or payload["checkpoint_format_version"] != CHECKPOINT_FORMAT_VERSION):
            raise ValueError("Unknown or missing corrected checkpoint format version")
        if expected_architecture is not None and payload.get("architecture") != expected_architecture:
            raise ValueError("Checkpoint architecture does not match the selected model")
        config, state = payload.get("model_config"), payload.get("state_dict")
        if not isinstance(config, dict) or config.get("scale_contract") != SPECIES_SCALE_CONTRACT:
            raise ValueError("Corrected checkpoint must declare species_v2 in model_config")
        scale_contract_from_config(config)
        if not isinstance(state, Mapping) or not state or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError("Checkpoint state_dict must contain tensors")
        return deepcopy(config), state

    if legacy_config is None:
        raise ValueError("Plain state dictionaries require an explicit legacy_config")
    config = deepcopy(legacy_config)
    if config.get("scale_contract", LEGACY_SCALE_CONTRACT) != LEGACY_SCALE_CONTRACT:
        raise ValueError("Plain state dictionaries cannot be loaded as species_v2")
    config["scale_contract"] = LEGACY_SCALE_CONTRACT
    if not payload or not all(torch.is_tensor(v) for v in payload.values()):
        raise ValueError("Legacy state dictionary must contain tensors")
    return config, payload


def write_checkpoint_payload(path, payload, *, overwrite=False):
    """Atomically publish a validated corrected checkpoint, preserving old files.

    An interrupted/failed serialization cannot replace the previous snapshot.
    No-overwrite is the default; replacement must be explicit for a run-owned
    progress snapshot. This is a weight-recovery artifact, not full optimizer/RNG
    state for exact training resumption.
    """
    unpack_checkpoint(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Atomic fail-if-present; never truncates a checkpoint.
    finally:
        temporary.unlink(missing_ok=True)
