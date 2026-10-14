"""CPU-only exploratory test of latent targeting, not a trained-model benchmark.

The two controls are community overshoot and relative late-time level. Neither a
stationary tail nor target attainment establishes a true equilibrium or GLV
validity. No neural-model training, noise/extinction patch, or test-family-scale sampling
is performed. A small ridge probe diagnoses latent/property predictability only.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import sys
import time

# These precede numerical imports and affect this process only, not other jobs.
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg", WANDB_MODE="disabled")

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.utils.trajectory_controls import summarize_trajectory_controls


CONTROL_NAMES = ("log1p_overshoot_ratio", "relative_terminal_level")


def property_matrix(summary):
    return np.column_stack([np.log1p(summary["overshoot_ratio"]), summary["terminal_level"]])


def eligible(summary):
    return summary["valid"] & summary["stationary"]


def finite_summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"n": 0, "min": None, "median": None, "max": None, "iqr": None}
    q = np.quantile(values, [.0, .25, .5, .75, 1.])
    return {"n": int(values.size), "min": float(q[0]), "median": float(q[2]),
            "max": float(q[4]), "iqr": float(q[3] - q[1])}


def population_summary(trajectories, summary):
    mask = eligible(summary)
    return {
        "n": int(len(trajectories)), "valid_n": int(summary["valid"].sum()),
        "stationary_and_valid_n": int(mask.sum()),
        "eligible_zero_overshoot_fraction": float(np.mean(summary["overshoot_ratio"][mask] <= 1e-12)) if mask.any() else None,
        "any_zero_species_n": int(np.count_nonzero(np.any(np.all(trajectories == 0, axis=2), axis=1))),
        "family_unit_peak_above_1p05_n": int(np.count_nonzero(trajectories.max(axis=(1, 2)) > 1.05)),
        "overshoot_ratio_eligible": finite_summary(summary["overshoot_ratio"][mask]),
        "terminal_level_eligible": finite_summary(summary["terminal_level"][mask]),
    }


def prior_latents(rng, n, dim, radius, *, with_rejections=False):
    """A standard normal restricted to the declared latent ball, by rejection."""
    chunks = []
    remaining = n
    rejected = 0
    for _ in range(100):
        batch = rng.normal(size=(remaining, dim))
        inside = np.linalg.norm(batch, axis=1) <= radius
        rejected += int((~inside).sum())
        batch = batch[inside]
        chunks.append(batch)
        remaining -= len(batch)
        if not remaining:
            result = np.concatenate(chunks).astype(np.float32)
            return (result, rejected) if with_rejections else result
    raise ValueError("Latent ball too restrictive for this rejection budget")


def local_latents(rng, parents, n, radius, jitter):
    chunks, remaining, rejected = [], n, 0
    for _ in range(100):
        selected = parents[rng.integers(0, len(parents), size=remaining)]
        batch = selected + rng.normal(0, jitter, selected.shape)
        inside = np.linalg.norm(batch, axis=1) <= radius
        rejected += int((~inside).sum())
        batch = batch[inside]
        chunks.append(batch)
        remaining -= len(batch)
        if not remaining:
            return np.concatenate(chunks).astype(np.float32), rejected
    raise ValueError("Local latent proposals exhausted the declared rejection budget")


def distances(properties, target, scales, mask):
    out = np.full(len(properties), np.inf)
    good = mask & np.isfinite(properties).all(axis=1)
    out[good] = np.sqrt(np.mean(((properties[good] - target) / scales) ** 2, axis=1))
    return out


def linear_probe(z_train, y_train, z_probe, y_probe):
    """Choose axes on calibration only; evaluate on fresh latent codes."""
    model = Ridge(alpha=1.0).fit(z_train, y_train)
    prediction = model.predict(z_probe)
    result = {"ridge_alpha": 1.0, "heldout_r2": {}, "single_axis": {}}
    for col, name in enumerate(CONTROL_NAMES):
        observed = float(r2_score(y_probe[:, col], prediction[:, col])) if np.ptp(y_probe[:, col]) > 1e-12 else np.nan
        result["heldout_r2"][name] = observed if np.isfinite(observed) else None
        correlations = np.array([spearmanr(z_train[:, j], y_train[:, col]).statistic
                                 for j in range(z_train.shape[1])])
        correlations = np.nan_to_num(correlations)
        axis = int(np.argmax(np.abs(correlations)))
        rho = float(spearmanr(z_probe[:, axis], y_probe[:, col]).statistic)
        result["single_axis"][name] = {
            "selected_axis": axis, "calibration_spearman": float(correlations[axis]),
            "heldout_spearman": rho if np.isfinite(rho) else None,
        }
    return result


def search_target(decoder, bank_z, bank_x, bank_y, target, scales, *, rng,
                  radius, budget, jitter, guided, tolerance):
    """Same bank incumbent and online decoder budget for guided/random methods."""
    bank_scores = distances(bank_y, target, scales, np.ones(len(bank_y), dtype=bool))
    best_bank = int(bank_scores.argmin())
    parents = bank_z[np.argsort(bank_scores)[:min(8, len(bank_z))]]
    candidate_z, candidate_x = [], []
    norm_rejections = 0
    rounds = 2 if guided else 1
    for round_index in range(rounds):
        count = budget // rounds + (round_index < budget % rounds)
        if guided:
            z, rejections = local_latents(rng, parents, count, radius, jitter)
        else:
            z, rejections = prior_latents(rng, count, bank_z.shape[1], radius, with_rejections=True)
        norm_rejections += rejections
        x = decoder(z)
        candidate_z.append(z)
        candidate_x.append(x)
        if guided:
            # Parent coordinates include initial bank points; decode outputs for
            # proposals are reused. Rank new candidates plus original bank only.
            proposed_z = np.concatenate(candidate_z)
            proposed_x = np.concatenate(candidate_x)
            measured = summarize_trajectory_controls(proposed_x)
            proposed_y = property_matrix(measured)
            proposed_scores = distances(proposed_y, target, scales, eligible(measured))
            combined_z = np.concatenate([bank_z, proposed_z])
            combined_scores = np.concatenate([bank_scores, proposed_scores])
            parents = combined_z[np.argsort(combined_scores)[:min(8, len(combined_z))]]

    z, x = np.concatenate(candidate_z), np.concatenate(candidate_x)
    measured = summarize_trajectory_controls(x)
    y = property_matrix(measured)
    score = distances(y, target, scales, eligible(measured))
    new_best = int(score.argmin())
    improved = bool(score[new_best] < bank_scores[best_bank])
    chosen_z = z[new_best] if improved else bank_z[best_bank]
    chosen_x = x[new_best] if improved else bank_x[best_bank]
    chosen_y = y[new_best] if improved else bank_y[best_bank]
    nearest_latent = float(np.linalg.norm(bank_z - chosen_z, axis=1).min())
    nn_trajectory = float(np.sqrt(np.mean((bank_x - chosen_x) ** 2, axis=(1, 2))).min())
    accepted = eligible(measured) & np.all(np.abs(y - target) / scales <= tolerance, axis=1)
    diversity = (float(np.median(pdist(x[accepted].reshape(int(accepted.sum()), -1)) / np.sqrt(x.shape[1] * x.shape[2])))
                 if accepted.sum() >= 2 else None)
    result = {
        "online_decodes": int(budget), "eligible_proposals": int(eligible(measured).sum()),
        "valid_proposals": int(measured["valid"].sum()),
        "latent_ball_rejections_before_decoding": norm_rejections,
        "lookup_only_error": float(bank_scores[best_bank]),
        "lookup_only_target_attained": bool(np.all(np.abs(bank_y[best_bank] - target) / scales <= tolerance)),
        "bank_codes_within_target_tolerance": int(np.all(np.abs(bank_y - target) / scales <= tolerance, axis=1).sum()),
        "best_error": float(min(score[new_best], bank_scores[best_bank])),
        "improved_over_lookup": improved,
        "selected_new_proposal": improved,
        "target_attained": bool(np.all(np.abs(chosen_y - target) / scales <= tolerance)),
        "normalized_coordinate_errors": ((chosen_y - target) / scales).tolist(),
        "new_proposals_within_tolerance": int(accepted.sum()),
        "accepted_proposal_pairwise_trajectory_rmse_median": diversity,
        "achieved_overshoot_ratio": float(np.expm1(chosen_y[0])),
        "achieved_relative_terminal_level": float(chosen_y[1]),
        "nearest_bank_latent_distance": nearest_latent,
        "nearest_bank_trajectory_rmse": nn_trajectory,
    }
    return result, chosen_z, chosen_x, z, x


class CPUDecoder:
    def __init__(self, checkpoint, batch_size):
        import torch
        from src.models.cvae import LSTM_VAE
        from src.utils.checkpoints import unpack_checkpoint
        torch.set_num_threads(1)
        self.torch = torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        legacy = {"n_curves": 7, "seq_len": 65, "latent_dim": 30,
                  "rnn_hidden_size": 256, "rnn_num_layers": 2,
                  "scale_prediction_mode": "log", "use_scale_conditioning": True}
        if "checkpoint_format_version" not in payload and checkpoint.name != "model_final_30_conditioned.pth":
            raise ValueError("This pilot only assumes the documented sorted v1 checkpoint; other plain checkpoints need verified configuration")
        config, state = unpack_checkpoint(payload, expected_architecture="LSTM_VAE", legacy_config=legacy)
        with redirect_stdout(io.StringIO()):
            self.model = LSTM_VAE(config).cpu().eval()
        self.model.load_state_dict(state)
        self.model.requires_grad_(False)
        self.batch_size = batch_size
        self.n_decoded = 0
        self.seconds = 0.0

    def __call__(self, z):
        start = time.perf_counter()
        arrays = []
        with self.torch.no_grad():
            for batch in range(0, len(z), self.batch_size):
                tensor = self.torch.as_tensor(z[batch:batch + self.batch_size], dtype=self.torch.float32, device="cpu")
                normalized, relative_scales = self.model.decode(tensor)
                arrays.append((normalized * relative_scales[:, :, None]).cpu().numpy())
        self.n_decoded += len(z)
        self.seconds += time.perf_counter() - start
        return np.concatenate(arrays)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_pilot(decoder, *, bank_size=512, probe_size=256, budget=64, seed=20261014,
              jitter=.25, tolerance=.15, repetitions=3):
    dim = decoder.model.latent_dim
    radius = float(np.sqrt(dim) + 3.)
    bank_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    probe_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    bank_z = prior_latents(bank_rng, bank_size, dim, radius)
    probe_z = prior_latents(probe_rng, probe_size, dim, radius)
    bank_x, probe_x = decoder(bank_z), decoder(probe_z)
    repeated = bank_z[:min(8, len(bank_z))]
    repeat_first, repeat_second = decoder(repeated), decoder(repeated)
    repeatable = bool(np.allclose(repeat_first, repeat_second, rtol=1e-6, atol=1e-7, equal_nan=False))
    bank_summary = summarize_trajectory_controls(bank_x)
    probe_summary = summarize_trajectory_controls(probe_x)
    bank_mask, probe_mask = eligible(bank_summary), eligible(probe_summary)
    bank_y, probe_y = property_matrix(bank_summary), property_matrix(probe_summary)
    midpoint = bank_size // 2
    seed_mask = bank_mask.copy()
    seed_mask[midpoint:] = False
    definition_mask = bank_mask.copy()
    definition_mask[:midpoint] = False
    report = {
        "status": "exploratory", "seed": seed, "device": "cpu", "threads": 1,
        "repeatability": {"passed": repeatable, "extra_decodes": int(2 * len(repeated)),
                          "max_absolute_difference": float(np.max(np.abs(repeat_first - repeat_second)))},
        "calibration": population_summary(bank_x, bank_summary),
        "probe": population_summary(probe_x, probe_summary),
        "protocol": {"latent_dim": dim, "latent_radius": radius, "bank_size": bank_size,
                     "probe_size": probe_size, "online_decodes_per_target_per_method": budget,
                     "search_repetitions": repetitions,
                     "calibration_split": "first half seeds/incumbents only; second half targets/IQR/ridge only",
                     "jitter_sigma": jitter, "coordinatewise_success_tolerance_in_calibration_IQR": tolerance,
                     "target_quantiles": [.2, .5, .8], "tail_points": 10,
                     "minimum_terminal_fraction_of_peak": .02,
                     "stationary_drift_tolerance": .05, "stationary_cv_tolerance": .05,
                     "coordinates": "decoded normalized curves times predicted relative factors; no absolute family-scale sampling",
                     "random_proposal": "standard normal restricted to latent ball",
                     "guided_proposal": "two local rounds around eight nearest codes in standardized control space; rejection to same latent ball",
                     "overshoot_definition": "post-initial rise above late level only when community starts below that level",
                     "community_weighting": "arithmetic mean in relative-family coordinates; per-species amplitude factors affect community behavior",
                     "decoder_conditioning": "no external decoder condition: the legacy scale conditioning enters the encoder; both shape and factors are functions of z",
                     "shared_bank_incumbent": True},
        "limitations": [
            "Single-checkpoint exploratory search, not a corrected-model benchmark or publication finding.",
            "Late-time level is not a verified equilibrium; aggregate stationarity can hide species-level changes.",
            "Target accuracy does not establish GLV validity, biological species identity, novelty, or diverse conditional generation.",
            "Offline calibration and probe costs are reported separately; no speedup over simulator rejection sampling is established.",
            "Joint quantile targets may be incompatible; all targets are retained, and failure is not proof of global unreachability.",
            "The v1 checkpoint uses peak-sorted species; relative terminal units are not absolute ecological abundances.",
            "A low fixed-alpha linear-probe R² does not show the latent is uncontrollable; search may be nonlinear.",
        ],
    }
    arrays = {"bank_z": bank_z, "bank_x": bank_x, "bank_properties": bank_y,
              "bank_eligible": bank_mask, "probe_z": probe_z, "probe_properties": probe_y,
              "probe_eligible": probe_mask}
    if not repeatable:
        report["status"] = "decoder_not_repeatable"
        return report, arrays
    report["search_bank_eligible_n"] = int(seed_mask.sum())
    report["target_definition_eligible_n"] = int(definition_mask.sum())
    if min(seed_mask.sum(), definition_mask.sum()) < max(32, dim + 2) or probe_mask.sum() < 16:
        report["status"] = "insufficient_stationary_valid_samples"
        return report, arrays
    z, x, y = bank_z[seed_mask], bank_x[seed_mask], bank_y[seed_mask]
    definition_y = bank_y[definition_mask]
    scales = np.quantile(definition_y, .75, axis=0) - np.quantile(definition_y, .25, axis=0)
    report["control_IQR"] = dict(zip(CONTROL_NAMES, scales.tolist()))
    if np.any(scales < 1e-5):
        report["status"] = "insufficient_control_variation"
        return report, arrays
    report["predictability"] = linear_probe(bank_z[definition_mask], definition_y, probe_z[probe_mask], probe_y[probe_mask])
    levels = np.quantile(definition_y, [.2, .5, .8], axis=0)
    targets = np.unique(np.array([[a, b] for a in levels[:, 0] for b in levels[:, 1]]), axis=0)
    report["distinct_joint_targets"] = len(targets)
    report["targets"] = []
    for trial in range(len(targets) * repetitions):
        target_index, repetition = divmod(trial, repetitions)
        target = targets[target_index]
        record = {"target_id": target_index,
                  "repetition": repetition,
                  "requested_overshoot_ratio": float(np.expm1(target[0])),
                  "requested_relative_terminal_level": float(target[1])}
        for guided, name in [(False, "random_plus_bank"), (True, "local_search_plus_bank")]:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 10 + int(guided), target_index, repetition]))
            result, chosen_z, chosen_x, proposed_z, proposed_x = search_target(
                decoder, z, x, y, target, scales, rng=rng, radius=radius,
                budget=budget, jitter=jitter, guided=guided, tolerance=tolerance,
            )
            record[name] = result
            arrays[f"{name}_{target_index}_{repetition}_selected_z"] = chosen_z
            arrays[f"{name}_{target_index}_{repetition}_selected_x"] = chosen_x
            arrays[f"{name}_{target_index}_{repetition}_proposed_z"] = proposed_z
            arrays[f"{name}_{target_index}_{repetition}_proposed_x"] = proposed_x
        report["targets"].append(record)
    for name in ["random_plus_bank", "local_search_plus_bank"]:
        entries = [row[name] for row in report["targets"]]
        report[name] = {
            "target_trials_attained": sum(row["target_attained"] for row in entries),
            "target_trials_total": len(entries), "median_normalized_error": float(np.median([row["best_error"] for row in entries])),
            "new_proposals_selected": sum(row["selected_new_proposal"] for row in entries),
        }
    report["offline_bank_decodes"] = bank_size
    report["independent_probe_decodes"] = probe_size
    differences = np.array([row["local_search_plus_bank"]["best_error"] - row["random_plus_bank"]["best_error"] for row in report["targets"]])
    report["local_vs_random_paired_trials"] = {"win": int((differences < -1e-12).sum()), "tie": int((np.abs(differences) <= 1e-12).sum()), "loss": int((differences > 1e-12).sum())}
    report["bank_only"] = {"distinct_targets_attained": sum(report["targets"][i * repetitions]["random_plus_bank"]["lookup_only_target_attained"] for i in range(len(targets))), "distinct_targets_total": len(targets)}
    report["online_decodes_each_method"] = len(targets) * budget * repetitions
    arrays["targets_transformed"] = targets
    return report, arrays


def plot_report(report, arrays, directory, reference_properties=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bank = arrays["bank_properties"][arrays["bank_eligible"]]
    if reference_properties is not None and len(reference_properties):
        axes[0].scatter(np.expm1(reference_properties[:, 0]), reference_properties[:, 1],
                        s=9, color="gray", alpha=.3, label="Noisy real reference (context only)")
    if len(bank):
        axes[0].scatter(np.expm1(bank[:, 0]), bank[:, 1], s=11, alpha=.5, label="Generated calibration bank")
    axes[0].set(xlabel="Community overshoot ratio", ylabel="Relative late-time community level",
                title="Available behavior range; not physical equilibrium")
    axes[0].legend(fontsize=7)
    rows = report.get("targets", [])
    for name, label in [("random_plus_bank", "Random + same bank"), ("local_search_plus_bank", "Local search + same bank")]:
        if rows:
            ids = sorted({r["target_id"] for r in rows})
            axes[1].plot(ids, [np.median([r[name]["best_error"] for r in rows if r["target_id"] == i]) for i in ids], "o-", label=label)
    axes[1].set(xlabel="Joint target", ylabel="Best normalized RMS target error", title="Equal online decode budgets; lower is better")
    if rows:
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "control_feasibility.png", dpi=160)
    plt.close(fig)
    if rows:
        fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
        first_repetition = [row for row in rows if row["repetition"] == 0]
        for index, axis in enumerate(axes.flat):
            if index >= len(first_repetition):
                axis.set_visible(False)
                continue
            for name, label in [("random_plus_bank", "Random"), ("local_search_plus_bank", "Local")]:
                trajectory = arrays[f"{name}_{index}_0_selected_x"].mean(axis=0)
                axis.plot(np.arange(len(trajectory)), trajectory, label=label)
            axis.axhline(first_repetition[index]["requested_relative_terminal_level"], color="gray", linestyle=":")
            axis.set_title(f"O={first_repetition[index]['requested_overshoot_ratio']:.2f}, late={first_repetition[index]['requested_relative_terminal_level']:.2f}", fontsize=9)
        axes.flat[0].legend(fontsize=7)
        fig.supxlabel("Observation index; first search repetition, not best-of-repetitions")
        fig.supylabel("Relative community abundance")
        fig.tight_layout()
        fig.savefig(directory / "targeted_trajectories.png", dpi=160)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "model_ckpts/model_final_30_conditioned.pth")
    parser.add_argument("--reference-data", type=Path, default=REPO_ROOT / "data/TEST_FINAL_PROCESSED.pkl")
    parser.add_argument("--bank-size", type=int, default=512)
    parser.add_argument("--probe-size", type=int, default=256)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    if min(args.bank_size, args.probe_size, args.batch_size, args.repetitions) < 1 or args.budget < 2 or args.seed < 0:
        parser.error("positive sample/batch sizes, budget>=2, and nonnegative seed are required")
    output = args.out_dir or REPO_ROOT / "results/runs" / ("latent-control-cpu-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if output.exists():
        parser.error("--out-dir must be a new directory; existing artifacts are never replaced")
    decoder = CPUDecoder(args.checkpoint, args.batch_size)
    report, arrays = run_pilot(decoder, bank_size=args.bank_size, probe_size=args.probe_size,
                               budget=args.budget, seed=args.seed, repetitions=args.repetitions)
    report["checkpoint"] = {"basename": args.checkpoint.name, "sha256": file_hash(args.checkpoint),
                            "scale_contract": decoder.model.scale_contract,
                            "configuration_assumption": "Documented legacy sorted v1 config unless embedded in a versioned checkpoint"}
    report["decoder_samples_total"] = decoder.n_decoded
    report["decoder_wall_seconds"] = decoder.seconds
    report["versions"] = {"numpy": np.__version__, "torch": str(decoder.torch.__version__)}
    report["source_sha256"] = {name: file_hash(REPO_ROOT / name) for name in [
        "analysis/latent_control_feasibility.py", "src/utils/trajectory_controls.py",
        "src/models/cvae.py", "src/models/scale_contract.py",
    ]}
    reference = None
    if args.reference_data.exists():
        # Trusted local pickle only. These observations are not used for targets,
        # latent search, ridge fitting, or threshold/hyperparameter selection.
        with args.reference_data.open("rb") as stream:
            package = pickle.load(stream)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, 99]))
        indices = rng.choice(len(package["data"]), size=min(512, len(package["data"])), replace=False)
        real = package["data"][indices] * package["reconstruction_max_values"][indices, :, None]
        summary = summarize_trajectory_controls(real)
        report["noisy_real_reference_context_only"] = population_summary(real, summary)
        report["reference_data"] = {"basename": args.reference_data.name, "sha256": file_hash(args.reference_data), "indices": indices.tolist()}
        reference = property_matrix(summary)[eligible(summary)]
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "samples.npz", **arrays)
    plot_report(report, arrays, output, reference)
    print(json.dumps({"output": str(output), "status": report["status"],
                      "calibration": report["calibration"], "predictability": report.get("predictability"),
                      "random_plus_bank": report.get("random_plus_bank"),
                      "local_search_plus_bank": report.get("local_search_plus_bank")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
