"""Tiny CPU-only scientific regression checks; no fitted model or local data.

The maintenance-only interpreter may skip these tests if numerical dependencies
are absent. CUDA is hidden before importing PyTorch, even in a GPU environment.
"""

import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None

if torch is not None:
    from data_generation.preprocessor import preprocess
    from src.models.cvae import LSTM_VAE
    from src.models.cvae_heteroscedastic import HeteroscedasticLSTMVAE
    from src.models.transformer_vae import TransformerVAE
    from src.models.scale_contract import (
        LEGACY_SCALE_CONTRACT, SPECIES_SCALE_CONTRACT,
        decode_scale_predictions, scale_reconstruction_loss,
        scale_target_slice, validate_scale_targets, validate_dataset_scale_contract,
    )
    from src.utils.checkpoints import make_checkpoint, unpack_checkpoint, write_checkpoint_payload


@unittest.skipIf(torch is None, "numerical dependencies required; use TimeSeries/bin/python")
class ScaleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.cuda_guard = mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU forbidden"))
        self.cuda_guard.start()
        self.addCleanup(self.cuda_guard.stop)
        self.config = {
            "n_curves": 7, "seq_len": 5, "latent_dim": 4,
            "rnn_hidden_size": 8, "rnn_num_layers": 1,
            "scale_prediction_mode": "log", "use_scale_conditioning": True,
            "scale_contract": SPECIES_SCALE_CONTRACT,
            "d_model": 8, "n_heads": 2, "n_enc_layers": 1, "n_dec_layers": 1,
            "ff_dim": 16, "dropout": 0.0,
        }

    def model(self, kind=None, **overrides):
        kind = LSTM_VAE if kind is None else kind
        with contextlib.redirect_stdout(io.StringIO()):
            return kind({**self.config, **overrides}).cpu()

    def test_every_species_can_be_the_dominant_scale(self):
        logits = torch.eye(7, dtype=torch.float64) * 3.0
        scales, transformed = decode_scale_predictions(logits, SPECIES_SCALE_CONTRACT, "log")
        self.assertEqual(scales.shape, (7, 7))
        torch.testing.assert_close(scales.argmax(dim=1), torch.arange(7))
        torch.testing.assert_close(scales.amax(dim=1), torch.ones(7, dtype=scales.dtype))
        self.assertTrue(bool((scales > 0).all() and (scales <= 1).all()))
        torch.testing.assert_close(transformed.mean(dim=1), torch.zeros(7, dtype=transformed.dtype))
        torch.testing.assert_close((transformed - transformed.amax(dim=1, keepdim=True)).exp(), scales)

    def test_species_contract_is_shift_and_permutation_consistent(self):
        logits = torch.tensor([[0.1, -2.0, 3.0, 0.5, -1.0, 2.0, -3.0]], dtype=torch.float64)
        perm = torch.tensor([6, 4, 1, 0, 5, 2, 3])
        scales, _ = decode_scale_predictions(logits, SPECIES_SCALE_CONTRACT, "log")
        shifted, _ = decode_scale_predictions(logits + 100.0, SPECIES_SCALE_CONTRACT, "log")
        permuted, _ = decode_scale_predictions(logits[:, perm], SPECIES_SCALE_CONTRACT, "log")
        torch.testing.assert_close(scales, shifted)
        torch.testing.assert_close(scales[:, perm], permuted)

    def test_extreme_logits_stay_finite_positive_and_differentiable(self):
        logits = torch.tensor([[10000.0, -10000.0, 0.0, 3.0, -2.0, 1.0, 5.0]], requires_grad=True)
        scales, transformed = decode_scale_predictions(logits, SPECIES_SCALE_CONTRACT, "log")
        self.assertTrue(bool(torch.isfinite(scales).all() and (scales > 0).all()))
        self.assertTrue(bool(torch.isfinite(transformed).all()))
        scales.sum().backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_legacy_scale_contract_matches_original_six_output_formula(self):
        raw = torch.tensor([[0.1, -0.2, 0.5, -1.0, 0.0, 1.0]])
        for mode in ["log", "linear", "exp"]:
            scales, transformed = decode_scale_predictions(raw, LEGACY_SCALE_CONTRACT, mode)
            partial = raw.exp() if mode == "log" else raw.clamp(min=1e-8).log() if mode == "exp" else raw
            torch.testing.assert_close(scales, torch.cat([torch.ones(1, 1), partial], dim=1))
            leading = torch.log(torch.ones(1, 1) + 1e-8) if mode == "log" else torch.ones(1, 1).exp() if mode == "exp" else torch.ones(1, 1)
            torch.testing.assert_close(transformed, torch.cat([leading, raw], dim=1))

    def test_unknown_contract_and_nonlog_corrected_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.model(scale_contract="guess")
        with self.assertRaises(ValueError):
            self.model(scale_prediction_mode="linear")

    def test_wrong_dominant_species_receives_gradient_under_centered_loss(self):
        logits = torch.tensor([[2., -1., 0., -.2, .4, -.5, .2]], requires_grad=True)
        targets = torch.tensor([[.2, 1., .3, .4, .5, .6, .7]])
        predicted, centered = decode_scale_predictions(logits, SPECIES_SCALE_CONTRACT, "log")
        loss = scale_reconstruction_loss(predicted, targets, SPECIES_SCALE_CONTRACT, transformed=centered)
        loss.backward()
        self.assertTrue(bool((logits.grad.abs() > 0).all()))
        self.assertGreater(float(logits.grad[0, 0]), 0)
        self.assertLess(float(logits.grad[0, 1]), 0)

    def test_legacy_loss_ignores_species_zero_but_corrected_loss_does_not(self):
        targets = torch.tensor([[.2, 1., .3, .4, .5, .6, .7]])
        predictions = targets.clone()
        predictions[:, 0] = .8
        legacy = scale_reconstruction_loss(predictions, targets, LEGACY_SCALE_CONTRACT)
        corrected = scale_reconstruction_loss(predictions, targets, SPECIES_SCALE_CONTRACT,
                                               transformed=predictions.log())
        self.assertEqual(float(legacy), 0)
        self.assertGreater(float(corrected), 0)

    def test_dataset_guard_requires_explicit_historical_override(self):
        unsorted = torch.tensor([[.2, 1., .3, .4, .5, .6, .7]])
        with self.assertRaises(ValueError):
            validate_dataset_scale_contract(unsorted, LEGACY_SCALE_CONTRACT)
        with self.assertRaises(ValueError):
            validate_dataset_scale_contract(torch.ones(1, 7), LEGACY_SCALE_CONTRACT, sort_applied=False)
        validate_dataset_scale_contract(unsorted, LEGACY_SCALE_CONTRACT, allow_legacy_on_unsorted=True)
        validate_dataset_scale_contract(torch.ones(1, 7), LEGACY_SCALE_CONTRACT)
        validate_dataset_scale_contract(unsorted, SPECIES_SCALE_CONTRACT, sort_applied=False)

    def test_zero_species_preprocessing_sentinel_is_not_treated_as_a_real_peak(self):
        raw = np.ones((1, 7, 5))
        raw[:, 2, :] = 0
        pkg = preprocess(raw, sort_curves=False)
        self.assertEqual(pkg["reconstruction_max_values"][0, 2], 1e-8)
        with self.assertRaises(ValueError):
            validate_dataset_scale_contract(
                torch.from_numpy(pkg["reconstruction_max_values"]), SPECIES_SCALE_CONTRACT,
                normalized_curves=torch.from_numpy(pkg["data"]),
            )

    def test_centered_loss_is_shift_invariant_and_floor_is_finite(self):
        logits = torch.tensor([[0., -1., 2., 1., -2., .3, -.3]], requires_grad=True)
        targets = torch.tensor([[1e-12, 1., .2, .3, .4, .5, .6]])
        values, centered = decode_scale_predictions(logits, SPECIES_SCALE_CONTRACT, "log")
        first = scale_reconstruction_loss(values, targets, SPECIES_SCALE_CONTRACT, transformed=centered)
        shifted = scale_reconstruction_loss(values, targets, SPECIES_SCALE_CONTRACT, transformed=centered + 4.)
        torch.testing.assert_close(first, shifted)
        self.assertTrue(bool(torch.isfinite(first)))
        first.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_legacy_trainer_loss_matches_original_and_corrected_path_scores_all_species(self):
        import train_cvae
        x = torch.zeros(1, 7, 5)
        z = torch.zeros(1, 4)
        targets = torch.tensor([[.2, 1., .3, .4, .5, .6, .7]])
        predictions = targets.clone()
        predictions[0, 0] = 1.
        legacy = train_cvae.vae_loss(x, x, z, z, predictions, targets, .01, .5)
        self.assertEqual(float(legacy[0]), 0.)
        corrected = train_cvae.vae_loss(
            x, x, z, z, predictions, targets, .01, .5,
            scale_contract=SPECIES_SCALE_CONTRACT, max_vals_pred_transformed=predictions.log(),
        )
        self.assertGreater(float(corrected[3]), 0.)
        torch.testing.assert_close(corrected[0], .5 * corrected[3])

    def test_no_sort_preprocessing_round_trip_for_every_dominant_index(self):
        raw = np.ones((7, 7, 5), dtype=np.float64)
        for dominant in range(7):
            raw[dominant, dominant] *= 5.0
        pkg = preprocess(raw, sort_curves=False)
        recovered = (pkg["data"] * pkg["reconstruction_max_values"][:, :, None]
                     * pkg["family_max_values"][:, None, None])
        np.testing.assert_allclose(recovered, raw)
        np.testing.assert_array_equal(pkg["reconstruction_max_values"].argmax(axis=1), np.arange(7))
        validate_scale_targets(torch.from_numpy(pkg["reconstruction_max_values"]), SPECIES_SCALE_CONTRACT)

    def test_all_seven_targets_are_selected_only_for_corrected_contract(self):
        values = torch.arange(7.0).unsqueeze(0)
        self.assertEqual(values[:, scale_target_slice(SPECIES_SCALE_CONTRACT)].shape[1], 7)
        self.assertEqual(values[:, scale_target_slice(LEGACY_SCALE_CONTRACT)].shape[1], 6)

    def test_corrected_targets_reject_nonfinite_nonpositive_and_unnormalized_rows(self):
        for values in [torch.full((1, 7), 0.5), torch.zeros(1, 7), torch.tensor([[1., 1., 1., float("nan"), 1., 1., 1.]])]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_scale_targets(values, SPECIES_SCALE_CONTRACT)

    def test_selected_models_forward_backward_decode_and_generate_on_cpu(self):
        for kind in [LSTM_VAE, HeteroscedasticLSTMVAE, TransformerVAE]:
            with self.subTest(model=kind.__name__):
                model = self.model(kind)
                x = torch.full((2, 7, 5), 0.5)
                targets = torch.tensor([[.2, .3, .4, 1., .5, .6, .7], [1., .2, .3, .4, .5, .6, .7]])
                out = model(x, targets, teacher_forcing_ratio=0)
                self.assertEqual(model.max_value_dim, 7)
                self.assertEqual(out[3].shape, (2, 7))
                (out[0].square().mean() + (out[3] - targets).square().mean()).backward()
                self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
                model.eval()
                for _, scales in [model.decode(torch.zeros(2, 4)), model.generate(2, torch.device("cpu"))]:
                    self.assertEqual(scales.shape, (2, 7))
                    torch.testing.assert_close(scales.amax(dim=1), torch.ones(2))
                    self.assertEqual(scales.device.type, "cpu")

    def test_legacy_checkpoint_cannot_be_loaded_as_corrected_even_with_strict_false(self):
        legacy = self.model(scale_contract=LEGACY_SCALE_CONTRACT)
        corrected = self.model()
        self.assertEqual(legacy.max_value_dim, 6)
        with self.assertRaises(RuntimeError):
            corrected.load_state_dict(legacy.state_dict(), strict=False)
        with self.assertRaises(ValueError):
            unpack_checkpoint(legacy.state_dict())
        config, state = unpack_checkpoint(legacy.state_dict(), legacy_config={**self.config, "scale_contract": LEGACY_SCALE_CONTRACT})
        self.assertEqual(config["scale_contract"], LEGACY_SCALE_CONTRACT)
        self.assertEqual(set(state), set(legacy.state_dict()))

    def test_versioned_checkpoint_round_trip_and_architecture_validation(self):
        model = self.model()
        payload = make_checkpoint(model, training_metadata={"seed": 42})
        config, state = unpack_checkpoint(payload, expected_architecture="LSTM_VAE")
        self.assertEqual(config["scale_contract"], SPECIES_SCALE_CONTRACT)
        clone = self.model(**config)
        clone.load_state_dict(state)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.pth"
            torch.save(payload, path)
            restored = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(restored["training_metadata"]["seed"], 42)
        with self.assertRaises(ValueError):
            unpack_checkpoint(payload, expected_architecture="TransformerVAE")
        with self.assertRaises(ValueError):
            unpack_checkpoint({**payload, "checkpoint_format_version": 999})
        with self.assertRaises(ValueError):
            unpack_checkpoint({**payload, "checkpoint_format_version": 2.0})
        with self.assertRaises(ValueError):
            unpack_checkpoint({**payload, "model_config": {**config, "scale_contract": LEGACY_SCALE_CONTRACT}})

    def test_atomic_checkpoint_writes_never_destroy_an_existing_snapshot(self):
        payload = make_checkpoint(self.model(), training_metadata={"seed": 42})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested/fixture.pth"
            write_checkpoint_payload(path, payload)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_checkpoint_payload(path, payload)
            self.assertEqual(path.read_bytes(), original)
            with mock.patch("src.utils.checkpoints.torch.save", side_effect=RuntimeError("fixture serialization failure")):
                with self.assertRaises(RuntimeError):
                    write_checkpoint_payload(path, payload, overwrite=True)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob(".checkpoint-*.tmp")), [])

    def test_bad_corrected_metadata_is_rejected_before_training_or_device_transfer(self):
        import train_cvae
        model = self.model()
        model.config["invalid_metadata"] = object()
        with mock.patch.object(train_cvae.wandb, "init") as logging, self.assertRaises(TypeError):
            train_cvae.train(model)
        logging.assert_not_called()

    def test_evaluation_adapters_load_small_versioned_models_without_gpu(self):
        # Prevent even availability probing during imports of historical modules.
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            from analysis import evaluate_all_models as evaluation
        with tempfile.TemporaryDirectory() as temporary:
            for kind, adapter in [
                (LSTM_VAE, lambda p: evaluation.adapter_cvae(p, True, torch.device("cpu"))),
                (HeteroscedasticLSTMVAE, lambda p: evaluation.adapter_cvae_heteroscedastic(p, torch.device("cpu"))),
                (TransformerVAE, lambda p: evaluation.adapter_transformer(p, torch.device("cpu"))),
            ]:
                with self.subTest(model=kind.__name__), contextlib.redirect_stdout(io.StringIO()):
                    original = self.model(kind).eval()
                    path = Path(temporary) / f"{kind.__name__}.pth"
                    torch.save(make_checkpoint(original), path)
                    restored = adapter(path)
                    self.assertEqual(restored.scale_contract, SPECIES_SCALE_CONTRACT)
                    for key, value in original.state_dict().items():
                        torch.testing.assert_close(value, restored.state_dict()[key])

    def test_unsupported_stochastic_corrected_contract_fails_loudly(self):
        from src.models.cvae_stochastic import StochasticLSTMVAE
        with self.assertRaises(ValueError):
            self.model(StochasticLSTMVAE)

    def test_pivot_cli_invalid_contract_fails_before_seeding_or_training(self):
        import train_pivot
        for extra in [[], ["--scale-contract", SPECIES_SCALE_CONTRACT, "--model", "kan-vae"],
                      ["--scale-contract", SPECIES_SCALE_CONTRACT, "--scale-log-floor", "nan"]]:
            args = ["train_pivot.py", "--seed", "42", "--name", "cpu_validation_not_a_run", *extra]
            with self.subTest(arguments=extra), mock.patch("sys.argv", args), \
                    mock.patch.object(train_pivot, "set_seed") as seed, \
                    mock.patch.object(train_pivot.train_cvae, "train") as train, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                train_pivot.main()
            self.assertEqual(error.exception.code, 2)
            seed.assert_not_called()
            train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
