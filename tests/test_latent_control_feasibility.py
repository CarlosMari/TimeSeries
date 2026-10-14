"""Synthetic decoder fixtures for fair-budget latent search, no checkpoints/GPU."""

from types import SimpleNamespace
import unittest

try:
    import numpy as np
    import scipy
    import sklearn
except ImportError:
    np = None

if np is not None:
    from analysis.latent_control_feasibility import (
        distances, eligible, linear_probe, local_latents, prior_latents,
        property_matrix, run_pilot,
    )
    from src.utils.trajectory_controls import summarize_trajectory_controls


class SyntheticDecoder:
    def __init__(self, constant=False):
        self.model = SimpleNamespace(latent_dim=2)
        self.n_decoded = 0
        self.constant = constant

    def __call__(self, z):
        self.n_decoded += len(z)
        sigmoid = 1 / (1 + np.exp(-z))
        terminal = .2 + .4 * sigmoid[:, 0]
        overshoot = .3 + 1.5 * sigmoid[:, 1]
        if self.constant:
            terminal[:] = .4
            overshoot[:] = .7
        curve = np.repeat(terminal[:, None], 32, axis=1)
        curve[:, :8] = np.linspace(.05, 1., 8)[None, :] * terminal[:, None]
        curve[:, 8:16] = terminal[:, None] * (1 + overshoot[:, None] * np.sin(np.linspace(0, np.pi, 8))[None, :])
        return curve[:, None, :]


@unittest.skipIf(np is None, "NumPy/SciPy/scikit-learn required")
class LatentControlFeasibilityTests(unittest.TestCase):
    def test_prior_and_local_proposals_respect_same_ball_without_projection(self):
        rng = np.random.default_rng(42)
        z = prior_latents(rng, 50, 2, 2.)
        self.assertTrue(np.all(np.linalg.norm(z, axis=1) <= 2.))
        local, rejected = local_latents(rng, z[:4], 50, 2., .25)
        self.assertTrue(np.all(np.linalg.norm(local, axis=1) <= 2.))
        self.assertGreaterEqual(rejected, 0)

    def test_invalid_candidates_have_infinite_distance_not_zero_error(self):
        score = distances(np.array([[1., 2.], [np.nan, 2.], [1., 2.]]), [1., 2.], [1., 1.],
                          np.array([True, True, False]))
        np.testing.assert_array_equal(score, [0., np.inf, np.inf])

    def test_probe_evaluates_known_linear_controls_on_independent_codes(self):
        rng = np.random.default_rng(45)
        z, probe = rng.normal(size=(100, 3)), rng.normal(size=(50, 3))
        weights = np.array([[2., 0.], [0., -3.], [.2, .1]])
        result = linear_probe(z, z @ weights, probe, probe @ weights)
        self.assertTrue(all(v > .99 for v in result["heldout_r2"].values()))
        self.assertEqual(result["single_axis"]["log1p_overshoot_ratio"]["selected_axis"], 0)
        self.assertEqual(result["single_axis"]["relative_terminal_level"]["selected_axis"], 1)

    def test_pilot_retains_all_targets_and_equal_online_budgets(self):
        decoder = SyntheticDecoder()
        report, arrays = run_pilot(decoder, bank_size=128, probe_size=64, budget=8, repetitions=2)
        self.assertEqual(report["status"], "exploratory")
        self.assertEqual(report["distinct_joint_targets"], 9)
        self.assertEqual(len(report["targets"]), 18)
        self.assertEqual(report["online_decodes_each_method"], 9 * 8 * 2)
        self.assertEqual(decoder.n_decoded, 128 + 64 + 16 + 2 * 9 * 8 * 2)
        self.assertEqual(sum(report["local_vs_random_paired_trials"].values()), 18)
        for row in report["targets"]:
            for method in ["random_plus_bank", "local_search_plus_bank"]:
                record = row[method]
                self.assertEqual(record["online_decodes"], 8)
                self.assertLessEqual(record["best_error"], record["lookup_only_error"])
                name = f"{method}_{row['target_id']}_{row['repetition']}_selected_x"
                controls = summarize_trajectory_controls(arrays[name][None])
                self.assertTrue(eligible(controls)[0])
                self.assertAlmostEqual(controls["overshoot_ratio"][0], record["achieved_overshoot_ratio"])

    def test_constant_behavior_is_a_negative_result_not_artificial_control(self):
        report, arrays = run_pilot(SyntheticDecoder(constant=True), bank_size=128, probe_size=64, budget=8)
        self.assertEqual(report["status"], "insufficient_control_variation")
        self.assertNotIn("targets", report)


if __name__ == "__main__":
    unittest.main()
