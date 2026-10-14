"""Hand-computed relative-scale scoring checks; NumPy only, never GPU."""

import unittest

try:
    import numpy as np
    from src.utils.scale_metrics import relative_scale_metrics
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy required; use TimeSeries/bin/python")
class RelativeScaleMetricTests(unittest.TestCase):
    def test_species_zero_error_is_visible_only_in_all_species_metric(self):
        targets = np.array([[.2, 1., .3, .4, .5, .6, .7]])
        predicted = targets.copy()
        predicted[0, 0] = 1.
        legacy = relative_scale_metrics(targets, predicted, "legacy_sorted_v1")
        corrected = relative_scale_metrics(targets, predicted, "species_v2")
        self.assertEqual(legacy["max_val_MSE_pooled"], 0.)
        self.assertAlmostEqual(corrected["max_val_MSE_pooled"], .8**2 / 7)
        self.assertEqual(corrected["max_val_species_indices"], list(range(7)))
        self.assertEqual(legacy["max_val_species_indices"], list(range(1, 7)))
        self.assertEqual(corrected["scale_dominant_species_accuracy"], 0.)

    def test_perfect_scales_have_zero_error_and_tied_true_dominance_is_allowed(self):
        targets = np.array([[1., 1., .3], [.2, 1., .4]])
        result = relative_scale_metrics(targets, targets.copy(), "species_v2")
        self.assertEqual(result["scale_centered_log_ratio_MSE"], 0.)
        self.assertEqual(result["scale_dominant_species_accuracy"], 1.)
        self.assertEqual(result["max_val_R2_pooled"], 1.)

    def test_nonfinite_values_and_unknown_contracts_are_not_silently_scored(self):
        for contract in ["unknown", "species_v2"]:
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                relative_scale_metrics([[1., .5]], [[float("nan"), .4]], contract)

    def test_small_targets_are_counted_and_log_error_stays_finite(self):
        result = relative_scale_metrics([[1., 1e-12]], [[1., 1e-11]], "species_v2", log_floor=1e-8)
        self.assertTrue(np.isfinite(result["scale_centered_log_ratio_MSE"]))
        self.assertEqual(result["scale_target_at_or_below_log_floor_count"], 1)

    def test_floor_equality_is_counted_independently_of_dtype(self):
        for dtype in [np.float32, np.float64]:
            targets = np.array([[1., 1e-8]], dtype=dtype)
            result = relative_scale_metrics(targets, targets, "species_v2")
            self.assertEqual(result["scale_target_at_or_below_log_floor_count"], 1)

    def test_constant_species_r2_is_explicitly_undefined(self):
        targets = np.array([[1., .2], [1., .3]])
        result = relative_scale_metrics(targets, targets, "species_v2")
        self.assertIsNone(result["max_val_R2_per_species"]["0"])
        self.assertEqual(result["max_val_R2_per_species"]["1"], 1.)


if __name__ == "__main__":
    unittest.main()
