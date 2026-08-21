"""Deterministic checks for the sideband-region and subtraction mathematics."""

from __future__ import annotations

import unittest

import numpy as np

from datafactory.stat.sideband_ana import (
    FitResult1D,
    MassRegions1D,
    TransferFactor1D,
    calculate_transfer_factor_1d,
    calculate_transfer_factors_2d,
    subtract_sideband_1d,
    subtract_sideband_2d,
)


class FlatPlaneFit:
    """Minimal physical fit product with flat background region integrals."""

    x_fit_range = (0.0, 6.0)
    y_fit_range = (0.0, 6.0)
    parameter_values = np.asarray([0.0])
    parameter_covariance = np.zeros((1, 1))

    def axis_region_integrals(self, axis, regions, parameter_values=None):
        signal = {"S": 0.90, "L": 0.02, "H": 0.02}
        background = {"S": 2.0, "L": 1.0, "H": 1.0}
        labels = ["S"] + (["L"] if regions.sideband_low is not None else []) + (["H"] if regions.sideband_high is not None else [])
        return {
            "signal": {label: signal[label] for label in labels},
            "background": {label: background[label] for label in labels},
        }


class SidebandMathTest(unittest.TestCase):
    """Verify flat-background limits and the signed corner convention."""

    def setUp(self):
        self.double_regions = MassRegions1D(signal=(2.0, 4.0), sideband_low=(0.0, 1.0), sideband_high=(5.0, 6.0))

    def test_region_validation_rejects_overlap(self):
        with self.assertRaises(ValueError):
            MassRegions1D(signal=(2.0, 4.0), sideband_low=(1.0, 2.5))

    def test_flat_one_dimensional_transfer_is_unity(self):
        fit = FitResult1D(
            mass_edges=np.linspace(0.0, 6.0, 7),
            observed_counts=np.ones(6),
            observed_variances=np.ones(6),
            fit_range=(0.0, 6.0),
            background_fit_range=(0.0, 6.0),
            background_profiled=False,
            parameter_names=("background:a0",),
            parameter_values=np.asarray([1.0]),
            parameter_covariance=np.zeros((1, 1)),
            background_parameter_indices=(0,),
            peak_mean=3.0,
            peak_mean_variance=0.0,
            background_formula="1.0",
            background_parameterized_formula="a0",
            symbolfit_initial_values={"a0": 1.0},
            model_counts=np.ones(6),
            background_counts=np.ones(6),
            dense_mass=np.linspace(0.0, 6.0, 20),
            dense_model=np.ones(20),
            dense_background=np.ones(20),
            chi2=0.0,
            ndf=5,
            converged=True,
            background_model="constant",
        )
        result = calculate_transfer_factor_1d(fit, self.double_regions)
        self.assertAlmostEqual(result.r_combined, 1.0, places=12)
        self.assertAlmostEqual(result.variance_r_combined, 0.0, places=12)

    def test_flat_two_dimensional_transfer_has_negative_corner(self):
        result = calculate_transfer_factors_2d(FlatPlaneFit(), self.double_regions)
        np.testing.assert_allclose([result.w_H, result.w_V, result.w_C], [1.0, 1.0, -1.0], atol=1.0e-12)
        self.assertAlmostEqual(result.factorization_closure, 0.0, places=12)
        self.assertEqual(set(result.atomic_region_integrals["BxBy"]), {"SS", "SL", "SH", "LS", "LL", "LH", "HS", "HL", "HH"})

    def test_one_dimensional_subtraction_recovers_injected_signal(self):
        transfer = TransferFactor1D(self.double_regions, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0, np.zeros(1))
        result = subtract_sideband_1d(
            np.asarray([15.0, 26.0]),
            np.asarray([15.0, 26.0]),
            low_sideband_counts=np.asarray([2.0, 3.0]),
            low_sideband_variances=np.asarray([2.0, 3.0]),
            high_sideband_counts=np.asarray([3.0, 3.0]),
            high_sideband_variances=np.asarray([3.0, 3.0]),
            transfer=transfer,
        )
        np.testing.assert_allclose(result.subtracted_signal, [10.0, 20.0])

    def test_two_dimensional_inclusion_exclusion_recovers_injected_signal(self):
        transfer = calculate_transfer_factors_2d(FlatPlaneFit(), self.double_regions)
        region_counts = {}
        region_variances = {}
        for x_label in ("S", "L", "H"):
            for y_label in ("S", "L", "H"):
                mass_region_area = (2.0 if x_label == "S" else 1.0) * (2.0 if y_label == "S" else 1.0)
                region_counts[(x_label, y_label)] = mass_region_area * np.asarray([1.0, 2.0])
                region_variances[(x_label, y_label)] = mass_region_area * np.asarray([1.0, 2.0])
        region_counts[("S", "S")] = np.asarray([14.0, 28.0])
        result = subtract_sideband_2d(region_counts, region_variances, transfer)
        np.testing.assert_allclose(result.estimated_background, [4.0, 8.0])
        np.testing.assert_allclose(result.subtracted_signal, [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()
