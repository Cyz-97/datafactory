"""zfit profiling check using a fixed SymbolFit seed to keep the test fast."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from datafactory.stat.sideband_ana import MassRegions1D, calculate_transfer_factor_1d, fit_mass_spectrum_1d
from datafactory.stat.sideband_ana.fit import SymbolFitBackgroundSeed


class ProfiledBackgroundFitTest(unittest.TestCase):
    """Exercise the nominal profiled-background branch independently of Julia."""

    def test_linear_symbolfit_background_is_profiled_by_zfit(self):
        edges = np.linspace(1.08, 1.175, 101)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        mean, sigma_narrow, sigma_wide = 1.116, 0.0020, 0.0055
        background_density = 1.8e5 - 4.0e4 * centers
        signal_density = (
            3500.0 * np.exp(-0.5 * ((centers - mean) / sigma_narrow) ** 2) / (np.sqrt(2.0 * np.pi) * sigma_narrow)
            + 1500.0 * np.exp(-0.5 * ((centers - mean) / sigma_wide) ** 2) / (np.sqrt(2.0 * np.pi) * sigma_wide)
        )
        expected = (background_density + signal_density) * widths
        counts = np.random.RandomState(31).poisson(expected).astype(float)
        regions = MassRegions1D(signal=(mean - 0.008, mean + 0.008), sideband_high=(1.150, 1.166))
        seed = SymbolFitBackgroundSeed(
            parameterized_formula="a0 + a1*x0",
            fitted_formula="175000 - 35000*x0",
            parameter_names=("a0", "a1"),
            parameter_values=np.asarray([1.75e5, -3.5e4]),
            parameter_covariance=np.diag([2.0e6, 2.0e6]),
            training_chi2=100.0,
            training_ndf=80,
            selection_score=104.0,
        )
        with patch("datafactory.stat.sideband_ana.fit.select_symbolfit_background", return_value=seed):
            fit = fit_mass_spectrum_1d(
                edges,
                counts,
                counts,
                fit_range=(1.095, 1.145),
                regions=regions,
                profile_background=True,
                random_seed=31,
            )
        transfer = calculate_transfer_factor_1d(fit, regions)
        self.assertTrue(fit.converged)
        self.assertTrue(fit.background_profiled)
        self.assertGreater(transfer.r_combined, 0.0)
        self.assertGreaterEqual(transfer.variance_r_combined, 0.0)


if __name__ == "__main__":
    unittest.main()
