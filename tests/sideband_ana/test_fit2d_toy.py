"""Small zfit four-component closure check without the expensive SymbolFit stage."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import erf

from datafactory.stat.sideband_ana import FitResult1D, MassRegions1D, calculate_transfer_factors_2d, fit_mass_plane_2d


class FourComponentFitTest(unittest.TestCase):
    """Fit a deterministic Poisson toy and recover flat-background transfers."""

    def test_four_component_plane(self):
        edges = np.linspace(1.08, 1.175, 41)
        mean, sigma_narrow, delta_sigma, fraction = 1.116, 0.0022, 0.0040, 0.65
        sqrt_two = np.sqrt(2.0)
        narrow = 0.5 * (erf((edges[1:] - mean) / (sqrt_two * sigma_narrow)) - erf((edges[:-1] - mean) / (sqrt_two * sigma_narrow)))
        wide_sigma = sigma_narrow + delta_sigma
        wide = 0.5 * (erf((edges[1:] - mean) / (sqrt_two * wide_sigma)) - erf((edges[:-1] - mean) / (sqrt_two * wide_sigma)))
        signal = fraction * narrow / narrow.sum() + (1.0 - fraction) * wide / wide.sum()
        background = np.full(40, 1.0 / 40.0)
        yields = np.asarray([2500.0, 3500.0, 3000.0, 6000.0])
        expectation = (
            yields[0] * np.outer(signal, signal)
            + yields[1] * np.outer(background, signal)
            + yields[2] * np.outer(signal, background)
            + yields[3] * np.outer(background, background)
        )
        observed = np.random.RandomState(19).poisson(expectation)[None, :, :]
        seed = FitResult1D(
            mass_edges=edges,
            observed_counts=observed.sum(axis=(0, 2)),
            observed_variances=observed.sum(axis=(0, 2)),
            fit_range=(edges[0], edges[-1]),
            background_fit_range=(edges[0], edges[-1]),
            background_profiled=False,
            parameter_names=("narrow_yield", "wide_yield", "mean", "sigma_narrow", "delta_sigma", "background:a0"),
            parameter_values=np.asarray([1500.0, 800.0, mean, sigma_narrow, delta_sigma, 1.0]),
            parameter_covariance=np.eye(6) * 1.0e-8,
            background_parameter_indices=(5,),
            peak_mean=mean,
            peak_mean_variance=1.0e-8,
            background_formula="1.0",
            background_parameterized_formula="a0",
            symbolfit_initial_values={"a0": 1.0},
            model_counts=observed.sum(axis=(0, 2)),
            background_counts=np.ones(40),
            dense_mass=np.linspace(edges[0], edges[-1], 100),
            dense_model=np.ones(100),
            dense_background=np.ones(100),
            chi2=0.0,
            ndf=34,
            converged=True,
            background_model="constant",
        )
        fit = fit_mass_plane_2d(edges, edges, observed, x_seed=seed, fit_nbins=20, random_seed=19)
        regions = MassRegions1D(signal=(mean - 0.010, mean + 0.010), sideband_high=(mean + 0.0343, mean + 0.0543))
        transfer = calculate_transfer_factors_2d(fit, regions)
        self.assertTrue(fit.converged)
        np.testing.assert_allclose([transfer.w_H, transfer.w_V], [1.0, 1.0], rtol=0.12)
        np.testing.assert_allclose(transfer.w_C, -transfer.w_H * transfer.w_V, rtol=1.0e-10)


if __name__ == "__main__":
    unittest.main()
