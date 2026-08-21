"""Run the real DELPHI Lambda-Lambdabar sideband fixture end to end.

Example
-------
``conda activate root6.34`` and run
``python tests/sideband_ana/run_fixture_analysis.py`` from the repository root.
The expensive SymbolFit stage is deterministic for the supplied seed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from datafactory.stat.sideband_ana import (
    MassRegions1D,
    calculate_transfer_factor_1d,
    calculate_transfer_factors_2d,
    fit_mass_plane_2d,
    fit_mass_spectrum_1d,
    subtract_sideband_2d,
    write_fit_report_1d,
    write_fit_report_2d,
    write_transfer_summary,
)


parser = argparse.ArgumentParser(description=__doc__)
repository = Path(__file__).resolve().parents[2]
parser.add_argument("--fixture", type=Path, default=repository / "tests/data/sideband_ana/sideband_ll_data_cat0_llbar.npz")
parser.add_argument("--metadata", type=Path, default=repository / "tests/data/sideband_ana/sideband_ll_data_cat0_llbar.json")
parser.add_argument("--output-dir", type=Path, default=repository / "tests/results/sideband_ana/ll_data_cat0_llbar")
parser.add_argument("--symbolfit-iterations", type=int, default=100)
parser.add_argument("--profile-background", action="store_true", help="Profile SymbolFit parameters in zfit; default reproduces the 09 fixed-background path")
arguments = parser.parse_args()

fixture = np.load(arguments.fixture, allow_pickle=False)
metadata = json.loads(arguments.metadata.read_text(encoding="utf-8"))
arguments.output_dir.mkdir(parents=True, exist_ok=True)
peak = float(metadata["regions"]["peak_mean_gev"])
half_width = float(metadata["regions"]["signal_window_gev"])
sideband_offset = float(metadata["regions"]["sideband_offset_gev"])
regions = MassRegions1D(
    signal=(peak - half_width, peak + half_width),
    sideband_high=(peak + sideband_offset - half_width, peak + sideband_offset + half_width),
)

fit_1d = fit_mass_spectrum_1d(
    fixture["mass_fit_edges"],
    fixture["mass_fit_counts"],
    fixture["mass_fit_variances"],
    fit_range=(1.095, 1.145),
    regions=regions,
    random_seed=20260803,
    profile_background=arguments.profile_background,
    symbolfit_output_dir=arguments.output_dir / "symbolfit_1d",
    symbolfit_niterations=arguments.symbolfit_iterations,
)
transfer_1d = calculate_transfer_factor_1d(fit_1d, regions)
fit_2d = fit_mass_plane_2d(
    fixture["mass_plane_x_edges"],
    fixture["mass_plane_y_edges"],
    fixture["mass_plane_counts"][None, :, :],
    x_seed=fit_1d,
    fit_nbins=20,
    random_seed=20260813,
)
transfer_2d = calculate_transfer_factors_2d(fit_2d, regions)

region_order = metadata["regions"]["order"]
region_keys = {"SS": ("S", "S"), "BS": ("H", "S"), "SB": ("S", "H"), "BB": ("H", "H")}
delta_phi_counts = {region_keys[name]: fixture["delta_phi_thrust_counts"][region_order.index(name)] for name in region_order}
delta_phi_variances = {region_keys[name]: fixture["delta_phi_thrust_variances"][region_order.index(name)] for name in region_order}
subtraction = subtract_sideband_2d(delta_phi_counts, delta_phi_variances, transfer_2d)

sample_metadata = {
    "sample": "data",
    "sample_label": r"DELPHI $\Lambda\bar{\Lambda}$ data, all-event, $\Lambda\bar{\Lambda}$ channel",
    "sqrt_s_gev": 91.25,
    "years": metadata["years"],
    "selection": metadata["selection"],
    "cut_chain": metadata["cut_chain"],
}
write_fit_report_1d(fit_1d, transfer_1d, sample_metadata=sample_metadata, output_dir=arguments.output_dir, stem="ll_data_cat0_llbar")
write_fit_report_2d(fit_2d, transfer_2d, sample_metadata=sample_metadata, output_dir=arguments.output_dir, stem="ll_data_cat0_llbar")
write_transfer_summary({"ll_data_1d": transfer_1d, "ll_data_cat0_llbar_2d": transfer_2d}, analysis_metadata=sample_metadata, output_dir=arguments.output_dir)

reference = metadata["reference_results"]
reference_values = np.asarray([
    reference["one_dimensional"]["r"],
    reference["two_dimensional"]["w_H"],
    reference["two_dimensional"]["w_V"],
    reference["two_dimensional"]["w_C"],
])
reference_errors = np.asarray([
    reference["one_dimensional"]["sigma_r"],
    *np.sqrt(np.diag(np.asarray(reference["two_dimensional"]["weight_covariance"], dtype=float))),
])
fitted_values = np.asarray([transfer_1d.r_combined, transfer_2d.w_H, transfer_2d.w_V, transfer_2d.w_C])
tolerances = np.maximum(3.0 * reference_errors, 0.01 * np.abs(reference_values))
differences = fitted_values - reference_values
passed = np.abs(differences) <= tolerances
names = ("r", "w_H", "w_V", "w_C")
comparison = {
    "schema_version": "sideband_fixture_comparison_v1",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "produced_by": str(Path(__file__).resolve()),
    "fixture": str(arguments.fixture.resolve()),
    "profile_background": arguments.profile_background,
    "symbolfit_iterations": arguments.symbolfit_iterations,
    "comparison_rule": "abs(fitted-reference) <= max(3*reference_sigma, 0.01*abs(reference))",
    "period_note": "The fixture plane is the merged source plane and is fitted as one period; the stored 09 reference used four simultaneous periods.",
    "quantities": {
        name: {"fitted": float(value), "reference": float(reference_value), "difference": float(difference), "tolerance": float(tolerance), "pass": bool(ok)}
        for name, value, reference_value, difference, tolerance, ok in zip(names, fitted_values, reference_values, differences, tolerances, passed)
    },
    "overall_pass": bool(np.all(passed)),
}
(arguments.output_dir / "transfer_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
markdown = [
    "# Transfer-factor regression comparison",
    "",
    f"Overall: **{'PASS' if comparison['overall_pass'] else 'FAIL'}**",
    "",
    "| Quantity | New fit | 09 reference | Difference | Tolerance | Result |",
    "|---|---:|---:|---:|---:|---|",
]
for name in names:
    item = comparison["quantities"][name]
    markdown.append(f"| ${name}$ | {item['fitted']:.8g} | {item['reference']:.8g} | {item['difference']:.3g} | {item['tolerance']:.3g} | {'PASS' if item['pass'] else 'FAIL'} |")
markdown.extend(("", comparison["period_note"], ""))
(arguments.output_dir / "transfer_comparison.md").write_text("\n".join(markdown), encoding="utf-8")
np.savez(
    arguments.output_dir / "fit_and_transfer_results.npz",
    mass_edges=fit_1d.mass_edges,
    mass_observed=fit_1d.observed_counts,
    mass_model=fit_1d.model_counts,
    mass_background=fit_1d.background_counts,
    mass_parameter_values=fit_1d.parameter_values,
    mass_parameter_covariance=fit_1d.parameter_covariance,
    plane_x_edges=fit_2d.x_edges,
    plane_y_edges=fit_2d.y_edges,
    plane_observed=fit_2d.observed_counts_by_period,
    plane_model=fit_2d.model_counts_by_period,
    plane_parameter_values=fit_2d.parameter_values,
    plane_parameter_covariance=fit_2d.parameter_covariance,
    transfer_values=fitted_values,
    transfer_covariance=transfer_2d.weight_covariance,
    delta_phi_edges=fixture["delta_phi_thrust_edges"],
    delta_phi_observed_ss=subtraction.observed_signal_region,
    delta_phi_estimated_background=subtraction.estimated_background,
    delta_phi_subtracted=subtraction.subtracted_signal,
    delta_phi_subtracted_variance=subtraction.signal_variance,
)
print(json.dumps(comparison, indent=2))
