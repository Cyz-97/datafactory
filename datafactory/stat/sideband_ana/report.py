"""Read-only diagnostic reports for sideband fits and transfer coefficients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .fit import FitResult1D, FitResult2D
from .transfer import TransferFactor1D, TransferFactors2D


@dataclass(frozen=True)
class ReportArtifacts:
    """Files and self-contained captions created by one report call."""

    files: tuple[Path, ...]
    captions: dict[str, str]


def _apply_datafactory_style():
    """Load the exact style used by :mod:`datafactory.plot` without importing ROOT.

    ``datafactory.plot`` imports PyROOT at module load time; keeping report
    rendering ROOT-free prevents ROOT and PySR/Julia LLVM runtimes from being
    loaded into the same fit worker process.
    """
    import matplotlib.pyplot as plt
    style_path = Path(__file__).resolve().parents[2] / "style.mplstyle"
    if not style_path.is_file():
        raise FileNotFoundError(f"DataFactory plotting style not found: {style_path}")
    plt.style.use(style_path)


def _add_delphi_header(ax, metadata: dict, *, has_data: bool = True):
    """Place the two-line DELPHI identity block and return its artists."""
    state = "Open Data" if has_data else "Simulation"
    delphi = ax.text(0.02, 0.97, "DELPHI", transform=ax.transAxes, ha="left", va="top", family="sans-serif", weight="bold", fontsize=11)
    state_artist = ax.text(0.175, 0.97, state, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    parts = []
    if metadata.get("sqrt_s_gev") is not None:
        parts.append(rf"\sqrt{{s}}={float(metadata['sqrt_s_gev']):g}\,\mathrm{{GeV}}")
    if metadata.get("years"):
        years = metadata["years"]
        if isinstance(years, str):
            year_text = years
        elif len(years) == 1:
            year_text = str(years[0])
        else:
            year_text = rf"{years[0]}\text{{--}}{years[-1]}"
        parts.append(rf"\mathrm{{{year_text}}}")
    if metadata.get("lumi_invpb") is not None:
        parts.append(rf"\mathcal{{L}}={float(metadata['lumi_invpb']):g}\,\mathrm{{pb}}^{{-1}}")
    metadata_artist = ax.text(0.02, 0.90, "$" + r",\ ".join(parts) + "$" if parts else "", transform=ax.transAxes, ha="left", va="top", fontsize=7)
    return delphi, state_artist, metadata_artist


def _save_figure(fig, output_stem: Path, caption: str, source: str):
    """Save matching PDF/PNG diagnostics and embed an auditable PDF Subject."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path, png_path = output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300, transparent=True, metadata={"Subject": f"{caption} Source: {source}", "Creator": source})
    fig.savefig(png_path, bbox_inches="tight", dpi=300, transparent=True)
    return pdf_path, png_path


def _parameter_panel(names, values, covariance, chi2, ndf, *, maximum_parameters=10):
    """Format fitted values with covariance-derived one-sigma uncertainties."""
    errors = np.sqrt(np.maximum(np.diag(np.asarray(covariance, dtype=float)), 0.0))
    lines = [rf"$\chi^2/\mathrm{{ndf}}={chi2:.1f}/{ndf}$"]
    for name, value, error in zip(names[:maximum_parameters], values[:maximum_parameters], errors[:maximum_parameters]):
        latex_name = name.replace("background:", "b_").replace("_", r"\_")
        lines.append(rf"${latex_name}={value:.4g}\pm{error:.2g}$")
    if len(names) > maximum_parameters:
        lines.append(rf"$\text{{{len(names) - maximum_parameters} additional yield parameters in JSON}}$")
    return "\n".join(lines)


def _fit_residual(observed, model):
    """Calculate data/model minus one and a symmetric, bounded display range."""
    residual = np.divide(observed, model, out=np.full_like(observed, np.nan, dtype=float), where=model > 0.0) - 1.0
    finite = np.abs(residual[np.isfinite(residual)])
    limit = min(5.0, max(0.5, 1.2 * float(np.percentile(finite, 95.0)) if finite.size else 0.5))
    return residual, (-limit, limit)


def write_fit_report_1d(
    fit_result: FitResult1D,
    transfer_result: TransferFactor1D,
    *,
    sample_metadata: dict,
    output_dir,
    stem: str,
) -> ReportArtifacts:
    """Write a full-range 1-D fit, residual, regions, and parameter panel."""
    import matplotlib.pyplot as plt

    _apply_datafactory_style()
    centers = 0.5 * (fit_result.mass_edges[:-1] + fit_result.mass_edges[1:])
    errors = np.sqrt(fit_result.observed_variances)
    residual, residual_range = _fit_residual(fit_result.observed_counts, fit_result.model_counts)
    residual_error = np.divide(errors, fit_result.model_counts, out=np.zeros_like(errors), where=fit_result.model_counts > 0.0)
    fig, (main_ax, residual_ax) = plt.subplots(2, 1, sharex=True, figsize=(6.0, 6.6), gridspec_kw={"height_ratios": (4, 1), "hspace": 0.08})
    for interval in (transfer_result.regions.signal, transfer_result.regions.sideband_low, transfer_result.regions.sideband_high):
        if interval is not None:
            main_ax.axvspan(*interval, color="0.75", alpha=0.4, zorder=0)
            residual_ax.axvspan(*interval, color="0.75", alpha=0.4, zorder=0)
    marker_size, capsize = (1.6, 0.0) if centers.size > 80 else (3.0, 3.0)
    main_ax.errorbar(centers, fit_result.observed_counts, yerr=errors, color="black", marker="o", markersize=marker_size, capsize=capsize, capthick=0.64, elinewidth=0.8, linestyle="", label=r"$\text{Data}$")
    main_ax.plot(fit_result.dense_mass, fit_result.dense_model, color="black", linewidth=1.2, label=r"$\text{Total fit}$")
    main_ax.plot(fit_result.dense_mass, fit_result.dense_background, color=plt.get_cmap("tab10").colors[0], linewidth=1.2, label=r"$\text{Background}$")
    physical_maximum = float(np.nanmax(fit_result.observed_counts + errors))
    main_ax.set_ylim(0.0, max(1.0, physical_maximum / 0.76))
    main_ax.set_ylabel(r"$\text{Candidates / bin}$")
    main_ax.set_box_aspect(3 / 4)
    main_ax.grid(False)
    _add_delphi_header(main_ax, sample_metadata, has_data=sample_metadata.get("sample", "data") == "data")
    main_ax.legend(loc="upper right", fontsize=7, frameon=False)
    panel = _parameter_panel(fit_result.parameter_names, fit_result.parameter_values, fit_result.parameter_covariance, fit_result.chi2, fit_result.ndf)
    main_ax.text(0.02, 0.76, panel, transform=main_ax.transAxes, ha="left", va="top", fontsize=6, linespacing=1.05)
    residual_ax.errorbar(centers, residual, yerr=residual_error, color="black", marker="o", markersize=marker_size, capsize=capsize, capthick=0.64, elinewidth=0.8, linestyle="")
    residual_ax.axhline(0.0, color="black", linewidth=0.8)
    residual_ax.set_ylim(*residual_range)
    residual_ax.set_yticks(np.linspace(residual_range[0], residual_range[1], 5))
    residual_ax.set_ylabel(r"$\frac{\mathrm{Data}}{\mathrm{model}}-1$")
    residual_ax.set_xlabel(r"$m\,[\mathrm{GeV}]$")
    residual_ax.grid(False)
    fig.align_ylabels((main_ax, residual_ax))
    caption = (
        f"Binned mass spectrum for {sample_metadata.get('sample_label', sample_metadata.get('sample', 'the selected sample'))}. "
        "Points show data with Sumw2 uncertainties; the black and blue curves are the final zfit total and SymbolFit-selected background models. "
        f"Grey bands are the manually supplied signal and sideband intervals, and the lower panel is data/model minus one; the fitted background was {'profiled' if fit_result.background_profiled else 'fixed'} in zfit."
    )
    paths = _save_figure(fig, Path(output_dir) / f"{stem}_fit1d", caption, "datafactory.stat.sideband_ana.report.write_fit_report_1d")
    plt.close(fig)
    return ReportArtifacts(files=paths, captions={f"{stem}_fit1d": caption})


def write_fit_report_2d(
    fit_result: FitResult2D,
    transfer_result: TransferFactors2D,
    *,
    sample_metadata: dict,
    output_dir,
    stem: str,
) -> ReportArtifacts:
    """Write the mass plane plus x/y projection fit and residual diagnostics."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    _apply_datafactory_style()
    output_path = Path(output_dir)
    observed_plane = fit_result.observed_counts_by_period.sum(axis=0)
    model_plane = fit_result.model_counts_by_period.sum(axis=0)
    plane_residual, _ = _fit_residual(observed_plane, model_plane)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 6.2))
    labels = (r"$\text{Observed}$", r"$\text{Model}$", r"$\mathrm{Data}/\mathrm{model}-1$")
    values = (observed_plane, model_plane, plane_residual)
    for axis, label, plane in zip(axes, labels, values):
        mesh = axis.pcolormesh(fit_result.x_edges, fit_result.y_edges, plane.T, shading="auto", cmap="viridis" if label != labels[2] else "coolwarm")
        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes("right", size="4%", pad=0.08)
        fig.colorbar(mesh, cax=colorbar_axis)
        axis.set_aspect("equal")
        axis.set_xlabel(r"$m_x\,[\mathrm{GeV}]$")
        axis.set_ylabel(r"$m_y\,[\mathrm{GeV}]$")
        axis.text(0.97, 0.03, label, transform=axis.transAxes, ha="right", va="bottom", fontsize=8, color="black", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
        axis.grid(False)
        x_intervals = {"S": transfer_result.x_regions.signal, "L": transfer_result.x_regions.sideband_low, "H": transfer_result.x_regions.sideband_high}
        y_intervals = {"S": transfer_result.y_regions.signal, "L": transfer_result.y_regions.sideband_low, "H": transfer_result.y_regions.sideband_high}
        for x_label, x_interval in x_intervals.items():
            for y_label, y_interval in y_intervals.items():
                if x_interval is None or y_interval is None:
                    continue
                color = plt.get_cmap("tab10").colors[3] if (x_label, y_label) == ("S", "S") else plt.get_cmap("tab10").colors[0]
                axis.add_patch(Rectangle((x_interval[0], y_interval[0]), x_interval[1] - x_interval[0], y_interval[1] - y_interval[0], fill=False, edgecolor=color, linewidth=0.8))
    state = "Open Data" if sample_metadata.get("sample", "data") == "data" else "Simulation"
    years = sample_metadata.get("years", "")
    year_text = years if isinstance(years, str) else (str(years[0]) if len(years) == 1 else rf"{years[0]}\text{{--}}{years[-1]}")
    energy_text = "" if sample_metadata.get("sqrt_s_gev") is None else rf"$\sqrt{{s}}={float(sample_metadata['sqrt_s_gev']):g}\,\mathrm{{GeV}},\ \mathrm{{{year_text}}}$"
    fig.text(0.04, 0.965, "DELPHI", ha="left", va="top", family="sans-serif", weight="bold", fontsize=11)
    fig.text(0.13, 0.965, state, ha="left", va="top", fontsize=9)
    fig.text(0.04, 0.925, energy_text, ha="left", va="top", fontsize=7)
    valid = model_plane > 0.0
    bc_terms = np.where(observed_plane[valid] > 0.0, observed_plane[valid] * np.log(observed_plane[valid] / model_plane[valid]), 0.0)
    chi2 = float(2.0 * np.sum(bc_terms + model_plane[valid] - observed_plane[valid]))
    ndf = int(np.count_nonzero(valid) - len(fit_result.parameter_names))
    panel = _parameter_panel(fit_result.parameter_names, fit_result.parameter_values, fit_result.parameter_covariance, chi2, ndf, maximum_parameters=6)
    panel_lines = panel.splitlines()
    split_position = (len(panel_lines) + 1) // 2
    fig.text(0.10, 0.035, "\n".join(panel_lines[:split_position]), ha="left", va="bottom", fontsize=7.0, linespacing=1.05)
    fig.text(0.36, 0.035, "\n".join(panel_lines[split_position:]), ha="left", va="bottom", fontsize=7.0, linespacing=1.05)
    fig.subplots_adjust(bottom=0.28, top=0.84, wspace=0.42)
    plane_caption = (
        f"Observed and fitted two-dimensional mass plane for {sample_metadata.get('sample_label', sample_metadata.get('sample', 'the selected sample'))}. "
        "The model is the simultaneous extended-Poisson sum of SxSy, BxSy, SxBy, and BxBy; the third panel shows data/model minus one. "
        "Red marks SS and blue marks every available low/high sideband atom used by the transfer calculation."
    )
    plane_paths = _save_figure(fig, output_path / f"{stem}_fit2d_plane", plane_caption, "datafactory.stat.sideband_ana.report.write_fit_report_2d")
    plt.close(fig)

    all_paths = list(plane_paths)
    captions = {f"{stem}_fit2d_plane": plane_caption}
    projection_data = (
        ("x", fit_result.x_edges, fit_result.x_projection_observed, fit_result.x_projection_model, fit_result.x_projection_dense_mass, fit_result.x_projection_dense_model, fit_result.x_projection_dense_background),
        ("y", fit_result.y_edges, fit_result.y_projection_observed, fit_result.y_projection_model, fit_result.y_projection_dense_mass, fit_result.y_projection_dense_model, fit_result.y_projection_dense_background),
    )
    for axis_name, edges, observed, model, dense_mass, dense_model, dense_background in projection_data:
        centers = 0.5 * (edges[:-1] + edges[1:])
        errors = np.sqrt(np.maximum(observed, 0.0))
        residual, residual_range = _fit_residual(observed, model)
        residual_error = np.divide(errors, model, out=np.zeros_like(errors), where=model > 0.0)
        projection_fig, (main_ax, residual_ax) = plt.subplots(2, 1, sharex=True, figsize=(6.0, 6.6), gridspec_kw={"height_ratios": (4, 1), "hspace": 0.08})
        main_ax.errorbar(centers, observed, yerr=errors, color="black", marker="o", markersize=3.0, capsize=3.0, capthick=0.64, elinewidth=0.8, linestyle="", label=r"$\text{Data}$")
        main_ax.plot(dense_mass, dense_model, color="black", linewidth=1.2, label=r"$\text{Total fit}$")
        main_ax.plot(dense_mass, dense_background, color=plt.get_cmap("tab10").colors[0], linewidth=1.2, label=r"$\text{Background}$")
        physical_maximum = max(float(np.max(observed + errors)), float(np.max(dense_model)))
        main_ax.set_ylim(0.0, max(1.0, physical_maximum / 0.72))
        main_ax.set_ylabel(r"$\text{Candidates / bin}$")
        main_ax.set_box_aspect(3 / 4)
        main_ax.grid(False)
        _add_delphi_header(main_ax, sample_metadata, has_data=sample_metadata.get("sample", "data") == "data")
        main_ax.legend(loc="upper right", fontsize=7, frameon=False)
        main_ax.text(0.02, 0.75, panel, transform=main_ax.transAxes, ha="left", va="top", fontsize=5.5, linespacing=1.0)
        residual_ax.errorbar(centers, residual, yerr=residual_error, color="black", marker="o", markersize=3.0, capsize=3.0, capthick=0.64, elinewidth=0.8, linestyle="")
        residual_ax.axhline(0.0, color="black", linewidth=0.8)
        residual_ax.set_ylim(*residual_range)
        residual_ax.set_yticks(np.linspace(residual_range[0], residual_range[1], 5))
        residual_ax.set_ylabel(r"$\frac{\mathrm{Data}}{\mathrm{model}}-1$")
        residual_ax.set_xlabel(rf"$m_{axis_name}\,[\mathrm{{GeV}}]$")
        residual_ax.grid(False)
        projection_fig.align_ylabels((main_ax, residual_ax))
        projection_caption = (
            f"{axis_name}-axis projection of the fitted two-dimensional mass plane. "
            "Points have Poisson statistical uncertainties; black is the full four-component fit and blue is the sum of non-SxSy components. "
            "The lower panel is data/model minus one."
        )
        projection_paths = _save_figure(projection_fig, output_path / f"{stem}_fit2d_projection_{axis_name}", projection_caption, "datafactory.stat.sideband_ana.report.write_fit_report_2d")
        plt.close(projection_fig)
        all_paths.extend(projection_paths)
        captions[f"{stem}_fit2d_projection_{axis_name}"] = projection_caption
    return ReportArtifacts(files=tuple(all_paths), captions=captions)


def write_transfer_summary(
    named_results: dict[str, TransferFactor1D | TransferFactors2D],
    *,
    analysis_metadata: dict,
    output_dir,
    stem: str = "transfer_factors",
) -> ReportArtifacts:
    """Write self-explaining JSON, Markdown, and coefficient comparison figures."""
    import matplotlib.pyplot as plt

    if not named_results:
        raise ValueError("named_results must contain at least one transfer result")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_utc = datetime.now(timezone.utc).isoformat()
    records, plot_labels, plot_values, plot_errors = {}, [], [], []
    markdown = ["# Sideband transfer coefficients", "", f"Generated UTC: `{generated_utc}`", "", "| Scope | Quantity | Value | Statistical fit uncertainty |", "|---|---|---:|---:|"]
    for name, result in named_results.items():
        if isinstance(result, TransferFactor1D):
            records[name] = {
                "kind": "one_dimensional",
                "regions": {"signal": result.regions.signal, "sideband_low": result.regions.sideband_low, "sideband_high": result.regions.sideband_high},
                "integrals": {"signal": result.integral_signal, "sideband_low": result.integral_sideband_low, "sideband_high": result.integral_sideband_high, "sideband_combined": result.integral_sideband_combined},
                "r_low": result.r_low,
                "r_high": result.r_high,
                "r_combined": result.r_combined,
                "variance_r_combined": result.variance_r_combined,
            }
            markdown.append(f"| `{name}` | $r$ | {result.r_combined:.8g} | {result.sigma_r_combined:.3g} |")
            plot_labels.append(r"$r$")
            plot_values.append(result.r_combined)
            plot_errors.append(result.sigma_r_combined)
        elif isinstance(result, TransferFactors2D):
            errors = np.sqrt(np.maximum(np.diag(result.weight_covariance), 0.0))
            records[name] = {
                "kind": "two_dimensional",
                "x_regions": {"signal": result.x_regions.signal, "sideband_low": result.x_regions.sideband_low, "sideband_high": result.x_regions.sideband_high},
                "y_regions": {"signal": result.y_regions.signal, "sideband_low": result.y_regions.sideband_low, "sideband_high": result.y_regions.sideband_high},
                "weights": {"w_H": result.w_H, "w_V": result.w_V, "w_C": result.w_C},
                "weight_covariance": result.weight_covariance.tolist(),
                "weight_correlation": result.weight_correlation.tolist(),
                "atomic_region_integrals": result.atomic_region_integrals,
                "aggregated_region_integrals": result.aggregated_region_integrals,
                "factorization_closure": result.factorization_closure,
                "signal_leakage_by_region": result.signal_leakage_by_region,
            }
            for quantity, value, error in zip(("w_H", "w_V", "w_C"), (result.w_H, result.w_V, result.w_C), errors):
                markdown.append(f"| `{name}` | ${quantity}$ | {value:.8g} | {error:.3g} |")
                plot_labels.append(rf"${quantity}$")
                plot_values.append(value)
                plot_errors.append(error)
        else:
            raise TypeError(f"{name}: unsupported transfer result type {type(result).__name__}")
    document = {
        "schema_version": "datafactory_sideband_transfer_v1",
        "generated_utc": generated_utc,
        "produced_by": "datafactory.stat.sideband_ana.report.write_transfer_summary",
        "analysis_metadata": analysis_metadata,
        "results": records,
    }
    json_path = output_path / f"{stem}.json"
    markdown_path = output_path / f"{stem}.md"
    json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown.extend(("", "The quoted uncertainty is propagated from the final fit covariance. The signed corner coefficient is an inclusion-exclusion weight, not a probability.", ""))
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")

    _apply_datafactory_style()
    figure, axis = plt.subplots(figsize=(6.0, 4.5))
    positions = np.arange(len(plot_values))
    axis.errorbar(positions, plot_values, yerr=plot_errors, color="black", marker="o", markersize=3.0, capsize=3.0, capthick=0.64, elinewidth=0.8, linestyle="")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, plot_labels)
    axis.set_xlim(-0.5, len(plot_values) - 0.5)
    axis.set_ylabel(r"$\text{Transfer coefficient}$")
    axis.set_box_aspect(3 / 4)
    axis.grid(False)
    _add_delphi_header(axis, analysis_metadata, has_data=analysis_metadata.get("sample", "data") == "data")
    values_array, errors_array = np.asarray(plot_values), np.asarray(plot_errors)
    lower, upper = float(np.min(values_array - errors_array)), float(np.max(values_array + errors_array))
    padding = max(0.2, 0.25 * (upper - lower))
    axis.set_ylim(lower - padding, max(upper + padding, lower + 4.0 * padding))
    caption = "Transfer factors obtained by integrating the final fitted background model over the manually specified signal and sideband regions. Error bars propagate the final fit covariance; the signed corner term implements two-dimensional inclusion-exclusion."
    figure_paths = _save_figure(figure, output_path / stem, caption, "datafactory.stat.sideband_ana.report.write_transfer_summary")
    plt.close(figure)
    return ReportArtifacts(files=(json_path, markdown_path, *figure_paths), captions={stem: caption})
