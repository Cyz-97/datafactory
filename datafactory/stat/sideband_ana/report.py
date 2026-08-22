"""拟合与 transfer 的报告输出：PDF 图表 + Markdown/JSON 汇总。

只依赖 matplotlib（Agg 后端）和本包的数据契约，不引入 ROOT。所有 PDF 写入
Subject 元数据记录生成信息（caption、来源脚本、样本、选择、时间）。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from .transfer import MassRegions1D, TransferFactor1D, TransferFactors2D

__all__ = [
    "write_fit_report_1d",
    "write_fit_report_2d",
    "write_transfer_summary",
]

_STYLE_PATH = Path(__file__).resolve().parents[2] / "style.mplstyle"

_REGION_COLORS = {"S": "0.6", "L": "0.8", "H": "0.8"}
_REGION_LABELS = {
    "S": "signal",
    "L": "low sideband",
    "H": "high sideband",
}


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if _STYLE_PATH.exists():
        plt.style.use(str(_STYLE_PATH))
    return plt


def _metadata(sample_metadata: Mapping, extra: str = "") -> dict:
    parts = []
    if extra:
        parts.append(extra)
    for key in ("caption", "sample", "selection", "source_script"):
        if sample_metadata.get(key):
            parts.append(f"{key}={sample_metadata[key]}")
    parts.append(f"generated={datetime.now(timezone.utc).isoformat()}")
    return {"Subject": "; ".join(parts)}


def _format_regions(regions: MassRegions1D) -> list[str]:
    lines = []
    for label, (low, high) in regions.region_intervals().items():
        lines.append(
            f"{_REGION_LABELS[label]} [{low:.6g}, {high:.6g}]"
        )
    return lines


def _residual_axes_limits(residual: np.ndarray) -> tuple[float, float]:
    """残差 y 范围关于 0 对称且不超出 (-5, 5)。"""
    finite = residual[np.isfinite(residual)]
    if finite.size:
        magnitude = float(np.max(np.abs(finite)))
    else:
        magnitude = 1.0
    limit = min(5.0, max(magnitude * 1.2, 0.2))
    return (-limit, limit)


# ---------------------------------------------------------------------------
# 9.1 一维拟合报告
# ---------------------------------------------------------------------------

_SIGNAL_LATEX_1D = {
    "narrow_yield": r"$N_{\mathrm{narrow}}$",
    "wide_yield": r"$N_{\mathrm{wide}}$",
    "mean": r"$\mu$",
    "sigma_narrow": r"$\sigma_{\mathrm{narrow}}$",
    "delta_sigma": r"$\Delta\sigma$",
}


def write_fit_report_1d(
    fit_result,
    transfer_result: TransferFactor1D | None,
    *,
    sample_metadata: Mapping,
    output_dir: Path | str,
    stem: str,
) -> list[Path]:
    """输出一维质量谱拟合报告 PDF：主图 + 残差 + 参数面板。"""
    plt = _import_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}_fit1d.pdf"

    centers = 0.5 * (fit_result.mass_edges[:-1] + fit_result.mass_edges[1:])
    errors = np.sqrt(np.maximum(fit_result.observed_variances, 0.0))
    model = fit_result.model_counts
    valid_model = model > 0.0
    residual = np.full_like(centers, np.nan, dtype=float)
    residual[valid_model] = (
        fit_result.observed_counts[valid_model] / model[valid_model] - 1.0
    )
    residual_errors = np.full_like(centers, np.nan, dtype=float)
    residual_errors[valid_model] = errors[valid_model] / model[valid_model]

    figure = plt.figure(figsize=(11.0, 6.5))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[3.2, 1.4],
        height_ratios=[3.0, 1.0],
        hspace=0.08,
        wspace=0.06,
    )
    main_ax = figure.add_subplot(grid[0, 0])
    residual_ax = figure.add_subplot(grid[1, 0], sharex=main_ax)
    text_ax = figure.add_subplot(grid[:, 1])
    text_ax.axis("off")

    main_ax.errorbar(
        centers,
        fit_result.observed_counts,
        yerr=errors,
        fmt=".",
        color="black",
        ms=3,
        lw=0.8,
        label="data",
    )
    main_ax.plot(
        fit_result.dense_mass,
        fit_result.dense_model,
        color="black",
        lw=0.9,
        label="model",
    )
    main_ax.plot(
        fit_result.dense_mass,
        fit_result.dense_background,
        color="royalblue",
        lw=0.9,
        label="background",
    )
    for label, (low, high) in fit_result.regions.region_intervals().items():
        main_ax.axvspan(
            low,
            high,
            color=_REGION_COLORS[label],
            alpha=0.35,
            lw=0,
        )
    main_ax.set_ylabel("candidates / bin")
    main_ax.set_xlim(
        float(fit_result.mass_edges[0]), float(fit_result.mass_edges[-1])
    )
    main_ax.legend(frameon=False, fontsize="small", loc="upper right")
    caption = str(sample_metadata.get("caption", ""))
    if caption:
        main_ax.set_title(caption, fontsize="medium")

    residual_ax.axhline(0.0, color="black", lw=0.6)
    residual_ax.errorbar(
        centers,
        residual,
        yerr=residual_errors,
        fmt=".",
        color="black",
        ms=3,
        lw=0.8,
    )
    lower, upper = _residual_axes_limits(residual)
    residual_ax.set_ylim(lower, upper)
    residual_ax.yaxis.set_major_locator(MaxNLocator(5))
    residual_ax.set_xlabel(
        r"$m_{p\pi^-}$ [GeV]" if not sample_metadata.get("x_label")
        else str(sample_metadata["x_label"])
    )
    residual_ax.set_ylabel("data/model - 1")

    # 参数面板。
    lines = []
    errors_vec = np.sqrt(np.clip(np.diag(fit_result.parameter_covariance), 0.0, None))
    for index, name in enumerate(fit_result.parameter_names):
        latex = _SIGNAL_LATEX_1D.get(name, rf"${name}$")
        lines.append(
            f"{latex} = {fit_result.parameter_values[index]:.4g} "
            f"$\\pm$ {errors_vec[index]:.2g}"
        )
    lines.append("")
    lines.append(
        rf"$\chi^2/\mathrm{{ndf}} = {fit_result.chi2:.1f}/{fit_result.ndf}"
        rf" = {fit_result.chi2 / max(fit_result.ndf, 1):.3g}$"
    )
    lines.append(
        "background: "
        + ("profiled by zfit" if fit_result.background_profiled else "fixed (SymbolFit)")
    )
    lines.append("")
    lines.append("SymbolFit background:")
    lines.append(
        fit_result.background_model.formula.replace(
            "x0", r"m"
        )
    )
    for name, value in fit_result.symbolfit_initial_values.items():
        lines.append(f"  {name} = {value:.6g}")
    if transfer_result is not None:
        lines.append("")
        lines.append("transfer:")
        lines.append(
            f"  r = {transfer_result.r_combined:.4f} "
            f"$\\pm$ {transfer_result.sigma_r_combined:.4f}"
        )
        lines.extend(f"  {line}" for line in _format_regions(transfer_result.regions))
    text_ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize="small",
        transform=text_ax.transAxes,
    )

    figure.savefig(
        pdf_path,
        metadata=_metadata(sample_metadata, extra=f"1D mass fit: {stem}"),
    )
    plt.close(figure)
    return [pdf_path]


# ---------------------------------------------------------------------------
# 9.2 二维拟合报告
# ---------------------------------------------------------------------------


def _region_rectangles(
    ax,
    x_regions: MassRegions1D,
    y_regions: MassRegions1D,
):
    """在平面上画 signal（红实线）和 sideband（蓝实线）区域矩形。"""
    for x_label, (x_low, x_high) in x_regions.region_intervals().items():
        for y_label, (y_low, y_high) in y_regions.region_intervals().items():
            edge, lw, label = "red", 1.4, "signal"
            if not (x_label == "S" and y_label == "S"):
                edge, lw, label = "royalblue", 0.8, None
            ax.add_patch(
                Rectangle(
                    (x_low, y_low),
                    x_high - x_low,
                    y_high - y_low,
                    fill=False,
                    edgecolor=edge,
                    lw=lw,
                    label=label if label else None,
                )
            )


def _plane_panel(ax, x_centers, y_centers, values, title, vmin=None, vmax=None):
    mesh = ax.pcolormesh(x_centers, y_centers, values, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize="small")
    return mesh


def write_fit_report_2d(
    fit_result,
    transfer_result: TransferFactors2D | None,
    *,
    sample_metadata: Mapping,
    output_dir: Path | str,
    stem: str,
) -> list[Path]:
    """输出二维质量平面拟合报告。

    产出三个 PDF：平面三联图（observed/model 共用单个 colorbar，单页）、
    各区域 data vs model 事例数柱状图（独立 PDF）、x 投影、y 投影。
    拟合参数、区域积分表、w 代入式与区域事例数数值等文本信息以结构化
    表格打印到 terminal，不写入 PDF。
    """
    plt = _import_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    x_label = str(sample_metadata.get("x_label", "x mass [GeV]"))
    y_label = str(sample_metadata.get("y_label", "y mass [GeV]"))
    same_quantity = (
        sample_metadata.get("x_label") is not None
        and sample_metadata.get("x_label") == sample_metadata.get("y_label")
    )

    observed = fit_result.observed_counts_by_period.sum(axis=0)
    model = fit_result.model_counts_by_period.sum(axis=0)
    valid = model > 0.0
    ratio = np.full_like(model, np.nan, dtype=float)
    ratio[valid] = observed[valid] / model[valid] - 1.0

    x_edges = np.asarray(fit_result.x_edges, dtype=float)
    y_edges = np.asarray(fit_result.y_edges, dtype=float)
    x_bin_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_bin_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    components = ["SxSy", "BxSy", "SxBy", "BxBy"]

    # 各原子区域的 data bin-content 和与模型期望计数（产额 × 区域积分）。
    # data 按 bin 中心点归入区域；模型积分用精确区域边界。
    region_rows: list = []
    if transfer_result is not None:
        component_totals = {
            name: sum(
                yields[name]
                for yields in fit_result.component_yields_by_period
            )
            for name in components
        }
        for key in transfer_result.atomic_region_integrals:
            x_low, x_high = transfer_result.x_regions.region_intervals()[key[0]]
            y_low, y_high = transfer_result.y_regions.region_intervals()[key[1]]
            in_x = (x_bin_centers >= x_low) & (x_bin_centers < x_high)
            in_y = (y_bin_centers >= y_low) & (y_bin_centers < y_high)
            data_sum = float(observed[np.ix_(in_x, in_y)].sum())
            model_sum = float(sum(
                component_totals[name]
                * transfer_result.atomic_region_integrals[key][name]
                for name in components
            ))
            pull = (
                (data_sum - model_sum) / np.sqrt(model_sum)
                if model_sum > 0.0 else np.nan
            )
            region_rows.append((f"{key[0]}{key[1]}", data_sum, model_sum, pull))

    # ---- 文本信息打印到 terminal（结构化表格，不写入 PDF） -------------------
    tag = f"[write_fit_report_2d:{stem}]"

    def _section(title: str) -> None:
        print(f"\n{tag} ---- {title} " + "-" * max(0, 56 - len(title)))

    _section("fit information")
    print(f"{'periods':<26}: {fit_result.n_periods}")
    print(f"{'fit bins':<26}: {fit_result.fit_nbins} x {fit_result.fit_nbins}")
    print(f"{'NLL':<26}: {fit_result.nll_value:.2f}")

    _section("component yields")
    print(f"{'period':<10}" + "".join(f"{name:>14}" for name in components))
    for period, yields in enumerate(fit_result.component_yields_by_period):
        print(f"{'p' + str(period):<10}"
              + "".join(f"{yields[name]:>14.0f}" for name in components))

    _section("shape parameters")
    errors_vec = np.sqrt(np.clip(np.diag(fit_result.parameter_covariance), 0.0, None))
    print(f"{'name':<26}{'value':>14}{'error':>14}")
    for index, name in enumerate(fit_result.parameter_names):
        if name.startswith("N_"):
            continue
        print(f"{name:<26}{fit_result.parameter_values[index]:>14.4g}"
              f"{errors_vec[index]:>14.2g}")

    if transfer_result is not None:
        _section("component integrals per atomic region")
        print(f"{'region':<10}" + "".join(f"{name:>14}" for name in components))
        for key, values in transfer_result.atomic_region_integrals.items():
            print(f"{key[0] + key[1]:<10}"
                  + "".join(f"{values[name]:>14.4g}" for name in components))

        _section("aggregated region integrals")
        print(f"{'component':<10}"
              + "".join(f"{k:>14}" for k in ("SS", "BS", "SB", "BB")))
        aggregated = transfer_result.aggregated_region_integrals
        for component in components:
            entries = aggregated[component]
            print(f"{component:<10}"
                  + "".join(f"{entries[k]:>14.4g}" for k in ("SS", "BS", "SB", "BB")))

        _section("transfer coefficients")
        sigma_w = np.sqrt(
            np.clip(np.diag(transfer_result.weight_covariance), 0.0, None)
        )
        bxsy = aggregated["BxSy"]
        sxby = aggregated["SxBy"]
        bxby = aggregated["BxBy"]
        print(f"  w_H = {bxsy['SS']:.4g} / {bxsy['BS']:.4g} "
              f"= {transfer_result.w_H:.4f} ± {sigma_w[0]:.4f}")
        print(f"  w_V = {sxby['SS']:.4g} / {sxby['SB']:.4g} "
              f"= {transfer_result.w_V:.4f} ± {sigma_w[1]:.4f}")
        print(f"  w_C = ({bxby['SS']:.4g} - w_H*{bxby['BS']:.4g} "
              f"- w_V*{bxby['SB']:.4g}) / {bxby['BB']:.4g} "
              f"= {transfer_result.w_C:.4f} ± {sigma_w[2]:.4f}")
        print(f"  closure w_C + w_H*w_V = {transfer_result.factorization_closure:.4g}")

        _section("signal leakage f_SxSy(R) / f_SxSy(SS)")
        for key, value in transfer_result.signal_leakage_by_region.items():
            print(f"  {key[0]}{key[1]}: {value:.4g}")

    # ---- 平面三联图（单页 PDF；data/model 共享色标） ------------------------
    plane_path = output_dir / f"{stem}_fit2d_plane.pdf"
    paths.append(plane_path)
    figure = plt.figure(figsize=(11.5, 4.4))
    grid = figure.add_gridspec(1, 3, wspace=0.55)
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    vmax_common = float(max(observed.max(), model.max()))
    mesh_observed = _plane_panel(
        axes[0], x_edges, y_edges, observed.T, "observed",
        vmin=0.0, vmax=vmax_common,
    )
    mesh_model = _plane_panel(
        axes[1], x_edges, y_edges, model.T, "model",
        vmin=0.0, vmax=vmax_common,
    )
    mesh_ratio = _plane_panel(
        axes[2], x_edges, y_edges, ratio.T, "observed/model - 1"
    )
    figure.colorbar(mesh_model, ax=axes[:2], shrink=0.9,
                    label="candidates / bin")
    figure.colorbar(mesh_ratio, ax=axes[2], shrink=0.9)
    for ax in axes:
        if same_quantity:
            ax.set_aspect("equal")
        ax.set_xlabel(x_label, fontsize="small")
        ax.set_ylabel(y_label, fontsize="small")
    if transfer_result is not None:
        for ax in axes:
            _region_rectangles(ax, transfer_result.x_regions, transfer_result.y_regions)
        # 区域事例数标注：observed 面板标 data，model 面板标模型期望。
        label_style = dict(
            color="white", ha="center", va="center", fontsize="x-small",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, ec="none"),
        )
        for row, key in zip(region_rows, transfer_result.atomic_region_integrals):
            x_low, x_high = transfer_result.x_regions.region_intervals()[key[0]]
            y_low, y_high = transfer_result.y_regions.region_intervals()[key[1]]
            center_x = 0.5 * (x_low + x_high)
            center_y = 0.5 * (y_low + y_high)
            axes[0].text(center_x, center_y, f"{row[1]:,.0f}", **label_style)
            axes[1].text(center_x, center_y, f"{row[2]:,.0f}", **label_style)
    caption = str(sample_metadata.get("caption", ""))
    if caption:
        figure.suptitle(caption, fontsize="medium")
    figure.savefig(
        plane_path,
        metadata=_metadata(sample_metadata, extra=f"2D mass-plane fit: {stem}"),
    )
    plt.close(figure)

    # ---- 区域事例数对比：数值表打印到 terminal，柱状图为独立 PDF ------------
    if region_rows:
        _section("region content: data vs model")
        print(f"{'region':<10}{'data':>14}{'model':>14}{'pull':>10}")
        for label, data_sum, model_sum, pull in region_rows:
            pull_text = f"{pull:+.2f}" if np.isfinite(pull) else "n/a"
            print(f"{label:<10}{data_sum:>14.0f}{model_sum:>14.0f}{pull_text:>10}")
        print("  (data: bins assigned by center; model: yields x region integrals)")

        region_path = output_dir / f"{stem}_fit2d_region_content.pdf"
        paths.append(region_path)
        figure = plt.figure(figsize=(7.0, 4.8))
        ax_bars = figure.add_subplot(1, 1, 1)
        positions = np.arange(len(region_rows))
        bar_width = 0.38
        data_values = np.asarray([row[1] for row in region_rows])
        model_values = np.asarray([row[2] for row in region_rows])
        ax_bars.bar(
            positions - 0.5 * bar_width, data_values, width=bar_width,
            color="black", yerr=np.sqrt(np.maximum(data_values, 0.0)),
            error_kw={"lw": 0.9, "capsize": 3}, label="data",
        )
        ax_bars.bar(
            positions + 0.5 * bar_width, model_values, width=bar_width,
            color="royalblue", label="model",
        )
        for position, row in zip(positions, region_rows):
            if np.isfinite(row[3]):
                ax_bars.text(
                    position, max(row[1], row[2]), f"pull={row[3]:+.1f}",
                    ha="center", va="bottom", fontsize="x-small",
                )
        ax_bars.set_xticks(positions, [row[0] for row in region_rows])
        ax_bars.set_ylabel("candidates in region")
        ax_bars.legend(frameon=False, fontsize="small")
        ax_bars.set_title(
            "region content: data (bin centers) vs model (region integrals)",
            fontsize="small",
        )
        figure.tight_layout()
        figure.savefig(
            region_path,
            metadata=_metadata(sample_metadata, extra=f"2D fit region content: {stem}"),
        )
        plt.close(figure)

    # 投影图（x 和 y）。observed/model 带 period 轴 (n_periods, n_bins)，
    # background 已经是 period 求和后的 1D，只有 ndim==2 时才压缩。
    def _projection_total(array):
        array = np.asarray(array)
        return array.sum(axis=0) if array.ndim == 2 else array

    projections = (
        ("x", _projection_total(fit_result.x_projection_observed),
         _projection_total(fit_result.x_projection_model),
         fit_result.x_projection_dense_mass,
         fit_result.x_projection_dense_model,
         fit_result.x_projection_dense_background,
         x_bin_centers, x_label),
        ("y", _projection_total(fit_result.y_projection_observed),
         _projection_total(fit_result.y_projection_model),
         fit_result.y_projection_dense_mass,
         fit_result.y_projection_dense_model,
         fit_result.y_projection_dense_background,
         y_bin_centers, y_label),
    )
    for axis_name, observed_proj, model_proj, dense_mass, dense_model, dense_background, bin_centers, axis_label in projections:
        figure = plt.figure(figsize=(7.5, 6.0))
        grid = figure.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.08)
        main_ax = figure.add_subplot(grid[0])
        residual_ax = figure.add_subplot(grid[1], sharex=main_ax)

        observed_errors = np.sqrt(np.maximum(observed_proj, 0.0))
        main_ax.errorbar(
            bin_centers,
            observed_proj,
            yerr=observed_errors,
            fmt=".",
            color="black",
            ms=3,
            lw=0.8,
            label="data",
        )
        main_ax.plot(dense_mass, dense_model, color="black", lw=0.9, label="model")
        main_ax.plot(
            dense_mass, dense_background, color="royalblue", lw=0.9,
            label="background",
        )
        if transfer_result is not None:
            regions_for_axis = (
                transfer_result.x_regions
                if axis_name == "x"
                else transfer_result.y_regions
            )
            for label, (low, high) in regions_for_axis.region_intervals().items():
                main_ax.axvspan(
                    low, high, color=_REGION_COLORS[label], alpha=0.35, lw=0
                )
        main_ax.set_ylabel(f"candidates / bin ({axis_name} projection)")
        main_ax.legend(frameon=False, fontsize="small", loc="upper right")
        caption = str(sample_metadata.get("caption", ""))
        if caption:
            main_ax.set_title(f"{caption} — {axis_name} projection", fontsize="medium")

        valid_proj = model_proj > 0.0
        residual_proj = np.full_like(model_proj, np.nan, dtype=float)
        residual_proj[valid_proj] = (
            observed_proj[valid_proj] / model_proj[valid_proj] - 1.0
        )
        residual_proj_errors = np.full_like(model_proj, np.nan, dtype=float)
        residual_proj_errors[valid_proj] = (
            observed_errors[valid_proj] / model_proj[valid_proj]
        )
        residual_ax.axhline(0.0, color="black", lw=0.6)
        residual_ax.errorbar(
            bin_centers,
            residual_proj,
            yerr=residual_proj_errors,
            fmt=".",
            color="black",
            ms=3,
            lw=0.8,
        )
        lower, upper = _residual_axes_limits(residual_proj)
        residual_ax.set_ylim(lower, upper)
        residual_ax.yaxis.set_major_locator(MaxNLocator(5))
        residual_ax.set_xlabel(axis_label)
        residual_ax.set_ylabel("data/model - 1")

        projection_path = output_dir / f"{stem}_fit2d_projection_{axis_name}.pdf"
        paths.append(projection_path)
        figure.savefig(
            projection_path,
            metadata=_metadata(
                sample_metadata, extra=f"2D fit {axis_name} projection: {stem}"
            ),
        )
        plt.close(figure)

    return paths


# ---------------------------------------------------------------------------
# 9.3 transfer 汇总
# ---------------------------------------------------------------------------


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_transfer_summary(
    named_results: Sequence[Mapping],
    *,
    analysis_metadata: Mapping,
    output_dir: Path | str,
    stem: str = "transfer_factors",
) -> list[Path]:
    """汇总所有作用域的 transfer 结果，输出 Markdown / JSON / PDF。

    ``named_results`` 的每一项是包含以下 key 的映射::

        label          结果名称
        scope          作用域描述（category/subset 等）
        fallback       fallback 来源描述（无则为 None）
        regions_x      MassRegions1D（一维结果也用它）
        regions_y      MassRegions1D 或 None
        transfer_1d    TransferFactor1D 或 None
        transfer_2d    TransferFactors2D 或 None
    """
    plt = _import_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    pdf_path = output_dir / f"{stem}.pdf"
    paths = [markdown_path, json_path, pdf_path]

    # ---- Markdown ----------------------------------------------------------
    lines = ["# Sideband transfer factors", ""]
    for key, value in analysis_metadata.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    structured_entries = []
    for entry in named_results:
        label = str(entry["label"])
        lines.append(f"## {label}")
        lines.append(f"- scope: {entry.get('scope', '')}")
        lines.append(
            f"- fallback: {entry.get('fallback') or 'none'}"
        )
        regions_x: MassRegions1D = entry["regions_x"]
        lines.append(f"- x regions: {', '.join(_format_regions(regions_x))}")
        regions_y = entry.get("regions_y")
        if regions_y is not None:
            lines.append(f"- y regions: {', '.join(_format_regions(regions_y))}")
        transfer_1d = entry.get("transfer_1d")
        if transfer_1d is not None:
            lines.append("- 1D transfer:")
            lines.append(
                f"  - I_S = {transfer_1d.integral_signal:.6g}"
            )
            for key, value in (
                ("I_L", transfer_1d.integral_sideband_low),
                ("I_H", transfer_1d.integral_sideband_high),
            ):
                if value is not None:
                    lines.append(f"  - {key} = {value:.6g}")
            lines.append(
                f"  - I_B = {transfer_1d.integral_sideband_combined:.6g}"
            )
            for key, value in (
                ("r_L", transfer_1d.r_low),
                ("r_H", transfer_1d.r_high),
            ):
                if value is not None:
                    lines.append(f"  - {key} = {value:.6g}")
            lines.append(
                f"  - r = {transfer_1d.r_combined:.6g} "
                f"± {transfer_1d.sigma_r_combined:.2g}"
            )
            lines.append(
                f"  - background: `{transfer_1d.background_formula}`"
            )
        transfer_2d = entry.get("transfer_2d")
        if transfer_2d is not None:
            lines.append("- 2D transfer:")
            lines.append("  - atomic region integrals:")
            for key, values in transfer_2d.atomic_region_integrals.items():
                entries = ", ".join(
                    f"{name}={value:.4g}" for name, value in values.items()
                )
                lines.append(f"    - {key[0]}{key[1]}: {entries}")
            lines.append("  - aggregated integrals:")
            for component, values in transfer_2d.aggregated_region_integrals.items():
                entries = ", ".join(
                    f"{key}={value:.4g}" for key, value in values.items()
                )
                lines.append(f"    - {component}: {entries}")
            sigma_w = np.sqrt(
                np.clip(np.diag(transfer_2d.weight_covariance), 0.0, None)
            )
            lines.append(
                f"  - w_H = {transfer_2d.w_H:.6g} ± {sigma_w[0]:.2g}"
            )
            lines.append(
                f"  - w_V = {transfer_2d.w_V:.6g} ± {sigma_w[1]:.2g}"
            )
            lines.append(
                f"  - w_C = {transfer_2d.w_C:.6g} ± {sigma_w[2]:.2g}"
            )
            lines.append(
                f"  - closure w_C + w_H*w_V = "
                f"{transfer_2d.factorization_closure:.4g}"
            )
            lines.append("  - signal leakage:")
            for key, value in transfer_2d.signal_leakage_by_region.items():
                lines.append(f"    - {key[0]}{key[1]}: {value:.4g}")
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    # ---- JSON --------------------------------------------------------------
    for entry in named_results:
        structured_entries.append(
            {
                "label": str(entry["label"]),
                "scope": entry.get("scope", ""),
                "fallback": entry.get("fallback"),
                "regions_x": _jsonable(entry["regions_x"]),
                "regions_y": _jsonable(entry.get("regions_y")),
                "transfer_1d": _jsonable(entry.get("transfer_1d")),
                "transfer_2d": _jsonable(entry.get("transfer_2d")),
            }
        )
    import json

    json_path.write_text(
        json.dumps(
            {
                "analysis_metadata": _jsonable(analysis_metadata),
                "results": structured_entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---- PDF 汇总表 --------------------------------------------------------
    figure = plt.figure(figsize=(8.5, 2.2 + 0.5 * len(named_results)))
    ax = figure.add_subplot(1, 1, 1)
    ax.axis("off")
    cell_text = []
    for entry in named_results:
        transfer_1d = entry.get("transfer_1d")
        transfer_2d = entry.get("transfer_2d")
        r_text = (
            f"{transfer_1d.r_combined:.4f} ± {transfer_1d.sigma_r_combined:.4f}"
            if transfer_1d is not None
            else "-"
        )
        if transfer_2d is not None:
            sigma_w = np.sqrt(
                np.clip(np.diag(transfer_2d.weight_covariance), 0.0, None)
            )
            w_text = (
                f"{transfer_2d.w_H:.4f}±{sigma_w[0]:.4f} / "
                f"{transfer_2d.w_V:.4f}±{sigma_w[1]:.4f} / "
                f"{transfer_2d.w_C:.4f}±{sigma_w[2]:.4f}"
            )
        else:
            w_text = "-"
        cell_text.append([str(entry["label"]), r_text, w_text])
    ax.table(
        cellText=cell_text,
        colLabels=["label", "r (1D)", "w_H / w_V / w_C (2D)"],
        cellLoc="center",
        loc="upper center",
    )
    figure.suptitle("Sideband transfer factors", fontsize="medium")
    figure.savefig(pdf_path, metadata=_metadata(analysis_metadata))
    plt.close(figure)

    return paths
