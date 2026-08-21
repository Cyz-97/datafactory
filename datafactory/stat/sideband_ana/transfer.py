"""质量区域契约与 transfer factor / transfer coefficients 计算。

本文件只依赖 numpy，不依赖 TensorFlow/zfit/SymbolFit。拟合结果通过鸭子类型
传入：一维需要 ``background_model``（提供 ``evaluate_density`` / ``integrate``
和 ``parameter_names``）、``parameter_names`` / ``parameter_values`` /
``parameter_covariance``；二维需要 ``component_models``（每个分量提供归一化的
``x_pdf`` / ``y_pdf``）。因此纯数学部分可以用玩具模型独立测试。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "MassRegions1D",
    "regions_from_offsets",
    "TransferFactor1D",
    "TransferFactors2D",
    "calculate_transfer_factor_1d",
    "calculate_transfer_factors_2d",
]


# ---------------------------------------------------------------------------
# 4. 质量区域数据契约
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MassRegions1D:
    """一条质量轴上的 signal / sideband 显式区间（单位与质量轴一致）。

    ``signal`` 与两个 sideband 都写成 ``(low, high)`` 开闭不重要，区间按闭区间
    处理；至少要提供一个 sideband。无效配置在构造时立即抛出 ``ValueError``，
    不做自动裁剪或端点交换。
    """

    signal: tuple[float, float]
    sideband_low: tuple[float, float] | None = None
    sideband_high: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for name, (low, high) in self.region_intervals().items():
            if not low < high:
                raise ValueError(
                    f"MassRegions1D: {name} 区间必须满足 low < high，"
                    f"得到 ({low}, {high})"
                )
        if self.sideband_low is None and self.sideband_high is None:
            raise ValueError("MassRegions1D: 至少需要一个 sideband")
        signal_low, signal_high = self.signal
        if self.sideband_low is not None:
            low, high = self.sideband_low
            if high >= signal_low:
                raise ValueError(
                    f"MassRegions1D: low sideband ({low}, {high}) 必须整体位于 "
                    f"signal ({signal_low}, {signal_high}) 的低端"
                )
        if self.sideband_high is not None:
            low, high = self.sideband_high
            if low <= signal_high:
                raise ValueError(
                    f"MassRegions1D: high sideband ({low}, {high}) 必须整体位于 "
                    f"signal ({signal_low}, {signal_high}) 的高端"
                )
        if self.sideband_low is not None and self.sideband_high is not None:
            low_low, low_high = self.sideband_low
            high_low, high_high = self.sideband_high
            if low_high >= high_low:
                raise ValueError(
                    f"MassRegions1D: low sideband ({low_low}, {low_high}) 与 "
                    f"high sideband ({high_low}, {high_high}) 不得重叠"
                )

    def region_intervals(self) -> dict[str, tuple[float, float]]:
        """返回现有区域 label -> (low, high)。label 为 S / L / H。"""
        intervals: dict[str, tuple[float, float]] = {"S": self.signal}
        if self.sideband_low is not None:
            intervals["L"] = self.sideband_low
        if self.sideband_high is not None:
            intervals["H"] = self.sideband_high
        return intervals

    def labels(self) -> list[str]:
        """返回现有区域 label 列表，按 S, L, H 顺序。"""
        return list(self.region_intervals())

    def validate_within(self, mass_lo: float, mass_hi: float) -> None:
        """检查所有区间都位于 [mass_lo, mass_hi] 内。"""
        for label, (low, high) in self.region_intervals().items():
            if low < mass_lo or high > mass_hi:
                raise ValueError(
                    f"MassRegions1D: {label} 区间 ({low}, {high}) 超出质量范围 "
                    f"[{mass_lo}, {mass_hi}]"
                )


def regions_from_offsets(
    peak_mean: float,
    *,
    signal_half_width: float,
    sideband_low_offset: float | None = None,
    sideband_high_offset: float | None = None,
) -> MassRegions1D:
    """把"峰位加偏移量"的常用配置转换为显式区间对象。

    signal 为 ``peak_mean ± signal_half_width``；sideband 中心分别在
    ``peak_mean - sideband_low_offset`` 和 ``peak_mean + sideband_high_offset``，
    宽度与 signal 相同。offset 必须为正数。
    """
    if signal_half_width <= 0.0:
        raise ValueError(f"signal_half_width 必须为正，得到 {signal_half_width}")
    signal = (peak_mean - signal_half_width, peak_mean + signal_half_width)
    sideband_low = None
    if sideband_low_offset is not None:
        if sideband_low_offset <= 0.0:
            raise ValueError(f"sideband_low_offset 必须为正，得到 {sideband_low_offset}")
        center = peak_mean - sideband_low_offset
        sideband_low = (center - signal_half_width, center + signal_half_width)
    sideband_high = None
    if sideband_high_offset is not None:
        if sideband_high_offset <= 0.0:
            raise ValueError(f"sideband_high_offset 必须为正，得到 {sideband_high_offset}")
        center = peak_mean + sideband_high_offset
        sideband_high = (center - signal_half_width, center + signal_half_width)
    return MassRegions1D(
        signal=signal,
        sideband_low=sideband_low,
        sideband_high=sideband_high,
    )


# ---------------------------------------------------------------------------
# 7.1 一维 transfer factor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferFactor1D:
    """一维 transfer factor r = I_S / I_B 及其误差传播结果。"""

    regions: MassRegions1D
    background_formula: str
    integral_signal: float
    integral_sideband_low: float | None
    integral_sideband_high: float | None
    integral_sideband_combined: float
    r_low: float | None
    r_high: float | None
    r_combined: float
    variance_r_combined: float
    sigma_r_combined: float
    parameter_gradient: dict[str, float]


def calculate_transfer_factor_1d(fit_result, regions: MassRegions1D) -> TransferFactor1D:
    """对最终拟合本底 b(m; θ) 计算区域积分与 r = I_S / I_B。

    ``fit_result`` 需要 ``background_model``（含 ``formula``、``parameter_names``
    和 ``integrate(mass_lo, mass_hi, parameter_values)``）、``parameter_names``、
    ``parameter_values``、``parameter_covariance`` 以及 ``mass_edges``。
    误差由最终拟合 covariance 经数值梯度传播：V_r = ∇r^T Cov(θ) ∇r。
    """
    mass_lo = float(np.min(fit_result.mass_edges))
    mass_hi = float(np.max(fit_result.mass_edges))
    regions.validate_within(mass_lo, mass_hi)

    parameter_names = list(fit_result.parameter_names)
    parameter_values = np.asarray(fit_result.parameter_values, dtype=float)
    covariance = np.asarray(fit_result.parameter_covariance, dtype=float)
    if covariance.shape != (len(parameter_names), len(parameter_names)):
        raise ValueError(
            "calculate_transfer_factor_1d: covariance 形状 "
            f"{covariance.shape} 与参数数 {len(parameter_names)} 不匹配"
        )

    background_model = fit_result.background_model
    nominal = dict(zip(parameter_names, parameter_values))
    background_parameters = list(background_model.parameter_names)
    for name in background_parameters:
        if name not in nominal:
            raise ValueError(
                f"calculate_transfer_factor_1d: 本底参数 {name} 不在拟合参数中"
            )

    intervals = regions.region_intervals()

    def integrals_at(parameter_dict: Mapping[str, float]) -> dict[str, float]:
        return {
            label: background_model.integrate(low, high, parameter_dict)
            for label, (low, high) in intervals.items()
        }

    nominal_integrals = integrals_at(nominal)
    integral_signal = nominal_integrals["S"]
    if integral_signal <= 0.0:
        raise ValueError(
            f"calculate_transfer_factor_1d: signal 区域本底积分非正 ({integral_signal})"
        )

    integral_low = nominal_integrals.get("L")
    integral_high = nominal_integrals.get("H")
    sideband_terms = [
        value for value in (integral_low, integral_high) if value is not None
    ]
    integral_combined = float(sum(sideband_terms))
    if integral_combined <= 0.0:
        raise ValueError(
            f"calculate_transfer_factor_1d: 联合 sideband 本底积分非正 ({integral_combined})"
        )

    def r_combined_at(parameter_dict: Mapping[str, float]) -> float:
        integrals = integrals_at(parameter_dict)
        sideband = sum(
            integrals[label] for label in intervals if label != "S"
        )
        return integrals["S"] / sideband

    # 数值梯度：只对本底参数非零，其余参数梯度为 0（r 只依赖本底）。
    gradient = {name: 0.0 for name in parameter_names}
    for name in background_parameters:
        step = 1.0e-5 * max(abs(nominal[name]), 1.0)
        up = dict(nominal)
        down = dict(nominal)
        up[name] = nominal[name] + step
        down[name] = nominal[name] - step
        gradient[name] = (r_combined_at(up) - r_combined_at(down)) / (2.0 * step)

    gradient_vector = np.asarray([gradient[name] for name in parameter_names])
    variance = float(gradient_vector @ covariance @ gradient_vector)
    if variance < 0.0:
        raise ValueError(
            f"calculate_transfer_factor_1d: 传播得到的 r 方差为负 ({variance})"
        )

    return TransferFactor1D(
        regions=regions,
        background_formula=background_model.formula,
        integral_signal=float(integral_signal),
        integral_sideband_low=None if integral_low is None else float(integral_low),
        integral_sideband_high=None if integral_high is None else float(integral_high),
        integral_sideband_combined=integral_combined,
        r_low=None if integral_low is None else float(integral_signal / integral_low),
        r_high=None if integral_high is None else float(integral_signal / integral_high),
        r_combined=float(integral_signal / integral_combined),
        variance_r_combined=variance,
        sigma_r_combined=float(np.sqrt(variance)),
        parameter_gradient=gradient,
    )


# ---------------------------------------------------------------------------
# 7.2 二维 transfer coefficients
# ---------------------------------------------------------------------------

COMPONENT_NAMES = ("SxSy", "BxSy", "SxBy", "BxBy")


def _atomic_region_keys(
    x_regions: MassRegions1D, y_regions: MassRegions1D
) -> list[tuple[str, str]]:
    """枚举当前区域配置实际产生的 (x_region, y_region) 原子区域 key。

    双边带时为九个互斥区域；单侧边带时自动缩减。key 顺序固定为
    x 在前、y 在后，禁止使用数字编号。
    """
    return [
        (x_label, y_label)
        for x_label in x_regions.labels()
        for y_label in y_regions.labels()
    ]


@dataclass(frozen=True)
class TransferFactors2D:
    """二维 transfer coefficients w_H / w_V / w_C 及其协方差。"""

    x_regions: MassRegions1D
    y_regions: MassRegions1D
    atomic_region_integrals: dict[tuple[str, str], dict[str, float]]
    aggregated_region_integrals: dict[str, dict[str, float]]
    w_H: float
    w_V: float
    w_C: float
    weight_covariance: np.ndarray
    weight_correlation: np.ndarray
    parameter_gradient: dict[str, np.ndarray]
    signal_leakage_by_region: dict[tuple[str, str], float]
    factorization_closure: float


def _aggregated_integrals(
    atomic_integrals: Mapping[tuple[str, str], Mapping[str, float]],
    component: str,
) -> dict[str, float]:
    """把一个分量的原子区域积分聚合为 SS / BS / SB / BB。"""
    ss = atomic_integrals[("S", "S")][component]
    bs = sum(
        values[component]
        for (x_label, y_label), values in atomic_integrals.items()
        if x_label != "S" and y_label == "S"
    )
    sb = sum(
        values[component]
        for (x_label, y_label), values in atomic_integrals.items()
        if x_label == "S" and y_label != "S"
    )
    bb = sum(
        values[component]
        for (x_label, y_label), values in atomic_integrals.items()
        if x_label != "S" and y_label != "S"
    )
    return {"SS": float(ss), "BS": float(bs), "SB": float(sb), "BB": float(bb)}


def calculate_transfer_factors_2d(
    fit_result,
    x_regions: MassRegions1D,
    y_regions: MassRegions1D | None = None,
) -> TransferFactors2D:
    """计算二维容斥 transfer coefficients 及其协方差。

    ``fit_result`` 需要 ``component_models``（每个分量提供 ``name`` 和归一化的
    ``x_pdf`` / ``y_pdf``，各自带 ``region_integral(interval, parameter_values)``）、
    ``parameter_names`` / ``parameter_values`` / ``parameter_covariance``、
    ``x_edges`` 和 ``y_edges``。``y_regions=None`` 表示两轴共用同一套区间。
    """
    if y_regions is None:
        y_regions = x_regions

    x_lo, x_hi = float(np.min(fit_result.x_edges)), float(np.max(fit_result.x_edges))
    y_lo, y_hi = float(np.min(fit_result.y_edges)), float(np.max(fit_result.y_edges))
    x_regions.validate_within(x_lo, x_hi)
    y_regions.validate_within(y_lo, y_hi)

    parameter_names = list(fit_result.parameter_names)
    parameter_values = np.asarray(fit_result.parameter_values, dtype=float)
    covariance = np.asarray(fit_result.parameter_covariance, dtype=float)
    if covariance.shape != (len(parameter_names), len(parameter_names)):
        raise ValueError(
            "calculate_transfer_factors_2d: covariance 形状 "
            f"{covariance.shape} 与参数数 {len(parameter_names)} 不匹配"
        )

    components = {
        component.name: component for component in fit_result.component_models
    }
    for name in COMPONENT_NAMES:
        if name not in components:
            raise ValueError(
                f"calculate_transfer_factors_2d: 缺少拟合分量 {name}，"
                f"现有 {sorted(components)}"
            )

    x_intervals = x_regions.region_intervals()
    y_intervals = y_regions.region_intervals()
    atomic_keys = _atomic_region_keys(x_regions, y_regions)
    if len(atomic_keys) != len(set(atomic_keys)):
        raise ValueError("calculate_transfer_factors_2d: 原子区域 key 出现重复")

    def atomic_integrals_at(
        parameter_dict: Mapping[str, float],
    ) -> dict[tuple[str, str], dict[str, float]]:
        """在给定参数下计算每个原子区域内每个分量的归一化 PDF 积分。"""
        integrals: dict[tuple[str, str], dict[str, float]] = {}
        for (x_label, y_label) in atomic_keys:
            x_low, x_high = x_intervals[x_label]
            y_low, y_high = y_intervals[y_label]
            per_component = {}
            for component_name, component in components.items():
                x_integral = component.x_pdf.region_integral(
                    (x_low, x_high), parameter_dict
                )
                y_integral = component.y_pdf.region_integral(
                    (y_low, y_high), parameter_dict
                )
                per_component[component_name] = float(x_integral * y_integral)
            integrals[(x_label, y_label)] = per_component
        return integrals

    def weights_from(
        integrals: Mapping[tuple[str, str], Mapping[str, float]],
    ) -> np.ndarray:
        """由原子区域积分计算 (w_H, w_V, w_C)。"""
        aggregated = {
            component_name: _aggregated_integrals(integrals, component_name)
            for component_name in COMPONENT_NAMES
        }
        bxsy = aggregated["BxSy"]
        sxby = aggregated["SxBy"]
        bxby = aggregated["BxBy"]
        if bxsy["BS"] <= 0.0 or sxby["SB"] <= 0.0 or bxby["BB"] <= 0.0:
            raise ValueError(
                "calculate_transfer_factors_2d: 聚合 sideband 积分非正 "
                f"(BS={bxsy['BS']}, SB={sxby['SB']}, BB={bxby['BB']})"
            )
        w_h = bxsy["SS"] / bxsy["BS"]
        w_v = sxby["SS"] / sxby["SB"]
        w_c = (
            bxby["SS"] - w_h * bxby["BS"] - w_v * bxby["SB"]
        ) / bxby["BB"]
        return np.asarray([w_h, w_v, w_c], dtype=float)

    nominal = dict(zip(parameter_names, parameter_values))
    atomic_integrals = atomic_integrals_at(nominal)
    nominal_weights = weights_from(atomic_integrals)
    aggregated_integrals = {
        component_name: _aggregated_integrals(atomic_integrals, component_name)
        for component_name in COMPONENT_NAMES
    }

    signal_ss = atomic_integrals[("S", "S")]["SxSy"]
    if signal_ss <= 0.0:
        raise ValueError("calculate_transfer_factors_2d: SxSy 在 SS 的积分非正")
    signal_leakage = {
        key: values["SxSy"] / signal_ss
        for key, values in atomic_integrals.items()
        if key != ("S", "S")
    }

    # 数值 Jacobian：对全部拟合参数做中心差分。
    jacobian = np.zeros((3, len(parameter_names)), dtype=float)
    gradient: dict[str, np.ndarray] = {}
    for index, name in enumerate(parameter_names):
        step = 1.0e-5 * max(abs(nominal[name]), 1.0)
        up = dict(nominal)
        down = dict(nominal)
        up[name] = nominal[name] + step
        down[name] = nominal[name] - step
        column = (
            weights_from(atomic_integrals_at(up))
            - weights_from(atomic_integrals_at(down))
        ) / (2.0 * step)
        jacobian[:, index] = column
        gradient[name] = column

    weight_covariance = jacobian @ covariance @ jacobian.T
    weight_covariance = 0.5 * (weight_covariance + weight_covariance.T)
    if not np.all(np.isfinite(weight_covariance)) or np.any(
        np.diag(weight_covariance) < 0.0
    ):
        raise ValueError(
            "calculate_transfer_factors_2d: 传播得到的权重协方差无效 "
            f"({weight_covariance})"
        )

    diagonal = np.sqrt(np.diag(weight_covariance))
    weight_correlation = np.zeros_like(weight_covariance)
    nonzero = diagonal > 0.0
    if np.any(nonzero):
        scale = np.where(nonzero, diagonal, 1.0)
        weight_correlation = weight_covariance / np.outer(scale, scale)

    w_h, w_v, w_c = (float(value) for value in nominal_weights)

    return TransferFactors2D(
        x_regions=x_regions,
        y_regions=y_regions,
        atomic_region_integrals=atomic_integrals,
        aggregated_region_integrals=aggregated_integrals,
        w_H=w_h,
        w_V=w_v,
        w_C=w_c,
        weight_covariance=weight_covariance,
        weight_correlation=weight_correlation,
        parameter_gradient=gradient,
        signal_leakage_by_region=signal_leakage,
        factorization_closure=float(w_c + w_h * w_v),
    )
