"""Sideband subtraction 的纯数学实现：一维单边带和二维容斥式。

输入是区域计数（含 Sumw2 方差）和 :mod:`transfer` 输出的 transfer
factor / coefficients；不依赖任何拟合栈。保留负 bin，不做截断。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .transfer import TransferFactor1D, TransferFactors2D

__all__ = [
    "SubtractionResult",
    "subtract_sideband_1d",
    "subtract_sideband_2d",
]


@dataclass(frozen=True)
class SubtractionResult:
    """sideband 减除结果；所有数组与输入观测同长度。"""

    observed_signal_region: np.ndarray
    observed_variance: np.ndarray
    atomic_sideband_counts: dict
    aggregated_sideband_counts: dict
    estimated_background: np.ndarray
    background_variance: np.ndarray
    subtracted_signal: np.ndarray
    signal_variance: np.ndarray
    negative_bin_mask: np.ndarray


def _validated_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"subtract_sideband: {name} 含非有限值")
    return array


def _validated_variance(values, name: str) -> np.ndarray:
    array = _validated_array(values, name)
    if np.any(array < 0.0):
        raise ValueError(f"subtract_sideband: {name} 含负方差")
    return array


# ---------------------------------------------------------------------------
# 8.1 一维减除
# ---------------------------------------------------------------------------


def subtract_sideband_1d(
    signal_counts,
    signal_variances,
    *,
    low_sideband_counts=None,
    low_sideband_variances=None,
    high_sideband_counts=None,
    high_sideband_variances=None,
    transfer: TransferFactor1D,
) -> SubtractionResult:
    """按 N_bkg = r · N_B 做一维 sideband 减除。

    必须提供 transfer 所使用的那些 sideband（``integral_sideband_low/high``
    非空的边带），且不得提供 transfer 未使用的边带。方差按
    V_bkg = r² V_B + N_B² V_r 传播。
    """
    counts_s = _validated_array(signal_counts, "signal_counts")
    variance_s = _validated_variance(signal_variances, "signal_variances")
    if counts_s.shape != variance_s.shape:
        raise ValueError(
            f"subtract_sideband_1d: counts/variances 形状不一致 "
            f"{counts_s.shape} vs {variance_s.shape}"
        )

    sideband_inputs = {
        "L": (low_sideband_counts, low_sideband_variances),
        "H": (high_sideband_counts, high_sideband_variances),
    }
    transfer_used = {
        "L": transfer.integral_sideband_low is not None,
        "H": transfer.integral_sideband_high is not None,
    }
    for label, used in transfer_used.items():
        counts, variances = sideband_inputs[label]
        provided = counts is not None or variances is not None
        if used and not provided:
            raise ValueError(
                f"subtract_sideband_1d: transfer 使用了 {label} sideband，"
                "必须提供对应的计数和方差"
            )
        if not used and provided:
            raise ValueError(
                f"subtract_sideband_1d: transfer 未使用 {label} sideband，"
                "不应提供对应的计数或方差"
            )

    sideband_counts = {}
    sideband_variances = {}
    for label, used in transfer_used.items():
        if not used:
            continue
        counts, variances = sideband_inputs[label]
        counts = _validated_array(counts, f"{label}_sideband_counts")
        variances = _validated_variance(variances, f"{label}_sideband_variances")
        if counts.shape != counts_s.shape or variances.shape != counts_s.shape:
            raise ValueError(
                "subtract_sideband_1d: sideband 数组形状必须与 signal 区域一致 "
                f"({counts_s.shape})"
            )
        sideband_counts[label] = counts
        sideband_variances[label] = variances

    counts_b = sum(sideband_counts.values())
    variance_b = sum(sideband_variances.values())

    r = float(transfer.r_combined)
    variance_r = float(transfer.variance_r_combined)

    background = r * counts_b
    background_variance = r * r * variance_b + counts_b * counts_b * variance_r
    subtracted = counts_s - background
    subtracted_variance = variance_s + background_variance

    return SubtractionResult(
        observed_signal_region=counts_s,
        observed_variance=variance_s,
        atomic_sideband_counts=sideband_counts,
        aggregated_sideband_counts={"B": np.asarray(counts_b, dtype=float)},
        estimated_background=np.asarray(background, dtype=float),
        background_variance=np.asarray(background_variance, dtype=float),
        subtracted_signal=np.asarray(subtracted, dtype=float),
        signal_variance=np.asarray(subtracted_variance, dtype=float),
        negative_bin_mask=np.asarray(subtracted < 0.0, dtype=bool),
    )


# ---------------------------------------------------------------------------
# 8.2 二维减除
# ---------------------------------------------------------------------------


def subtract_sideband_2d(
    region_counts: Mapping[tuple[str, str], np.ndarray],
    region_variances: Mapping[tuple[str, str], np.ndarray],
    transfer: TransferFactors2D,
) -> SubtractionResult:
    """按容斥式 N_bkg = w_H·N_BS + w_V·N_SB + w_C·N_BB 做二维减除。

    ``region_counts`` / ``region_variances`` 的 key 必须与 transfer 的原子
    区域一一对应（例如双边带的九个 key 或单侧边带的四个 key）。方差按
    V_bkg = w^T V_N w + N^T Cov(w) N 逐 bin 传播。
    """
    expected_keys = set(transfer.atomic_region_integrals)
    provided_keys = set(region_counts)
    if provided_keys != expected_keys:
        missing = sorted(expected_keys - provided_keys)
        unexpected = sorted(provided_keys - expected_keys)
        raise ValueError(
            "subtract_sideband_2d: 区域 key 与 transfer 不一致 "
            f"(缺失 {missing}，多余 {unexpected})"
        )

    counts = {}
    variances = {}
    reference_shape = None
    for key in expected_keys:
        counts[key] = _validated_array(region_counts[key], f"counts{key}")
        variances[key] = _validated_variance(region_variances[key], f"variances{key}")
        if counts[key].shape != variances[key].shape:
            raise ValueError(
                f"subtract_sideband_2d: {key} 的 counts/variances 形状不一致"
            )
        if reference_shape is None:
            reference_shape = counts[key].shape
        elif counts[key].shape != reference_shape:
            raise ValueError(
                "subtract_sideband_2d: 所有区域数组形状必须一致 "
                f"{counts[key].shape} vs {reference_shape}"
            )

    # ponytail: 每轴至少一个 sideband 由 MassRegions1D 保证，聚合必非空。
    def aggregate(source, predicate):
        selected = [source[key] for key in sorted(expected_keys) if predicate(key)]
        return np.sum(selected, axis=0)

    counts_ss = counts[("S", "S")]
    counts_bs = aggregate(counts, lambda key: key[0] != "S" and key[1] == "S")
    counts_sb = aggregate(counts, lambda key: key[0] == "S" and key[1] != "S")
    counts_bb = aggregate(counts, lambda key: key[0] != "S" and key[1] != "S")
    variance_ss = variances[("S", "S")]
    variance_bs = aggregate(variances, lambda key: key[0] != "S" and key[1] == "S")
    variance_sb = aggregate(variances, lambda key: key[0] == "S" and key[1] != "S")
    variance_bb = aggregate(variances, lambda key: key[0] != "S" and key[1] != "S")

    w_h = float(transfer.w_H)
    w_v = float(transfer.w_V)
    w_c = float(transfer.w_C)

    background = w_h * counts_bs + w_v * counts_sb + w_c * counts_bb

    # 计数方差部分（各聚合区域计数方差独立求和）。
    background_variance = (
        w_h * w_h * variance_bs
        + w_v * w_v * variance_sb
        + w_c * w_c * variance_bb
    )
    # 权重协方差部分：N^T Cov(w) N，N = (N_BS, N_SB, N_BB) 逐 bin。
    stacked = np.stack([counts_bs, counts_sb, counts_bb], axis=-1)
    background_variance = background_variance + np.einsum(
        "...i,ij,...j->...", stacked, transfer.weight_covariance, stacked
    )

    subtracted = counts_ss - background
    subtracted_variance = variance_ss + background_variance

    atomic_sideband_counts = {
        key: counts[key] for key in sorted(expected_keys) if key != ("S", "S")
    }

    return SubtractionResult(
        observed_signal_region=counts_ss,
        observed_variance=variance_ss,
        atomic_sideband_counts=atomic_sideband_counts,
        aggregated_sideband_counts={
            "BS": np.asarray(counts_bs, dtype=float),
            "SB": np.asarray(counts_sb, dtype=float),
            "BB": np.asarray(counts_bb, dtype=float),
        },
        estimated_background=np.asarray(background, dtype=float),
        background_variance=np.asarray(background_variance, dtype=float),
        subtracted_signal=np.asarray(subtracted, dtype=float),
        signal_variance=np.asarray(subtracted_variance, dtype=float),
        negative_bin_mask=np.asarray(subtracted < 0.0, dtype=bool),
    )
