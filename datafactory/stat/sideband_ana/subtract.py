"""Sideband subtraction in a target observable distinct from fitted mass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transfer import TransferFactor1D, TransferFactors2D


@dataclass(frozen=True)
class SubtractionResult:
    """Observed, estimated-background, and background-subtracted bin values."""

    observed_signal_region: np.ndarray
    observed_variance: np.ndarray
    atomic_sideband_counts: dict[str, np.ndarray]
    aggregated_sideband_counts: dict[str, np.ndarray]
    estimated_background: np.ndarray
    background_variance: np.ndarray
    subtracted_signal: np.ndarray
    signal_variance: np.ndarray
    negative_bin_mask: np.ndarray


def _validated_histogram_pair(counts, variances, name: str, expected_shape=None) -> tuple[np.ndarray, np.ndarray]:
    """Validate one binned observable and its Sumw2 marginal variances."""
    values = np.asarray(counts, dtype=float)
    errors2 = np.asarray(variances, dtype=float)
    if values.ndim != 1 or values.shape != errors2.shape:
        raise ValueError(f"{name} counts and variances must be same-shape 1-D arrays")
    if expected_shape is not None and values.shape != expected_shape:
        raise ValueError(f"{name} shape {values.shape} does not match {expected_shape}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(errors2)) or np.any(errors2 < 0.0):
        raise ValueError(f"{name} contains non-finite values or negative variances")
    return values, errors2


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
    """Subtract a single- or double-sideband estimate from another observable."""
    signal, signal_var = _validated_histogram_pair(signal_counts, signal_variances, "signal")
    supplied_low = low_sideband_counts is not None or low_sideband_variances is not None
    supplied_high = high_sideband_counts is not None or high_sideband_variances is not None
    if supplied_low != (low_sideband_counts is not None and low_sideband_variances is not None):
        raise ValueError("low-sideband counts and variances must be supplied together")
    if supplied_high != (high_sideband_counts is not None and high_sideband_variances is not None):
        raise ValueError("high-sideband counts and variances must be supplied together")
    if supplied_low != (transfer.regions.sideband_low is not None) or supplied_high != (transfer.regions.sideband_high is not None):
        raise ValueError("observable sideband inputs must match the transfer-region configuration")
    atomic_counts, atomic_variances = {}, {}
    if supplied_low:
        atomic_counts["L"], atomic_variances["L"] = _validated_histogram_pair(
            low_sideband_counts, low_sideband_variances, "low sideband", signal.shape
        )
    if supplied_high:
        atomic_counts["H"], atomic_variances["H"] = _validated_histogram_pair(
            high_sideband_counts, high_sideband_variances, "high sideband", signal.shape
        )
    combined = sum(atomic_counts.values(), np.zeros_like(signal))
    combined_var = sum(atomic_variances.values(), np.zeros_like(signal_var))
    background = transfer.r_combined * combined
    background_var = (
        transfer.r_combined**2 * combined_var
        + combined**2 * transfer.variance_r_combined
    )
    subtracted = signal - background
    subtracted_var = signal_var + background_var
    return SubtractionResult(
        observed_signal_region=signal,
        observed_variance=signal_var,
        atomic_sideband_counts=atomic_counts,
        aggregated_sideband_counts={"B": combined},
        estimated_background=background,
        background_variance=background_var,
        subtracted_signal=subtracted,
        signal_variance=subtracted_var,
        negative_bin_mask=subtracted < 0.0,
    )


def subtract_sideband_2d(
    region_counts: dict[tuple[str, str], np.ndarray],
    region_variances: dict[tuple[str, str], np.ndarray],
    transfer: TransferFactors2D,
) -> SubtractionResult:
    """Apply the nine-region horizontal/vertical/corner inclusion-exclusion formula."""
    x_labels = ["S"] + (["L"] if transfer.x_regions.sideband_low is not None else []) + (["H"] if transfer.x_regions.sideband_high is not None else [])
    y_labels = ["S"] + (["L"] if transfer.y_regions.sideband_low is not None else []) + (["H"] if transfer.y_regions.sideband_high is not None else [])
    expected = {(x_label, y_label) for x_label in x_labels for y_label in y_labels}
    if set(region_counts) != expected or set(region_variances) != expected:
        raise ValueError(f"region maps must contain exactly {sorted(expected)}")
    validated_counts, validated_variances, shape = {}, {}, None
    for key in sorted(expected):
        counts, variances = _validated_histogram_pair(
            region_counts[key], region_variances[key], f"region {key}", shape
        )
        shape = counts.shape
        validated_counts[key], validated_variances[key] = counts, variances
    zeros = np.zeros(shape, dtype=float)
    horizontal_keys = [(label, "S") for label in x_labels if label != "S"]
    vertical_keys = [("S", label) for label in y_labels if label != "S"]
    corner_keys = [(x_label, y_label) for x_label in x_labels if x_label != "S" for y_label in y_labels if y_label != "S"]
    aggregated_counts = {
        "BS": sum((validated_counts[key] for key in horizontal_keys), zeros.copy()),
        "SB": sum((validated_counts[key] for key in vertical_keys), zeros.copy()),
        "BB": sum((validated_counts[key] for key in corner_keys), zeros.copy()),
    }
    aggregated_variances = {
        "BS": sum((validated_variances[key] for key in horizontal_keys), zeros.copy()),
        "SB": sum((validated_variances[key] for key in vertical_keys), zeros.copy()),
        "BB": sum((validated_variances[key] for key in corner_keys), zeros.copy()),
    }
    weights = np.asarray([transfer.w_H, transfer.w_V, transfer.w_C], dtype=float)
    region_vector = np.stack([aggregated_counts[name] for name in ("BS", "SB", "BB")])
    variance_vector = np.stack([aggregated_variances[name] for name in ("BS", "SB", "BB")])
    background = np.einsum("i,ij->j", weights, region_vector)
    counting_variance = np.einsum("i,ij->j", weights**2, variance_vector)
    weight_variance = np.einsum("ib,ij,jb->b", region_vector, transfer.weight_covariance, region_vector)
    background_var = counting_variance + weight_variance
    signal = validated_counts[("S", "S")] - background
    signal_var = validated_variances[("S", "S")] + background_var
    return SubtractionResult(
        observed_signal_region=validated_counts[("S", "S")],
        observed_variance=validated_variances[("S", "S")],
        atomic_sideband_counts={"".join(key): value for key, value in validated_counts.items() if key != ("S", "S")},
        aggregated_sideband_counts=aggregated_counts,
        estimated_background=background,
        background_variance=background_var,
        subtracted_signal=signal,
        signal_variance=signal_var,
        negative_bin_mask=signal < 0.0,
    )
