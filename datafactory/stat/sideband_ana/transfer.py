"""Mass-region definitions and fitted-background transfer coefficients.

The public objects in this module name the physics regions explicitly.  A
sideband is never inferred from histogram bin numbers: callers provide the
mass intervals used by the event selection, and the fitted PDF is integrated
over those same intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .fit import FitResult1D, FitResult2D


@dataclass(frozen=True)
class MassRegions1D:
    """One signal interval and one or two disjoint mass sidebands.

    ``sideband_low`` and ``sideband_high`` describe the physical location of
    the sideband relative to the signal peak.  At least one must be present.
    Intervals follow the histogram convention ``[low, high)``.
    """

    signal: tuple[float, float]
    sideband_low: tuple[float, float] | None = None
    sideband_high: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        intervals = {
            "signal": self.signal,
            "sideband_low": self.sideband_low,
            "sideband_high": self.sideband_high,
        }
        if self.sideband_low is None and self.sideband_high is None:
            raise ValueError("MassRegions1D requires at least one sideband")
        for name, interval in intervals.items():
            if interval is None:
                continue
            if len(interval) != 2 or not np.all(np.isfinite(interval)):
                raise ValueError(f"{name} must contain two finite mass boundaries")
            if not interval[0] < interval[1]:
                raise ValueError(f"{name} must satisfy low < high")
        if self.sideband_low is not None and not self.sideband_low[1] <= self.signal[0]:
            raise ValueError("sideband_low must lie below and not overlap signal")
        if self.sideband_high is not None and not self.signal[1] <= self.sideband_high[0]:
            raise ValueError("sideband_high must lie above and not overlap signal")
        if (
            self.sideband_low is not None
            and self.sideband_high is not None
            and not self.sideband_low[1] <= self.sideband_high[0]
        ):
            raise ValueError("low and high sidebands must not overlap")

    def validate_within(self, fit_range: tuple[float, float]) -> None:
        """Check that all selected regions are inside the fitted mass range."""
        if len(fit_range) != 2 or not np.all(np.isfinite(fit_range)) or not fit_range[0] < fit_range[1]:
            raise ValueError("fit_range must contain two increasing finite boundaries")
        for name, interval in (
            ("signal", self.signal),
            ("sideband_low", self.sideband_low),
            ("sideband_high", self.sideband_high),
        ):
            if interval is not None and not (fit_range[0] <= interval[0] < interval[1] <= fit_range[1]):
                raise ValueError(f"{name}={interval} lies outside fit_range={fit_range}")


def regions_from_offsets(
    peak_mean: float,
    *,
    signal_half_width: float,
    sideband_low_offset: float | None = None,
    sideband_high_offset: float | None = None,
) -> MassRegions1D:
    """Convert peak-centred offsets into the explicit interval contract.

    Offsets are non-negative distances from the fitted peak to each sideband
    centre.  All three windows use ``signal_half_width``; callers needing
    unequal widths should instantiate :class:`MassRegions1D` directly.
    """
    values = [peak_mean, signal_half_width]
    values.extend(value for value in (sideband_low_offset, sideband_high_offset) if value is not None)
    if not np.all(np.isfinite(values)) or signal_half_width <= 0.0:
        raise ValueError("peak, half-width, and supplied offsets must be finite and positive")
    if sideband_low_offset is not None and sideband_low_offset <= signal_half_width * 2.0:
        raise ValueError("low-sideband centre offset must exceed two window half-widths")
    if sideband_high_offset is not None and sideband_high_offset <= signal_half_width * 2.0:
        raise ValueError("high-sideband centre offset must exceed two window half-widths")
    return MassRegions1D(
        signal=(peak_mean - signal_half_width, peak_mean + signal_half_width),
        sideband_low=None if sideband_low_offset is None else (
            peak_mean - sideband_low_offset - signal_half_width,
            peak_mean - sideband_low_offset + signal_half_width,
        ),
        sideband_high=None if sideband_high_offset is None else (
            peak_mean + sideband_high_offset - signal_half_width,
            peak_mean + sideband_high_offset + signal_half_width,
        ),
    )


@dataclass(frozen=True)
class TransferFactor1D:
    """Background integral ratio from the combined sideband to signal region."""

    regions: MassRegions1D
    integral_signal: float
    integral_sideband_low: float | None
    integral_sideband_high: float | None
    integral_sideband_combined: float
    r_low: float | None
    r_high: float | None
    r_combined: float
    variance_r_combined: float
    sigma_r_combined: float
    parameter_gradient: np.ndarray


@dataclass(frozen=True)
class TransferFactors2D:
    """Horizontal, vertical, and signed corner inclusion-exclusion weights."""

    x_regions: MassRegions1D
    y_regions: MassRegions1D
    atomic_region_integrals: dict[str, dict[str, float]]
    aggregated_region_integrals: dict[str, dict[str, float]]
    w_H: float
    w_V: float
    w_C: float
    weight_covariance: np.ndarray
    weight_correlation: np.ndarray
    parameter_gradient: np.ndarray
    signal_leakage_by_region: dict[str, float]
    factorization_closure: float


def _integrate_interval(evaluator, interval: tuple[float, float], nodes: np.ndarray, weights: np.ndarray) -> float:
    """Integrate a fitted one-dimensional density with Gauss--Legendre nodes."""
    low, high = interval
    masses = 0.5 * (low + high) + 0.5 * (high - low) * nodes
    values = np.asarray(evaluator(masses), dtype=float)
    if values.shape != masses.shape or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise RuntimeError("fitted PDF is non-finite or negative inside a mass region")
    return float(0.5 * (high - low) * np.dot(weights, values))


def calculate_transfer_factor_1d(fit_result: "FitResult1D", regions: MassRegions1D) -> TransferFactor1D:
    """Integrate the final zfit background and propagate its fit covariance."""
    regions.validate_within(fit_result.background_fit_range)
    nodes, weights = np.polynomial.legendre.leggauss(64)

    def integrals(parameter_values: np.ndarray) -> tuple[float, float | None, float | None, float]:
        evaluator = lambda masses: fit_result.evaluate_background(masses, parameter_values)
        signal = _integrate_interval(evaluator, regions.signal, nodes, weights)
        low = None if regions.sideband_low is None else _integrate_interval(evaluator, regions.sideband_low, nodes, weights)
        high = None if regions.sideband_high is None else _integrate_interval(evaluator, regions.sideband_high, nodes, weights)
        combined = (0.0 if low is None else low) + (0.0 if high is None else high)
        if signal <= 0.0 or combined <= 0.0:
            raise RuntimeError("signal and combined-sideband background integrals must be positive")
        return signal, low, high, combined

    parameters = np.asarray(fit_result.parameter_values, dtype=float)
    integral_signal, integral_low, integral_high, integral_combined = integrals(parameters)
    ratio = integral_signal / integral_combined
    gradient = np.zeros_like(parameters)
    for index, value in enumerate(parameters):
        step = 1.0e-5 * max(abs(value), 1.0)
        values_up, values_down = parameters.copy(), parameters.copy()
        values_up[index] += step
        values_down[index] -= step
        signal_up, _, _, side_up = integrals(values_up)
        signal_down, _, _, side_down = integrals(values_down)
        gradient[index] = (signal_up / side_up - signal_down / side_down) / (2.0 * step)
    covariance = np.asarray(fit_result.parameter_covariance, dtype=float)
    variance = float(gradient @ covariance @ gradient)
    if not np.isfinite(variance) or variance < -1.0e-12:
        raise RuntimeError(f"invalid propagated 1-D transfer variance {variance}")
    variance = max(variance, 0.0)
    return TransferFactor1D(
        regions=regions,
        integral_signal=integral_signal,
        integral_sideband_low=integral_low,
        integral_sideband_high=integral_high,
        integral_sideband_combined=integral_combined,
        r_low=None if integral_low is None else integral_signal / integral_low,
        r_high=None if integral_high is None else integral_signal / integral_high,
        r_combined=ratio,
        variance_r_combined=variance,
        sigma_r_combined=float(np.sqrt(variance)),
        parameter_gradient=gradient,
    )


def calculate_transfer_factors_2d(
    fit_result: "FitResult2D",
    x_regions: MassRegions1D,
    y_regions: MassRegions1D | None = None,
) -> TransferFactors2D:
    """Calculate the RooFit-equivalent nine-region inclusion-exclusion weights."""
    y_regions = x_regions if y_regions is None else y_regions
    x_regions.validate_within(fit_result.x_fit_range)
    y_regions.validate_within(fit_result.y_fit_range)

    def weights_and_integrals(parameter_values: np.ndarray):
        x_parts = fit_result.axis_region_integrals("x", x_regions, parameter_values)
        y_parts = fit_result.axis_region_integrals("y", y_regions, parameter_values)
        x_labels = [label for label in ("S", "L", "H") if label in x_parts["signal"]]
        y_labels = [label for label in ("S", "L", "H") if label in y_parts["signal"]]
        components = {
            "SxSy": (x_parts["signal"], y_parts["signal"]),
            "BxSy": (x_parts["background"], y_parts["signal"]),
            "SxBy": (x_parts["signal"], y_parts["background"]),
            "BxBy": (x_parts["background"], y_parts["background"]),
        }
        atomic = {
            component: {
                x_label + y_label: float(x_values[x_label] * y_values[y_label])
                for x_label in x_labels
                for y_label in y_labels
            }
            for component, (x_values, y_values) in components.items()
        }
        aggregated = {}
        for component, values in atomic.items():
            aggregated[component] = {
                "SS": values["SS"],
                "BS": sum(values.get(label + "S", 0.0) for label in ("L", "H")),
                "SB": sum(values.get("S" + label, 0.0) for label in ("L", "H")),
                "BB": sum(values.get(x_label + y_label, 0.0) for x_label in ("L", "H") for y_label in ("L", "H")),
            }
        horizontal = aggregated["BxSy"]["SS"] / aggregated["BxSy"]["BS"]
        vertical = aggregated["SxBy"]["SS"] / aggregated["SxBy"]["SB"]
        corner_values = aggregated["BxBy"]
        corner = (
            corner_values["SS"]
            - horizontal * corner_values["BS"]
            - vertical * corner_values["SB"]
        ) / corner_values["BB"]
        return np.asarray([horizontal, vertical, corner]), atomic, aggregated

    parameters = np.asarray(fit_result.parameter_values, dtype=float)
    nominal, atomic, aggregated = weights_and_integrals(parameters)
    jacobian = np.zeros((3, parameters.size), dtype=float)
    for index, value in enumerate(parameters):
        step = 1.0e-5 * max(abs(value), 1.0)
        values_up, values_down = parameters.copy(), parameters.copy()
        values_up[index] += step
        values_down[index] -= step
        jacobian[:, index] = (
            weights_and_integrals(values_up)[0] - weights_and_integrals(values_down)[0]
        ) / (2.0 * step)
    covariance = jacobian @ np.asarray(fit_result.parameter_covariance, dtype=float) @ jacobian.T
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) < -1.0e-12):
        raise RuntimeError("invalid propagated 2-D transfer covariance")
    covariance[np.diag_indices(3)] = np.maximum(np.diag(covariance), 0.0)
    errors = np.sqrt(np.diag(covariance))
    correlation = np.divide(
        covariance,
        np.outer(errors, errors),
        out=np.zeros_like(covariance),
        where=np.outer(errors, errors) > 0.0,
    )
    return TransferFactors2D(
        x_regions=x_regions,
        y_regions=y_regions,
        atomic_region_integrals=atomic,
        aggregated_region_integrals=aggregated,
        w_H=float(nominal[0]),
        w_V=float(nominal[1]),
        w_C=float(nominal[2]),
        weight_covariance=covariance,
        weight_correlation=correlation,
        parameter_gradient=jacobian,
        signal_leakage_by_region=atomic["SxSy"],
        factorization_closure=float(nominal[2] + nominal[0] * nominal[1]),
    )
