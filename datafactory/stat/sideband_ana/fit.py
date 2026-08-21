"""SymbolFit-seeded zfit models for one- and two-dimensional mass spectra.

SymbolFit is used only to select a smooth, non-negative background expression
and its numerical starting point.  zfit performs the final parameter estimate.
The one-dimensional fit can either profile the selected background parameters
or keep their SymbolFit estimate fixed for compatibility studies.  The
two-dimensional fit follows the four physical components used by the DELPHI
analysis: ``SxSy``, ``BxSy``, ``SxBy``, and ``BxBy``.
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import erf

from .transfer import MassRegions1D


@dataclass(frozen=True)
class SymbolFitBackgroundSeed:
    """Selected symbolic background expression before the final zfit stage."""

    parameterized_formula: str
    fitted_formula: str
    parameter_names: tuple[str, ...]
    parameter_values: np.ndarray
    parameter_covariance: np.ndarray
    training_chi2: float
    training_ndf: int
    selection_score: float


@dataclass(frozen=True)
class FitResult1D:
    """Complete binned one-dimensional Signal+Background fit product."""

    mass_edges: np.ndarray
    observed_counts: np.ndarray
    observed_variances: np.ndarray
    fit_range: tuple[float, float]
    background_fit_range: tuple[float, float]
    background_profiled: bool
    parameter_names: tuple[str, ...]
    parameter_values: np.ndarray
    parameter_covariance: np.ndarray
    background_parameter_indices: tuple[int, ...]
    peak_mean: float
    peak_mean_variance: float
    background_formula: str
    background_parameterized_formula: str
    symbolfit_initial_values: dict[str, float]
    model_counts: np.ndarray
    background_counts: np.ndarray
    dense_mass: np.ndarray
    dense_model: np.ndarray
    dense_background: np.ndarray
    chi2: float
    ndf: int
    converged: bool
    background_model: str

    def evaluate_background(self, masses, parameter_values=None):
        """Evaluate the fitted background density in events per mass unit."""
        values = self.parameter_values if parameter_values is None else np.asarray(parameter_values, dtype=float)
        background_parameters = {
            name.split("background:", 1)[1]: values[index]
            for name, index in zip(
                (self.parameter_names[index] for index in self.background_parameter_indices),
                self.background_parameter_indices,
            )
        }
        evaluated = evaluate_symbolfit_expression(
            self.background_parameterized_formula,
            np.asarray(masses, dtype=float),
            background_parameters,
            np,
        )
        return np.full_like(np.asarray(masses, dtype=float), float(evaluated)) if np.ndim(evaluated) == 0 else evaluated


@dataclass(frozen=True)
class FitResult2D:
    """Four-component simultaneous extended-Poisson mass-plane fit product."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    observed_counts_by_period: np.ndarray
    model_counts_by_period: np.ndarray
    component_names: tuple[str, ...]
    component_yields_by_period: np.ndarray
    parameter_names: tuple[str, ...]
    parameter_values: np.ndarray
    parameter_covariance: np.ndarray
    x_signal_parameter_indices: tuple[int, int, int]
    y_signal_parameter_indices: tuple[int, int, int]
    x_background_parameter_indices: tuple[int, int, int]
    y_background_parameter_indices: tuple[int, int, int]
    x_peak_mean: float
    y_peak_mean: float
    x_fit_range: tuple[float, float]
    y_fit_range: tuple[float, float]
    symbolfit_initial_values_x: dict[str, float]
    symbolfit_initial_values_y: dict[str, float]
    nll_value: float
    fit_nbins: int
    n_periods: int
    converged: bool
    x_projection_observed: np.ndarray
    x_projection_model: np.ndarray
    x_projection_background: np.ndarray
    x_projection_dense_mass: np.ndarray
    x_projection_dense_model: np.ndarray
    x_projection_dense_background: np.ndarray
    y_projection_observed: np.ndarray
    y_projection_model: np.ndarray
    y_projection_background: np.ndarray
    y_projection_dense_mass: np.ndarray
    y_projection_dense_model: np.ndarray
    y_projection_dense_background: np.ndarray
    component_models: tuple[str, ...]

    def axis_region_integrals(self, axis: str, regions: MassRegions1D, parameter_values=None):
        """Return normalized signal/background integrals for S, L, and H."""
        values = self.parameter_values if parameter_values is None else np.asarray(parameter_values, dtype=float)
        if axis == "x":
            fit_range, mean = self.x_fit_range, self.x_peak_mean
            signal_indices = self.x_signal_parameter_indices
            background_indices = self.x_background_parameter_indices
        elif axis == "y":
            fit_range, mean = self.y_fit_range, self.y_peak_mean
            signal_indices = self.y_signal_parameter_indices
            background_indices = self.y_background_parameter_indices
        else:
            raise ValueError("axis must be 'x' or 'y'")
        sigma_narrow, delta_sigma, narrow_fraction = values[list(signal_indices)]
        background_coefficients = values[list(background_indices)]
        intervals = {"S": regions.signal}
        if regions.sideband_low is not None:
            intervals["L"] = regions.sideband_low
        if regions.sideband_high is not None:
            intervals["H"] = regions.sideband_high
        return {
            "signal": {
                label: _double_gaussian_integral(
                    interval,
                    fit_range,
                    mean,
                    sigma_narrow,
                    delta_sigma,
                    narrow_fraction,
                )
                for label, interval in intervals.items()
            },
            "background": {
                label: _exp_chebyshev_integral(interval, fit_range, background_coefficients)
                for label, interval in intervals.items()
            },
        }


def _evaluate_ast(node, x, parameters, backend):
    """Evaluate the strict SymbolFit expression subset for NumPy or TensorFlow."""
    if isinstance(node, ast.Expression):
        return _evaluate_ast(node.body, x, parameters, backend)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "x0":
            return x
        if node.id in parameters:
            return parameters[node.id]
        raise ValueError(f"unknown SymbolFit symbol {node.id!r}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_ast(node.operand, x, parameters, backend)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        left = _evaluate_ast(node.left, x, parameters, backend)
        right = _evaluate_ast(node.right, x, parameters, backend)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        exponent = ast.literal_eval(node.right)
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("SymbolFit powers must be non-negative integers")
        return _evaluate_ast(node.left, x, parameters, backend) ** exponent
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1:
        value = _evaluate_ast(node.args[0], x, parameters, backend)
        if node.func.id == "exp":
            return backend.exp(value)
        if node.func.id == "square":
            return value * value
    raise ValueError(f"unsupported SymbolFit expression node: {ast.dump(node)}")


def evaluate_symbolfit_expression(formula: str, x, parameters: dict[str, object], backend):
    """Evaluate a formula containing only ``+ - * exp square`` and integer powers."""
    tree = ast.parse(formula, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.Dict, ast.List, ast.Tuple)):
            raise ValueError(f"unsupported SymbolFit syntax: {ast.dump(node)}")
    return _evaluate_ast(tree, x, parameters, backend)


def select_symbolfit_background(
    mass_centers,
    density,
    density_errors,
    *,
    fit_range: tuple[float, float],
    excluded_interval: tuple[float, float],
    random_seed: int,
    output_dir: str | os.PathLike | None = None,
    niterations: int = 100,
) -> SymbolFitBackgroundSeed:
    r"""Select the finite non-negative SymbolFit candidate with minimum $\chi^2+2k$."""
    from pysr import PySRRegressor
    import sympy
    from symbolfit.symbolfit import SymbolFit

    centers = np.asarray(mass_centers, dtype=float)
    values = np.asarray(density, dtype=float)
    errors = np.asarray(density_errors, dtype=float)
    fit_mask = (centers >= fit_range[0]) & (centers < fit_range[1])
    excluded = (centers >= excluded_interval[0]) & (centers < excluded_interval[1])
    training = fit_mask & ~excluded
    if centers.ndim != 1 or centers.shape != values.shape or values.shape != errors.shape:
        raise ValueError("SymbolFit centers, density, and errors must be same-shape 1-D arrays")
    if np.count_nonzero(training) < 8 or np.any(errors[training] <= 0.0):
        raise ValueError("SymbolFit requires at least eight training bins with positive uncertainty")
    pysr_config = PySRRegressor(
        model_selection="accuracy",
        niterations=int(niterations),
        maxsize=15,
        binary_operators=["+", "-", "*"],
        unary_operators=["exp", "square(x) = x*x"],
        constraints={"exp": 5},
        nested_constraints={"exp": {"exp": 0, "square": 0}, "square": {"square": 0}},
        extra_sympy_mappings={"square": lambda value: value**2},
        elementwise_loss="loss(y, y_pred, weights) = (y - y_pred)^2 * weights",
    )
    model = SymbolFit(
        x=centers[training].reshape(-1, 1),
        y=values[training].reshape(-1, 1),
        y_up=errors[training].reshape(-1, 1),
        y_down=errors[training].reshape(-1, 1),
        pysr_config=pysr_config,
        max_complexity=15,
        input_rescale=True,
        scale_y_by="mean",
        max_stderr=20,
        fit_y_unc=True,
        random_seed=int(random_seed),
    )
    owned_temporary = None
    if output_dir is None:
        owned_temporary = tempfile.TemporaryDirectory(prefix="datafactory_symbolfit_")
        output_path = Path(owned_temporary.name)
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    work_path = output_path / "symbolfit_work"
    work_path.mkdir(parents=True, exist_ok=True)
    original_directory = Path.cwd()
    try:
        os.chdir(work_path)
        model.fit()
    finally:
        os.chdir(original_directory)
    if model.func_candidates.empty:
        raise RuntimeError("SymbolFit returned no background formula")

    mass_symbol = sympy.symbols("x0")
    dense_mass = np.linspace(fit_range[0], fit_range[1], 2000)
    candidate_records = []
    for _, candidate in model.func_candidates.iterrows():
        parameterized = str(candidate["Parameterized equation, unscaled"])
        fit_parameters = candidate["Parameters: (best-fit, +1, -1)"]
        names = tuple(fit_parameters)
        parameter_values = np.asarray([float(fit_parameters[name][0]) for name in names])
        parameter_map = dict(zip(names, parameter_values))
        try:
            center_prediction = np.asarray(evaluate_symbolfit_expression(parameterized, centers, parameter_map, np), dtype=float)
            dense_prediction = np.asarray(evaluate_symbolfit_expression(parameterized, dense_mass, parameter_map, np), dtype=float)
            if center_prediction.ndim == 0:
                center_prediction = np.full_like(centers, float(center_prediction))
            if dense_prediction.ndim == 0:
                dense_prediction = np.full_like(dense_mass, float(dense_prediction))
            valid = np.all(np.isfinite(center_prediction)) and np.all(np.isfinite(dense_prediction)) and np.all(dense_prediction >= 0.0)
        except (ValueError, TypeError, OverflowError):
            valid = False
            center_prediction = np.full_like(centers, np.nan)
        chi2 = float(np.sum(np.square((values[training] - center_prediction[training]) / errors[training]))) if valid else np.inf
        score = chi2 + 2.0 * float(candidate["Complexity"]) if valid else np.inf
        candidate_records.append((score, chi2, parameterized, names, parameter_values, fit_parameters, candidate))
    finite_candidates = [record for record in candidate_records if np.isfinite(record[0])]
    if not finite_candidates:
        raise RuntimeError("SymbolFit returned no finite non-negative supported formula")
    score, chi2, parameterized, names, parameter_values, fit_parameters, candidate = min(finite_candidates, key=lambda record: record[0])
    covariance = np.zeros((len(names), len(names)), dtype=float)
    for index, name in enumerate(names):
        covariance[index, index] = (0.5 * (abs(float(fit_parameters[name][1])) + abs(float(fit_parameters[name][2])))) ** 2
    for name_pair, covariance_value in candidate["Covariance"].items():
        first, second = (name.strip() for name in name_pair.split(","))
        covariance[names.index(first), names.index(second)] = float(covariance_value)
        covariance[names.index(second), names.index(first)] = float(covariance_value)
    fitted_expression = sympy.sympify(parameterized, locals={"x0": mass_symbol}).subs(
        {sympy.symbols(name): value for name, value in zip(names, parameter_values)}
    )
    if output_dir is not None:
        model.func_candidates["Background selection score"] = [record[0] for record in candidate_records]
        model.save_to_csv(output_dir=str(output_path))
    if owned_temporary is not None:
        owned_temporary.cleanup()
    return SymbolFitBackgroundSeed(
        parameterized_formula=parameterized,
        fitted_formula=str(fitted_expression),
        parameter_names=names,
        parameter_values=parameter_values,
        parameter_covariance=covariance,
        training_chi2=chi2,
        training_ndf=int(candidate["NDF"]),
        selection_score=float(score),
    )


def _double_gaussian_bin_fractions(edges, fit_range, mean, sigma_narrow, delta_sigma, narrow_fraction, backend):
    """Normalized double-Gaussian probability in each supplied mass bin."""
    sqrt_two = np.sqrt(2.0)
    sigma_wide = sigma_narrow + delta_sigma
    erf_function = backend.math.erf if hasattr(backend, "math") else erf
    narrow = 0.5 * (erf_function((edges[1:] - mean) / (sqrt_two * sigma_narrow)) - erf_function((edges[:-1] - mean) / (sqrt_two * sigma_narrow)))
    wide = 0.5 * (erf_function((edges[1:] - mean) / (sqrt_two * sigma_wide)) - erf_function((edges[:-1] - mean) / (sqrt_two * sigma_wide)))
    narrow_norm = 0.5 * (erf_function((fit_range[1] - mean) / (sqrt_two * sigma_narrow)) - erf_function((fit_range[0] - mean) / (sqrt_two * sigma_narrow)))
    wide_norm = 0.5 * (erf_function((fit_range[1] - mean) / (sqrt_two * sigma_wide)) - erf_function((fit_range[0] - mean) / (sqrt_two * sigma_wide)))
    return narrow_fraction * narrow / narrow_norm + (1.0 - narrow_fraction) * wide / wide_norm


def _double_gaussian_integral(interval, fit_range, mean, sigma_narrow, delta_sigma, narrow_fraction):
    """Return the normalized signal probability inside one physical interval.

    The narrow and wide Gaussian fractions are integrated analytically through
    their error functions, then normalized over the full fitted mass range.
    This is the signal analogue of the background-region integral used by the
    transfer-factor calculation.
    """
    return float(_double_gaussian_bin_fractions(np.asarray(interval), fit_range, mean, sigma_narrow, delta_sigma, narrow_fraction, np)[0])


def _exp_chebyshev_integral(interval, fit_range, coefficients):
    """Normalized exp(Chebyshev-3) integral used by the 2-D background."""
    nodes, weights = np.polynomial.legendre.leggauss(48)
    bounds = np.asarray((interval, fit_range), dtype=float)
    widths = bounds[:, 1] - bounds[:, 0]
    masses = 0.5 * (bounds[:, :1] + bounds[:, 1:]) + 0.5 * widths[:, None] * nodes[None, :]
    scaled = 2.0 * (masses - fit_range[0]) / (fit_range[1] - fit_range[0]) - 1.0
    basis = np.stack((scaled, 2.0 * scaled**2 - 1.0, 4.0 * scaled**3 - 3.0 * scaled))
    raw_density = np.exp(np.tensordot(np.asarray(coefficients), basis, axes=(0, 0)))
    integrals = 0.5 * widths * np.sum(weights[None, :] * raw_density, axis=1)
    return float(integrals[0] / integrals[1])


def _exp_chebyshev_density(masses, fit_range, coefficients):
    """Evaluate the normalized exp(Chebyshev-3) background density."""
    masses = np.asarray(masses, dtype=float)
    scaled = 2.0 * (masses - fit_range[0]) / (fit_range[1] - fit_range[0]) - 1.0
    basis = np.stack((scaled, 2.0 * scaled**2 - 1.0, 4.0 * scaled**3 - 3.0 * scaled))
    raw = np.exp(np.asarray(coefficients) @ basis)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    integration_mass = 0.5 * (fit_range[0] + fit_range[1]) + 0.5 * (fit_range[1] - fit_range[0]) * nodes
    integration_scaled = 2.0 * (integration_mass - fit_range[0]) / (fit_range[1] - fit_range[0]) - 1.0
    integration_basis = np.stack((integration_scaled, 2.0 * integration_scaled**2 - 1.0, 4.0 * integration_scaled**3 - 3.0 * integration_scaled))
    raw_normalization = 0.5 * (fit_range[1] - fit_range[0]) * np.dot(weights, np.exp(np.asarray(coefficients) @ integration_basis))
    return raw / raw_normalization


def _validated_spectrum(edges, counts, variances):
    """Validate a binned mass spectrum without changing its statistical content."""
    edges = np.asarray(edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if edges.ndim != 1 or counts.ndim != 1 or variances.shape != counts.shape or edges.size != counts.size + 1:
        raise ValueError("mass edges, counts, and variances have inconsistent one-dimensional shapes")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
        raise ValueError("mass edges must be finite and strictly increasing")
    if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(variances)) or np.any(variances < 0.0):
        raise ValueError("mass counts/variances must be finite and variances non-negative")
    return edges, counts, variances


def fit_mass_spectrum_1d(
    mass_edges,
    counts,
    variances,
    *,
    fit_range: tuple[float, float],
    regions: MassRegions1D,
    signal_model: str = "double_gaussian",
    random_seed: int = 1,
    profile_background: bool = True,
    symbolfit_output_dir: str | os.PathLike | None = None,
    symbolfit_niterations: int = 100,
) -> FitResult1D:
    """Fit a binned mass spectrum with a SymbolFit background and zfit peak."""
    if signal_model != "double_gaussian":
        raise ValueError("the first sideband_ana release supports only signal_model='double_gaussian'")
    edges, observed, observed_variances = _validated_spectrum(mass_edges, counts, variances)
    if not (edges[0] <= fit_range[0] < fit_range[1] <= edges[-1]):
        raise ValueError("fit_range must lie within mass_edges")
    regions.validate_within((float(edges[0]), float(edges[-1])))
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    fit_mask = (centers >= fit_range[0]) & (centers < fit_range[1])
    if np.count_nonzero(fit_mask) < 10 or observed[fit_mask].sum() <= 0.0:
        raise ValueError("one-dimensional fit range is empty or too coarsely binned")
    seed = select_symbolfit_background(
        centers,
        observed / widths,
        np.sqrt(observed_variances) / widths,
        fit_range=(float(edges[0]), float(edges[-1])),
        excluded_interval=regions.signal,
        random_seed=random_seed,
        output_dir=symbolfit_output_dir,
        niterations=symbolfit_niterations,
    )

    import tensorflow as tf
    import zfit

    zfit.run.set_graph_mode(False)
    tf.config.run_functions_eagerly(True)
    fit_edges = edges[np.r_[np.flatnonzero(fit_mask), np.flatnonzero(fit_mask)[-1] + 1]]
    fit_counts = observed[fit_mask]
    fit_errors = np.maximum(np.sqrt(observed_variances[fit_mask]), 1.0)
    fit_widths = widths[fit_mask]
    total_count = float(fit_counts.sum())
    signal_parameters = [
        zfit.Parameter("narrow_yield", 0.20 * total_count, 0.0, 2.0 * total_count),
        zfit.Parameter("wide_yield", 0.10 * total_count, 0.0, 2.0 * total_count),
        zfit.Parameter("mean", 0.5 * (regions.signal[0] + regions.signal[1]), regions.signal[0], regions.signal[1]),
        zfit.Parameter("sigma_narrow", 0.0025, 0.0002, 0.020),
        zfit.Parameter("delta_sigma", 0.0050, 0.0001, 0.030),
    ]
    background_parameters = []
    for index, (name, value) in enumerate(zip(seed.parameter_names, seed.parameter_values)):
        sigma = np.sqrt(max(seed.parameter_covariance[index, index], 0.0))
        half_span = max(10.0 * sigma, 0.5 * abs(value), 1.0e-6)
        background_parameters.append(zfit.Parameter(f"background_{name}", value, value - half_span, value + half_span))
    centers_tf = tf.constant(centers[fit_mask], dtype=tf.float64)
    counts_tf = tf.constant(fit_counts, dtype=tf.float64)
    errors_tf = tf.constant(fit_errors, dtype=tf.float64)
    widths_tf = tf.constant(fit_widths, dtype=tf.float64)
    norm = np.sqrt(2.0 * np.pi)

    def chi2_objective():
        narrow_yield, wide_yield, mean, sigma_narrow, delta_sigma = signal_parameters
        sigma_wide = sigma_narrow + delta_sigma
        narrow = narrow_yield * widths_tf / (norm * sigma_narrow) * tf.exp(-0.5 * tf.square((centers_tf - mean) / sigma_narrow))
        wide = wide_yield * widths_tf / (norm * sigma_wide) * tf.exp(-0.5 * tf.square((centers_tf - mean) / sigma_wide))
        parameter_map = {
            name: parameter if profile_background else tf.constant(value, dtype=tf.float64)
            for name, value, parameter in zip(seed.parameter_names, seed.parameter_values, background_parameters)
        }
        background_density = evaluate_symbolfit_expression(seed.parameterized_formula, centers_tf, parameter_map, tf)
        if getattr(background_density, "shape", None) == ():
            background_density = tf.ones_like(centers_tf) * background_density
        model = narrow + wide + background_density * widths_tf
        invalid_penalty = 1.0e12 * tf.reduce_sum(tf.nn.relu(-background_density) + tf.nn.relu(1.0e-12 - model))
        objective = tf.reduce_sum(tf.square((counts_tf - model) / errors_tf)) + invalid_penalty
        tf.debugging.assert_all_finite(objective, "non-finite 1-D Signal+Background objective")
        return objective

    floating_parameters = signal_parameters + (background_parameters if profile_background else [])
    loss = zfit.loss.SimpleLoss(chi2_objective, floating_parameters, errordef=1.0, jit=False)
    result = zfit.minimize.Minuit(tol=1.0e-4, mode=2, maxiter=20_000, verbosity=0).minimize(loss)
    if not result.converged or not result.valid:
        raise RuntimeError(f"zfit one-dimensional fit did not converge: {result}")
    fitted_signal = np.asarray([float(np.asarray(parameter.value())) for parameter in signal_parameters])
    fitted_background = np.asarray([float(np.asarray(parameter.value())) for parameter in background_parameters]) if profile_background else seed.parameter_values.copy()
    parameter_names = ("narrow_yield", "wide_yield", "mean", "sigma_narrow", "delta_sigma") + tuple(f"background:{name}" for name in seed.parameter_names)
    parameter_values = np.concatenate((fitted_signal, fitted_background))
    covariance = np.zeros((parameter_values.size, parameter_values.size), dtype=float)
    floating_covariance = np.asarray(result.covariance(params=floating_parameters), dtype=float)
    if profile_background:
        covariance[:, :] = floating_covariance
    else:
        covariance[:5, :5] = floating_covariance
        covariance[5:, 5:] = seed.parameter_covariance
    background_indices = tuple(range(5, parameter_values.size))
    background_map = dict(zip(seed.parameter_names, fitted_background))
    background_density = np.asarray(evaluate_symbolfit_expression(seed.parameterized_formula, centers, background_map, np), dtype=float)
    if background_density.ndim == 0:
        background_density = np.full_like(centers, float(background_density))
    background_counts = background_density * widths
    narrow_yield, wide_yield, mean, sigma_narrow, delta_sigma = fitted_signal
    signal_density = narrow_yield / (norm * sigma_narrow) * np.exp(-0.5 * ((centers - mean) / sigma_narrow) ** 2) + wide_yield / (norm * (sigma_narrow + delta_sigma)) * np.exp(-0.5 * ((centers - mean) / (sigma_narrow + delta_sigma)) ** 2)
    model_counts = background_counts + signal_density * widths
    representative_width = float(np.median(widths))
    dense_mass = np.linspace(edges[0], edges[-1], 2000)
    dense_background_density = np.asarray(evaluate_symbolfit_expression(seed.parameterized_formula, dense_mass, background_map, np), dtype=float)
    if dense_background_density.ndim == 0:
        dense_background_density = np.full_like(dense_mass, float(dense_background_density))
    dense_signal_density = narrow_yield / (norm * sigma_narrow) * np.exp(-0.5 * ((dense_mass - mean) / sigma_narrow) ** 2) + wide_yield / (norm * (sigma_narrow + delta_sigma)) * np.exp(-0.5 * ((dense_mass - mean) / (sigma_narrow + delta_sigma)) ** 2)
    chi2 = float(np.sum(np.square((observed[fit_mask] - model_counts[fit_mask]) / fit_errors)))
    ndf = int(np.count_nonzero(fit_mask) - len(floating_parameters))
    if ndf <= 0:
        raise RuntimeError(f"one-dimensional fit has non-positive ndf={ndf}")
    return FitResult1D(
        mass_edges=edges,
        observed_counts=observed,
        observed_variances=observed_variances,
        fit_range=fit_range,
        background_fit_range=(float(edges[0]), float(edges[-1])),
        background_profiled=profile_background,
        parameter_names=parameter_names,
        parameter_values=parameter_values,
        parameter_covariance=covariance,
        background_parameter_indices=background_indices,
        peak_mean=float(mean),
        peak_mean_variance=float(covariance[2, 2]),
        background_formula=seed.fitted_formula,
        background_parameterized_formula=seed.parameterized_formula,
        symbolfit_initial_values=dict(zip(seed.parameter_names, seed.parameter_values)),
        model_counts=model_counts,
        background_counts=background_counts,
        dense_mass=dense_mass,
        dense_model=(dense_background_density + dense_signal_density) * representative_width,
        dense_background=dense_background_density * representative_width,
        chi2=chi2,
        ndf=ndf,
        converged=True,
        background_model="symbolfit_expression_profiled" if profile_background else "symbolfit_expression_fixed",
    )


def _rebin_mass_plane(counts_by_period, fit_nbins):
    """Sum adjacent square mass bins while preserving per-period Poisson counts."""
    n_periods, n_xbins, n_ybins = counts_by_period.shape
    if n_xbins % fit_nbins != 0 or n_ybins % fit_nbins != 0:
        raise ValueError("both mass-axis bin counts must be divisible by fit_nbins")
    group_x, group_y = n_xbins // fit_nbins, n_ybins // fit_nbins
    return counts_by_period.reshape(n_periods, fit_nbins, group_x, fit_nbins, group_y).sum(axis=(2, 4))


def fit_mass_plane_2d(
    x_edges,
    y_edges,
    counts_by_period,
    *,
    x_seed: FitResult1D,
    y_seed: FitResult1D | None = None,
    fit_nbins: int = 20,
    random_seed: int = 1,
) -> FitResult2D:
    """Fit the four physical Signal/Background products to raw mass-plane counts."""
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)
    observed = np.asarray(counts_by_period, dtype=float)
    if observed.ndim != 3 or observed.shape[0] == 0 or observed.shape[1:] != (x_edges.size - 1, y_edges.size - 1):
        raise ValueError("counts_by_period must have shape (n_periods, n_xbins, n_ybins)")
    if not np.all(np.isfinite(observed)) or np.any(observed < 0.0) or not np.allclose(observed, np.rint(observed), atol=1.0e-8):
        raise ValueError("2-D zfit input must be finite, non-negative, unweighted Poisson counts")
    if not np.all(np.diff(x_edges) > 0.0) or not np.all(np.diff(y_edges) > 0.0) or fit_nbins <= 0:
        raise ValueError("mass-plane edges and fit_nbins are invalid")
    same_axis_model = y_seed is None
    y_seed = x_seed if y_seed is None else y_seed
    planes = _rebin_mass_plane(observed, fit_nbins)
    fit_x_edges = x_edges[:: (x_edges.size - 1) // fit_nbins]
    fit_y_edges = y_edges[:: (y_edges.size - 1) // fit_nbins]
    if fit_x_edges.size != fit_nbins + 1:
        fit_x_edges = np.r_[fit_x_edges, x_edges[-1]]
    if fit_y_edges.size != fit_nbins + 1:
        fit_y_edges = np.r_[fit_y_edges, y_edges[-1]]

    def chebyshev_seed(seed_result, axis_edges):
        centers = 0.5 * (axis_edges[:-1] + axis_edges[1:])
        density = np.maximum(seed_result.evaluate_background(centers), 1.0e-30)
        scaled = 2.0 * (centers - axis_edges[0]) / (axis_edges[-1] - axis_edges[0]) - 1.0
        basis = np.column_stack((scaled, 2.0 * scaled**2 - 1.0, 4.0 * scaled**3 - 3.0 * scaled))
        return np.clip(np.linalg.lstsq(basis, np.log(density), rcond=None)[0], -15.0, 15.0)

    x_cheb_seed = chebyshev_seed(x_seed, x_edges)
    y_cheb_seed = x_cheb_seed.copy() if same_axis_model else chebyshev_seed(y_seed, y_edges)
    import tensorflow as tf
    import zfit
    zfit.run.set_graph_mode(False)
    tf.config.run_functions_eagerly(True)
    x_narrow_seed, x_wide_seed = x_seed.parameter_values[:2]
    x_fraction_seed = float(np.clip(x_narrow_seed / max(x_narrow_seed + x_wide_seed, 1.0e-12), 0.30, 0.95))
    x_shape = [
        zfit.Parameter("x_sigma_narrow", float(np.clip(x_seed.parameter_values[3], 0.001, 0.005)), 0.0005, 0.010),
        zfit.Parameter("x_delta_sigma", float(np.clip(x_seed.parameter_values[4], 0.0002, 0.015)), 0.0001, 0.025),
        zfit.Parameter("x_narrow_fraction", x_fraction_seed, 0.20, 0.98),
    ]
    x_background = [zfit.Parameter(f"x_background_c{index + 1}", value, -20.0, 20.0) for index, value in enumerate(x_cheb_seed)]
    if same_axis_model:
        y_shape, y_background = x_shape, x_background
    else:
        y_narrow_seed, y_wide_seed = y_seed.parameter_values[:2]
        y_fraction_seed = float(np.clip(y_narrow_seed / max(y_narrow_seed + y_wide_seed, 1.0e-12), 0.20, 0.98))
        y_shape = [
            zfit.Parameter("y_sigma_narrow", float(np.clip(y_seed.parameter_values[3], 0.0005, 0.010)), 0.0005, 0.010),
            zfit.Parameter("y_delta_sigma", float(np.clip(y_seed.parameter_values[4], 0.0001, 0.025)), 0.0001, 0.025),
            zfit.Parameter("y_narrow_fraction", y_fraction_seed, 0.20, 0.98),
        ]
        y_background = [zfit.Parameter(f"y_background_c{index + 1}", value, -20.0, 20.0) for index, value in enumerate(y_cheb_seed)]
    shape_parameters = x_shape + x_background + ([] if same_axis_model else y_shape + y_background)
    rng = np.random.RandomState(random_seed)
    yield_parameters = []
    for period in range(planes.shape[0]):
        total = max(float(planes[period].sum()), 1.0)
        for component in ("SxSy", "BxSy", "SxBy", "BxBy"):
            yield_parameters.append(zfit.Parameter(f"yield_{component}_period{period}", total * 0.25 * (1.0 + 0.01 * rng.randn()), 0.0, 1.0e9))
    all_parameters = shape_parameters + yield_parameters
    x_edges_tf = tf.constant(fit_x_edges, dtype=tf.float64)
    y_edges_tf = tf.constant(fit_y_edges, dtype=tf.float64)
    observed_tf = tf.constant(planes, dtype=tf.float64)
    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(8)

    def background_fractions(axis_edges, coefficients):
        low, high = axis_edges[:-1], axis_edges[1:]
        masses = 0.5 * (low + high)[:, None] + 0.5 * (high - low)[:, None] * quadrature_nodes
        scaled = 2.0 * (masses - axis_edges[0]) / (axis_edges[-1] - axis_edges[0]) - 1.0
        basis = np.stack((scaled, 2.0 * scaled**2 - 1.0, 4.0 * scaled**3 - 3.0 * scaled), axis=-1)
        basis_tf = tf.constant(basis, dtype=tf.float64)
        bin_weights_tf = tf.constant(0.5 * (high - low)[:, None] * quadrature_weights, dtype=tf.float64)
        raw = tf.reduce_sum(bin_weights_tf * tf.exp(tf.linalg.matvec(basis_tf, tf.stack(coefficients))), axis=1)
        return raw / tf.reduce_sum(raw)

    x_background_fraction = lambda: background_fractions(fit_x_edges, x_background)
    y_background_fraction = x_background_fraction if same_axis_model else lambda: background_fractions(fit_y_edges, y_background)
    component_order = ("SxSy", "BxSy", "SxBy", "BxBy")

    def poisson_nll():
        x_signal_fraction = _double_gaussian_bin_fractions(x_edges_tf, (fit_x_edges[0], fit_x_edges[-1]), x_seed.peak_mean, *x_shape, tf)
        y_signal_fraction = x_signal_fraction if same_axis_model else _double_gaussian_bin_fractions(y_edges_tf, (fit_y_edges[0], fit_y_edges[-1]), y_seed.peak_mean, *y_shape, tf)
        x_background_values = x_background_fraction()
        y_background_values = y_background_fraction()
        total_nll = tf.constant(0.0, dtype=tf.float64)
        for period in range(planes.shape[0]):
            period_yields = yield_parameters[4 * period:4 * period + 4]
            expectation = (
                period_yields[0] * tf.einsum("i,j->ij", x_signal_fraction, y_signal_fraction)
                + period_yields[1] * tf.einsum("i,j->ij", x_background_values, y_signal_fraction)
                + period_yields[2] * tf.einsum("i,j->ij", x_signal_fraction, y_background_values)
                + period_yields[3] * tf.einsum("i,j->ij", x_background_values, y_background_values)
            )
            total_nll += tf.reduce_sum(expectation - observed_tf[period] * tf.math.log(expectation + 1.0e-12))
        tf.debugging.assert_all_finite(total_nll, "non-finite 2-D extended Poisson NLL")
        return total_nll

    loss = zfit.loss.SimpleLoss(poisson_nll, all_parameters, errordef=0.5, jit=False)
    result = zfit.minimize.Minuit(tol=1.0e-3, mode=2, maxiter=20_000, verbosity=0).minimize(loss)
    if not result.converged or not result.valid:
        raise RuntimeError(f"zfit two-dimensional fit did not converge: {result}")
    parameter_values = np.asarray([float(np.asarray(parameter.value())) for parameter in all_parameters])
    covariance = np.asarray(result.covariance(params=all_parameters), dtype=float)
    parameter_names = tuple(parameter.name for parameter in all_parameters)
    x_signal_indices = tuple(parameter_names.index(parameter.name) for parameter in x_shape)
    x_background_indices = tuple(parameter_names.index(parameter.name) for parameter in x_background)
    y_signal_indices = x_signal_indices if same_axis_model else tuple(parameter_names.index(parameter.name) for parameter in y_shape)
    y_background_indices = x_background_indices if same_axis_model else tuple(parameter_names.index(parameter.name) for parameter in y_background)
    x_signal_fraction = np.asarray(_double_gaussian_bin_fractions(fit_x_edges, (fit_x_edges[0], fit_x_edges[-1]), x_seed.peak_mean, *parameter_values[list(x_signal_indices)], np))
    y_signal_fraction = x_signal_fraction if same_axis_model else np.asarray(_double_gaussian_bin_fractions(fit_y_edges, (fit_y_edges[0], fit_y_edges[-1]), y_seed.peak_mean, *parameter_values[list(y_signal_indices)], np))
    x_background_fraction_values = np.asarray([_exp_chebyshev_integral((fit_x_edges[index], fit_x_edges[index + 1]), (fit_x_edges[0], fit_x_edges[-1]), parameter_values[list(x_background_indices)]) for index in range(fit_nbins)])
    y_background_fraction_values = x_background_fraction_values if same_axis_model else np.asarray([_exp_chebyshev_integral((fit_y_edges[index], fit_y_edges[index + 1]), (fit_y_edges[0], fit_y_edges[-1]), parameter_values[list(y_background_indices)]) for index in range(fit_nbins)])
    fitted_yields = parameter_values[-4 * planes.shape[0]:].reshape(planes.shape[0], 4)
    model = np.empty_like(planes)
    background_model = np.empty_like(planes)
    for period, period_yields in enumerate(fitted_yields):
        components = np.stack((
            period_yields[0] * np.outer(x_signal_fraction, y_signal_fraction),
            period_yields[1] * np.outer(x_background_fraction_values, y_signal_fraction),
            period_yields[2] * np.outer(x_signal_fraction, y_background_fraction_values),
            period_yields[3] * np.outer(x_background_fraction_values, y_background_fraction_values),
        ))
        model[period] = components.sum(axis=0)
        background_model[period] = components[1:].sum(axis=0)
    dense_x = np.linspace(fit_x_edges[0], fit_x_edges[-1], 1000)
    dense_y = np.linspace(fit_y_edges[0], fit_y_edges[-1], 1000)
    x_sigma, x_delta, x_fraction = parameter_values[list(x_signal_indices)]
    y_sigma, y_delta, y_fraction = parameter_values[list(y_signal_indices)]
    x_narrow_norm = 0.5 * (erf((fit_x_edges[-1] - x_seed.peak_mean) / (np.sqrt(2.0) * x_sigma)) - erf((fit_x_edges[0] - x_seed.peak_mean) / (np.sqrt(2.0) * x_sigma)))
    x_wide_norm = 0.5 * (erf((fit_x_edges[-1] - x_seed.peak_mean) / (np.sqrt(2.0) * (x_sigma + x_delta))) - erf((fit_x_edges[0] - x_seed.peak_mean) / (np.sqrt(2.0) * (x_sigma + x_delta))))
    y_narrow_norm = 0.5 * (erf((fit_y_edges[-1] - y_seed.peak_mean) / (np.sqrt(2.0) * y_sigma)) - erf((fit_y_edges[0] - y_seed.peak_mean) / (np.sqrt(2.0) * y_sigma)))
    y_wide_norm = 0.5 * (erf((fit_y_edges[-1] - y_seed.peak_mean) / (np.sqrt(2.0) * (y_sigma + y_delta))) - erf((fit_y_edges[0] - y_seed.peak_mean) / (np.sqrt(2.0) * (y_sigma + y_delta))))
    x_signal_density = x_fraction * np.exp(-0.5 * ((dense_x - x_seed.peak_mean) / x_sigma) ** 2) / (np.sqrt(2.0 * np.pi) * x_sigma * x_narrow_norm) + (1.0 - x_fraction) * np.exp(-0.5 * ((dense_x - x_seed.peak_mean) / (x_sigma + x_delta)) ** 2) / (np.sqrt(2.0 * np.pi) * (x_sigma + x_delta) * x_wide_norm)
    y_signal_density = y_fraction * np.exp(-0.5 * ((dense_y - y_seed.peak_mean) / y_sigma) ** 2) / (np.sqrt(2.0 * np.pi) * y_sigma * y_narrow_norm) + (1.0 - y_fraction) * np.exp(-0.5 * ((dense_y - y_seed.peak_mean) / (y_sigma + y_delta)) ** 2) / (np.sqrt(2.0 * np.pi) * (y_sigma + y_delta) * y_wide_norm)
    x_background_density = _exp_chebyshev_density(dense_x, (fit_x_edges[0], fit_x_edges[-1]), parameter_values[list(x_background_indices)])
    y_background_density = _exp_chebyshev_density(dense_y, (fit_y_edges[0], fit_y_edges[-1]), parameter_values[list(y_background_indices)])
    summed_yields = fitted_yields.sum(axis=0)
    x_bin_width = float(np.median(np.diff(fit_x_edges)))
    y_bin_width = float(np.median(np.diff(fit_y_edges)))
    x_dense_total = x_bin_width * ((summed_yields[0] + summed_yields[2]) * x_signal_density + (summed_yields[1] + summed_yields[3]) * x_background_density)
    x_dense_background = x_bin_width * (summed_yields[2] * x_signal_density + (summed_yields[1] + summed_yields[3]) * x_background_density)
    y_dense_total = y_bin_width * ((summed_yields[0] + summed_yields[1]) * y_signal_density + (summed_yields[2] + summed_yields[3]) * y_background_density)
    y_dense_background = y_bin_width * (summed_yields[1] * y_signal_density + (summed_yields[2] + summed_yields[3]) * y_background_density)
    return FitResult2D(
        x_edges=fit_x_edges,
        y_edges=fit_y_edges,
        observed_counts_by_period=planes,
        model_counts_by_period=model,
        component_names=component_order,
        component_yields_by_period=fitted_yields,
        parameter_names=parameter_names,
        parameter_values=parameter_values,
        parameter_covariance=covariance,
        x_signal_parameter_indices=x_signal_indices,
        y_signal_parameter_indices=y_signal_indices,
        x_background_parameter_indices=x_background_indices,
        y_background_parameter_indices=y_background_indices,
        x_peak_mean=x_seed.peak_mean,
        y_peak_mean=y_seed.peak_mean,
        x_fit_range=(float(fit_x_edges[0]), float(fit_x_edges[-1])),
        y_fit_range=(float(fit_y_edges[0]), float(fit_y_edges[-1])),
        symbolfit_initial_values_x=x_seed.symbolfit_initial_values,
        symbolfit_initial_values_y=y_seed.symbolfit_initial_values,
        nll_value=float(result.fmin),
        fit_nbins=fit_nbins,
        n_periods=planes.shape[0],
        converged=True,
        x_projection_observed=planes.sum(axis=(0, 2)),
        x_projection_model=model.sum(axis=(0, 2)),
        x_projection_background=background_model.sum(axis=(0, 2)),
        x_projection_dense_mass=dense_x,
        x_projection_dense_model=x_dense_total,
        x_projection_dense_background=x_dense_background,
        y_projection_observed=planes.sum(axis=(0, 1)),
        y_projection_model=model.sum(axis=(0, 1)),
        y_projection_background=background_model.sum(axis=(0, 1)),
        y_projection_dense_mass=dense_y,
        y_projection_dense_model=y_dense_total,
        y_projection_dense_background=y_dense_background,
        component_models=("S_x S_y", "B_x S_y", "S_x B_y", "B_x B_y"),
    )
