"""质量谱 / 质量平面拟合：SymbolFit 本底选择 + zfit 最终估计。

TensorFlow、zfit、PySR、SymbolFit 一律在拟合函数内部局部导入，导入本模块
本身不需要任何拟合栈。本底解析式通过白名单 AST 转换器编译成 numpy（本模块
内部使用）和 TensorFlow（zfit 损失函数使用）两种求值器。

SymbolFit（PySR/Julia）在子进程中运行（见 ``_symbolfit_worker.py``）：
Julia 的 LLVM 与 ROOT/libCling 的 LLVM 在同一进程内会发生
``cl::opt`` 重复注册冲突（进程直接 abort），子进程隔离是唯一稳妥的共存
方式。
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import erf

from ._symbolfit import (
    BackgroundModel,
    _compile_expression,
    _run_symbolfit_selection,
)
from .transfer import MassRegions1D

__all__ = [
    "BackgroundModel",
    "NormalizedDensity1D",
    "ComponentModel2D",
    "FitResult1D",
    "FitResult2D",
    "fit_mass_spectrum_1d",
    "fit_mass_plane_2d",
]

# 每次拟合生成唯一的 zfit 参数名后缀，避免同一进程内多次拟合重名。
_FIT_COUNTER = itertools.count()


# ---------------------------------------------------------------------------
# 表达式白名单 AST 转换器
# ---------------------------------------------------------------------------
#
# 支持的语法：数字常量、x0、参数符号 a1/a2/...、+ - *、**(非负整数次幂)、
# exp()、square()。每个 ** 的指数必须是非负整数，且底数/指数内部不得再嵌
# 套 **；exp() 的参数必须是 x0 的次数 <= 1 的多项式。其余节点立即报错。


# ---------------------------------------------------------------------------
# 数值求值模型（供 transfer / report / 测试使用）
# ---------------------------------------------------------------------------


def _gauss_legendre_points(mass_lo: float, mass_hi: float, n_points: int):
    """返回 (积分节点, 积分权重)，节点已映射到 [mass_lo, mass_hi]。"""
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    points = 0.5 * (mass_lo + mass_hi) + 0.5 * (mass_hi - mass_lo) * nodes
    mapped_weights = 0.5 * (mass_hi - mass_lo) * weights
    return points, mapped_weights


class NormalizedDensity1D:
    """归一化到 [mass_lo, mass_hi] 的 1D 密度，可按任意参数值求值。

    用于二维 transfer：``region_integral(interval, parameter_values)`` 返回
    密度在该区间内的积分除以全区间积分（与显式 ProductPDF 的归一化定义一致）。
    """

    def __init__(
        self,
        mass_lo: float,
        mass_hi: float,
        parameter_names: list[str],
        density_fn: Callable,
        n_points: int = 64,
    ):
        self.mass_lo = float(mass_lo)
        self.mass_hi = float(mass_hi)
        self.parameter_names = [str(name) for name in parameter_names]
        self._density_fn = density_fn
        self.n_points = int(n_points)

    def total_integral(self, parameter_values: Mapping[str, float]) -> float:
        points, weights = _gauss_legendre_points(
            self.mass_lo, self.mass_hi, self.n_points
        )
        values = np.asarray(
            self._density_fn(points, parameter_values), dtype=float
        )
        return float(np.dot(weights, values))

    def region_integral(
        self,
        interval: tuple[float, float],
        parameter_values: Mapping[str, float],
    ) -> float:
        """密度在区间内的积分 / 全区间积分。"""
        low, high = float(interval[0]), float(interval[1])
        total = self.total_integral(parameter_values)
        if total <= 0.0:
            raise ValueError("NormalizedDensity1D: 全区间积分非正")
        points, weights = _gauss_legendre_points(low, high, self.n_points)
        values = np.asarray(
            self._density_fn(points, parameter_values), dtype=float
        )
        return float(np.dot(weights, values) / total)


def _double_gaussian_density(
    mean: float, parameter_names: tuple[str, str, str]
) -> Callable:
    """固定共峰位的 double-Gaussian 密度。

    ``parameter_names`` 是 (sigma_narrow, delta_sigma, narrow_frac) 三个参数
    的实际名字；宽 Gauss 宽度为 sigma_narrow + delta_sigma。
    """

    def density(mass, params: Mapping[str, float]) -> np.ndarray:
        mass_array = np.asarray(mass, dtype=float)
        sigma_narrow = params[parameter_names[0]]
        sigma_wide = sigma_narrow + params[parameter_names[1]]
        narrow_frac = params[parameter_names[2]]
        narrow = np.exp(
            -0.5 * np.square((mass_array - mean) / sigma_narrow)
        ) / (np.sqrt(2.0 * np.pi) * sigma_narrow)
        wide = np.exp(
            -0.5 * np.square((mass_array - mean) / sigma_wide)
        ) / (np.sqrt(2.0 * np.pi) * sigma_wide)
        return narrow_frac * narrow + (1.0 - narrow_frac) * wide

    return density


# ---------------------------------------------------------------------------
# 拟合结果数据契约
# ---------------------------------------------------------------------------


@dataclass
class FitResult1D:
    """一维质量谱拟合结果。"""

    mass_edges: np.ndarray
    observed_counts: np.ndarray
    observed_variances: np.ndarray
    fit_range: tuple[float, float]
    regions: MassRegions1D
    background_profiled: bool
    parameter_names: list[str]
    parameter_values: np.ndarray
    parameter_covariance: np.ndarray
    peak_mean: float
    peak_mean_variance: float
    background_formula: str
    symbolfit_initial_values: dict[str, float]
    model_counts: np.ndarray
    background_counts: np.ndarray
    dense_mass: np.ndarray
    dense_model: np.ndarray
    dense_background: np.ndarray
    chi2: float
    ndf: int
    converged: bool
    background_model: BackgroundModel


@dataclass
class ComponentModel2D:
    """一个二维分量的两个归一化轴密度。"""

    name: str
    x_pdf: NormalizedDensity1D
    y_pdf: NormalizedDensity1D


@dataclass
class FitResult2D:
    """二维质量平面四分量拟合结果。"""

    x_edges: np.ndarray
    y_edges: np.ndarray
    observed_counts_by_period: np.ndarray
    model_counts_by_period: np.ndarray
    component_names: list[str]
    component_yields_by_period: list[dict[str, float]]
    parameter_names: list[str]
    parameter_values: np.ndarray
    parameter_covariance: np.ndarray
    symbolfit_initial_values_x: dict[str, float]
    symbolfit_initial_values_y: dict[str, float]
    nll_value: float
    fit_nbins: int
    n_periods: int
    converged: bool
    x_projection_observed: np.ndarray
    x_projection_model: np.ndarray
    x_projection_background: np.ndarray
    y_projection_observed: np.ndarray
    y_projection_model: np.ndarray
    y_projection_background: np.ndarray
    x_projection_dense_mass: np.ndarray
    x_projection_dense_model: np.ndarray
    x_projection_dense_background: np.ndarray
    y_projection_dense_mass: np.ndarray
    y_projection_dense_model: np.ndarray
    y_projection_dense_background: np.ndarray
    component_models: list[ComponentModel2D]
    x_background_formula: str = ""
    y_background_formula: str = ""
    signal_yield_profile_interval: tuple[float, float] | None = None


# ---------------------------------------------------------------------------
# 6.1 SymbolFit 本底选择
# ---------------------------------------------------------------------------


def _select_symbolfit_background(
    *,
    centers: np.ndarray,
    density: np.ndarray,
    density_errors: np.ndarray,
    training_mask: np.ndarray,
    mass_lo: float,
    mass_hi: float,
    random_seed: int,
    output_dir: Path | None,
) -> tuple[BackgroundModel, np.ndarray]:
    """在子进程中运行 SymbolFit 候选式搜索并返回最优本底模型。

    返回 (BackgroundModel, 参数协方差)。
    """
    config = {
        "mass_lo": float(mass_lo),
        "mass_hi": float(mass_hi),
        "random_seed": int(random_seed),
        "output_dir": str(output_dir) if output_dir is not None else None,
    }
    with tempfile.TemporaryDirectory(prefix="symbolfit_payload_") as tmp:
        payload_path = Path(tmp) / "payload.npz"
        result_path = Path(tmp) / "result.json"
        np.savez(
            payload_path,
            centers=np.asarray(centers, dtype=float),
            density=np.asarray(density, dtype=float),
            density_errors=np.asarray(density_errors, dtype=float),
            training_mask=np.asarray(training_mask, dtype=bool),
            config_json=np.array(json.dumps(config)),
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "_symbolfit_worker.py"),
                str(payload_path),
                str(result_path),
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if completed.returncode != 0 or not result_path.exists():
            stdout_tail = completed.stdout[-2000:]
            stderr_tail = completed.stderr[-2000:]
            raise RuntimeError(
                "SymbolFit 子进程失败 "
                f"(returncode={completed.returncode}):\n"
                f"stdout: {stdout_tail}\nstderr: {stderr_tail}"
            )
        result = json.loads(result_path.read_text())

    if not result.get("ok"):
        raise RuntimeError(f"SymbolFit 选择失败: {result.get('error')}")

    background_model = BackgroundModel(
        result["formula"],
        result["parameter_names"],
        result["initial_values"],
    )
    covariance = np.asarray(result["covariance"], dtype=float)
    return background_model, covariance


# ---------------------------------------------------------------------------
# 6.2 一维质量谱拟合
# ---------------------------------------------------------------------------


def fit_mass_spectrum_1d(
    mass_edges,
    counts,
    variances,
    *,
    fit_range: tuple[float, float],
    regions: MassRegions1D,
    signal_model: str = "double_gaussian",
    random_seed: int = 0,
    profile_background: bool = True,
    symbolfit_output_dir: Path | str | None = None,
) -> FitResult1D:
    """拟合一维质量谱：SymbolFit 本底 + 共峰位 double-Gaussian 信号。

    ``profile_background=True`` 时把 SymbolFit 表达式转换为可微 PDF，由 zfit
    同时估计信号和本底参数；``False`` 时固定 SymbolFit 本底形状，仅拟合信号
    （复现 09 脚本的一维路径）。损失为 chi2（误差取 max(sqrt(Sumw2), 1)）。
    """
    if signal_model != "double_gaussian":
        raise NotImplementedError(
            f"暂不支持信号模型 {signal_model}，当前只实现 double_gaussian"
        )

    mass_edges = np.asarray(mass_edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    variances = np.asarray(variances, dtype=float)
    n_bins = counts.size
    if mass_edges.ndim != 1 or mass_edges.size != n_bins + 1:
        raise ValueError(f"mass_edges 长度应为 {n_bins + 1}（counts 长度 + 1）")
    if variances.shape != counts.shape:
        raise ValueError("counts 与 variances 形状不一致")
    if not np.all(np.isfinite(mass_edges)) or not np.all(np.isfinite(counts)):
        raise ValueError("mass_edges / counts 含非有限值")
    if np.any(variances < 0.0) or not np.all(np.isfinite(variances)):
        raise ValueError("variances 必须为非负有限值")
    if np.any(np.diff(mass_edges) <= 0.0):
        raise ValueError("mass_edges 必须严格递增")

    mass_lo, mass_hi = float(mass_edges[0]), float(mass_edges[-1])
    fit_lo, fit_hi = float(fit_range[0]), float(fit_range[1])
    if not (mass_lo <= fit_lo < fit_hi <= mass_hi):
        raise ValueError(
            f"fit_range ({fit_lo}, {fit_hi}) 必须位于质量范围 [{mass_lo}, {mass_hi}] 内"
        )
    regions.validate_within(mass_lo, mass_hi)
    if regions.signal[0] < fit_lo or regions.signal[1] > fit_hi:
        raise ValueError(
            f"signal 区域 {regions.signal} 必须位于 fit_range ({fit_lo}, {fit_hi}) 内"
        )

    widths = np.diff(mass_edges)
    centers = 0.5 * (mass_edges[:-1] + mass_edges[1:])
    density = counts / widths
    density_errors = np.sqrt(variances) / widths

    signal_low, signal_high = regions.signal
    training_mask = (centers < signal_low) | (centers >= signal_high)

    output_dir = (
        Path(symbolfit_output_dir) if symbolfit_output_dir is not None else None
    )
    background_model, background_covariance = _select_symbolfit_background(
        centers=centers,
        density=density,
        density_errors=density_errors,
        training_mask=training_mask,
        mass_lo=mass_lo,
        mass_hi=mass_hi,
        random_seed=random_seed,
        output_dir=output_dir,
    )

    # ---- zfit 最终拟合 -----------------------------------------------------
    import tensorflow as tf
    import zfit

    zfit.run.set_graph_mode(False)
    tf.config.run_functions_eagerly(True)

    uid = next(_FIT_COUNTER)
    total_count = float(np.sum(counts))
    signal_half_width = 0.5 * (signal_high - signal_low)
    signal_center = 0.5 * (signal_low + signal_high)

    narrow_yield = zfit.Parameter(
        f"narrow_yield_{uid}", 0.20 * total_count, 0.0, 2.0 * total_count
    )
    wide_yield = zfit.Parameter(
        f"wide_yield_{uid}", 0.10 * total_count, 0.0, 2.0 * total_count
    )
    mean = zfit.Parameter(f"mean_{uid}", signal_center, signal_low, signal_high)
    sigma_narrow = zfit.Parameter(
        f"sigma_narrow_{uid}",
        0.25 * signal_half_width,
        0.05 * signal_half_width,
        signal_half_width,
    )
    delta_sigma = zfit.Parameter(
        f"delta_sigma_{uid}",
        0.50 * signal_half_width,
        0.02 * signal_half_width,
        2.5 * signal_half_width,
    )
    signal_parameters = [narrow_yield, wide_yield, mean, sigma_narrow, delta_sigma]
    signal_parameter_names = [
        "narrow_yield",
        "wide_yield",
        "mean",
        "sigma_narrow",
        "delta_sigma",
    ]

    background_tf_evaluator = _compile_expression(
        background_model.formula,
        background_model.parameter_names,
        tf.exp,
        tf.square,
    )
    background_parameters = []
    if profile_background:
        # 本底参数作为 zfit 参数浮动，初值取 SymbolFit 结果。
        background_tf_params = {}
        for name in background_model.parameter_names:
            parameter = zfit.Parameter(
                f"{name}_{uid}", background_model.symbolfit_initial_values[name]
            )
            background_parameters.append(parameter)
            background_tf_params[name] = parameter
    else:
        # 本底固定为 SymbolFit 曲线。
        background_tf_params = {
            name: tf.constant(
                background_model.symbolfit_initial_values[name], dtype=tf.float64
            )
            for name in background_model.parameter_names
        }

    fit_mask = (centers >= fit_lo) & (centers < fit_hi)
    fit_centers = centers[fit_mask]
    fit_counts = counts[fit_mask]
    fit_widths = widths[fit_mask]
    fit_errors = np.maximum(np.sqrt(variances[fit_mask]), 1.0)
    if fit_counts.size == 0:
        raise ValueError("fit_range 内没有任何 bin")

    centers_tf = tf.constant(fit_centers, dtype=tf.float64)
    counts_tf = tf.constant(fit_counts, dtype=tf.float64)
    widths_tf = tf.constant(fit_widths, dtype=tf.float64)
    errors_tf = tf.constant(fit_errors, dtype=tf.float64)
    norm = float(np.sqrt(2.0 * np.pi))

    def chi2_objective():
        # 本底曲线必须在目标函数内部求值：profile 模式下它依赖 zfit 参数，
        # 提前 eager 求值会把它冻结在初值上（参数不进入梯度）。
        background_counts_tf = (
            background_tf_evaluator(centers_tf, background_tf_params) * widths_tf
        )
        sigma_wide = sigma_narrow + delta_sigma
        narrow = (
            narrow_yield
            * widths_tf
            / (norm * sigma_narrow)
            * tf.exp(-0.5 * tf.square((centers_tf - mean) / sigma_narrow))
        )
        wide = (
            wide_yield
            * widths_tf
            / (norm * sigma_wide)
            * tf.exp(-0.5 * tf.square((centers_tf - mean) / sigma_wide))
        )
        residual = (counts_tf - narrow - wide - background_counts_tf) / errors_tf
        objective = tf.reduce_sum(tf.square(residual))
        tf.debugging.assert_all_finite(objective, "chi2 目标函数含非有限值")
        return objective

    floating_parameters = signal_parameters + background_parameters
    loss = zfit.loss.SimpleLoss(
        chi2_objective, floating_parameters, errordef=1.0, jit=False
    )
    result = zfit.minimize.Minuit(
        tol=1.0e-4, mode=2, maxiter=20_000, verbosity=0
    ).minimize(loss)
    fit_converged = bool(result.converged and result.valid)
    if not fit_converged:
        warnings.warn(f"zfit 一维拟合未收敛: {result}；返回当前参数值", RuntimeWarning)

    def parameter_value(parameter):
        return float(np.asarray(parameter.value()))

    signal_values = [parameter_value(p) for p in signal_parameters]
    signal_covariance = np.asarray(
        result.covariance(params=signal_parameters), dtype=float
    )
    if signal_covariance.shape != (5, 5) or not np.all(
        np.isfinite(signal_covariance)
    ):
        warnings.warn(
            "zfit 返回的一维信号 covariance 无效，使用零矩阵", RuntimeWarning
        )
        signal_covariance = np.zeros((5, 5), dtype=float)
        fit_converged = False

    parameter_names = signal_parameter_names + list(background_model.parameter_names)
    if profile_background:
        parameter_values = np.asarray(
            signal_values + [parameter_value(p) for p in background_parameters],
            dtype=float,
        )
        parameter_covariance = np.asarray(
            result.covariance(params=floating_parameters), dtype=float
        )
    else:
        parameter_values = np.asarray(
            signal_values
            + [
                background_model.symbolfit_initial_values[name]
                for name in background_model.parameter_names
            ],
            dtype=float,
        )
        parameter_covariance = np.zeros(
            (len(parameter_values), len(parameter_values)), dtype=float
        )
        signal_block = slice(0, 5)
        background_block = slice(5, len(parameter_values))
        parameter_covariance[signal_block, signal_block] = signal_covariance
        parameter_covariance[background_block, background_block] = (
            background_covariance
        )
    if (
        parameter_covariance.shape != (len(parameter_values),) * 2
        or not np.all(np.isfinite(parameter_covariance))
    ):
        warnings.warn("zfit 返回的一维 covariance 无效，使用零矩阵", RuntimeWarning)
        parameter_covariance = np.zeros(
            (len(parameter_values), len(parameter_values)), dtype=float
        )
        fit_converged = False

    mean_index = parameter_names.index("mean")
    peak_mean = float(parameter_values[mean_index])
    peak_mean_variance = float(parameter_covariance[mean_index, mean_index])

    # ---- 名义模型曲线（numpy，全谱 bin 中心 + 稠密采样） -------------------
    nominal = dict(zip(parameter_names, parameter_values))
    background_counts = background_model.evaluate_density(centers, nominal) * widths

    def signal_density(masses):
        mass_array = np.asarray(masses, dtype=float)
        core_sigma = nominal["sigma_narrow"]
        tail_sigma = nominal["sigma_narrow"] + nominal["delta_sigma"]
        core = (
            nominal["narrow_yield"]
            / (np.sqrt(2.0 * np.pi) * core_sigma)
            * np.exp(-0.5 * np.square((mass_array - peak_mean) / core_sigma))
        )
        tail = (
            nominal["wide_yield"]
            / (np.sqrt(2.0 * np.pi) * tail_sigma)
            * np.exp(-0.5 * np.square((mass_array - peak_mean) / tail_sigma))
        )
        return core + tail

    model_counts = background_counts + signal_density(centers) * widths
    dense_mass = np.linspace(mass_lo, mass_hi, 2000)
    average_width = (mass_hi - mass_lo) / n_bins
    dense_background = (
        background_model.evaluate_density(dense_mass, nominal) * average_width
    )
    dense_model = dense_background + signal_density(dense_mass) * average_width

    chi2_value = float(
        np.sum(np.square((fit_counts - model_counts[fit_mask]) / fit_errors))
    )
    ndf = int(fit_counts.size - len(floating_parameters))
    if ndf <= 0:
        raise RuntimeError(f"一维拟合自由度非正 (ndf={ndf})")

    return FitResult1D(
        mass_edges=mass_edges,
        observed_counts=counts,
        observed_variances=variances,
        fit_range=(fit_lo, fit_hi),
        regions=regions,
        background_profiled=bool(profile_background),
        parameter_names=parameter_names,
        parameter_values=parameter_values,
        parameter_covariance=parameter_covariance,
        peak_mean=peak_mean,
        peak_mean_variance=peak_mean_variance,
        background_formula=background_model.formula,
        symbolfit_initial_values=dict(background_model.symbolfit_initial_values),
        model_counts=model_counts,
        background_counts=background_counts,
        dense_mass=dense_mass,
        dense_model=dense_model,
        dense_background=dense_background,
        chi2=chi2_value,
        ndf=ndf,
        converged=fit_converged,
        background_model=background_model,
    )


# ---------------------------------------------------------------------------
# 6.3 二维质量平面拟合
# ---------------------------------------------------------------------------


def _signal_axis_fractions_tf(
    edges_tf,
    mean_value: float,
    sigma_narrow,
    delta_sigma,
    narrow_frac,
    mass_lo: float,
    mass_hi: float,
    tf,
):
    """TF：共峰 double-Gaussian 在给定 bin 边缘上的归一化 bin 分数。"""
    sqrt2 = float(np.sqrt(2.0))
    sigma_wide = sigma_narrow + delta_sigma

    def erf_difference(edges, sigma):
        return tf.math.erf(
            (edges[1:] - mean_value) / (sigma * sqrt2)
        ) - tf.math.erf((edges[:-1] - mean_value) / (sigma * sqrt2))

    erf_narrow = erf_difference(edges_tf, sigma_narrow)
    erf_wide = erf_difference(edges_tf, sigma_wide)
    erf_narrow_full = tf.math.erf(
        (mass_hi - mean_value) / (sigma_narrow * sqrt2)
    ) - tf.math.erf((mass_lo - mean_value) / (sigma_narrow * sqrt2))
    erf_wide_full = tf.math.erf(
        (mass_hi - mean_value) / (sigma_wide * sqrt2)
    ) - tf.math.erf((mass_lo - mean_value) / (sigma_wide * sqrt2))
    return (
        narrow_frac * erf_narrow / erf_narrow_full
        + (1.0 - narrow_frac) * erf_wide / erf_wide_full
    )


def _background_axis_fractions_tf(
    edges: np.ndarray,
    mass_lo: float,
    mass_hi: float,
    tf_evaluator: Callable,
    tf_params: Mapping,
    tf,
    n_quad: int = 5,
):
    """TF：SymbolFit 表达式在各 bin 的归一化分数（逐 bin Gauss-Legendre）。"""
    n_bins = edges.size - 1
    bin_lo = edges[:-1]
    bin_hi = edges[1:]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
    quad_points = (
        0.5 * (bin_lo + bin_hi)[:, None]
        + 0.5 * (bin_hi - bin_lo)[:, None] * gl_nodes[None, :]
    )
    quad_weights = 0.5 * (bin_hi - bin_lo)[:, None] * gl_weights[None, :]
    n_full = max(n_quad * n_bins, 60)
    full_points, full_weights = _gauss_legendre_points(mass_lo, mass_hi, n_full)
    quad_values = tf.reshape(
        tf_evaluator(
            tf.constant(quad_points.reshape(-1), dtype=tf.float64), tf_params
        ),
        (n_bins, n_quad),
    )
    bin_integrals = tf.reduce_sum(
        tf.constant(quad_weights, dtype=tf.float64) * quad_values, axis=1
    )
    full_values = tf_evaluator(
        tf.constant(full_points, dtype=tf.float64), tf_params
    )
    full_integral = tf.reduce_sum(
        tf.constant(full_weights, dtype=tf.float64) * full_values
    )
    return bin_integrals / full_integral


def fit_mass_plane_2d(
    x_edges,
    y_edges,
    counts_by_period,
    *,
    x_seed: FitResult1D,
    y_seed: FitResult1D | None = None,
    fit_nbins: int,
    random_seed: int = 0,
    profile_signal_yield: bool = False,
) -> FitResult2D:
    """拟合计数质量平面：SxSy / BxSy / SxBy / BxBy 四分量模型。

    每个时期具有独立、非负的四分量产额；信号形状（共峰 double-Gaussian，
    由一维结果标定）固定，本底形状参数跨时期共享并浮动。
    ``y_seed=None`` 表示两轴共享同一套一维模型（要求两轴 binning 一致）。
    损失为逐时期 extended Poisson binned NLL。
    ``profile_signal_yield=True`` 仅对单时期的 N_SxSy 计算 68.3% MINOS 区间。
    """
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)
    counts_by_period = np.asarray(counts_by_period, dtype=float)
    shared_axes = y_seed is None

    if counts_by_period.ndim != 3:
        raise ValueError("counts_by_period 形状应为 (n_periods, n_xbins, n_ybins)")
    n_periods, n_xbins, n_ybins = counts_by_period.shape
    if profile_signal_yield and n_periods != 1:
        raise ValueError("N_SS 轮廓区间目前只支持单时期质量平面")
    if x_edges.size != n_xbins + 1 or y_edges.size != n_ybins + 1:
        raise ValueError("x_edges / y_edges 长度与 counts_by_period 不匹配")
    if not np.all(np.isfinite(counts_by_period)) or np.any(
        counts_by_period < 0.0
    ):
        raise ValueError("counts_by_period 必须为非负有限值")
    if np.any(np.diff(x_edges) <= 0.0) or np.any(np.diff(y_edges) <= 0.0):
        raise ValueError("x_edges / y_edges 必须严格递增")
    if shared_axes and not np.array_equal(x_edges, y_edges):
        raise ValueError("两轴共享一维模型时 x_edges 与 y_edges 必须完全一致")
    if n_xbins % fit_nbins != 0 or n_ybins % fit_nbins != 0:
        raise ValueError(
            f"fit_nbins={fit_nbins} 必须能整除两轴 bin 数 ({n_xbins}, {n_ybins})"
        )

    x_lo, x_hi = float(x_edges[0]), float(x_edges[-1])
    y_lo, y_hi = float(y_edges[0]), float(y_edges[-1])
    if not (x_lo <= x_seed.peak_mean <= x_hi):
        raise ValueError("x_seed 的峰位不在 x 轴质量范围内")
    if y_seed is not None and not (y_lo <= y_seed.peak_mean <= y_hi):
        raise ValueError("y_seed 的峰位不在 y 轴质量范围内")

    # 重分箱到拟合网格。
    group_x = n_xbins // fit_nbins
    group_y = n_ybins // fit_nbins
    planes_fit = counts_by_period.reshape(
        n_periods, fit_nbins, group_x, fit_nbins, group_y
    ).sum(axis=(2, 4))
    fit_edges_x = np.linspace(x_lo, x_hi, fit_nbins + 1)
    fit_edges_y = np.linspace(y_lo, y_hi, fit_nbins + 1)

    import tensorflow as tf
    import zfit

    zfit.run.set_graph_mode(False)
    tf.config.run_functions_eagerly(True)

    uid = next(_FIT_COUNTER)
    rng = np.random.RandomState(random_seed)

    def seed_parameter_dict(seed: FitResult1D) -> dict[str, float]:
        return dict(
            zip(
                seed.parameter_names,
                np.asarray(seed.parameter_values, dtype=float),
            )
        )

    seed_x_values = seed_parameter_dict(x_seed)
    seed_y_values = (
        seed_parameter_dict(y_seed) if y_seed is not None else seed_x_values
    )

    # ---- 信号形状参数（由独立一维质量拟合标定） --------------------------
    def signal_shape_seeds(seed_values: Mapping[str, float], half_width: float):
        sigma_seed = float(
            np.clip(seed_values["sigma_narrow"], 0.1 * half_width, 0.5 * half_width)
        )
        delta_seed = float(
            np.clip(
                seed_values["delta_sigma"], 0.02 * half_width, 1.5 * half_width
            )
        )
        fraction_seed = float(
            np.clip(
                seed_values["narrow_yield"]
                / max(
                    seed_values["narrow_yield"] + seed_values["wide_yield"],
                    1.0e-12,
                ),
                0.31,
                0.94,
            )
        )
        return sigma_seed, delta_seed, fraction_seed

    def half_width_of(seed: FitResult1D) -> float:
        low, high = seed.regions.signal
        return 0.5 * (high - low)

    def make_signal_axis(
        seed_values: Mapping[str, float], half_width: float, suffix: str
    ):
        """为一根轴创建由一维标定固定的 double-Gaussian 参数。"""
        sigma_name, delta_name, fraction_name = (
            base + suffix
            for base in ("sigma_narrow", "delta_sigma", "narrow_frac")
        )
        names = (sigma_name, delta_name, fraction_name)
        sigma_seed, delta_seed, fraction_seed = signal_shape_seeds(
            seed_values, half_width
        )
        parameters = [
            zfit.Parameter(f"{names[0]}_{uid}", sigma_seed, floating=False),
            zfit.Parameter(f"{names[1]}_{uid}", delta_seed, floating=False),
            zfit.Parameter(f"{names[2]}_{uid}", fraction_seed, floating=False),
        ]
        return names, parameters

    if shared_axes:
        signal_names_x, signal_parameters_x = make_signal_axis(
            seed_x_values, half_width_of(x_seed), ""
        )
        signal_names_y = signal_names_x
        signal_parameters_y = signal_parameters_x
    else:
        signal_names_x, signal_parameters_x = make_signal_axis(
            seed_x_values, half_width_of(x_seed), "_x"
        )
        signal_names_y, signal_parameters_y = make_signal_axis(
            seed_y_values, half_width_of(y_seed), "_y"
        )

    # ---- 本底参数（SymbolFit 曲线种子化 exp(Chebyshev-6)，zfit profile） ----
    background_model_x = x_seed.background_model
    background_model_y = (
        y_seed.background_model if y_seed is not None else background_model_x
    )

    def make_background_axis(
        model: BackgroundModel,
        seed_values: Mapping[str, float],
        suffix: str,
        mass_lo_value: float,
        mass_hi_value: float,
    ):
        """用 SymbolFit 曲线初始化正定、低阶且可 profile 的背景密度。"""
        seed_mass = np.linspace(mass_lo_value, mass_hi_value, 257)
        seed_density = model.evaluate_density(seed_mass, seed_values)
        if not np.all(np.isfinite(seed_density)) or np.any(seed_density <= 0.0):
            raise RuntimeError("SymbolFit 本底在二维拟合范围内非有限或非正")
        scaled_mass = 2.0 * (
            seed_mass - 0.5 * (mass_lo_value + mass_hi_value)
        ) / (mass_hi_value - mass_lo_value)
        chebyshev = np.polynomial.chebyshev.chebvander(scaled_mass, 6)[:, 1:]
        log_density = np.log(seed_density)
        initial, _, _, _ = np.linalg.lstsq(
            chebyshev, log_density - np.mean(log_density), rcond=None
        )
        initial = np.clip(initial, -15.0, 15.0)
        scaled_formula = (
            f"({2.0 / (mass_hi_value - mass_lo_value):.17g}) * "
            f"(x0 - ({0.5 * (mass_lo_value + mass_hi_value):.17g}))"
        )
        # 指数链接严格正定；六阶 Chebyshev 补足 loose 选择下的宽尺度曲率，
        # signal-window harness 仍会拒绝局部振荡。
        terms = (
            scaled_formula,
            f"2 * ({scaled_formula})**2 - 1",
            f"4 * ({scaled_formula})**3 - 3 * ({scaled_formula})",
            f"8 * ({scaled_formula})**4 - 8 * ({scaled_formula})**2 + 1",
            f"16 * ({scaled_formula})**5 - 20 * ({scaled_formula})**3 + 5 * ({scaled_formula})",
            f"32 * ({scaled_formula})**6 - 48 * ({scaled_formula})**4 + 18 * ({scaled_formula})**2 - 1",
        )
        names = [f"c{index}" for index in range(1, 7)]
        model = BackgroundModel(
            "exp(" + " + ".join(
                f"{name} * ({term})" for name, term in zip(names, terms)
            ) + ")",
            names,
            dict(zip(names, map(float, initial))),
        )
        seed_values = model.symbolfit_initial_values
        renamed = [f"{name}{suffix}" for name in model.parameter_names]
        tf_evaluator = _compile_expression(
            model.formula, model.parameter_names, tf.exp, tf.square
        )
        tf_params = {}
        initial_values = {}
        for original_name, name in zip(model.parameter_names, renamed):
            value = float(seed_values[original_name])
            tf_params[original_name] = zfit.Parameter(
                f"{name}_{uid}", value, -20.0, 20.0
            )
            initial_values[name] = value
        return model, renamed, tf_evaluator, tf_params, initial_values

    if shared_axes:
        background_model_x, bg_names_x, bg_evaluator_x, bg_params_x, bg_initial_x = (
            make_background_axis(
                background_model_x, seed_x_values, "", x_lo, x_hi
            )
        )
        bg_names_y, bg_evaluator_y, bg_params_y, bg_initial_y = (
            bg_names_x,
            bg_evaluator_x,
            bg_params_x,
            dict(bg_initial_x),
        )
        background_model_y = background_model_x
    else:
        background_model_x, bg_names_x, bg_evaluator_x, bg_params_x, bg_initial_x = (
            make_background_axis(
                background_model_x, seed_x_values, "_x", x_lo, x_hi
            )
        )
        background_model_y, bg_names_y, bg_evaluator_y, bg_params_y, bg_initial_y = (
            make_background_axis(
                background_model_y, seed_y_values, "_y", y_lo, y_hi
            )
        )

    # ---- 产额参数（每时期每分量，非负） ------------------------------------
    component_names = ["SxSy", "BxSy", "SxBy", "BxBy"]
    yield_parameter_names = []
    yield_parameters = []
    for period in range(n_periods):
        total = max(float(np.sum(planes_fit[period])), 1.0)
        for component in component_names:
            name = f"N_{component}_p{period}"
            parameter = zfit.Parameter(
                f"{name}_{uid}",
                float(total * 0.25 * (1.0 + 0.01 * rng.randn())),
                0.0,
                1.0e8,
            )
            yield_parameter_names.append(name)
            yield_parameters.append(parameter)

    # ---- TF bin 分数 --------------------------------------------------------
    mean_x = float(x_seed.peak_mean)
    mean_y = float(y_seed.peak_mean) if y_seed is not None else mean_x
    edges_x_tf = tf.constant(fit_edges_x, dtype=tf.float64)
    edges_y_tf = tf.constant(fit_edges_y, dtype=tf.float64)

    # 共享轴模式下 x/y 参数本就是同一组对象，闭包无需区分。
    sigma_narrow_x, delta_sigma_x, narrow_frac_x = signal_parameters_x
    sigma_narrow_y, delta_sigma_y, narrow_frac_y = signal_parameters_y

    def signal_fraction_x():
        return _signal_axis_fractions_tf(
            edges_x_tf, mean_x, sigma_narrow_x, delta_sigma_x, narrow_frac_x,
            x_lo, x_hi, tf,
        )

    def signal_fraction_y():
        return _signal_axis_fractions_tf(
            edges_y_tf, mean_y, sigma_narrow_y, delta_sigma_y, narrow_frac_y,
            y_lo, y_hi, tf,
        )

    def background_fraction_x():
        return _background_axis_fractions_tf(
            fit_edges_x, x_lo, x_hi, bg_evaluator_x, bg_params_x, tf
        )

    def background_fraction_y():
        return _background_axis_fractions_tf(
            fit_edges_y, y_lo, y_hi, bg_evaluator_y, bg_params_y, tf
        )

    observed_tf = [
        tf.constant(planes_fit[period], dtype=tf.float64)
        for period in range(n_periods)
    ]
    yield_lookup = {}
    index = 0
    for period in range(n_periods):
        yield_lookup[period] = {}
        for component in component_names:
            yield_lookup[period][component] = yield_parameters[index]
            index += 1

    def nll_func():
        """Extended Poisson binned NLL（差数据常数项）。"""
        fx = signal_fraction_x()
        bx = background_fraction_x()
        fy = signal_fraction_y()
        by = background_fraction_y()
        total = tf.constant(0.0, dtype=tf.float64)
        for period in range(n_periods):
            yields = yield_lookup[period]
            mu = (
                yields["SxSy"].value() * tf.einsum("i,j->ij", fx, fy)
                + yields["BxSy"].value() * tf.einsum("i,j->ij", bx, fy)
                + yields["SxBy"].value() * tf.einsum("i,j->ij", fx, by)
                + yields["BxBy"].value() * tf.einsum("i,j->ij", bx, by)
            )
            counts = observed_tf[period]
            total += tf.reduce_sum(mu - counts * tf.math.log(mu + 1.0e-10))
        return total

    # 信号参数作为一维标定常数进入 NLL；只有本底和产额参与最小化。
    if shared_axes:
        fixed_parameters = signal_parameters_x
        floating_parameters = list(bg_params_x.values()) + yield_parameters
    else:
        fixed_parameters = signal_parameters_x + signal_parameters_y
        floating_parameters = (
            list(bg_params_x.values()) + list(bg_params_y.values()) + yield_parameters
        )
    loss = zfit.loss.SimpleLoss(
        nll_func, floating_parameters, errordef=0.5, jit=False
    )
    result = zfit.minimize.Minuit(
        tol=1.0e-3, mode=2, maxiter=10_000, verbosity=0
    ).minimize(loss)
    fit_converged = bool(result.converged and result.valid)
    if not fit_converged:
        warnings.warn(f"zfit 二维拟合未收敛: {result}；返回当前参数值", RuntimeWarning)

    def parameter_value(parameter):
        return float(np.asarray(parameter.value()))

    fixed_values = [parameter_value(p) for p in fixed_parameters]
    floating_values = [parameter_value(p) for p in floating_parameters]
    floating_covariance = np.asarray(
        result.covariance(params=floating_parameters), dtype=float
    )
    if floating_covariance.shape != (len(floating_parameters),) * 2 or not np.all(
        np.isfinite(floating_covariance)
    ):
        warnings.warn("zfit 返回的二维 covariance 无效，使用零矩阵", RuntimeWarning)
        floating_covariance = np.zeros((len(floating_parameters),) * 2, dtype=float)
        fit_converged = False

    # ---- 参数汇总（干净名字，顺序与 all_parameters 一致） ------------------
    if shared_axes:
        parameter_names = (
            list(signal_names_x) + bg_names_x + yield_parameter_names
        )
    else:
        parameter_names = (
            list(signal_names_x)
            + list(signal_names_y)
            + bg_names_x
            + bg_names_y
            + yield_parameter_names
        )
    parameter_values = np.asarray(fixed_values + floating_values, dtype=float)
    covariance = np.zeros((len(parameter_values),) * 2, dtype=float)
    covariance[
        len(fixed_parameters) :, len(fixed_parameters) :
    ] = floating_covariance
    nominal = dict(zip(parameter_names, parameter_values))

    component_yields = []
    yield_start = len(parameter_values) - len(yield_parameters)
    index = 0
    for period in range(n_periods):
        per_component = {}
        for component in component_names:
            per_component[component] = float(parameter_values[yield_start + index])
            index += 1
        component_yields.append(per_component)

    # ---- 原始 binning 上的模型和投影（numpy 后验评估） ---------------------
    def numpy_signal_fractions(
        edges: np.ndarray,
        mean_value: float,
        names: tuple[str, str, str],
        mass_lo_value: float,
        mass_hi_value: float,
    ) -> np.ndarray:
        sigma_narrow_value = nominal[names[0]]
        sigma_wide_value = sigma_narrow_value + nominal[names[1]]
        fraction_value = nominal[names[2]]
        sqrt2 = np.sqrt(2.0)

        def erf_bin(sigma):
            return erf((edges[1:] - mean_value) / (sigma * sqrt2)) - erf(
                (edges[:-1] - mean_value) / (sigma * sqrt2)
            )

        def erf_full(sigma):
            return erf((mass_hi_value - mean_value) / (sigma * sqrt2)) - erf(
                (mass_lo_value - mean_value) / (sigma * sqrt2)
            )

        return (
            fraction_value * erf_bin(sigma_narrow_value) / erf_full(sigma_narrow_value)
            + (1.0 - fraction_value)
            * erf_bin(sigma_wide_value)
            / erf_full(sigma_wide_value)
        )

    def numpy_background_fractions(
        edges: np.ndarray,
        mass_lo_value: float,
        mass_hi_value: float,
        model: BackgroundModel,
        renamed: list[str],
    ) -> np.ndarray:
        n_bins_local = edges.size - 1
        bin_lo = edges[:-1]
        bin_hi = edges[1:]
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(5)
        quad_points = (
            0.5 * (bin_lo + bin_hi)[:, None]
            + 0.5 * (bin_hi - bin_lo)[:, None] * gl_nodes[None, :]
        )
        quad_weights = 0.5 * (bin_hi - bin_lo)[:, None] * gl_weights[None, :]

        def values_at(points):
            params = {
                original: nominal[name]
                for original, name in zip(model.parameter_names, renamed)
            }
            result = np.asarray(
                model._numpy_evaluator(np.asarray(points, dtype=float), params),
                dtype=float,
            )
            if result.shape != np.shape(points):
                result = np.broadcast_to(result, np.shape(points)).copy()
            return result

        quad_values = values_at(quad_points.reshape(-1)).reshape(n_bins_local, 5)
        bin_integrals = np.sum(quad_weights * quad_values, axis=1)
        full_points, full_weights = _gauss_legendre_points(
            mass_lo_value, mass_hi_value, max(5 * n_bins_local, 60)
        )
        full_integral = float(np.dot(full_weights, values_at(full_points)))
        return bin_integrals / full_integral

    sig_frac_x = numpy_signal_fractions(x_edges, mean_x, signal_names_x, x_lo, x_hi)
    sig_frac_y = numpy_signal_fractions(y_edges, mean_y, signal_names_y, y_lo, y_hi)
    bg_frac_x = numpy_background_fractions(
        x_edges, x_lo, x_hi, background_model_x, bg_names_x
    )
    bg_frac_y = numpy_background_fractions(
        y_edges, y_lo, y_hi, background_model_y, bg_names_y
    )

    model_planes = np.zeros_like(counts_by_period)
    for period in range(n_periods):
        yields = component_yields[period]
        model_planes[period] = (
            yields["SxSy"] * np.outer(sig_frac_x, sig_frac_y)
            + yields["BxSy"] * np.outer(bg_frac_x, sig_frac_y)
            + yields["SxBy"] * np.outer(sig_frac_x, bg_frac_y)
            + yields["BxBy"] * np.outer(bg_frac_x, bg_frac_y)
        )

    x_projection_observed = counts_by_period.sum(axis=2)
    x_projection_model = model_planes.sum(axis=2)
    x_projection_background = sum(
        yields["BxSy"] + yields["BxBy"] for yields in component_yields
    ) * bg_frac_x
    y_projection_observed = counts_by_period.sum(axis=1)
    y_projection_model = model_planes.sum(axis=1)
    y_projection_background = sum(
        yields["SxBy"] + yields["BxBy"] for yields in component_yields
    ) * bg_frac_y

    # ---- 稠密投影曲线（诊断图光滑绘制） -------------------------------------
    # 曲线值 = [m−Δ/2, m+Δ/2] 内的期望计数（Δ 为平均原始 bin 宽），与逐 bin
    # 期望同一定义：在原始 bin 中心处与投影数组一致，同时随 m 连续光滑。
    def signal_axis_cdf(mass, mean_value, names, lo, hi):
        sqrt2 = np.sqrt(2.0)
        sigma_narrow_value = nominal[names[0]]
        sigma_wide_value = sigma_narrow_value + nominal[names[1]]
        fraction_value = nominal[names[2]]

        def erf_growth(values, sigma):
            return erf((values - mean_value) / (sigma * sqrt2)) - erf(
                (lo - mean_value) / (sigma * sqrt2)
            )

        norm = fraction_value * erf_growth(hi, sigma_narrow_value) + (
            1.0 - fraction_value
        ) * erf_growth(hi, sigma_wide_value)
        return (
            fraction_value * erf_growth(mass, sigma_narrow_value)
            + (1.0 - fraction_value) * erf_growth(mass, sigma_wide_value)
        ) / norm

    def background_axis_density(mass, model, renamed, lo, hi):
        params = {
            original: nominal[name]
            for original, name in zip(model.parameter_names, renamed)
        }
        raw = np.asarray(model._numpy_evaluator(np.asarray(mass), params), dtype=float)
        if raw.shape != np.shape(mass):
            raw = np.broadcast_to(raw, np.shape(mass)).copy()
        nodes, weights = _gauss_legendre_points(lo, hi, 256)
        node_values = np.asarray(model._numpy_evaluator(nodes, params), dtype=float)
        return raw / float(np.dot(weights, node_values))

    def background_axis_cdf(mass, model, renamed, lo, hi):
        grid = np.linspace(lo, hi, 4001)
        density = background_axis_density(grid, model, renamed, lo, hi)
        cdf = np.concatenate(
            ([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid)))
        )
        return np.interp(mass, grid, cdf)

    def axis_dense_curves(axis):
        if axis == "x":
            lo, hi, n_bins = x_lo, x_hi, n_xbins
            mean_value, names = mean_x, signal_names_x
            model_bg, renamed_bg = background_model_x, bg_names_x
        else:
            lo, hi, n_bins = y_lo, y_hi, n_ybins
            mean_value, names = mean_y, signal_names_y
            model_bg, renamed_bg = background_model_y, bg_names_y
        # 曲线定义域收缩到 bin 中心范围：滑动半宽窗口必须完整落在拟合区间内。
        step = 0.5 * (hi - lo) / n_bins
        dense_mass = np.linspace(lo + step, hi - step, 2000)
        edges_low = dense_mass - step
        edges_high = dense_mass + step
        signal_fraction = signal_axis_cdf(
            edges_high, mean_value, names, lo, hi
        ) - signal_axis_cdf(edges_low, mean_value, names, lo, hi)
        background_fraction = background_axis_cdf(
            edges_high, model_bg, renamed_bg, lo, hi
        ) - background_axis_cdf(edges_low, model_bg, renamed_bg, lo, hi)
        return (
            dense_mass,
            signal_total[axis] * signal_fraction
            + background_total[axis] * background_fraction,
            background_total[axis] * background_fraction,
        )

    signal_total = {
        "x": sum(yields["SxSy"] + yields["SxBy"] for yields in component_yields),
        "y": sum(yields["SxSy"] + yields["BxSy"] for yields in component_yields),
    }
    background_total = {
        "x": sum(yields["BxSy"] + yields["BxBy"] for yields in component_yields),
        "y": sum(yields["SxBy"] + yields["BxBy"] for yields in component_yields),
    }
    (
        x_projection_dense_mass,
        x_projection_dense_model,
        x_projection_dense_background,
    ) = axis_dense_curves("x")
    (
        y_projection_dense_mass,
        y_projection_dense_model,
        y_projection_dense_background,
    ) = axis_dense_curves("y")

    # ---- 分量模型（供 transfer 使用） ---------------------------------------
    def make_signal_pdf(
        mass_lo_value: float,
        mass_hi_value: float,
        names: tuple[str, str, str],
        mean_value: float,
    ) -> NormalizedDensity1D:
        density = _double_gaussian_density(mean_value, names)
        return NormalizedDensity1D(
            mass_lo_value, mass_hi_value, list(names), density
        )

    def make_background_pdf(
        model: BackgroundModel,
        mass_lo_value: float,
        mass_hi_value: float,
        renamed: list[str],
    ) -> NormalizedDensity1D:
        original_names = model.parameter_names

        def density(mass, params):
            values = {
                original: params[name]
                for original, name in zip(original_names, renamed)
            }
            result = np.asarray(
                model._numpy_evaluator(np.asarray(mass, dtype=float), values),
                dtype=float,
            )
            if result.shape != np.shape(mass):
                result = np.broadcast_to(result, np.shape(mass)).copy()
            return result

        return NormalizedDensity1D(
            mass_lo_value, mass_hi_value, list(renamed), density
        )

    signal_pdf_x = make_signal_pdf(x_lo, x_hi, signal_names_x, mean_x)
    signal_pdf_y = make_signal_pdf(y_lo, y_hi, signal_names_y, mean_y)
    background_pdf_x = make_background_pdf(
        background_model_x, x_lo, x_hi, bg_names_x
    )
    background_pdf_y = make_background_pdf(
        background_model_y, y_lo, y_hi, bg_names_y
    )

    component_models = [
        ComponentModel2D("SxSy", signal_pdf_x, signal_pdf_y),
        ComponentModel2D("BxSy", background_pdf_x, signal_pdf_y),
        ComponentModel2D("SxBy", signal_pdf_x, background_pdf_y),
        ComponentModel2D("BxBy", background_pdf_x, background_pdf_y),
    ]

    # MINOS 在固定 N_SS 时重新极小化其余本底与产额参数；边界处的区间
    # 因而不需要把协方差误差的负半轴解释为物理产额。默认关闭以免拖慢批量拟合。
    signal_yield_profile_interval = None
    if profile_signal_yield:
        signal_parameter = yield_lookup[0]["SxSy"]
        try:
            profile_errors, new_result = result.errors(
                params=[signal_parameter], method="minuit_minos", cl=0.682689492
            )
            profile = profile_errors[signal_parameter]
            if new_result is not None:
                fit_converged = False
                raise RuntimeError("MINOS 找到了新的极小值，原拟合已标记为未收敛")
            if not profile["is_valid"]:
                raise RuntimeError("MINOS 未能给出有效的 N_SS 区间")
            signal_yield = component_yields[0]["SxSy"]
            signal_yield_profile_interval = (
                max(0.0, signal_yield + float(profile["lower"])),
                signal_yield + float(profile["upper"]),
            )
        except Exception as exc:
            warnings.warn(
                f"N_SS 轮廓区间缺失，保留原拟合中心值与协方差误差：{exc!r}",
                RuntimeWarning,
            )

    return FitResult2D(
        x_edges=x_edges,
        y_edges=y_edges,
        observed_counts_by_period=counts_by_period,
        model_counts_by_period=model_planes,
        component_names=component_names,
        component_yields_by_period=component_yields,
        parameter_names=parameter_names,
        parameter_values=parameter_values,
        parameter_covariance=covariance,
        symbolfit_initial_values_x=dict(bg_initial_x),
        symbolfit_initial_values_y=dict(bg_initial_y),
        nll_value=float(result.fmin),
        fit_nbins=int(fit_nbins),
        n_periods=int(n_periods),
        converged=fit_converged,
        x_projection_observed=x_projection_observed,
        x_projection_model=x_projection_model,
        x_projection_background=x_projection_background,
        y_projection_observed=y_projection_observed,
        y_projection_model=y_projection_model,
        y_projection_background=y_projection_background,
        x_projection_dense_mass=x_projection_dense_mass,
        x_projection_dense_model=x_projection_dense_model,
        x_projection_dense_background=x_projection_dense_background,
        y_projection_dense_mass=y_projection_dense_mass,
        y_projection_dense_model=y_projection_dense_model,
        y_projection_dense_background=y_projection_dense_background,
        component_models=component_models,
        x_background_formula=background_model_x.formula,
        y_background_formula=background_model_y.formula,
        signal_yield_profile_interval=signal_yield_profile_interval,
    )
