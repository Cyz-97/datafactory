"""sideband_ana 模块的 notebook-style 流程测试。

以 tests/data/sideband_ana/ 的 DELPHI Λ-Λbar 真实数据夹具为输入，按
分析流程逐步执行：区域契约 -> 表达式编译器 -> 一维质量谱拟合（固定 /
profile 本底）-> 一维 transfer -> 二维质量平面拟合 -> 二维 transfer ->
二维减除 -> 报告输出。每个 cell 打印中间量；全部图表通过
datafactory.stat.sideband_ana.report 的 API 输出（tests/sideband_ana/
output/reports/ 下的 PDF），transfer 系数只打印到 CLI。

运行（需要 root6.34 环境，约 3-8 分钟，PySR 搜索占大头）::

    python tests/sideband_ana/test_sideband_ana.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from datafactory.stat.sideband_ana import (  # noqa: E402
    MassRegions1D,
    calculate_transfer_factor_1d,
    calculate_transfer_factors_2d,
    regions_from_offsets,
    subtract_sideband_1d,
    subtract_sideband_2d,
)
from datafactory.stat.sideband_ana.fit import (  # noqa: E402
    BackgroundModel,
    ComponentModel2D,
    NormalizedDensity1D,
    _compile_expression,
    fit_mass_plane_2d,
    fit_mass_spectrum_1d,
)
from datafactory.stat.sideband_ana.report import (  # noqa: E402
    write_fit_report_1d,
    write_fit_report_2d,
    write_transfer_summary,
)

FIXTURE_DIR = REPO_ROOT / "tests" / "data" / "sideband_ana"
OUTPUT_DIR = REPO_ROOT / "tests" / "sideband_ana" / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
RANDOM_SEED = 42

_region_indices = {"SS": 0, "BS": 1, "SB": 2, "BB": 3}
# 模块 key -> 夹具 region 下标（B 是上边带，对应模块的 H）。
_region_index_map = {
    ("S", "S"): 0,
    ("H", "S"): 1,
    ("S", "H"): 2,
    ("H", "H"): 3,
}


def cell(title: str) -> None:
    print(f"\n{'=' * 78}\n# {title}\n{'=' * 78}")


CHECK_FAILURES: list[str] = []


def check(description: str, condition, detail: str = "") -> None:
    """只记录结果，不中断流程；最后统一汇总。"""
    status = "PASS" if bool(condition) else "FAIL"
    suffix = f"  [{detail}]" if detail else ""
    print(f"  [{status}] {description}{suffix}")
    if not condition:
        CHECK_FAILURES.append(f"{description}{suffix}")


# ---------------------------------------------------------------------------
cell("Cell 0: 载入夹具与元数据")
# ---------------------------------------------------------------------------

data = np.load(FIXTURE_DIR / "sideband_ll_data_cat0_llbar.npz")
metadata = json.loads(
    (FIXTURE_DIR / "sideband_ll_data_cat0_llbar.json").read_text()
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sample_metadata = {
    "caption": "DELPHI Lambda-Lambdabar, all-event, llbar, tight "
               "(sideband_ana module test)",
    "sample": "data",
    "selection": "tight",
    "source_script": "tests/sideband_ana/test_sideband_ana.py",
    "x_label": r"$m_{p\pi^-}$ [GeV]",
    "y_label": r"$m_{p\pi^-}$ [GeV]",
}

PEAK_MEAN = metadata["mass_fit"]["peak_mean_gev"]
SIGNAL_HALF_WIDTH = metadata["regions"]["signal_window_gev"]
SIDEBAND_OFFSET = metadata["regions"]["sideband_offset_gev"]
FIT_RANGE = (1.095, 1.145)

print(f"peak mean = {PEAK_MEAN:.7f} GeV（夹具 09 拟合参考值）")
print(f"signal half width = {SIGNAL_HALF_WIDTH} GeV, "
      f"sideband offset = {SIDEBAND_OFFSET} GeV")
print(f"mass spectrum total = {data['mass_fit_counts'].sum():.0f}")
print(f"mass plane total   = {data['mass_plane_counts'].sum():.0f}")

regions = regions_from_offsets(
    PEAK_MEAN,
    signal_half_width=SIGNAL_HALF_WIDTH,
    sideband_high_offset=SIDEBAND_OFFSET,
)
print(f"regions: {regions.region_intervals()}")

# ---------------------------------------------------------------------------
cell("Cell 1: 区域契约验证")
# ---------------------------------------------------------------------------

good = MassRegions1D(signal=(1.0, 1.1), sideband_low=(0.9, 0.95), sideband_high=(1.15, 1.2))
check("双边带正常构造", good.labels() == ["S", "L", "H"], str(good.labels()))

invalid_configs = [
    ("无 sideband", dict(signal=(1.0, 1.1))),
    ("low > high", dict(signal=(1.1, 1.0), sideband_high=(1.2, 1.3))),
    ("low sideband 覆盖 signal", dict(signal=(1.0, 1.1), sideband_low=(1.05, 1.2))),
    ("high sideband 覆盖 signal", dict(signal=(1.0, 1.1), sideband_high=(0.95, 1.05))),
]
for name, config in invalid_configs:
    try:
        MassRegions1D(**config)
        check(f"拒绝无效配置: {name}", False)
    except ValueError:
        check(f"拒绝无效配置: {name}", True)

try:
    good.validate_within(1.0, 1.05)
    check("拒绝越界区间", False)
except ValueError:
    check("拒绝越界区间", True)

# ---------------------------------------------------------------------------
cell("Cell 2: 表达式白名单编译器")
# ---------------------------------------------------------------------------

supported = {
    "a1 + a2*x0": {("a1", 2.0), ("a2", 3.0)},
    "a1*(x0 - a2)**2": {("a1", 2.0), ("a2", 1.0)},
    "exp(a1*x0 + a2)": {("a1", -0.5), ("a2", 0.0)},
    "square(a1*x0) + 3": {("a1", 1.5)},
    "12615500.0*(13.0457622375 - 10.5615*x0)**2*(10.5615*x0 - 12.5465822375)**2*(10.5615*x0 - 11.4536207375) + 11049424.4455": set(),
}
for formula, params in supported.items():
    names = sorted(name for name, _ in params)
    evaluator = _compile_expression(formula, names, np.exp, np.square)
    values = {name: value for name, value in params}
    result = evaluator(np.array([1.0, 2.0]), values)
    check(f"编译支持的表达式: {formula[:50]}{'...' if len(formula) > 50 else ''}",
          np.all(np.isfinite(result)))

rejected = [
    "a1/x0",                # 除法
    "a1**x0",               # 幂指数非常量
    "a1**2.5",              # 幂指数非整数
    "a1**(x0**2)",          # 嵌套幂
    "exp(a1*x0**2 + a2)",   # exp 参数次数 > 1
    "exp(exp(x0))",         # 嵌套 exp（白名单外）
    "sin(x0)",              # 未知函数
    "a1 + b1",              # 未知参数符号
]
for formula in rejected:
    try:
        _compile_expression(formula, ["a1"], np.exp, np.square)
        check(f"拒绝表达式: {formula}", False)
    except ValueError:
        check(f"拒绝表达式: {formula}", True)

model = BackgroundModel("a1*(x0 - a2)**2 + a3", ["a1", "a2", "a3"],
                        {"a1": 2.0, "a2": 1.0, "a3": 0.5})
integral = model.integrate(0.0, 2.0, {"a1": 2.0, "a2": 1.0, "a3": 0.5})
# ∫ 2(m-1)^2 + 0.5 dm over [0,2] = 2*(2/3) + 1 = 7/3
check("BackgroundModel GL 积分与解析值一致", abs(integral - 7.0 / 3.0) < 1e-10,
      f"GL={integral:.10f}")

# 夹具 JSON 里的 09 参考本底公式必须能被编译（回归保护；该式已把参数
# 数值代入，不含 a{N} 符号）。
reference_formula = metadata["mass_fit"]["background_formula"]
reference_model = BackgroundModel(reference_formula, [])
reference_prediction = reference_model.evaluate_density(
    np.array([1.10, 1.12, 1.16]), {}
)
check("09 参考本底公式通过白名单编译",
      np.all(np.isfinite(reference_prediction)) and np.all(reference_prediction >= 0))

# ---------------------------------------------------------------------------
cell("Cell 3: 平坦本底的解析传递检验（r=1, w_H=w_V=1, w_C=-1）")
# ---------------------------------------------------------------------------


class _FlatFit:
    mass_edges = np.linspace(1.0, 1.3, 31)
    parameter_names = ["c0"]
    parameter_values = np.array([5.0])
    parameter_covariance = np.array([[0.01]])
    background_model = BackgroundModel("c0", ["c0"], {"c0": 5.0})


flat_regions = regions_from_offsets(
    1.15, signal_half_width=0.01, sideband_low_offset=0.03, sideband_high_offset=0.03
)
tf_flat = calculate_transfer_factor_1d(_FlatFit, flat_regions)
# 平坦本底: r = I_S / (I_L + I_H) = w_S / (w_L + w_H) = 0.5（两侧边带等宽）。
check("平坦本底 r = 0.5（等宽双边带）", abs(tf_flat.r_combined - 0.5) < 1e-12,
      f"r={tf_flat.r_combined:.12f}")


def _flat_density(mass, params):
    return np.ones_like(np.asarray(mass, dtype=float))


def _peak_density(mass, params):
    return np.exp(-0.5 * np.square((np.asarray(mass) - 1.15) / 0.002))


_bg_pdf = NormalizedDensity1D(1.0, 1.3, [], _flat_density)
_sig_pdf = NormalizedDensity1D(1.0, 1.3, ["s"], _peak_density)


class _FlatFit2D:
    x_edges = np.linspace(1.0, 1.3, 31)
    y_edges = np.linspace(1.0, 1.3, 31)
    parameter_names = ["s"]
    parameter_values = np.array([0.002])
    parameter_covariance = np.array([[1e-8]])
    component_models = [
        ComponentModel2D("SxSy", _sig_pdf, _sig_pdf),
        ComponentModel2D("BxSy", _bg_pdf, _sig_pdf),
        ComponentModel2D("SxBy", _sig_pdf, _bg_pdf),
        ComponentModel2D("BxBy", _bg_pdf, _bg_pdf),
    ]


tf2_flat = calculate_transfer_factors_2d(_FlatFit2D, flat_regions)
# 平坦本底（等宽双边带）解析值:
#   w_H = w_V = f(S) / f(L∪H) = 0.5，
#   w_C = (f(S)² - w_H·f(B)f(S) - w_V·f(S)f(B)) / f(B)² = -0.25。
check("平坦本底 w_H = 0.5（等宽双边带）", abs(tf2_flat.w_H - 0.5) < 1e-12,
      f"{tf2_flat.w_H:.12f}")
check("平坦本底 w_V = 0.5（等宽双边带）", abs(tf2_flat.w_V - 0.5) < 1e-12,
      f"{tf2_flat.w_V:.12f}")
check("平坦本底 w_C = -0.25（等宽双边带）", abs(tf2_flat.w_C + 0.25) < 1e-12,
      f"{tf2_flat.w_C:.12f}")
check("平坦本底 closure = 0", abs(tf2_flat.factorization_closure) < 1e-12)
check("双边带九区域枚举", len(tf2_flat.atomic_region_integrals) == 9,
      f"{len(tf2_flat.atomic_region_integrals)} 个")

# ---------------------------------------------------------------------------
cell("Cell 4: 一维质量谱拟合（本底固定，复现 09 一维路径）")
# ---------------------------------------------------------------------------

fit1d_fixed = fit_mass_spectrum_1d(
    data["mass_fit_edges"],
    data["mass_fit_counts"],
    data["mass_fit_variances"],
    fit_range=FIT_RANGE,
    regions=regions,
    random_seed=RANDOM_SEED,
    profile_background=False,
    symbolfit_output_dir=OUTPUT_DIR / "symbolfit_1d_fixed",
)
print(f"  SymbolFit 本底式: {fit1d_fixed.background_formula}")
print(f"  chi2/ndf = {fit1d_fixed.chi2:.1f}/{fit1d_fixed.ndf} = "
      f"{fit1d_fixed.chi2 / fit1d_fixed.ndf:.3f}")
print(f"  peak mean = {fit1d_fixed.peak_mean:.7f} ± "
      f"{np.sqrt(fit1d_fixed.peak_mean_variance):.2e} GeV")
check("固定本底: chi2/ndf < 2.5", fit1d_fixed.chi2 / fit1d_fixed.ndf < 2.5,
      f"{fit1d_fixed.chi2 / fit1d_fixed.ndf:.3f}")
check("固定本底: 峰位与 09 参考一致 (<0.3 MeV)",
      abs(fit1d_fixed.peak_mean - PEAK_MEAN) < 3.0e-4,
      f"|Δm| = {abs(fit1d_fixed.peak_mean - PEAK_MEAN) * 1000:.3f} MeV")

transfer_1d = calculate_transfer_factor_1d(fit1d_fixed, regions)
print(f"  r = {transfer_1d.r_combined:.4f} ± {transfer_1d.sigma_r_combined:.4f}")
check("固定本底: r 在合理范围", 0.3 < transfer_1d.r_combined < 3.0,
      f"r = {transfer_1d.r_combined:.4f}")
check("固定本底: r 误差为正", transfer_1d.sigma_r_combined > 0)

# 一维诊断图通过 report API 输出（数据/模型/本底/残差/参数面板）。
artifacts_1d_fixed = write_fit_report_1d(
    fit1d_fixed,
    transfer_1d,
    sample_metadata=sample_metadata,
    output_dir=REPORTS_DIR,
    stem="llbar_1d_fixed_bg",
)
for path in artifacts_1d_fixed.paths:
    print(f"  -> {path}")

# ---------------------------------------------------------------------------
cell("Cell 5: 一维质量谱拟合（本底 profile，zfit 联合估计）")
# ---------------------------------------------------------------------------

fit1d_profiled = fit_mass_spectrum_1d(
    data["mass_fit_edges"],
    data["mass_fit_counts"],
    data["mass_fit_variances"],
    fit_range=FIT_RANGE,
    regions=regions,
    random_seed=RANDOM_SEED,
    profile_background=True,
    symbolfit_output_dir=OUTPUT_DIR / "symbolfit_1d_profiled",
)
print(f"  SymbolFit 本底式: {fit1d_profiled.background_formula}")
print(f"  chi2/ndf = {fit1d_profiled.chi2:.1f}/{fit1d_profiled.ndf} = "
      f"{fit1d_profiled.chi2 / fit1d_profiled.ndf:.3f}")
print(f"  peak mean = {fit1d_profiled.peak_mean:.7f} GeV")
for name, value in zip(fit1d_profiled.parameter_names,
                       fit1d_profiled.parameter_values):
    print(f"    {name} = {value:.5g}")

check("profile 本底: chi2/ndf < 2.5",
      fit1d_profiled.chi2 / fit1d_profiled.ndf < 2.5,
      f"{fit1d_profiled.chi2 / fit1d_profiled.ndf:.3f}")
check("profile 本底: 峰位与 09 参考一致 (<0.3 MeV)",
      abs(fit1d_profiled.peak_mean - PEAK_MEAN) < 3.0e-4,
      f"|Δm| = {abs(fit1d_profiled.peak_mean - PEAK_MEAN) * 1000:.3f} MeV")
check("profile 本底: covariance 为有限正定",
      np.all(np.isfinite(fit1d_profiled.parameter_covariance))
      and np.all(np.diag(fit1d_profiled.parameter_covariance) > 0))

transfer_1d_profiled = calculate_transfer_factor_1d(fit1d_profiled, regions)
print(f"  r = {transfer_1d_profiled.r_combined:.4f} ± "
      f"{transfer_1d_profiled.sigma_r_combined:.4f}")

# profile 本底诊断图同样走 report API。
artifacts_1d_profiled = write_fit_report_1d(
    fit1d_profiled,
    transfer_1d_profiled,
    sample_metadata=sample_metadata,
    output_dir=REPORTS_DIR,
    stem="llbar_1d_profiled_bg",
)
for path in artifacts_1d_profiled.paths:
    print(f"  -> {path}")

# ---------------------------------------------------------------------------
cell("Cell 6: 二维质量平面拟合（共享轴，四分量）")
# ---------------------------------------------------------------------------

counts_by_period = data["mass_plane_counts"][np.newaxis, :, :]
fit2d = fit_mass_plane_2d(
    data["mass_plane_x_edges"],
    data["mass_plane_y_edges"],
    counts_by_period,
    x_seed=fit1d_profiled,
    y_seed=None,
    fit_nbins=20,
    random_seed=RANDOM_SEED,
)
observed_total = float(counts_by_period.sum())
model_total = float(fit2d.model_counts_by_period.sum())
print(f"  NLL = {fit2d.nll_value:.2f}")
for period, yields in enumerate(fit2d.component_yields_by_period):
    print(f"  period {period} yields: "
          + ", ".join(f"{k}={v:.0f}" for k, v in yields.items()))
print(f"  observed total = {observed_total:.0f}, model total = {model_total:.0f}")

check("2D 拟合: 产额全部非负",
      all(v >= 0 for yields in fit2d.component_yields_by_period
          for v in yields.values()))
check("2D 拟合: 模型总量与观测一致 (<0.5%)",
      abs(model_total - observed_total) / observed_total < 5.0e-3,
      f"{abs(model_total - observed_total) / observed_total:.4%}")
check("2D 拟合: covariance 为有限正定",
      np.all(np.isfinite(fit2d.parameter_covariance))
      and np.all(np.diag(fit2d.parameter_covariance) >= 0))

# 投影残差检验：在拟合网格（20 个合并 bin）上计算 pull。细网格 bin 间
# 的局部结构（尤其 [1.08, 1.085] 陡峭边缘）不是平滑模型能逐 bin 描述的。
x_model = fit2d.x_projection_model.sum(axis=0)
x_observed = fit2d.x_projection_observed.sum(axis=0)
merge = x_observed.size // 20
x_model_fitgrid = x_model.reshape(20, merge).sum(axis=1)
x_observed_fitgrid = x_observed.reshape(20, merge).sum(axis=1)
x_pull = (x_observed_fitgrid - x_model_fitgrid) / np.sqrt(
    np.maximum(x_model_fitgrid, 1.0)
)
check("2D 拟合: x 投影逐 bin pull 合理（拟合网格）", np.max(np.abs(x_pull)) < 5.0,
      f"max|pull| = {np.max(np.abs(x_pull)):.2f}, chi2/20 = {np.sum(x_pull**2) / 20:.2f}")

# 二维诊断图（平面三联图 + 区域事例数标注 + 投影图）在 transfer 算完后
# 通过 write_fit_report_2d 输出（见 Cell 7）。稠密曲线一致性检查：在原始
# bin 中心处应与逐 bin 期望一致（同一积分定义）。
for axis_name, dense_m, dense_mod, perbin_mod, centers in (
    ("x", fit2d.x_projection_dense_mass, fit2d.x_projection_dense_model,
     fit2d.x_projection_model.sum(axis=0),
     0.5 * (fit2d.x_edges[:-1] + fit2d.x_edges[1:])),
    ("y", fit2d.y_projection_dense_mass, fit2d.y_projection_dense_model,
     fit2d.y_projection_model.sum(axis=0),
     0.5 * (fit2d.y_edges[:-1] + fit2d.y_edges[1:])),
):
    dense_at_centers = np.interp(centers, dense_m, dense_mod)
    max_rel = float(np.max(np.abs(dense_at_centers / perbin_mod - 1.0)))
    check(f"2D 投影 {axis_name}: 稠密曲线在 bin 中心与逐 bin 期望一致 (<0.5%)",
          max_rel < 5.0e-3, f"max rel dev = {max_rel:.3%}")

# ---------------------------------------------------------------------------
cell("Cell 7: 二维 transfer coefficients")
# ---------------------------------------------------------------------------

transfer_2d = calculate_transfer_factors_2d(fit2d, regions)
sigma_w = np.sqrt(np.clip(np.diag(transfer_2d.weight_covariance), 0.0, None))
print(f"  w_H = {transfer_2d.w_H:.4f} ± {sigma_w[0]:.4f}")
print(f"  w_V = {transfer_2d.w_V:.4f} ± {sigma_w[1]:.4f}")
print(f"  w_C = {transfer_2d.w_C:.4f} ± {sigma_w[2]:.4f}")
print(f"  closure w_C + w_H*w_V = {transfer_2d.factorization_closure:.4f}")
print("  signal leakage:")
for key, value in transfer_2d.signal_leakage_by_region.items():
    print(f"    {key}: {value:.5f}")
print("  aggregated integrals:")
for component, values in transfer_2d.aggregated_region_integrals.items():
    print(f"    {component}: "
          + ", ".join(f"{k}={v:.4g}" for k, v in values.items()))

check("2D transfer: w_H 在合理范围", 0.3 < transfer_2d.w_H < 3.0,
      f"{transfer_2d.w_H:.4f}")
check("2D transfer: w_V 在合理范围", 0.3 < transfer_2d.w_V < 3.0,
      f"{transfer_2d.w_V:.4f}")
check("2D transfer: w_C 在合理范围", -3.0 < transfer_2d.w_C < 1.0,
      f"{transfer_2d.w_C:.4f}")
check("2D transfer: 权重协方差有限", np.all(np.isfinite(transfer_2d.weight_covariance)))
check("2D transfer: 相关系数在 [-1, 1]",
      np.all(np.abs(transfer_2d.weight_correlation) <= 1.0 + 1e-9))
max_leakage = max(transfer_2d.signal_leakage_by_region.values())
check("2D transfer: 信号泄漏 < 5%", max_leakage < 0.05,
      f"max leakage = {max_leakage:.4f}")

# transfer 系数只打印到 CLI，不画柱状图。
# 二维诊断图（平面三联图 + 区域事例数标注 + 积分/w 页 + 投影图）：
artifacts_2d = write_fit_report_2d(
    fit2d,
    transfer_2d,
    sample_metadata=sample_metadata,
    output_dir=REPORTS_DIR,
    stem="llbar_plane",
)
for path in artifacts_2d.paths:
    print(f"  -> {path}")

# ---------------------------------------------------------------------------
cell("Cell 8: 二维减除（Δφ_thrust 分布，4 个质量区域 × 10 bin）")
# ---------------------------------------------------------------------------

delta_phi_edges = data["delta_phi_thrust_edges"]
delta_phi_counts = {
    key: data["delta_phi_thrust_counts"][index]
    for key, index in _region_index_map.items()
}
delta_phi_variances = {
    key: data["delta_phi_thrust_variances"][index]
    for key, index in _region_index_map.items()
}
for key, counts in delta_phi_counts.items():
    print(f"  {key}: total = {counts.sum():.0f}")

subtraction = subtract_sideband_2d(delta_phi_counts, delta_phi_variances,
                                   transfer_2d)
subtracted_total = float(subtraction.subtracted_signal.sum())
subtracted_total_variance = float(subtraction.signal_variance.sum())
fitted_signal_ss = fit2d.component_yields_by_period[0]["SxSy"]
print(f"  减除后信号总数 = {subtracted_total:.0f} ± "
      f"{np.sqrt(subtracted_total_variance):.0f}")
print(f"  2D 拟合 SS 信号产额 = {fitted_signal_ss:.0f}")
print(f"  SS 观测总数 = {subtraction.observed_signal_region.sum():.0f}")
print(f"  估计本底总数 = {subtraction.estimated_background.sum():.0f} ± "
      f"{np.sqrt(subtraction.background_variance.sum()):.0f}")
print(f"  负 bin 数 = {int(subtraction.negative_bin_mask.sum())}")

check("2D 减除: 信号总数为正", subtracted_total > 0, f"{subtracted_total:.0f}")
check("2D 减除: 信号总数小于 SS 观测", subtracted_total < subtraction.observed_signal_region.sum())
check("2D 减除: 与 2D 拟合 SS 产额一致 (<5σ)",
      abs(subtracted_total - fitted_signal_ss)
      < 5.0 * np.sqrt(subtracted_total_variance + fitted_signal_ss),
      f"|Δ| = {abs(subtracted_total - fitted_signal_ss):.0f}")
per_bin_pull = subtraction.subtracted_signal / np.sqrt(subtraction.signal_variance)
check("2D 减除: 无严重负 bin (> -4σ)", np.min(per_bin_pull) > -4.0,
      f"min pull = {np.min(per_bin_pull):.2f}")

# 一维减除交叉检验：用 r 对同一 Δφ 分布做单边带减除。
subtraction_1d = subtract_sideband_1d(
    delta_phi_counts[("S", "S")],
    delta_phi_variances[("S", "S")],
    high_sideband_counts=delta_phi_counts[("H", "S")],
    high_sideband_variances=delta_phi_variances[("H", "S")],
    transfer=transfer_1d_profiled,
)
print(f"  1D r 减除信号总数 = {subtraction_1d.subtracted_signal.sum():.0f} ± "
      f"{np.sqrt(subtraction_1d.signal_variance.sum()):.0f}")

# ---------------------------------------------------------------------------
cell("9: transfer 汇总与产物检查")
# ---------------------------------------------------------------------------

artifacts_summary = write_transfer_summary(
    [
        {
            "label": "llbar_fixed_bg",
            "scope": "cat0/llbar/tight (background fixed)",
            "fallback": None,
            "regions_x": regions,
            "regions_y": None,
            "transfer_1d": transfer_1d,
            "transfer_2d": None,
        },
        {
            "label": "llbar_profiled_bg",
            "scope": "cat0/llbar/tight (background profiled)",
            "fallback": None,
            "regions_x": regions,
            "regions_y": None,
            "transfer_1d": transfer_1d_profiled,
            "transfer_2d": None,
        },
        {
            "label": "llbar_2d",
            "scope": "cat0/llbar/tight (mass-plane 2D)",
            "fallback": None,
            "regions_x": regions,
            "regions_y": regions,
            "transfer_1d": None,
            "transfer_2d": transfer_2d,
        },
    ],
    analysis_metadata={
        "analysis": "sideband_ana module test",
        "channel": metadata.get("channel", ""),
        "selection": metadata.get("selection", ""),
        "peak_mean_gev": PEAK_MEAN,
    },
    output_dir=REPORTS_DIR,
    stem="transfer_factors",
)
all_report_paths = (
    artifacts_1d_fixed.paths
    + artifacts_1d_profiled.paths
    + artifacts_2d.paths
    + artifacts_summary.paths
)
for path in all_report_paths:
    check(f"报告文件存在: {path.name}", path.exists() and path.stat().st_size > 0,
          f"{path.stat().st_size if path.exists() else 0} bytes")

# JSON 汇总应可解析且包含三个条目。
summary_json = json.loads((REPORTS_DIR / "transfer_factors.json").read_text())
check("transfer JSON 可解析且含 3 个条目", len(summary_json["results"]) == 3)

# ---------------------------------------------------------------------------
cell("Cell 10: 总结")
# ---------------------------------------------------------------------------

print(f"\n输出目录: {OUTPUT_DIR}")
if CHECK_FAILURES:
    print(f"完成，但有 {len(CHECK_FAILURES)} 项检查未通过：")
    for failure in CHECK_FAILURES:
        print(f"  - {failure}")
else:
    print("全部检查通过。")
