# `datafactory.stat.sideband_ana` 架构与实施计划

## 1. 文档目的

本文档定义一个面向通用一维、二维质量边带分析的统计模块。模块从已经形成的质量谱、质量平面和区域化目标 observable 出发，完成：

1. 一维和二维 Signal+Background 拟合；
2. 由拟合 PDF 的区域积分计算 transfer factor / transfer coefficients；
3. 在另一个 observable 维度执行一维边带减除或二维容斥减除；
4. 输出可复核的拟合、区域积分和 transfer coefficient 诊断报告。

统计原理必须与以下参考实现一致：

- `lambda-ana/scripts/pair_distributions/09_pair_jet_distribution_summary.py`；
- `lambda-ana/scripts/lambda_lambda/shared/symbolfit_background_estimator.py`；
- 用户提供的 RooFit 二维四分量拟合与九区域容斥示例。

新模块使用 SymbolFit 选择本底函数形式和初值，使用 zfit 完成最终参数估计。RooFit 示例仅作为四分量模型、区域积分和容斥公式的物理参考，不作为运行依赖。

## 2. 范围与非目标

### 2.1 模块负责

- 对 binned 一维质量谱进行 Signal+Background 拟合；
- 对逐时期、未加权二维质量平面进行 simultaneous extended Poisson 拟合；
- 支持手动设置 signal、低端 sideband 和高端 sideband 区间；
- 支持仅低端、仅高端或低端加高端联合 sideband；
- 从最终拟合分量的区域积分计算 transfer coefficients 及协方差；
- 对另一个 binned observable 的区域分布执行减除和误差传播；
- 输出拟合曲线、二维平面、残差、区域积分、参数和 transfer coefficient 报告。

### 2.2 模块不负责

- 事件选择、对象重建、truth matching 或 cut chain；
- `RDataFrame.Define`、区域 record 构造或扩展轴 slot 编码；
- Data/MC 年份权重和通道合并；
- category/subset 循环；
- 稀疏样本阈值和 global-fit fallback 策略；
- 最终分析 ROOT 文件的对象命名和发布目录管理；
- 通用 PDF factory、模型 registry、插件系统或 workflow manager。

上述内容继续由具体分析脚本显式控制，避免把分析策略误装成通用统计行为。

## 3. 包结构

Python 包名不能包含连字符，因此使用 `sideband_ana`：

```text
datafactory/
├── statistic.py
└── stat/
    ├── __init__.py
    └── sideband_ana/
        ├── __init__.py
        ├── fit.py
        ├── transfer.py
        ├── subtract.py
        └── report.py
```

- 现有 `datafactory/statistic.py` 暂不迁移，避免扩大修改范围。
- `sideband_ana/__init__.py` 仅重导出稳定公共 API，不承载计算逻辑。
- TensorFlow、zfit、SymbolFit 和 PySR 在拟合函数或 worker 内局部导入，避免普通 `datafactory` import 强制加载拟合栈。

数据依赖关系保持单向：

```text
mass spectrum / mass plane
            |
            v
          fit.py
            |
            v
       transfer.py ----------> report.py
            |                     ^
            v                     |
        subtract.py --------------+
```

`report.py` 只读取上游完整结果，不重新拟合、积分或传播误差。

## 4. 质量区域数据契约

### 4.1 手动区间是唯一规范表示

在 `transfer.py` 定义：

```python
@dataclass(frozen=True)
class MassRegions1D:
    signal: tuple[float, float]
    sideband_low: tuple[float, float] | None = None
    sideband_high: tuple[float, float] | None = None
```

典型双边带配置：

```python
regions = MassRegions1D(
    signal=(1.105, 1.125),
    sideband_low=(1.080, 1.100),
    sideband_high=(1.135, 1.155),
)
```

仅高端边带：

```python
regions = MassRegions1D(
    signal=(1.105, 1.125),
    sideband_high=(1.150, 1.170),
)
```

当前分析常用的“峰位加偏移量”通过便利函数转换成相同的显式区间对象：

```python
regions_from_offsets(
    peak_mean,
    *,
    signal_half_width,
    sideband_low_offset=None,
    sideband_high_offset=None,
) -> MassRegions1D
```

后续函数只接收明确的区间边界，不再解释 offset。

### 4.2 输入验证

构造区域时必须检查：

- 每个区间满足 `low < high`；
- 至少存在一个 sideband；
- signal 与各 sideband 不重叠；
- low/high sideband 互不重叠；
- low sideband 位于 signal 低端；
- high sideband 位于 signal 高端；
- 所有区间位于拟合质量范围内。

无效配置立即抛出 `ValueError`，不得自动裁剪、交换端点或隐藏配置错误。

## 5. SymbolFit 到 zfit 的拟合链

### 5.1 统一原则

拟合严格分成两个统计阶段：

1. SymbolFit 在 signal window 之外选择非负本底解析式及参数初值；
2. 将选中表达式转换为 TensorFlow/zfit 可微 PDF，由 zfit 完成最终 Signal+Background 参数估计和 covariance 计算。

SymbolFit operator 集合限制为已经验证的：

```text
+  -  *  exp  square
```

不实现任意 Python、SymPy 或字符串表达式执行器。表达式转换器只接受白名单 AST 节点；遇到不支持的运算符立即失败并报告 SymbolFit 候选式。

### 5.2 一维拟合

公共接口：

```python
fit_mass_spectrum_1d(
    mass_edges,
    counts,
    variances,
    *,
    fit_range,
    regions,
    signal_model,
    random_seed,
    profile_background=True,
) -> FitResult1D
```

输入统计意义：

- `mass_edges`：长度为 `n_bins + 1` 的质量 bin edges；
- `counts`：质量 bin 内的候选数或加权产额；
- `variances`：对应 Sumw2 方差；
- `fit_range`：最终 Signal+Background 拟合范围；
- `regions`：显式 signal/sideband 区间；
- `signal_model`：第一版只实现当前已验证的共峰位 double-Gaussian；
- `profile_background`：是否在 zfit 中继续拟合 SymbolFit 本底参数。

`profile_background=True` 是新模块的名义模式：SymbolFit 参数作为初值，最终值和 covariance 来自 zfit。这与附件中同时拟合信号和本底的原则一致。

`profile_background=False` 是兼容模式：固定 SymbolFit 本底 shape，仅用 zfit 拟合信号，用于复现当前 `09_pair_jet_distribution_summary.py` 的一维路径。报告必须明确记录该选择。

`FitResult1D` 至少包含：

```text
mass_edges
observed_counts
observed_variances
fit_range
background_profiled
parameter_names
parameter_values
parameter_covariance
peak_mean
peak_mean_variance
background_formula
symbolfit_initial_values
model_counts
background_counts
dense_mass
dense_model
dense_background
chi2
ndf
converged
background_model
```

`dense_*` 数组由拟合阶段产生并持久化，报告层不得重新计算曲线。

### 5.3 二维四分量拟合

公共接口：

```python
fit_mass_plane_2d(
    x_edges,
    y_edges,
    counts_by_period,
    *,
    x_seed: FitResult1D,
    y_seed: FitResult1D | None = None,
    fit_nbins,
    random_seed,
) -> FitResult2D
```

输入要求：

- `counts_by_period.shape == (n_periods, n_xbins, n_ybins)`；
- 输入必须是逐时期、未加权、有限且非负的原始计数；
- 不允许把按年份权重合并后的 MC 平面作为 Poisson 数据；
- `y_seed=None` 表示两轴使用同一个一维模型并共享 shape，适用于相同粒子的质量对；
- 指定 `y_seed` 表示两轴使用不同的一维模型，适用于附件中的不同粒子质量对。

二维模型固定为四个具有明确物理意义的分量：

$$
S_xS_y,\qquad B_xS_y,\qquad S_xB_y,\qquad B_xB_y.
$$

- 每个时期具有独立、非负的四分量产额；
- 信号和本底 shape 参数在时期之间共享；
- 使用 simultaneous extended binned Poisson NLL；
- ProductPDF 的 bin 积分可以利用一维积分的乘积计算，但必须与显式 ProductPDF 的归一化定义一致；
- 二维本底沿用 `09` 已验证的稳定路径：由一维 SymbolFit 最优曲线拟合
  `exp(Chebyshev-3)` 的初始 shape，随后由 zfit profile 三个 Chebyshev
  系数。SymbolFit 曲线因此是二维本底初值的统计来源，而不是只用于画图；
  一维名义路径则直接 profile SymbolFit 选出的解析式参数。

`FitResult2D` 至少包含：

```text
x_edges
y_edges
observed_counts_by_period
model_counts_by_period
component_names
component_yields_by_period
parameter_names
parameter_values
parameter_covariance
symbolfit_initial_values_x
symbolfit_initial_values_y
nll_value
fit_nbins
n_periods
converged
x_projection_observed/model/background
y_projection_observed/model/background
component_models
```

## 6. 二维九区域与容斥定义

### 6.1 原子区域

每条质量轴使用：

- `S`：signal；
- `L`：low sideband；
- `H`：high sideband。

二维最多形成九个互斥原子区域：

```text
SS  LS  HS
SL  LL  HL
SH  LH  HH
```

核心 API 使用 `(x_region, y_region)` key，例如 `("L", "S")`，禁止使用附件中的 `region 1...9` 数字编号。数字编号容易交换轴和高低端；附件活动代码中的一个 `range_9` 还重复了 high/high 并遗漏 high/low。实现必须验证原子区域 key 唯一，不能复现该错误。

### 6.2 聚合区域

定义：

$$
N_{BS}=N_{LS}+N_{HS},
$$

$$
N_{SB}=N_{SL}+N_{SH},
$$

$$
N_{BB}=N_{LL}+N_{LH}+N_{HL}+N_{HH}.
$$

单侧边带时，上述求和只包含实际存在的原子区域。

## 7. Transfer factor / coefficients

### 7.1 一维 transfer factor

公共接口：

```python
calculate_transfer_factor_1d(
    fit_result,
    regions,
) -> TransferFactor1D
```

对最终 zfit 本底 PDF $b(m;\theta)$ 定义：

$$
I_S=\int_S b(m;\theta)\,dm,
$$

$$
I_B=\int_L b(m;\theta)\,dm+\int_H b(m;\theta)\,dm,
$$

$$
r=\frac{I_S}{I_B}.
$$

只有单侧边带时，$I_B$ 自动退化为对应单项，但不得伪造缺失侧的观测量。

误差由最终拟合 covariance 传播：

$$
V_r=
\nabla_\theta r^\mathsf{T}
\operatorname{Cov}(\theta)
\nabla_\theta r.
$$

`TransferFactor1D` 至少包含：

```text
integral_signal
integral_sideband_low
integral_sideband_high
integral_sideband_combined
r_low
r_high
r_combined
variance_r_combined
sigma_r_combined
parameter_gradient
```

`r_low` 和 `r_high` 仅用于稳定性诊断；名义减除使用 `r_combined`。

### 7.2 二维 transfer coefficients

公共接口：

```python
calculate_transfer_factors_2d(
    fit_result,
    x_regions,
    y_regions=None,
) -> TransferFactors2D
```

`y_regions=None` 表示两轴共用同一套区间。

令 $f_A(R)$ 表示拟合分量 $A$ 在区域 $R$ 内的归一化 PDF 积分。水平方向系数为：

$$
w_H=
\frac{f_{B_xS_y}(SS)}
{f_{B_xS_y}(LS)+f_{B_xS_y}(HS)}.
$$

垂直方向系数为：

$$
w_V=
\frac{f_{S_xB_y}(SS)}
{f_{S_xB_y}(SL)+f_{S_xB_y}(SH)}.
$$

corner 容斥系数为：

$$
w_C=
\frac{
f_{B_xB_y}(SS)
-w_H\left[f_{B_xB_y}(LS)+f_{B_xB_y}(HS)\right]
-w_V\left[f_{B_xB_y}(SL)+f_{B_xB_y}(SH)\right]
}{
f_{B_xB_y}(LL)+f_{B_xB_y}(LH)
+f_{B_xB_y}(HL)+f_{B_xB_y}(HH)
}.
$$

以上分别对应附件中的 `weight_h`、`weight_v`、`weight_d`，也对应现有分析中的 `w_H`、`w_V`、`w_C`。

$w_C$ 是有符号的容斥系数，不是必须为正的概率。对完全因子化本底应满足：

$$
w_C=-w_Hw_V.
$$

因此 corner 项负责加回被水平和垂直减除重复扣除的部分。

权重 covariance 由最终 zfit covariance 传播：

$$
\operatorname{Cov}(\mathbf w)
=J_{\mathbf w,\theta}
\operatorname{Cov}(\theta)
J_{\mathbf w,\theta}^{\mathsf T},
$$

其中 $\mathbf w=(w_H,w_V,w_C)$。

`TransferFactors2D` 至少包含：

```text
atomic_region_integrals
aggregated_region_integrals
w_H
w_V
w_C
weight_covariance
weight_correlation
parameter_gradient
signal_leakage_by_region
factorization_closure
```

`factorization_closure` 至少记录 $w_C+w_Hw_V$，用于识别数值积分、模型定义或轴映射错误。

## 8. 在另一个 observable 上执行减除

### 8.1 一维质量选择对应的减除

公共接口：

```python
subtract_sideband_1d(
    signal_counts,
    signal_variances,
    *,
    low_sideband_counts=None,
    low_sideband_variances=None,
    high_sideband_counts=None,
    high_sideband_variances=None,
    transfer,
) -> SubtractionResult
```

定义：

$$
N_B=N_L+N_H,
\qquad
V_B=V_L+V_H,
$$

$$
N_{\mathrm{bkg}}=rN_B,
$$

$$
V_{\mathrm{bkg}}=r^2V_B+N_B^2V_r,
$$

$$
N_{\mathrm{sig}}=N_S-N_{\mathrm{bkg}},
\qquad
V_{\mathrm{sig}}=V_S+V_{\mathrm{bkg}}.
$$

### 8.2 二维质量选择对应的容斥减除

公共接口：

```python
subtract_sideband_2d(
    region_counts,
    region_variances,
    transfer,
) -> SubtractionResult
```

输入映射允许的 key 为现有区域配置实际产生的 `(x_region, y_region)`。双边带的完整集合为：

```python
("S", "S")
("L", "S")
("H", "S")
("S", "L")
("S", "H")
("L", "L")
("L", "H")
("H", "L")
("H", "H")
```

本底与信号估计严格采用：

$$
N_{\mathrm{bkg}}
=w_HN_{BS}+w_VN_{SB}+w_CN_{BB},
$$

$$
N_{\mathrm{sig}}
=N_{SS}-N_{\mathrm{bkg}}.
$$

令：

$$
\mathbf w=(w_H,w_V,w_C),
\qquad
\mathbf N=(N_{BS},N_{SB},N_{BB}),
$$

则逐 bin 方差为：

$$
V_{\mathrm{bkg}}
=\mathbf w^\mathsf{T}V_N\mathbf w
+\mathbf N^\mathsf{T}
\operatorname{Cov}(\mathbf w)
\mathbf N.
$$

对互斥区域且不考虑区域间统计相关性：

$$
V_N=\operatorname{diag}(V_{BS},V_{SB},V_{BB}).
$$

最后：

$$
V_{\mathrm{sig}}=V_{SS}+V_{\mathrm{bkg}}.
$$

`SubtractionResult` 统一包含：

```text
observed_signal_region
observed_variance
atomic_sideband_counts
aggregated_sideband_counts
estimated_background
background_variance
subtracted_signal
signal_variance
negative_bin_mask
```

负 bin 必须保留并报告，不得裁剪到零。第一版只返回逐 bin 边际方差；共享 transfer coefficient 引起的目标 observable bin-bin covariance 暂不构造。

## 9. 诊断报告

`report.py` 提供：

```python
write_fit_report_1d(
    fit_result,
    transfer_result,
    *,
    sample_metadata,
    output_dir,
    stem,
) -> ReportArtifacts
```

```python
write_fit_report_2d(
    fit_result,
    transfer_result,
    *,
    sample_metadata,
    output_dir,
    stem,
) -> ReportArtifacts
```

```python
write_transfer_summary(
    named_results,
    *,
    analysis_metadata,
    output_dir,
    stem="transfer_factors",
) -> ReportArtifacts
```

### 9.1 一维拟合报告

- 展示完整质量区间；
- data 使用 errorbar；
- 总拟合使用黑色光滑曲线；
- 本底使用蓝色光滑曲线；
- 不单独绘制 signal component 曲线；
- 主图下方绘制 `data / model - 1` residual；
- residual y 范围关于 0 对称，不超出 $(-5,5)$，ticks 不超过 6 个；
- 参数面板使用 LaTeX，显示参数及误差、$\chi^2/\mathrm{ndf}$；
- signal、low-sideband、high-sideband 使用灰色半透明竖直条带；
- 写出 SymbolFit 公式、SymbolFit 初值、最终 zfit 参数以及 background fixed/profiled 状态。

### 9.2 二维拟合报告

- 使用 `pcolormesh` 分别展示 observed、model 和 `data / model - 1`；
- colorbar 使用独立 axes；
- 两轴为相同物理量时使用 equal aspect；
- signal region 使用红框；
- 所有 sideband 原子区域使用蓝框；
- 输出 x/y 投影，每个投影具有主图和 residual；
- 输出四分量逐时期产额、NLL、参数 covariance/correlation；
- 输出四分量乘九区域积分表；
- 输出聚合后的 SS、BS、SB、BB 积分；
- 输出 $w_H$、$w_V$、$w_C$ 的代入式和 $w_C+w_Hw_V$ 闭合量。

### 9.3 Transfer summary

- 一维的 $I_S$、$I_L$、$I_H$、$r_L$、$r_H$ 和 $r_{\mathrm{combined}}$；
- 二维的原子区域积分、聚合积分和 coefficient covariance；
- signal leakage；
- global/category/subset 作用域和 fallback 来源；
- 手动使用的所有质量区间；
- Markdown 和 JSON 两种机器可读/人类可读结果。

建议输出：

```text
<label>_fit1d.pdf
<label>_fit2d_plane.pdf
<label>_fit2d_projection_x.pdf
<label>_fit2d_projection_y.pdf
transfer_factors.pdf
transfer_factors.md
transfer_factors.json
```

所有 PDF 的 `Subject` 元数据写入 caption、来源脚本、样本、选择和生成时间。PNG 只用于视觉检查，发布产品优先保留 PDF。

## 10. 与现有分析脚本的迁移边界

从 `09_pair_jet_distribution_summary.py` 迁移：

- `fit_lambda_mass` 和对应 SymbolFit/zfit worker 核心到 `fit.py`；
- `fit_ll_mass_plane` 和二维四分量 worker 核心到 `fit.py`；
- 一维 background transfer 积分到 `transfer.py`；
- `w_H`、`w_V`、`w_C` 及协方差计算到 `transfer.py`；
- 目标 observable 的一维和二维减除公式到 `subtract.py`；
- 拟合曲线、transfer Markdown/JSON 和诊断图到 `report.py`。

继续留在分析脚本：

- `RDataFrame` 事件记录和质量区域标记；
- 扩展轴 histogram 的构造、slot 切片和通道求和；
- Data/MC 和年份权重；
- category/subset 循环；
- `MIN_EVENTS_LL_PLANE`、`MIN_EVENTS_LP_SPECTRUM`；
- 拟合失败后的 global fallback；
- 最终 ROOT 文件布局。

迁移后现有脚本应显式形成以下流程：

```python
fit = fit_mass_spectrum_1d(...)
transfer = calculate_transfer_factor_1d(fit, regions)
result = subtract_sideband_1d(..., transfer=transfer)
write_fit_report_1d(fit, transfer, ...)
```

二维流程同理，不增加隐藏 orchestration。

## 11. 数值验证和验收标准

### 11.1 最小单元检查

1. 平坦一维本底、signal 与联合 sideband 等宽时，必须得到 $r=1$。
2. 平坦可因子化二维本底、两轴 signal 与联合 sideband 等宽时，必须得到 $w_H=w_V=1$、$w_C=-1$。
3. 双边带九区域必须恰好包含四个不同 corner，禁止重复或遗漏 `(L,L)`、`(L,H)`、`(H,L)`、`(H,H)`。
4. 对人工构造的二维区域计数，容斥减除必须恢复注入信号。
5. 数值 Jacobian 传播结果必须与固定种子的 toy covariance 在统计精度内一致。

### 11.2 现有分析回归

- 使用当前 `09` 的单高端 sideband 和 `profile_background=False`，一维 $r$、峰位和减除结果应在数值容差内复现；
- 使用当前 LL 平面配置，二维 $w_H$、$w_V$、$w_C$ 和 covariance 应在数值容差内复现；
- 开启低端加高端 sideband 后，九区域计数和四区域聚合必须逐项可核对；
- Data 和 MC 分别验证，不允许用加权 MC 平面替代逐年份 Poisson 输入。

### 11.3 报告验收

- 所有拟合图包含参数误差和拟合优度；
- 所有一维投影具有 `data / model - 1` residual；
- 所有 transfer coefficient 能从报告中的区域积分直接复算；
- 报告明确 background fixed/profiled、手动区间、样本和 fallback；
- PDF caption 和元数据自解释；
- 绘图使用 `datafactory.plot` 风格并通过项目科研绘图静态检查和视觉检查。

### 11.4 固定测试数据契约、来源、生成与验收

仓库提交一套小型、固定且可审计的真实分析 fixture，供 `fit`、`transfer`、
`subtract` 和 `report` 四个模块做回归检查：

```text
tests/data/sideband_ana/
├── sideband_ll_data_cat0_llbar.npz
├── sideband_ll_data_cat0_llbar.json
├── extract_fixture.py
└── README.md
```

#### 分析切片与来源

- 来源脚本是
  `lambda-ana/scripts/pair_distributions/09_pair_jet_distribution_summary.py`，
  输入是其 tight 选择的 1992--1995 合并 ROOT 发布产物；源 ROOT 路径、
  metadata UTC、SHA-256 和 fixture UTC 写在 sidecar JSON 的 `source` 中。
- 样本固定为 `data`，年份为 `1992, 1993, 1994, 1995`，合并权重从源
  metadata 复制（data 每年均为 1）；`category=all_event (cat0)`、
  `subset=llbar`、`channel=lambda_lambdabar (ch1)`，因此没有隐藏的通道求和。
- `mass_fit_*` 是 `mass_fit_ll_data` 的单 Lambda 质量谱（拟合范围
  `[1.08,1.175] GeV`）；`mass_plane_*` 是 `mass_plane_ll_data_cat0_llbar`
  的 100×100 未加权合并二维端点质量平面。该 plane 在未来二维拟合测试中
  明确按 `n_periods=1` 使用，不冒充逐年份 Poisson 输入。
- 目标 observable 是脚本定义的 `delta_phi_thrust`（弧度）：两条腿动量
  投影到类别 thrust 轴垂直平面后取夹角；`all_event` 使用
  `Btag_thrustVector`。不要把该字段误称为未投影的 `delta_phi`。

#### NPZ 数组契约

所有数组为 `float64`，边界数组长度比对应 bin 数多一；`counts` 与
`variances` 形状逐项相同。区域轴（axis 0）固定为
`["SS", "BS", "SB", "BB"]`，不得依赖 ROOT 对象枚举顺序：

| 数组 | shape | 含义/单位 |
|---|---:|---|
| `mass_fit_edges`, `mass_fit_counts`, `mass_fit_variances` | `(301,)`, `(300,)`, `(300,)` | 单 Lambda 质量拟合；GeV；`variances` 为 Sumw2 |
| `mass_plane_x_edges`, `mass_plane_y_edges` | `(101,)`, `(101,)` | 两个交换后 Lambda 端点质量边界；GeV |
| `mass_plane_counts`, `mass_plane_variances` | `(100,100)`, `(100,100)` | 原始二维质量平面；GeV×GeV；Sumw2 |
| `mass_region_edges` | `(31,)` | `mass_5gev` 成对不变质量边界；GeV |
| `mass_region_counts`, `mass_region_variances` | `(4,30)`, `(4,30)` | SS/BS/SB/BB 的成对质量 observable；GeV；Sumw2 |
| `delta_phi_thrust_edges` | `(11,)` | `[0,π]` 的 10 个等宽 bin；rad |
| `delta_phi_thrust_counts`, `delta_phi_thrust_variances` | `(4,10)`, `(4,10)` | SS/BS/SB/BB 的目标 observable；rad；Sumw2 |

#### 区域朝向与聚合关系

fixture 保留 09 产物的四个二端点区域对象：`reg0=SS`、`reg1=BS`、
`reg2=SB`、`reg3=BB`，并在 JSON 的 `regions.source_objects` 中逐项列出 ROOT
对象名。这里每个端点只有一个上侧带：令 `B` 表示
`|m-(mean+0.0443)|<0.010 GeV`，则

$$
SS=(S,S),\quad BS=(B,S),\quad SB=(S,B),\quad BB=(B,B).
$$

因此该 fixture 是 09 的高侧带 2×2 路径，不是把低/高侧带九个原子区域偷偷
合并得到的替代品；完整九区域 API 仍必须遵守第 6 节的
`SS, LS, HS, SL, LL, HL, SH, LH, HH` 约定。fixture 的 `aggregation` 字段
只说明区域对象复制关系，不隐瞒任何通道或年份合并。

#### 生成/刷新方式

在 `root6.34` 环境中运行：

```sh
source /home/cheyuzhi/opt/miniconda3/etc/profile.d/conda.sh
conda activate root6.34
python tests/data/sideband_ana/extract_fixture.py \
  --source-product /path/to/pair_jet_distribution_products.root \
  --source-metadata /path/to/pair_jet_distribution_metadata.json
```

`extract_fixture.py` 只调用 ROOT `TH1D/TH2D` 的 bin edge、content 和
`GetBinError()**2`，不生成 toy、不重分箱、不四舍五入；NPZ 与 JSON 必须一起
刷新，JSON 的 source checksum 和 UTC 是刷新审计依据。ROOT 产物本身不提交到
DataFactory 仓库。

#### Fixture 验收检查

提交或刷新前至少运行以下检查（不依赖拟合栈）：

1. `np.load(..., allow_pickle=False)` 可读；JSON 的数组 shape 与 NPZ 实际
   shape 一致，所有边界严格递增且与 contents 的 bin 数匹配。
2. `mass_region_counts.shape == mass_region_variances.shape == (4, 30)`，
   `delta_phi_thrust_counts.shape == delta_phi_thrust_variances.shape == (4, 10)`，
   区域顺序完整包含 SS/BS/SB/BB；二维 plane 两个 counts/variance 形状相同。
3. 所有 counts、variances、edges 均为有限数，variances 非负；data 的
   Sumw2 方差可与 Poisson 计数逐 bin 对照，但不能据此修改数组。
4. `mass_fit`、峰位/窗口、区域朝向、样本、年份权重、category、subset、
   observable 和单位在 JSON 中均有明确字段；`git diff --check` 通过。

## 12. 实施阶段

### Phase 1：纯数学核心

- 建立包目录和区域数据契约；
- 实现单双侧边带验证、区域聚合；
- 实现一维、二维 transfer 公式；
- 实现一维减除和二维容斥减除；
- 完成平坦本底和九区域最小测试。

### Phase 2：拟合迁移

- 移入 SymbolFit 本底选择逻辑；
- 实现白名单表达式到 TensorFlow/zfit 的转换；
- 实现一维 profiled/fixed 两条明确路径；
- 移入二维四分量 simultaneous zfit；
- 验证 worker 进程隔离和序列化结果。

### Phase 3：报告

- 实现一维拟合和 residual 报告；
- 实现二维平面、投影和区域框报告；
- 实现 transfer Markdown/JSON/PDF；
- 补充 PDF metadata、caption 和视觉检查。

### Phase 4：分析脚本接入

- 先用当前单高端 sideband 做无物理变化迁移；
- 对比原脚本和新模块的 fit、transfer、subtraction 结果；
- 再启用手动低端加高端 sideband；
- 保留现有 fallback 策略并在报告中显式标注；
- 删除脚本中已由 DataFactory 覆盖的重复数学实现。

## 13. 明确推迟的内容

- 任意信号 PDF 和任意背景 PDF 的 factory/registry；
- 非矩形二维 signal/sideband 区域；
- 目标 observable 完整 bin-bin covariance；
- 自动 sideband 优化或扫描；
- 自动 category/subset fallback；
- 超过二维的通用容斥系统。

只有出现真实分析需求和可验证输入时再增加这些能力。

## 14. 当前实现状态（2026-08-21）

- Phase 1--3 已实现于 `datafactory/stat/sideband_ana/`；公共 API 由该目录
  `__init__.py` 导出，现有 `datafactory/statistic.py` 未迁移。
- 一维支持 SymbolFit 参数在 zfit 中 fixed/profiled 两条路径；二维支持相同
  或不同质量轴、逐时期四分量 extended-Poisson zfit，以及单侧/双侧边带积分。
- 固定测试包含 5 个纯数学断言、1 个 profiled-background zfit toy、1 个
  四分量二维 zfit toy，以及真实 DELPHI fixture 的完整回归。
- 正式 fixture 运行命令为：

  ```sh
  source /home/cheyuzhi/opt/miniconda3/etc/profile.d/conda.sh
  conda activate root6.34
  PYTHONPATH=. python tests/sideband_ana/run_fixture_analysis.py
  ```

- 正式回归结果位于
  `tests/results/sideband_ana/ll_data_cat0_llbar/transfer_comparison.md`；$r$、
  $w_H$、$w_V$、$w_C$ 均通过预先规定的
  `max(3 sigma_reference, 1% |reference|)` 门槛。
- Phase 4 中对 `09_pair_jet_distribution_summary.py` 的调用替换尚未执行；
  当前真实 fixture runner 已完成同输入的迁移验证，待分析脚本显式切换后再删除
  原脚本中的重复实现。
