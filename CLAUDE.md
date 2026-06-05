# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

DataFactory 是一个用于高能物理（HEP）数据-蒙特卡洛（MC）分析的 Python 框架，基于 `ROOT::RDataFrame` 构建，面向 BESIII 等 e+e- 对撞实验的数据分析流程。采用 editable mode 安装，边开发边使用。

## 安装与环境

```bash
# 需要已安装 Python 3.8+ 和 ROOT（含 PyROOT 支持）
conda activate <your_env>
pip install -e .
```

依赖：`hepunits`、`matplotlib`、`numpy`、`pandas`、`scipy`（以及可选的 `zfit`、`hist`、`particle`）。

## 测试

无自动化测试框架，测试通过 `test/test.ipynb` Jupyter Notebook 进行。

## 架构概览

代码遵循 **Staff / Factory 双层结构**：

- **`Staff`**（抽象基类，`core.py`）：代表单个物理样本（data 或 MC process）。子类必须实现 `load()` 和 `save()`。
- **`Factory`**（抽象基类，`core.py`）：管理一组 `Staff`，支持 `[]` 访问和迭代。

有两套并行的具体实现：

| 层级 | 处理阶段 | Staff 类 | Factory 类 | 所在文件 |
|---|---|---|---|---|
| RDF 层 | 原始 ROOT 文件 → 事例筛选 | `RDFStaff` | `RDFFactory` | `rdf.py` |
| Hist 层 | 直方图操作与可视化 | `HistStaff` | `HistFactory` | `hist.py` |

典型分析流程：`RDFFactory` → `get_histfactory()` → `HistFactory` → `compare_mc_data()` 绘图。

### 关键类详解

**`CutFlow`**（`rdf.py`）：数据类，封装一个 ROOT Filter 条件及其"旁观者"直方图（bystander）。`apply_on_rdf()` 会同时记录 cut 前后的分布和事例数，供后续 cut chain 统计使用。`list_bystander` 字典的 key 可以是字符串（1D）或元组（2D）。

**`RDFStaff`**（`rdf.py`）：数据类，在 `__post_init__` 中自动调用 `load()` 和 `pre_selection()`。若 ROOT 文件中找不到指定 tree，会自动创建空的 fake RDF 以保证下游流程不中断。`set_cuts()` 重建整个 filter chain；`define()` 同时更新 `rdf` 和最末 cut 的 `sample_final`。

**`RDFFactory`**（`rdf.py`）：持有 `path_dict`、`xsec_dict`、`luminosity`，以及可选的 `classify_dict`（真实分类，用于将单个 ROOT 文件按 truth 切分为多个成分）。`get_weights()` 计算 `w = L * xsec / N_eff`；`classify_dict` 中的 cut 不出现在 cut chain 表中，而是折叠为 `TruthClassification` 列。

**`DataInfo`**（`plot.py`）：存储质心系能量（MeV）和亮度（pb⁻¹），`print_label()` 输出 LaTeX 格式字符串，直接用于图中的标注（`str(datainfo)` 即可）。`core.py` 中也有同名类，但 `plot.py` 中的版本更完整，实际使用应优先 import 后者。

**`HistStaff`**（`hist.py`）：支持三种初始化方式：传入 ROOT TH1/TH2 对象、传入 `path:objectName` 字符串从文件加载、或传入 `numpy_tuple=(x_edge, content, err)`。支持 `+/-/*//` 运算符和 `get_uhi()` 转换为 boost-histogram 格式供 zfit 使用。

### 单位约定

所有物理量通过 `hepunits` 包携带单位：
- 能量：MeV（`from hepunits import MeV, GeV`）
- 亮度：pb⁻¹（`invpb`）
- 截面：nb（`nb`）
- `DataInfo` 接受 MeV 为单位的字符串；`RDFStaff.xsec` 单位为 nb

### 绘图函数（`plot.py`）

- `compare_mc_data(stack_mc: HistFactory, data: HistStaff, xlabel, ...)` — 标准 Data/MC 叠加堆图 + ratio panel
- `compare_hist1d(hist_a, hist_b, xlabel, ...)` — 两个分布的直接对比 + ratio（支持 χ²/ndf 或 KS 检验）
- `compare_hist1d_multi2one(hist_a_list, hist_b, xlabel, ...)` — 多对一比较
- `compare_mc_data2d(...)` — 2D 数据 vs MC 对比图
- `apply_style()` — 加载 `style.mplstyle`（需在绘图前调用）

所有绘图函数通过 `**kargs` 接受参数，常用键包括 `weights`、`datainfo`、`xlim/ylim`、`yscale`、`figsize`、`save`、`norm_by_width`、`legend_font_size`。

### 统计工具（`statistic.py`）

- `bayes_divide(y_pass, y_tot)` — 贝叶斯效率与置信区间（返回 eff, lower_err, upper_err）
- `cut_chain_to_eff_pur(table)` — 将 cut chain DataFrame 转换为效率/纯度百分比表
- `fit_mc_data(mc_factory, data, ...)` — 基于 zfit 的模板拟合（需额外安装 zfit）

## 开发事项

- commit comments规范：https://www.conventionalcommits.org/en/v1.0.0/
- 代码注释：为新生成的每个函数、类、关键代码段附加清晰专业的注释，包括目的、原理、数学公式、输入输出、使用案例。
- 编程风格：面相物理分析编程，用带注释的代码段代替5行以内的函数，不写物理意义不完整的helper函数。
- 尽可能复用已有的函数和代码段。
- 测试环境： `conda activate root6.34`