
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from hepunits import MeV, GeV, invpb, invnb, invfb, nb, pb, fb
from .hist import HistStaff, HistFactory
# from .stat import get_chi2

import ROOT as R

import numpy as np
import matplotlib.pyplot as plt
import os
# plt.style.use('HadTauAlg-00-01/script/datafactory/style.mplstyle')

from .core import StaffType


def apply_style():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    style_path = os.path.join(script_dir, 'style.mplstyle')
    plt.style.use(style_path)


@dataclass
class DataInfo:
    """Class representing data information including CMS energy and luminosity.

    Attributes:
        cms_energy: Center-of-mass energy in MeV (provided as string)
        luminosity: Data luminosity in pb^{-1}
        path: Optional path to the data
        mc_process: Optional MC process description
        cut: List of cuts applied (default empty list)
    """
    cms_energy_MeV: str
    luminosity_invpb: float
    path: Optional[str] = None
    mc_process: Optional[str] = None
    cut: list = field(default_factory=list)
    en_unit: Optional[str] = "MeV"
    lumi_unit: Optional[str] = "invpb"

    def __post_init__(self):
        """Convert string energy to float and apply units after initialization."""
        try:
            self.CMSEnergy = float(self.cms_energy_MeV) * MeV
        except:
            self.CMSEnergy = self.cms_energy_MeV
        self.Luminosity = self.luminosity_invpb * invpb

    def set_cuts(self, cut_flow):
        """Set the cuts for this data.

        Args:
            cut_flow: CutFlow object containing the cuts
        """
        self.cut = cut_flow

    def __str__(self):
        return self.print_label(en_unit=self.en_unit, lumi_unit=self.lumi_unit)

    def __repr__(self):
        return self.print_label()

    def print_label(self, en_unit="MeV", lumi_unit="invpb"):
        """Generate a formatted label string with energy and luminosity.

        Args:
            en_unit: Energy unit ("MeV" or "GeV")
            lumi_unit: Luminosity unit ("invpb" or "invnb")

        Returns:
            Formatted string with energy and luminosity
        """
        if en_unit == "MeV":
            str_en = f"{self.CMSEnergy / MeV:.0f} \\mathrm{{~MeV}}"
        elif en_unit == "GeV":
            str_en = f"{self.CMSEnergy / GeV:.3f} \\mathrm{{~GeV}}"
        else:
            str_en = f"{self.CMSEnergy:.0f} \\mathrm{{\\textcolor{{red}}{{Bad unit}}}}"

        if lumi_unit == "invpb":
            str_lumi = f"{self.Luminosity / invpb:.1f} \\mathrm{{~pb^{{-1}}}}"
        elif lumi_unit == "invnb":
            str_lumi = f"{self.Luminosity / invnb:.4f} \\mathrm{{~nb^{{-1}}}}"
        else:
            str_lumi = f"{self.Luminosity / invpb:.1f} \\mathrm{{\\textcolor{{red}}{{Bad unit}}}}"

        return str_en + "~(" + str_lumi + ")"

# ==========================
#   Compare two 1D hists    #
# ==========================


def compare_hist1d(hist_a: HistStaff, hist_b: HistStaff, xlabel: str, **kargs):
    """
    将两个 HistStaff 的 1D 直方图在同一张图上进行对比，并在下方给出 A/B 的比值。

    参数:
        hist_a (HistStaff): 第一个直方图（例如 Data 或 MC-A 的总直方图）。
        hist_b (HistStaff): 第二个直方图（例如 MC-B 的总直方图）。
        xlabel (str): x 轴标签。

    关键字参数:
        weight_a (float): A 的整体权重，默认 1.0。
        weight_b (float): B 的整体权重，默认 1.0。
        datainfo (str | object): 显示在主图右上角（可为自定义对象，使用 str()）。
        xlim (tuple): x 轴范围。
        ylim (tuple): y 轴范围（主图）。
        yscale (str): y 轴标度，默认 "linear"。
        ylabel (str): y 轴标签，默认 r"$\\mathrm{Count}$"。
        norm_by_width (bool): 是否按 bin 宽度归一，默认 False。
        figsize (tuple): 画布大小，默认 (4,4)。
        legend_title (str): 图例标题。
        label_a (str): A 的图例名称，默认 "A"。
        label_b (str): B 的图例名称，默认 "B"。
        plot_chi2_pos (tuple|None): 在比值图中放置 χ²/ndf 的位置；若为 None 则不显示。
        plot_integral_pos (tuple|None): 在比值图中显示积分信息的位置；若为 None 则不显示。
        save (dict|None): 若提供，保存图片。键包含 {"path","name","prefix","fmt"}。
        gof (str|None): 若提供，拟合优度检验方式，"chi2" 或 "ks"（ROOT TH1::KolmogorovTest）。
    返回:
        (ax1, ax2): 上下两个轴对象。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # 解析参数
    weight_a = kargs.get("weight_a", 1.0)
    weight_b = kargs.get("weight_b", 1.0)
    datainfo = kargs.get("datainfo", None)
    xlim = kargs.get("xlim", None)
    ylim = kargs.get("ylim", None)
    yscale = kargs.get("yscale", "linear")
    ylabel = kargs.get("ylabel", r"$\mathrm{Count}$")
    norm_by_width = kargs.get("norm_by_width", False)
    figsize = kargs.get("figsize", (4, 4))
    legend_title = kargs.get("legend_title", None)
    legend_font_size = kargs.get("legend_font_size", 'x-small')
    legend_ncol = kargs.get("legend_ncol", 3)
    label_a = kargs.get("label_a", hist_a.name)
    label_b = kargs.get("label_b", hist_b.name)
    plot_chi2_pos = kargs.get("plot_chi2_pos", (0.98, 1.23))
    # 拟合优度（可选）：chi2 或 Kolmogorov–Smirnov (KS)
    gof = kargs.get("gof", "chi2")
    plot_integral_pos = kargs.get("plot_integral_pos", (0.02, 1.23))
    save = kargs.get("save", None)
    show_diff = kargs.get("show_diff", False)
    diff_ylim = kargs.get("diff_ylim", None)
    diff_ylabel = kargs.get("diff_ylabel", r"$\text{Diff.}$")

    # 确保值已准备
    hist_a._get_value(hist_a)
    hist_b._get_value(hist_b)

    # 提取 numpy 数据
    x_a, y_a, ye_a, xedge_a = hist_a.get_numpy()
    x_b, y_b, ye_b, xedge_b = hist_b.get_numpy()

    # 检查 binning 一致性
    import numpy as _np
    if not (_np.allclose(xedge_a, xedge_b) and len(xedge_a) == len(xedge_b)):
        raise ValueError(
            "compare_hist1d: A 与 B 的 bin 边界不一致，请先在外部对齐/重建到相同 binning 后再比较。")

    bin_w = _np.diff(xedge_a)

    # 归一/加权
    if norm_by_width:
        yA = (y_a / bin_w) * weight_a
        eA = (ye_a / bin_w) * weight_a
        yB = (y_b / bin_w) * weight_b
        eB = (ye_b / bin_w) * weight_b
    else:
        yA = y_a * weight_a
        eA = ye_a * weight_a
        yB = y_b * weight_b
        eB = ye_b * weight_b

    # 根据 show_diff 动态调整子图数量与画布高度
    if show_diff:
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 1.25))  # 适当增高
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True,
                                     gridspec_kw={'height_ratios': [4, 1, 1], 'hspace': 0.12})
    else:
        fig = plt.figure(figsize=figsize)
        ax1, ax2 = fig.subplots(2, 1, sharex=True,
                                gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.12})

    # 主图：stair + 误差带
    edges = _np.hstack([x_a[0] - bin_w[0]/2, x_a + bin_w/2])
    # print(label_a, label_b)

    ax1.errorbar(x_a, yA, yerr=eA, label=f"${label_a}$", marker="o", ms=1.5,
                 color="black", ls="", lw=0.4)
    ax1.stairs(yB, edges, label=f"${label_b}$", lw=0.8, color="black")

    # 误差带（灰色）
    ax1.bar(x_b, 2*eB, width=bin_w, bottom=yB-eB,
            hatch="//////////", hatch_linewidth=0.6, fill=False, lw=0, ls="",
            facecolor="black", ec="black", alpha=0.6, label=f"${label_b}" + r"\pm \sigma_{\text{stat}}$")

    # DataInfo
    if datainfo is not None:
        ax1.text(1, 1.02, "$" + str(datainfo) + "$", fontsize="x-small",
                 ha='right', transform=ax1.transAxes)

    # 轴范围
    if "log" in yscale and ylim is None:
        ymin, ymax = 0.8, max(_np.max(yA), _np.max(yB)) * 200
    elif ylim is None:
        ymin, ymax = 0, max(_np.max(yA), _np.max(yB)) * 2
    else:
        ymin, ymax = ylim
    ax1.set(ylabel=ylabel, ylim=(ymin, ymax), yscale=yscale)

    # 图例
    ax1.legend(title=legend_title, loc="best", ncol=legend_ncol,
               handlelength=1.5, fontsize=legend_font_size, columnspacing=0.8)

    # 比值与误差传播 r = A/B
    with _np.errstate(divide='ignore', invalid='ignore'):
        ratio = _np.divide(yA, yB, where=yB != 0, out=_np.ones_like(yA))
        ratio_err = _np.abs(_np.divide(
            eA, yB, where=yB != 0, out=_np.ones_like(yA)))

    # 在比值图中画出 B 的相对误差带（灰色区），中心为 1
    residual_b = _np.divide(eB, yB, where=yB != 0, out=_np.zeros_like(yB))
    ax2.bar(x_a, 2*residual_b, width=bin_w, bottom=1-residual_b,
            hatch="//////////", hatch_linewidth=0.6, fill=False, lw=0, ls="",
            facecolor="black", ec="black", alpha=0.6)

    # A/B 误差棒
    ax2.errorbar(x_a, ratio, xerr=0, yerr=ratio_err, marker="o", ms=1.5,
                 color="black", ls="", lw=0.4)

    if plot_chi2_pos is not None and gof is not None:
        if str(gof).lower() == "chi2":
            denom = _np.hypot(eA, eB)
            chi2 = _np.divide((yA - yB)**2, denom**2,
                              where=(yA * yB) > 0,
                              out=_np.zeros_like(denom))
            # remove outlayers
            chi2 = _np.sort(chi2)[0:-2]
            ndf = int(_np.sum((yA * yB) > 0))
            # print(chi2, len(chi2), ndf)
            chi2_ndf = _np.sum(chi2)/ndf if ndf > 0 else 0.0
            ax2.text(plot_chi2_pos[0], plot_chi2_pos[1], r"$\chi^2/\text{NDF} = %.3f$" % chi2_ndf,
                     fontsize="x-small", ha='right', va='top', transform=ax2.transAxes)
        else:
            # KS 两样本检验（直接使用 ROOT 的 TH1::KolmogorovTest）
            # 使用原始 ROOT 直方图进行形状比较；按整体权重缩放，但保持 ROOT 默认的归一（shape-only）
            hA = hist_a.histogram.Clone("ks_tmpA")
            hB = hist_b.histogram.Clone("ks_tmpB")
            try:
                if float(weight_a) != 1.0:
                    hA.Scale(float(weight_a))
                if float(weight_b) != 1.0:
                    hB.Scale(float(weight_b))
                # ROOT 缺省会对两者归一化后进行 KS 检验；空字符串即为标准设置
                pval = float(hA.KolmogorovTest(hB, ""))
            finally:
                pass  # 交由 Python GC 回收临时克隆

            ax2.text(plot_chi2_pos[0], plot_chi2_pos[1],
                     rf"$\text{{KS test}}:~p={pval:.3f}$",
                     fontsize="x-small", ha='right', va='top', transform=ax2.transAxes)

    # 积分信息（使用 ROOT 积分并乘整体权重，更符合事件统计）
    if plot_integral_pos is not None:
        int_a = hist_a.histogram.Integral() * float(weight_a)
        int_b = hist_b.histogram.Integral() * float(weight_b)
        ratio_int = (int_a / int_b) if int_b != 0 else _np.nan
        ax2.text(plot_integral_pos[0], plot_integral_pos[1],
                 fr"$\text{{Integral: }} {int_a:.2f}/{int_b:.2f} \sim {ratio_int:.2f}$",
                 fontsize="x-small", ha='left', va='top', transform=ax2.transAxes)

    # 轴标签与范围
    if show_diff:
        ax2.set(ylabel=r"$\text{Ratio}$", xlim=xlim, ylim=(0.2, 1.8))
    else:
        ax2.set(xlabel=xlabel,
                ylabel=r"$\text{Ratio}$", xlim=xlim, ylim=(0.2, 1.8))
    ax2.axhline(y=1, color='black', linestyle='-', lw=0.5)
    ax2.grid(0)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(4))

    if show_diff:
        # 差值面板: A - B 及其误差
        # 背景：B 的绝对误差带（中心 0）；点误差：A 的误差
        diff = yA - yB
        diff_err = eA

        # 用 B 的误差在 0 附近画灰色带
        ax3.bar(x_a, 2*eB, width=bin_w, bottom=-eB,
                hatch="//////////", hatch_linewidth=0.6, fill=False, lw=0, ls="",
                facecolor="black", ec="black", alpha=0.6)

        # (A-B) 误差棒
        ax3.errorbar(x_a, diff, xerr=0, yerr=diff_err, marker="o", ms=1.5,
                     color="black", ls="", lw=0.4)

        # 零线
        ax3.axhline(y=0, color='black', linestyle='-', lw=0.5)

        # y 轴范围
        if diff_ylim is None:
            _max = _np.nanmax(_np.abs(diff)) if diff.size else 1.0
            _band = _np.nanmax(eB) if eB.size else 0.0
            _m = max(1e-12, _max + _band)
            ax3.set_ylim(-1.3*_m, 1.3*_m)
        else:
            ax3.set_ylim(diff_ylim)

        # 轴标签
        ax3.set(ylabel=diff_ylabel, xlabel=xlabel)

        # 风格与刻度
        ax3.grid(0)
        ax3.yaxis.set_major_locator(plt.MaxNLocator(3))

    # 保存（可选）
    if save is not None:
        out_name = f"{save.get('prefix', 'compare')}_{save.get('name', 'hist1d')}.{save.get('fmt', 'png')}"
        plt.savefig(os.path.join(save['path'], out_name))

    # 调整位置
    ax1.yaxis.set_label_coords(-0.1, 0.5)
    ax2.yaxis.set_label_coords(-0.1, 0.5)
    if show_diff:
        ax3.yaxis.set_label_coords(-0.1, 0.5)
        return ax1, ax2, ax3
    else:
        return ax1, ax2


def compare_hist1d_multi2one(hist_a_list: list, hist_b, xlabel: str, **kargs):
    """
    将一个 HistStaff 列表与一个基准 HistStaff (hist_b) 进行对比。
    主图显示所有直方图，下方子图显示 list 中每个 hist 与 hist_b 的比值或差值。

    参数:
        hist_a_list (list[HistStaff]): 直方图列表（作为比较对象）。
        hist_b (HistStaff): 基准直方图（作为分母/减数，通常是 Data 或 Standard MC）。
        xlabel (str): x 轴标签。

    关键字参数 (kargs):
        weight_a (float | list[float]): A 列表的权重。若是标量则应用到所有，若是列表需与 hist_a_list 等长。
        weight_b (float): B 的整体权重，默认 1.0。
        label_a (str | list[str]): A 列表的图例名称。若是列表需等长；若未提供则尝试读取 hist.name。
        label_b (str): B 的图例名称，默认 hist_b.name。
        datainfo (str | object): 显示在主图右上角（可为自定义对象，使用 str()）。
        xlim (tuple): x 轴范围。
        ylim (tuple): y 轴范围（主图）。
        yscale (str): y 轴标度，默认 "linear"。
        ylabel (str): y 轴标签，默认 r"$\\mathrm{Count}$"。
        norm_by_width (bool): 是否按 bin 宽度归一，默认 False。
        figsize (tuple): 画布大小，默认 (4,4)。
        legend_title (str): 图例标题。
        label_a (str): A 的图例名称，默认 "A"。
        label_b (str): B 的图例名称，默认 "B"。
        plot_chi2_pos (tuple|None): 在比值图中放置 χ²/ndf 的位置；若为 None 则不显示。
        plot_integral_pos (tuple|None): 在比值图中显示积分信息的位置；若为 None 则不显示。
        save (dict|None): 若提供，保存图片。键包含 {"path","name","prefix","fmt"}。
        hspace (float): 子图之间的空隙
        gof (str|None): 若提供，拟合优度检验方式，"chi2" 或 "ks"（ROOT TH1::KolmogorovTest）。
    返回:
        (ax1, ax2): 上下两个轴对象。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from itertools import cycle

    # --- 兼容性处理：如果传入的是单个对象，转为列表 ---
    if not isinstance(hist_a_list, (list, tuple)):
        hist_a_list = [hist_a_list]

    # --- 解析通用参数 ---
    weight_a_in = kargs.get("weight_a", 1.0)
    weight_b = kargs.get("weight_b", 1.0)
    label_a_in = kargs.get("label_a", None)
    label_b = kargs.get("label_b", getattr(hist_b, "name", "Ref"))

    # 处理权重列表
    if isinstance(weight_a_in, (list, tuple)):
        if len(weight_a_in) != len(hist_a_list):
            raise ValueError("weight_a 列表长度必须与 hist_a_list 一致")
        weights_a = weight_a_in
    else:
        weights_a = [weight_a_in] * len(hist_a_list)

    # 处理标签列表
    if isinstance(label_a_in, (list, tuple)):
        if len(label_a_in) != len(hist_a_list):
            raise ValueError("label_a 列表长度必须与 hist_a_list 一致")
        labels_a = label_a_in
    else:
        # 如果没给 list，尝试用 hist.name，如果 hist.name 也没有，则用 A_0, A_1...
        labels_a = []
        for i, h in enumerate(hist_a_list):
            if label_a_in is not None:
                labels_a.append(f"{label_a_in}_{i}")
            else:
                labels_a.append(getattr(h, "name", f"A_{i}"))

    # 其他绘图参数
    datainfo = kargs.get("datainfo", None)
    xlim = kargs.get("xlim", None)
    ylim = kargs.get("ylim", None)
    yscale = kargs.get("yscale", "linear")
    ylabel = kargs.get("ylabel", r"$\mathrm{Count}$")
    norm_by_width = kargs.get("norm_by_width", False)
    figsize = kargs.get("figsize", (4, 4))
    legend_title = kargs.get("legend_title", None)

    plot_chi2_pos = kargs.get("plot_chi2_pos", (0.98, 1.23))
    gof = kargs.get("gof", "chi2")
    plot_integral_pos = kargs.get("plot_integral_pos", (0.02, 1.23))
    save = kargs.get("save", None)
    show_diff = kargs.get("show_diff", False)
    diff_ylim = kargs.get("diff_ylim", None)
    diff_ylabel = kargs.get("diff_ylabel", r"$\text{Diff.}$")
    hspace = kargs.get("hspace", 0.12)

    # --- 准备基准数据 (Hist B) ---
    hist_b._get_value(hist_b)
    x_b, y_b, ye_b, xedge_b = hist_b.get_numpy()
    bin_w = np.diff(xedge_b)

    # 归一化/加权 B
    if norm_by_width:
        yB = (y_b / bin_w) * weight_b
        eB = (ye_b / bin_w) * weight_b
    else:
        yB = y_b * weight_b
        eB = ye_b * weight_b

    # --- 初始化画布 ---
    if show_diff:
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 1.25))
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True,
                                     gridspec_kw={'height_ratios': [4, 1, 1], 'hspace': hspace})
    else:
        fig = plt.figure(figsize=figsize)
        ax1, ax2 = fig.subplots(2, 1, sharex=True,
                                gridspec_kw={'height_ratios': [4, 1], 'hspace': hspace})

    # --- 绘制基准 (Hist B) ---
    # 使用 stairs 绘制 B 的轮廓
    edges_b = np.hstack([x_b[0] - bin_w[0]/2, x_b + bin_w/2])
    ax1.stairs(yB, edges_b, label=f"${label_b}$",
               lw=0.8, color="black", zorder=0)

    # B 的误差带（灰色阴影）
    ax1.bar(x_b, 2*eB, width=bin_w, bottom=yB-eB,
            hatch="//////////", hatch_linewidth=0.6, fill=False, lw=0, ls="",
            facecolor="black", ec="black", alpha=0.6, label=f"${label_b}" + r"\pm \sigma_{\text{stat}}$", zorder=0)

    # --- 准备循环绘制 A 列表 ---
    # 获取颜色循环
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])

    # 用于收集统计信息的文本列表
    chi2_texts = []
    int_texts = []

    # 用于自动计算 Y 轴范围
    max_y_val = np.max(yB)

    # --- 循环处理每个 Hist A ---
    for i, (hist_a, w_a, lbl_a) in enumerate(zip(hist_a_list, weights_a, labels_a)):
        curr_color = next(colors)

        # 1. 数据准备
        hist_a._get_value(hist_a)
        x_a, y_a_raw, ye_a_raw, xedge_a = hist_a.get_numpy()

        # 检查 Binning
        if not (np.allclose(xedge_a, xedge_b) and len(xedge_a) == len(xedge_b)):
            raise ValueError(f"Hist A[{i}] ({lbl_a}) 与 B 的 bin 边界不一致。")

        # 归一化/加权 A
        if norm_by_width:
            yA = (y_a_raw / bin_w) * w_a
            eA = (ye_a_raw / bin_w) * w_a
        else:
            yA = y_a_raw * w_a
            eA = ye_a_raw * w_a

        max_y_val = max(max_y_val, np.max(yA))

        # 2. 绘制主图 (Ax1)
        ax1.errorbar(x_a, yA, yerr=eA, label=f"${lbl_a}$", marker="o", ms=1.5,
                     color=curr_color, ls="", lw=0.4, zorder=10+i)

        # 3. 计算比值 (Ratio)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(yA, yB, where=yB != 0, out=np.ones_like(yA))
            # 误差传播: (A/B) * sqrt((dA/A)^2 + (dB/B)^2)
            # 这里仅绘制 A 的统计误差对 Ratio 的贡献，通常 B 的误差画在背景带上
            ratio_err = np.abs(
                np.divide(eA, yB, where=yB != 0, out=np.zeros_like(yA)))

        # 绘制 Ratio 点
        ax2.errorbar(x_a, ratio, xerr=0, yerr=ratio_err, marker="o", ms=1.5,
                     color=curr_color, ls="", lw=0.4, zorder=10+i)

        # 添加箭头指示超出 ylim 范围的点
        ratio_ylim = (0.2, 1.8)
        for x, y, yerr in zip(x_a, ratio, ratio_err):
            upper = y
            lower = y
            if upper > ratio_ylim[1]:
                # 向上箭头，指示超过上界
                ax2.annotate('', xy=(x, ratio_ylim[1]), xytext=(x, 1.2),
                             arrowprops=dict(arrowstyle='simple',
                                             color=curr_color,
                                             lw=0.2, alpha=0.5,
                                             mutation_scale=4),
                             ha='center')
            elif lower < ratio_ylim[0]:
                # 向下箭头，指示低于下界
                ax2.annotate('', xy=(x, ratio_ylim[0]*1.0), xytext=(x, 0.8),
                             arrowprops=dict(arrowstyle='simple',
                                             color=curr_color,
                                             alpha=0.5,
                                             lw=0.4,
                                             mutation_scale=4),
                             ha='center')

        # 4. 计算差值 (Diff) - 可选
        if show_diff:
            diff = yA - yB
            diff_err = eA  # 仅 A 的误差，B 的误差在背景
            ax3.errorbar(x_a, diff, yerr=diff_err, xerr=0, marker="o", ms=1.5,
                         color=curr_color, ls="", lw=0.4)

        # 5. 统计检验 (Chi2 / KS)
        stat_str = ""
        if plot_chi2_pos is not None and gof is not None:
            if str(gof).lower() == "chi2":
                # Chi2 计算
                denom = np.hypot(eA, eB)
                chi2_arr = np.divide(
                    (yA - yB)**2, denom**2, where=(yA * yB > 0), out=np.zeros_like(denom))
                # 简单的去除异常值策略（可根据需要移除）
                # chi2_arr = np.sort(chi2_arr)[0:-2]
                ndf = int(np.sum((yA * yB) > 0))
                chi2_val = np.sum(chi2_arr)
                chi2_ndf = chi2_val/ndf if ndf > 0 else 0.0
                stat_str = r"$\chi^2/\text{ndf}(%s) = %.2f$" % (lbl_a,
                                                                chi2_ndf)
            else:
                # KS Test (ROOT)
                hA_clone = hist_a.histogram.Clone(f"ks_A_{i}")
                hB_clone = hist_b.histogram.Clone(f"ks_B_{i}")
                try:
                    if float(w_a) != 1.0:
                        hA_clone.Scale(float(w_a))
                    if float(weight_b) != 1.0:
                        hB_clone.Scale(float(weight_b))
                    pval = float(hA_clone.KolmogorovTest(hB_clone, ""))
                finally:
                    pass
                stat_str = rf"$\text{{KS}}(%s) p=%.3f$" % (lbl_a, pval)

            # 设置颜色以便区分
            # 为避免 LaTeX 解析颜色名称的麻烦，这里简单存储字符串，颜色可以后续手动加或不加
            chi2_texts.append(stat_str)

        # 6. 积分信息
        if plot_integral_pos is not None:
            # 注意：这里用 ROOT 的原始积分 * 权重，更准确
            int_a = hist_a.histogram.Integral() * float(w_a)
            # B 的积分只需要算一次，但为了格式化方便，这里每次循环如果不算 B 就很难对齐
            # 所以我们只存 A 的积分，最后统一显示
            int_texts.append(f"{lbl_a}: {int_a:.1f}")

    # --- 绘制公共部分 (Ratio/Diff 背景) ---

    # Ratio 背景：B 的相对误差带 (中心为1)
    residual_b = np.divide(eB, yB, where=yB != 0, out=np.zeros_like(yB))
    ax2.bar(x_b, 2*residual_b, width=bin_w, bottom=1-residual_b,
            hatch="//////////", hatch_linewidth=0.6, fill=False, lw=0, ls="",
            facecolor="black", ec="black", alpha=0.4, zorder=0)
    ax2.axhline(y=1, color='black', linestyle='-', lw=0.4)

    # Diff 背景：B 的绝对误差带 (中心为0)
    if show_diff:
        ax3.bar(x_b, 2*eB, width=bin_w, bottom=-eB,
                hatch="//////////", hatch_linewidth=0.5, fill=False, lw=0, ls="",
                facecolor="black", ec="black", alpha=0.4, zorder=0)
        ax3.axhline(y=0, color='black', linestyle='-', lw=0.4)

    # --- 打印统计文本 ---
    if chi2_texts:
        # 将列表拼接为多行字符串
        full_chi2_str = "\n".join(chi2_texts)
        ax2.text(plot_chi2_pos[0], plot_chi2_pos[1], full_chi2_str,
                 fontsize="x-small", ha='right', va='bottom', transform=ax2.transAxes)

    if plot_integral_pos is not None:
        int_b = hist_b.histogram.Integral() * float(weight_b)
        # 首行显示 B，后续行显示 A list
        header = f"${label_b}: {int_b:.1f}$"
        body = "\n".join([f"${t}$" for t in int_texts])
        full_int_str = header + "\n" + body
        ax2.text(plot_integral_pos[0], plot_integral_pos[1], full_int_str,
                 fontsize="x-small", ha='left', va='bottom', transform=ax2.transAxes)

    # --- 设置样式与范围 ---

    # DataInfo
    if datainfo is not None:
        ax1.text(1, 1.02, "$" + str(datainfo) + "$", fontsize="x-small",
                 ha='right', transform=ax1.transAxes)

    # 主图 Y 轴范围
    if "log" in yscale and ylim is None:
        ymin, ymax = 0.8, max_y_val * 200
    elif ylim is None:
        ymin, ymax = 0, max(np.max(yA), np.max(yB)) * 2
    else:
        ymin, ymax = ylim
    ax1.set(ylabel=ylabel, ylim=(ymin, ymax), yscale=yscale)

    # 图例 (根据 list 长度动态调整列数)
    ncol = 2 if len(hist_a_list) < 4 else 3
    ax1.legend(title=legend_title, loc="best", ncol=ncol,
               handlelength=1.5, fontsize=6, columnspacing=0.8)

    # Ratio 轴设置
    ax2.set(ylabel=r"$\text{Ratio}$", xlim=xlim, ylim=(0.2, 1.8))
    if not show_diff:
        ax2.set(xlabel=xlabel)
    ax2.grid(False)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(4))

    # Diff 轴设置
    if show_diff:
        if diff_ylim is None:
            # 简单估算范围，避免 NaN
            _max = 0
            # 需重新遍历获取最大 diff (略繁琐，这里简化处理，取最后一组的量级或固定)
            # 更稳妥的是在上面循环中记录 max_diff
            _max = np.nanmax(np.abs(diff)) if diff.size else 1.0
            _band = np.nanmax(eB) if eB.size else 0.0
            _m = max(1e-12, _max + _band)
            ax3.set_ylim(-2*_m, 2*_m)
        else:
            ax3.set_ylim(diff_ylim)
        ax3.set(ylabel=diff_ylabel, xlabel=xlabel)
        ax3.grid(False)
        ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax3.yaxis.set_label_coords(-0.1, 0.5)

    # 调整 Label 位置
    ax1.yaxis.set_label_coords(-0.1, 0.5)
    ax2.yaxis.set_label_coords(-0.1, 0.5)

    # --- 保存 ---
    if save is not None:
        out_name = f"{save.get('prefix', 'compare')}_{save.get('name', 'hist1d')}.{save.get('fmt', 'png')}"
        if not os.path.exists(save['path']):
            os.makedirs(save['path'])
        plt.savefig(os.path.join(save['path'], out_name), bbox_inches='tight')

    if show_diff:
        return ax1, ax2, ax3
    else:
        return ax1, ax2


def compare_mc_data(stack_mc, data, xlabel, get_color = None, **kargs):
    """
    绘制蒙特卡洛数据与实际数据的对比图。

    参数:
        stack_mc (dict): 包含蒙特卡洛数据的字典，键为样本名称，值为TH1F对象。
        weights (dict): 包含样本权重的字典，键为样本名称，值为权重。
        data (TH1F): 实际数据的TH1F对象。
        **kargs: 其他关键字参数。

    可选参数:
        dataInfo (datainfo): 数据的luminosity, center-of-mass energy 等信息。
        xlim (tuple): x 轴范围。
        xlabel (str): x 轴标签。
        ylabel (str): y 轴标签。
        file_title (str): 文件标题。
    返回:
        ax1, ax2: 主图（MC、data对比图）和副图（Residual）
    """
    import numpy as np
    import matplotlib.pyplot as plt

    script_dir = os.path.dirname(os.path.abspath(__file__))
    style_path = os.path.join(script_dir, 'style.mplstyle')

    # 获取 weight, 如果未提供则为 1
    weights = kargs.get(
        "weights", {key: 1 for key in stack_mc.staff_dict.keys()})
    # 获取 DataInfo 对象，如果未提供则为 None
    datainfo = kargs.get("datainfo", None)
    # 获取 x 轴范围，如果未提供则默认为 (0, 3.5)
    xlim = kargs.get("xlim", None)
    # 获取 y 轴范围，如果未提供则默认为 (0, 3.5)
    ylim = kargs.get("ylim", None)
    # 获取 y 轴标度，如果未提供则默认为 linear
    yscale = kargs.get("yscale", "linear")
    # 获取 y 轴标签，如果未提供则默认为 r"$\mathrm{Count}$"
    ylabel = kargs.get("ylabel", r"$\mathrm{Count}$")
    # 获取 y 轴标签，如果未提供则默认为 r"$\mathrm{Count}$"
    stack = kargs.get("stack", True)
    # 设置是否按bin宽归一
    norm_by_width = kargs.get("norm_by_width", False)
    # 获取图片保存路径和名称
    save = kargs.get("save", None)
    # 设置图例标题
    legend_title = kargs.get("legend_title", None)
    # 设置成分高亮
    highlight_channel = kargs.get("highlight_channel", None)
    # 设置图片大小
    figsize = kargs.get("figsize", (4, 4))
    # 是否在 pull plot 里放 chi2
    plot_chi2_pos = kargs.get("plot_chi2_pos", (0.98, 1.23))
    # 是否在 pull plot 上放 MC和Data总数
    plot_integral_pos = kargs.get("plot_integral_pos", (0.02, 1.23))
    # 约定MC中信号、本底排列顺序
    mc_order = kargs.get(
        "mc_order", [StaffType.background, StaffType.signal, StaffType.other])
    # legend字体大小
    legend_font_size = kargs.get("legend_font_size", 5)
    # legend列数
    legend_ncol = kargs.get("legend_ncol", 4)

    if get_color is None:
        colors = {key: f"C{i}" for i, key in enumerate(stack_mc.staff_dict.keys())}
        get_color = lambda x: colors[x]

    stack_mc._get_value()
    data._get_value(data)

    # 将 TH1F 对象转换为 numpy 数组
    x_data, y_data, yerr_data, x_edge_data = data.get_numpy()
    # 计算每个 bin 的宽度
    x_width_data = np.diff(x_edge_data)
    if norm_by_width:
        # 归一化 y 轴数据
        y_data_norm = y_data/x_width_data
        # 归一化 y 轴误差
        yerr_data_norm = yerr_data/x_width_data
    else:
        # 归一化 y 轴数据
        y_data_norm = y_data
        # 归一化 y 轴误差
        yerr_data_norm = yerr_data
    # 初始化字典用于存储蒙特卡洛数据
    x_mc_col, y_mc_col, yerr_mc_col, x_width_mc_col = {}, {}, {}, {}

    # 初始化总计数数组
    tot = np.zeros_like(y_data)
    # 遍历蒙特卡洛数据字典
    for name, mc in stack_mc.staff_dict.items():
        # 将 TH1F 对象转换为 numpy 数组
        x_mc, y_mc, yerr_mc, x_edge_mc = mc.get_numpy()
        # 计算每个 bin 的宽度
        x_width_mc = np.diff(x_edge_mc)
        # 存储 x 轴数据
        x_mc_col[name] = x_mc
        if norm_by_width:
            # 存储归一化的 y 轴误差
            yerr_mc_col[name] = yerr_mc * weights[name]/x_width_mc
            # 存储归一化的 y 轴数据
            y_mc_col[name] = y_mc * weights[name]/x_width_mc
        else:
            # 存储归一化的 y 轴误差
            yerr_mc_col[name] = yerr_mc * weights[name]
            # 存储归一化的 y 轴数据
            y_mc_col[name] = y_mc * weights[name]
        # 存储 x 轴宽度
        x_width_mc_col[name] = x_width_mc
        # 累加总计数
        tot += y_mc_col[name]

    # 创建子图
    fig = plt.figure(figsize=figsize)
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={
                            'height_ratios': [4, 1], "hspace": 0.1})
    # 初始化基线数组
    baseline = np.zeros_like(x_data)
    # 初始化基线误差数组
    baseline_err = np.zeros_like(x_data)
    # 遍历蒙特卡洛数据字典
    for component_type in mc_order:
        print(component_type)
        for i in x_mc_col.keys():
            # 绘制柱状图
            if stack_mc.staff_dict[i].type == component_type:
                if (stack):
                    print(get_color(i))
                    ax1.bar(x_edge_mc[:-1], y_mc_col[i], width=x_width_mc_col[i], bottom=baseline,
                            label="$"+i+"$", lw=0, alpha=0.8, color=get_color(i),
                            edgecolor='white', align='edge',
                            hatch="/////\\\\\\\\\\" if i == highlight_channel else "")
                else:
                    ax1.stairs(y_mc_col[i], np.hstack([x_mc_col[i][0] - x_width_mc_col[i][0]/2, x_mc_col[i] + x_width_mc_col[i]/2]),
                               label="$"+i+"$", lw=0.6, alpha=1, color=get_color(i))
                # 累加基线
                baseline += y_mc_col[i]
                # 累加基线误差
                baseline_err = np.hypot(baseline_err, yerr_mc_col[i])

    # 求和后基线表示总 MC histogram
    ax1.bar(x_mc, 2*baseline_err, width=x_width_mc,
            bottom=baseline - baseline_err,
            hatch="//////////", hatch_linewidth=0.6,
            fill=False, lw=0, ls="",
            facecolor="gray", alpha=0.6, label=r"$\text{MC error}$")
    # 绘制实际数据的误差棒图
    ax1.errorbar(x_data, y_data_norm, xerr=0, yerr=yerr_data_norm,
                 marker="o", ms=1.5, color="black", label=r"$\text{Data}$", ls="", lw=0.4)

    # 计算 MC data 的 chi2
    chi2 = np.sum(
        np.divide((baseline - y_data_norm)**2,
                  (np.hypot(baseline_err, yerr_data_norm))**2,
                  where=baseline_err*yerr_data_norm != 0,
                  out=np.zeros_like(baseline)
                  )
    )
    # ndf 非零 bin 的个数
    ndf = np.sum((baseline * y_data_norm)**2 > 0)
    chi2_ndf = chi2/ndf

    if (stack == False):
        ax1.stairs(baseline, np.hstack([x_mc_col[i][0] - x_width_mc_col[i][0]/2, x_mc_col[i] + x_width_mc_col[i]/2]),
                   label="Mix MC", lw=0.6, alpha=1, color="black")

    # 在图中添加 DataInfo 信息
    ax1.text(1, 1.02, "$" + str(datainfo) + "$",
             fontsize="x-small", horizontalalignment='right',
             transform=ax1.transAxes)

    # 设置 y 轴标签和范围
    ymin, ymax = None, None
    if "log" in yscale and ylim is None:
        ymin = 0.8
        ymax = np.max(y_data_norm)*200
    elif ylim == None:
        ymin = 0
        ymax = np.max(y_data_norm)*2
    else:
        ymin, ymax = ylim
    ax1.set(ylabel=ylabel,
            ylim=(ymin, ymax),
            yscale=yscale)

    # 添加图例
    legend = ax1.legend(title=legend_title,
                        loc="best", ncol=legend_ncol, handlelength=1.5, fontsize=legend_font_size, columnspacing=0.5)

    # 设置图例标题颜色
    if highlight_channel:
        plt.setp(legend.get_title(), color="#" + get_color(highlight_channel))
    # 关闭网格线
    ax1.grid(0)

    # 计算差异
    diff = np.divide(y_data_norm, baseline, where=baseline !=
                     0, out=np.ones_like(y_data_norm))
    # 计算差异误差
    diff_err = np.divide(yerr_data_norm, baseline,
                         where=baseline != 0, out=np.zeros_like(yerr_data))
    # # 绘制差异柱状图
    # ax2.bar(x_data, diff, width = np.diff(x_edge_data)*0.8, alpha = 0.8, color = "blue", ls  = "")

    # 绘制差异误差棒图
    residual = np.divide(
        baseline_err,
        baseline,
        where=baseline != 0,
        out=np.zeros_like(baseline)
    )
    ax2.bar(x_mc, 2*residual, width=x_width_data,
            bottom=1-residual,
            hatch="//////////", hatch_linewidth=0.6,
            fill=False, lw=0, ls="",
            facecolor="gray", alpha=0.6)
    # ax2.stairs(baseline_err/baseline, edges=x_edge_mc,
    #        baseline=1-baseline_err/2/baseline,
    #        fill=True, # 保持与 bar 相同的 fill 状态
    #        color="gray", # stairs 没有 facecolor，使用 color 来设置线条颜色
    #        linewidth = 0.6, # 对应 hatch_linewidth，作为边框线宽
    #        linestyle = "-", # stairs 默认是实线，这里保持
    #        alpha = 0.6
    #        )

    ax2.errorbar(x_data, diff,
                 xerr=0, yerr=diff_err,
                 marker="o", ms=1.5, color="black", ls="", lw=0.4)

    # 在图中添加 chi2 信息
    if plot_chi2_pos is not None:
        ax2.text(plot_chi2_pos[0], plot_chi2_pos[1], fr"$\chi^2/\text{{NDF}} = {chi2_ndf:.3f}$",
                 fontsize="x-small", horizontalalignment='right', verticalalignment='top',
                 transform=ax2.transAxes)

    if plot_integral_pos:
        ax2.text(
            plot_integral_pos[0], plot_integral_pos[1],
            fr"$\text{{Total Ratio: }} \text{{Data}}/\text{{MC}} = {data.histogram.Integral():.0f}/{(stack_mc * weights).sum().histogram.Integral():.0f} \sim {data.histogram.Integral()/(stack_mc * weights).sum().histogram.Integral():.2f}$",
            fontsize="x-small", horizontalalignment='left', verticalalignment='top',
            transform=ax2.transAxes
        )

    # 设置 x 轴和 y 轴标签以及范围
    ax2.set(xlabel=xlabel, ylabel=r"$\text{Data/MC}$",
            xlim=xlim, ylim=(0.5, 1.5))

    # 添加箭头指示超出 ylim 范围的点
    ratio_ylim = ax2.get_ylim()
    for x, y, yerr in zip(x_data, diff, diff_err):
        upper = y
        lower = y
        if upper > ratio_ylim[1]:
            # 向上箭头，指示超过上界
            ax2.annotate('', xy=(x, ratio_ylim[1]*1.0), xytext=(x, 1.2),
                         arrowprops=dict(arrowstyle='simple',
                                         color='blue', lw=0.4, mutation_scale=4),
                         ha='center')
        elif lower < ratio_ylim[0]:
            # 向下箭头，指示低于下界
            ax2.annotate('', xy=(x, ratio_ylim[0]*1.0), xytext=(x, 0.8),
                         arrowprops=dict(arrowstyle='simple',
                                         color='blue', lw=0.4, mutation_scale=4),
                         ha='center')

    # 绘制水平参考线
    ax2.axhline(y=1, color='black', linestyle='-', lw=0.5)
    # # 绘制水平区域
    # ax2.axhspan(ymin=-1, ymax=1, color='gray', linestyle='-', lw = 0.5, zorder = 0, alpha = 0.3)
    # ax2.axhspan(ymin=-2, ymax=2, color='gray', linestyle='-', lw = 0.5, zorder = 0, alpha = 0.3)
    # ax2.axhspan(ymin=-3, ymax=3, color='gray', linestyle='-', lw = 0.5, zorder = 0, alpha = 0.3)

    # 关闭网格线
    ax2.grid(0)
    # 设置 y 轴刻度
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))

    # 如果 save 参数不为 None，则保存图片
    if save != None:
        plt.savefig(os.path.join(
            save['path'], f"{save['prefix']}_{n_pi:.0f}pi{n_pi0:.0f}pi0_{save['name']}.{save['fmt']}"))

    # 调整位置
    ax1.yaxis.set_label_coords(-0.1, 0.5)
    ax2.yaxis.set_label_coords(-0.1, 0.5)

    # 显示图片
    return ax1, ax2


# =========================
#          2D Plot       #
# =========================

def compare_mc_data2d(stack_mc, data, weights=None, **kargs):
    """
    2D: Data vs (weighted) mixed MC, and show top-N (by weighted 
    integral) MC components.

    Parameters
    ----------
    stack_mc : HistFactory
        stack_mc.staff_dict: {name: HistStaff(2D)}
    data : HistStaff
        2D HistStaff for data
    weights : dict | None
        {name: weight}. If None -> all 1.0

    Keyword args (kargs)
    --------------------
    datainfo : object | None
    xlim, ylim : tuple | None
    xlabel, ylabel : str
    figsize : tuple, default (9, 3)
    highlight_channel : str | None
    legend_title : str | None   (kept for API compatibility; not used in this 2D layout)
    norm_by_width : bool, default False
        If True, divide counts by bin-area (dx*dy) for all panels (data + MC).
    cmap : str, default "Spectral_r"
    norm : {"linear","log"} or matplotlib norm object, default "linear"
    vmin, vmax : float | None
        If None -> auto from data (and mixed MC), with vmin>=0 for linear and >0 for log
    right_grid : int | None
        If None -> floor(sqrt(Nmc)) columns; shown panels = grid^2
    save : dict | None
        {"path","name","prefix","fmt"}; saved with bbox_inches='tight'

    Returns
    -------
    fig, (ax_data, ax_mc), axs_components, cbar
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    # --- args ---
    datainfo = kargs.get("datainfo", None)
    xlim = kargs.get("xlim", None)
    ylim = kargs.get("ylim", None)
    xlabel = kargs.get("xlabel", r"$x$")
    ylabel = kargs.get("ylabel", r"$y$")
    figsize = kargs.get("figsize", (9, 3))
    highlight_channel = kargs.get("highlight_channel", None)
    norm_by_width = kargs.get("norm_by_width", False)

    cmap = kargs.get("cmap", "Spectral_r")
    norm_in = kargs.get("norm", "linear")
    vmin = kargs.get("vmin", None)
    vmax = kargs.get("vmax", None)

    right_grid = kargs.get("right_grid", None)
    save = kargs.get("save", None)

    # --- weights ---
    if weights is None:
        weights = {k: 1.0 for k in stack_mc.staff_dict.keys()}
    else:
        # fill missing -> 1
        for k in stack_mc.staff_dict.keys():
            if k not in weights:
                weights[k] = 1.0

    # --- prepare values ---
    stack_mc._get_value()
    data._get_value(data)

    data_xe, data_ye, data_z, _ = data.get_numpy()   # edges_x, edges_y, counts (Ny, Nx)
    mc_np = {k: v.get_numpy() for k, v in stack_mc.staff_dict.items()}

    # --- bin area (for density-like view) ---
    if norm_by_width:
        dx = np.diff(data_xe)  # (Nx,)
        dy = np.diff(data_ye)  # (Ny,)
        area = dy[:, None] * dx[None, :]  # (Ny, Nx)
        area = np.where(area > 0, area, 1.0)
        data_z_plot = data_z / area
    else:
        data_z_plot = data_z

    # --- mixed MC ---
    z_mix = np.zeros_like(data_z, dtype=float)
    for k, (xe, ye, z, _) in mc_np.items():
        if not (np.allclose(xe, data_xe) and np.allclose(ye, data_ye)):
            raise ValueError(
                f"compare_mc_data2d: binning mismatch between data and MC component '{k}'.")
        z_mix += z * float(weights[k])

    if norm_by_width:
        z_mix_plot = z_mix / area
    else:
        z_mix_plot = z_mix

    # --- normalization object ---
    if hasattr(norm_in, "__class__") and not isinstance(norm_in, str):
        norm_obj = norm_in
    else:
        norm_key = str(norm_in).lower()
        # auto vmin/vmax
        if vmax is None:
            vmax = float(np.nanmax([np.nanmax(data_z_plot), np.nanmax(
                z_mix_plot)])) if data_z_plot.size else 1.0
        if vmin is None:
            if "log" in norm_key:
                # strictly positive
                pos = data_z_plot[data_z_plot > 0]
                vmin = float(np.nanmin(pos)) if pos.size else 1e-3
            else:
                vmin = 0.0

        if "log" in norm_key:
            # ensure valid for LogNorm
            vmin = max(float(vmin), 1e-12)
            vmax = max(float(vmax), vmin * 1.01)
            norm_obj = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm_obj = Normalize(vmin=float(vmin), vmax=float(vmax))

    # --- layout ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.03)

    # left: Data vs Mixed MC
    gs_left = gs[0].subgridspec(1, 2, wspace=0.06)
    ax_data = fig.add_subplot(gs_left[0])
    ax_mc = fig.add_subplot(gs_left[1], sharex=ax_data)

    pcm_data = ax_data.pcolormesh(
        data_xe, data_ye, data_z_plot, shading="flat", cmap=cmap, norm=norm_obj, zorder = 0)
    pcm_mc = ax_mc.pcolormesh(
        data_xe, data_ye, z_mix_plot, shading="flat", cmap=cmap, norm=norm_obj, zorder = 0)

    ax_data.text(0.02,  1.02, r"$\text{Data}$", fontsize="x-small",
                 transform=ax_data.transAxes, ha="left", va="bottom")
    ax_mc.text(0.02, 1.02, r"$\text{Mixed MC}$", fontsize="x-small",
               transform=ax_mc.transAxes, ha="left", va="bottom")

    if xlim is None:
        xlim = (data_xe[0], data_xe[-1])
    if ylim is None:
        ylim = (data_ye[0], data_ye[-1])

    ax_data.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    ax_mc.set(xlabel=xlabel, xlim=xlim, ylim=ylim)
    ax_mc.set_yticklabels([])
    ax_data.grid(True)
    ax_mc.grid(True)

    if datainfo is not None:
        ax_data.text(1.0, 1.02, "$" + str(datainfo) + "$",
                   fontsize="x-small", ha="right", va="bottom",
                   transform=ax_mc.transAxes)
        ax_mc.text(1.0, 1.02, "$" + str(datainfo) + "$",
                   fontsize="x-small", ha="right", va="bottom",
                   transform=ax_mc.transAxes)

    # right: top components
    n_mc = len(mc_np)
    if right_grid is None:
        right_grid = int(np.floor(np.sqrt(max(1, n_mc))))
        right_grid = max(1, right_grid)

    gs_right = gs[1].subgridspec(
        right_grid, right_grid, hspace=0.0, wspace=0.0)

    # weighted integrals
    integrals = {}
    for k, (_, _, z, _) in mc_np.items():
        w = float(weights[k])
        integrals[k] = float(np.nansum(z * w))

    top_keys = sorted(integrals, key=integrals.get, reverse=True)[
        : right_grid * right_grid]

    axs_components = []
    last_pcm = None
    for i, k in enumerate(top_keys):
        ax = fig.add_subplot(gs_right[i])
        xe, ye, z, _ = mc_np[k]
        z_plot = (z * float(weights[k]))
        if norm_by_width:
            z_plot = z_plot / area

        last_pcm = ax.pcolormesh(
            xe, ye, z_plot, shading="flat", cmap=cmap, norm=norm_obj, zorder = 0)

        ax.text(0.95, 0.97, f"${k}$",
                fontsize="small",
                color=("r" if k == highlight_channel else "k"),
                bbox=dict(boxstyle="square", facecolor="w", ls="", alpha=0.7),
                transform=ax.transAxes, ha="right", va="top")

        ax.set(xlim=(xe[0], xe[-1]), ylim=(ye[0], ye[-1]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        axs_components.append(ax)

    for ax in axs_components:
        ax.label_outer()

    # colorbar attached to right panels (fallback to left if no right panels)
    cbar_ax_list = axs_components if axs_components else [ax_mc]
    cbar = fig.colorbar(pcm_mc, ax=cbar_ax_list, location="right", pad=0.02)
    cbar.set_label(
        r"$\text{Counts}$" if not norm_by_width else r"$\text{Counts}/(\Delta x \Delta y)$")

    # save
    if save is not None:
        if not os.path.exists(save["path"]):
            os.makedirs(save["path"])
        out_name = f"{save.get('prefix', 'compare2d')}_{save.get('name', 'hist2d')}.{save.get('fmt', 'png')}"
        plt.savefig(os.path.join(save["path"], out_name), bbox_inches="tight")

    return fig, (ax_data, ax_mc), axs_components, cbar


def project_2d_histogram(hist, slice_axis, slice_num):
    """
    将二维直方图沿一个轴均匀切片，并将结果投影到另一个轴上。

    参数:
    ----------
    hist : tuple
        一个包含 (x_edges, y_edges, counts) 的元组。
        - x_edges (array-like): x轴的箱体边界。
        - y_edges (array-like): y轴的箱体边界。
        - counts (2D array-like): 直方图的计数值，形状应为 (len(y_edges)-1, len(x_edges)-1)。
          注意：如果使用 np.histogram2d，其输出的 counts 数组需要转置。
    slice_axis : {'x', 'y'}
        定义切片的坐标轴。
    slice_num : int
        要将 slice_axis 分成的切片数量。

    返回:
    -------
    dict
        一个字典，其中键是切片范围的字符串表示，值是表示一维直方图的元组 (bin_edges, counts)。
    """
    x_edges, y_edges, counts = hist
    counts = np.asarray(counts)

    # 验证 counts 数组的形状
    expected_shape = (len(y_edges) - 1, len(x_edges) - 1)
    if counts.shape != expected_shape:
        raise ValueError(
            f"counts 的形状 {counts.shape} 与箱体边界不匹配 "
            f"（期望形状: {expected_shape}）。"
            "如果 counts 来自 np.histogram2d，请先将其转置。"
        )

    projected_hists = {}

    if slice_axis == 'y':
        slicing_edges = y_edges
        projection_edges = x_edges
        sum_axis = 0  # 沿y轴对箱体求和
    elif slice_axis == 'x':
        slicing_edges = x_edges
        projection_edges = y_edges
        sum_axis = 1  # 沿x轴对箱体求和
    else:
        raise ValueError("slice_axis 必须是 'x' 或 'y'")

    # 根据 slice_num 自动生成切片范围
    slice_min, slice_max = slicing_edges[0], slicing_edges[-1]
    slice_boundaries = np.linspace(slice_min, slice_max, slice_num + 1)
    slice_ranges = list(zip(slice_boundaries[:-1], slice_boundaries[1:]))

    for v_min, v_max in slice_ranges:
        # 查找与切片范围对应的箱体索引
        start_idx = np.searchsorted(slicing_edges, v_min, side='left')
        end_idx = np.searchsorted(slicing_edges, v_max, side='right')

        if start_idx >= end_idx:
            continue

        # 选择切片并投影
        if slice_axis == 'y':
            data_slice = counts[start_idx:end_idx, :]
        else:  # slice_axis == 'x'
            data_slice = counts[:, start_idx:end_idx]

        projected_counts = np.sum(data_slice, axis=sum_axis)

        # 准备输出
        key = f"{v_min:.1f} to {v_max:.1f}"
        projected_hists[key] = (projection_edges, projected_counts)

    return projected_hists


def plot_projections(projection_groups, slice_axis_name, xlabel, normalize_to=None, **kwargs):
    """
    绘制多组一维投影直方图的对比图。

    参数:
    ----------
    projection_groups : dict
        一个字典，键是组的标签（如 "Data", "MC"），值是 project_2d_histogram 返回的字典。
    slice_axis_name : str
        切片轴的名称（如 "Truth", "Reco"），用于生成标签。
    normalize_to : float, optional
        如果提供，所有直方图的面积将被归一化到此值。默认为 None（不归一化）。
    **kwargs :
        传递给 plt.figure 的其他关键字参数，例如 figsize。
    """
    if not projection_groups:
        print("Warning: projection_groups 字典为空，无法绘图。")
        return

    first_group = next(iter(projection_groups.values()))
    slice_keys = list(first_group.keys())
    num_plots = len(slice_keys)

    # 1. 创建纵向排列、共享x轴的子图
    fig, axes = plt.subplots(num_plots, 1, sharex=True,
                             gridspec_kw={"hspace": 0.3}, **kwargs)
    axes = np.atleast_1d(axes).flatten()

    for i, slice_key in enumerate(slice_keys):
        ax = axes[-i-1]
        for group_label, projections in projection_groups.items():
            if slice_key not in projections:
                continue

            edges, counts = projections[slice_key]

            if normalize_to is not None:
                bin_widths = np.diff(edges)
                integral = np.sum(counts)
                if integral > 0:
                    scale = normalize_to / integral
                    counts = counts * scale

            ax.stairs(counts, edges, label=group_label, fill=False)

        # 2. 将切片范围用text标注在子图里
        v_min_str, v_max_str = slice_key.split(' to ')
        label_text = slice_axis_name + fr"$\in ({v_min_str}, {v_max_str})$"
        ax.text(1, 1.01, label_text, transform=ax.transAxes,
                fontsize='x-small', ha='right', va='bottom',)

        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[0].legend()
    # 3. 共用x轴，只在最下方的图标注xlabel
    axes[-1].set_xlabel(xlabel)
    # 4. 共用y轴，只标注一个ylabel
    fig.supylabel(
        fr"$\text{{Counts (Norm. to {normalize_to})}}$" if normalize_to is not None else r"$\text{Counts}$")

    return fig


def plot_2d_slides(hist: HistFactory, slice_axis, slice_num,
                   slice_axis_name, xlabel, normalize_to=None, **kwargs):
    if isinstance(hist, HistFactory):
        groups = {}
        for key, val in hist.staff_dict.items():
            groups[key] = project_2d_histogram(
                val.get_numpy(), slice_axis, slice_num)

        return plot_projections(groups, slice_axis_name, xlabel, normalize_to=normalize_to, **kwargs)
    elif isinstance(hist, HistStaff):
        groups = {
            hist.name: project_2d_histogram(
                hist.histogram, slice_axis, slice_num)
        }
        return plot_projections(groups, slice_axis_name, xlabel, normalize_to=normalize_to, **kwargs)

def add_particle_mass_ticks(ax, particle_names, squared=False, axis_position='top', unit='GeV', ticks_param = {}):
    """
    Add particle mass (or mass squared) ticks to a matplotlib axis using known particle masses.

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axis to which the particle mass ticks will be added.
    particle_names : list of str
        List of particle names recognized by the `particle` package (e.g., 'pi0', 'eta', 'f(0)(980)').
    squared : bool, optional
        If True, place ticks at mass squared values. Otherwise, at mass values. Default is False.
    axis_position : {'bottom', 'top'}, optional
        Position for the new axis. Default is 'bottom'.
    unit : str, optional
        Unit of the axis for display ('GeV', 'MeV', etc.). Currently used only for label display. Default is 'GeV'.

    Returns:
    -------
    secax : matplotlib.axis.Axis
        The secondary x-axis with particle mass ticks.

    Example:
    -------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0.1, 1.2], [1, 2])
    >>> add_particle_mass_ticks(ax, ['pi0', 'eta', 'f(0)(980)'], squared=False)
    >>> plt.show()

    Notes:
    -----
    - The function uses the `Particle.from_name()` method to resolve particle masses.
    - Masses are automatically converted to GeV.
    - If `squared=True`, the tick positions will be set at m² [GeV²].
    - If a particle is not found, a warning will be printed and it will be skipped.
    """
    
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from particle import Particle

    # Get particle masses
    masses = []
    labels = []
    for name in particle_names:
        try:
            p = Particle.from_name(name)
            mass_GeV = p.mass / 1e3  # convert MeV to GeV
            val = mass_GeV**2 if squared else mass_GeV
            masses.append(val)
            labels.append(f"${p.latex_name}$")
        except Exception as e:
            print(f"Warning: Could not find particle '{name}': {e}")

    # Set ticks on the chosen axis
    twin_ax = ax.twiny() if axis_position == 'top' else ax.twiny()
    # print(ax.get_xlim())
    twin_ax.set_xticks(masses)
    twin_ax.set_xticklabels(labels, **ticks_param)
    twin_ax.minorticks_off()
    twin_ax.tick_params(axis='x', direction='out', length=3)

    if axis_position == 'bottom':
        twin_ax.xaxis.set_ticks_position('bottom')
        twin_ax.xaxis.set_label_position('bottom')
        twin_ax.spines['bottom'].set_position(('outward', 0))
    else:
        twin_ax.xaxis.set_ticks_position('top')
        twin_ax.xaxis.set_label_position('top')
    
    twin_ax.set(xlim = ax.get_xlim())

    return twin_ax