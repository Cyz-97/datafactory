from .hist import HistStaff, HistFactory
# from .stat import get_chi2

import ROOT as R

import numpy as np
import matplotlib.pyplot as plt
import os 
# plt.style.use('HadTauAlg-00-01/script/datafactory/style.mplstyle')

def apply_style():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    style_path = os.path.join(script_dir, 'style.mplstyle')
    plt.style.use(style_path)


def compare_mc_data(stack_mc, data, xlabel, **kargs):
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
        None
    """
    from metadata import get_color
    import numpy as np
    import matplotlib.pyplot as plt
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    style_path = os.path.join(script_dir, 'style.mplstyle')

    # 获取 weight, 如果未提供则为 1
    weights = kargs.get("weights", {key: 1 for key in stack_mc.staff_dict.keys()})
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
    figsize = kargs.get("figsize", (4,4))
    # 是否在 pull plot 里放 chi2
    plot_chi2_pos = kargs.get("plot_chi2_pos", None)
        
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
    x_mc_col, y_mc_col, yerr_mc_col, x_width_mc_col = {},{},{},{}
    
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
    fig = plt.figure(figsize = figsize)
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1], "hspace": 0.1})
    # 初始化基线数组
    baseline = np.zeros_like(x_data)
    # 初始化基线误差数组
    baseline_err = np.zeros_like(x_data)
    # 遍历蒙特卡洛数据字典
    for i in x_mc_col.keys():
        # 绘制柱状图
        if (stack):
            ax1.bar(x_edge_mc[:-1], y_mc_col[i], width = x_width_mc_col[i], bottom = baseline,
                    label = "$"+i+"$", lw = 0, alpha = 0.8, color = "#"+get_color(i),
                    edgecolor='white', align='edge',
                    hatch = "/////\\\\\\\\\\" if i == highlight_channel else "")
        else:
            ax1.stairs(y_mc_col[i],np.hstack([ x_mc_col[i][0] - x_width_mc_col[i][0]/2, x_mc_col[i] + x_width_mc_col[i]/2 ]), 
                    label = "$"+i+"$", lw = 0.6, alpha = 0.8, color = "#"+get_color(i))
        # 累加基线
        baseline += y_mc_col[i]
        # 累加基线误差
        baseline_err = np.hypot(baseline_err , yerr_mc_col[i])
        
    # 求和后基线表示总 MC histogram
    ax1.bar(x_mc, baseline_err, width = x_width_mc, 
            bottom = baseline - baseline_err/2, 
            hatch = "//////////", hatch_linewidth = 0.6, 
            fill = False, lw = 0, ls = "", 
            facecolor = "gray", alpha = 0.6, label = r"$\text{MC error}$")
    # 绘制实际数据的误差棒图
    ax1.errorbar(x_data, y_data_norm, xerr = 0, yerr = yerr_data_norm,
            marker = "o", ms = 1.5, color = "black", label = r"$\text{Data}$", ls = "", lw = 0.4)
    
    # 计算 MC data 的 chi2
    chi2 = np.sum(
        np.divide( (baseline - y_data_norm)**2, 
                   (np.linalg.norm([baseline_err, yerr_data_norm]))**2,
                   where = np.linalg.norm([baseline_err, yerr_data_norm]) != 0,
                   out = np.zeros_like(baseline)
        )
    )
    
    if (stack == False):
        ax1.stairs(baseline, np.hstack([x_mc_col[i][0] - x_width_mc_col[i][0]/2, x_mc_col[i] + x_width_mc_col[i]/2]),
                    label = "Mix MC", lw = 0.6, alpha = 1, color = "black")
    
    # 在图中添加 DataInfo 信息
    ax1.text(1, 1.02, "$" + str(datainfo) + "$",
             fontsize = "x-small", horizontalalignment='right',
             transform = ax1.transAxes)
    
    # 设置 y 轴标签和范围
    ax1.set(ylabel = ylabel, 
            ylim = (0 if yscale != "log" else None,
                    np.max(y_data_norm)*1.8) if ylim == None else ylim,
            yscale = yscale)
    # 添加图例
    legend = ax1.legend(title = legend_title,
               loc = "best", ncol=4, handlelength=1.5, fontsize = 5, columnspacing = 0.5)
    
    # 设置图例标题颜色
    if highlight_channel:
        plt.setp(legend.get_title(), color="#" + get_color(highlight_channel))
    # 关闭网格线
    ax1.grid(0)

    # 计算差异
    diff = np.divide(y_data_norm, baseline, where = yerr_data_norm != 0, out = np.ones_like(y_data_norm))
    # 计算差异误差
    diff_err = np.divide(yerr_data_norm, baseline , where = baseline != 0, out = np.zeros_like(yerr_data))
    # # 绘制差异柱状图
    # ax2.bar(x_data, diff, width = np.diff(x_edge_data)*0.8, alpha = 0.8, color = "blue", ls  = "")

    # 绘制差异误差棒图
    ax2.bar(x_mc, baseline_err/baseline, width = x_width_data,
            bottom = 1-baseline_err/2/baseline,
            hatch = "//////////", hatch_linewidth = 0.6, 
            fill = False, lw = 0, ls = "", 
            facecolor = "gray", alpha = 0.6)
    # ax2.stairs(baseline_err/baseline, edges=x_edge_mc,
    #        baseline=1-baseline_err/2/baseline,
    #        fill=True, # 保持与 bar 相同的 fill 状态
    #        color="gray", # stairs 没有 facecolor，使用 color 来设置线条颜色
    #        linewidth = 0.6, # 对应 hatch_linewidth，作为边框线宽
    #        linestyle = "-", # stairs 默认是实线，这里保持
    #        alpha = 0.6
    #        )

    ax2.errorbar(x_data, diff,
                 xerr = 0, yerr = diff_err,
               marker = "o", ms = 1.5, color = "black", ls = "", lw = 0.4)

    # 在图中添加 chi2 信息
    if plot_chi2_pos is not None:
        ax2.text(plot_chi2_pos[0], plot_chi2_pos[1], fr"$\chi^2/N_{{\text{{bins}}}} = {chi2/len(x_mc):.3f}$",
                fontsize = "x-small", horizontalalignment='right', verticalalignment='top',
                transform = ax2.transAxes)

    # 设置 x 轴和 y 轴标签以及范围
    ax2.set(xlabel = xlabel, ylabel = r"$\text{Data/MC}$",
            xlim = xlim, ylim = (0.5, 1.5))
    
    # 添加箭头指示超出 ylim 范围的点
    ratio_ylim = ax2.get_ylim()
    for x, y, yerr in zip(x_data, diff, diff_err):
        upper = y
        lower = y
        if upper > ratio_ylim[1]:
            # 向上箭头，指示超过上界
            ax2.annotate('', xy=(x, ratio_ylim[1]*1.0), xytext=(x, 1.2),
                         arrowprops=dict(arrowstyle='simple', color='blue', lw=0.4, mutation_scale=4),
                         ha='center')
        elif lower < ratio_ylim[0]:
            # 向下箭头，指示低于下界
            ax2.annotate('', xy=(x, ratio_ylim[0]*1.0), xytext=(x, 0.8),
                         arrowprops=dict(arrowstyle='simple', color='blue', lw=0.4, mutation_scale=4),
                         ha='center')
    
    
    # 绘制水平参考线
    ax2.axhline(y=1, color='black', linestyle='-', lw = 0.5)
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
        plt.savefig(os.path.join(save['path'], f"{save['prefix']}_{n_pi:.0f}pi{n_pi0:.0f}pi0_{save['name']}.{save['fmt']}"))

    # 调整位置
    ax1.yaxis.set_label_coords(-0.1, 0.5)
    ax2.yaxis.set_label_coords(-0.1, 0.5)

    # 显示图片
    return ax1, ax2


#========================= 
#          2D Plot       #
#=========================
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
    fig.supylabel(fr"$\text{{Counts (Norm. to {normalize_to})}}$" if normalize_to is not None else r"$\text{Counts}$")

    return fig

def plot_2d_slides(hist: HistFactory, slice_axis, slice_num, 
                   slice_axis_name, xlabel, normalize_to=None, **kwargs):
    if isinstance(hist, HistFactory):
        groups = {}
        for key, val in hist.staff_dict.items():
            groups[key] = project_2d_histogram(val.get_numpy(), slice_axis, slice_num)

        return plot_projections(groups, slice_axis_name, xlabel, normalize_to=normalize_to, **kwargs)
    elif isinstance(hist, HistStaff):
        groups = {
            hist.name: project_2d_histogram(hist.histogram, slice_axis, slice_num)
        }
        return plot_projections(groups, slice_axis_name, xlabel, normalize_to=normalize_to, **kwargs)