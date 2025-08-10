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


def compare_mc_data(stack_mc, data, **kargs):
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
    # 获取 x 轴标签，如果未提供则默认为 r"$s ~\mathrm{[GeV^{2}]}$"
    xlabel = kargs.get("xlabel", r"$s ~\mathrm{[GeV^{2}]}$")
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
    plot_chi2 = kargs.get("plot_chi2", None)
        
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
            facecolor = "gray", alpha = 0.6, label = "MC error")
    # 绘制实际数据的误差棒图
    ax1.errorbar(x_data, y_data_norm, xerr = 0, yerr = yerr_data_norm,
            marker = "o", ms = 1.5, color = "black", label = "Data", ls = "", lw = 0.4)
    
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
    if plot_chi2 is not None:
        ax2.text(plot_chi2[0], plot_chi2[1], fr"$\chi^2/N_{{\text{{bins}}}} = {chi2/len(x_mc):.3f}$",
                fontsize = "x-small", horizontalalignment='right', verticalalignment='top',
                transform = fig.transFigure)

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
