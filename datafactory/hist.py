from .core import Staff, StaffType, Factory
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Self
from copy import copy, deepcopy
from numbers import Number
import os

import ROOT as R
import numpy as np


def TH12Numpy(hist):
    """
    Converts a ROOT histogram object into NumPy arrays for easier manipulation in Python.

    This function takes a ROOT histogram object as input and returns four NumPy arrays:
    - x: the centers of the histogram bins
    - content: the content (counts or weights) of the histogram bins
    - err: the errors associated with each bin's content
    - x_edge: the lower edges of the histogram bins

    The function assumes that the input histogram is a 1D histogram.

    Args:
        hist (ROOT.TH1): A ROOT histogram object.

    Returns:
        tuple: A tuple containing four NumPy arrays (x, content, err, x_edge).

    Raises:
        AttributeError: If the input object does not have the required histogram methods.

    Example:
        >>> import ROOT
        >>> hist = ROOT.TH1F("example", "Example Histogram", 100, 0.0, 10.0)
        >>> # Fill the histogram with some data...
        >>> x, content, err, x_edge = TH12Numpy(hist)
        >>> print(x.shape, content.shape, err.shape, x_edge.shape)
        (100,) (100,) (100,) (101,)

    Note:
        This function relies on the ROOT framework being available and properly configured.
        It also assumes that the input histogram is not empty and has been properly initialized.

    See Also:
        ROOT.TH1: The ROOT histogram class.
        numpy.array: The NumPy array class.
    """
    import ROOT as R
    import numpy as np
    x = np.array([hist.GetBinCenter(i+1) for i in range(hist.GetNbinsX())])
    x_edge = np.array([hist.GetBinLowEdge(i+1)
                      for i in range(hist.GetNbinsX()+1)])
    x_width = np.diff(x_edge)
    content = np.array([hist.GetBinContent(i+1)
                       for i in range(hist.GetNbinsX())])
    err = np.array([hist.GetBinError(i+1) for i in range(hist.GetNbinsX())])
    return x, content, err, x_edge


def Numpy2TH1(x_edge, content, err, name=""):
    import array
    # 确保x_edge转换为C兼容的double数组
    x_edge_arr = array.array('d', x_edge)
    nbins = len(x_edge_arr) - 1

    # 创建非均匀分箱的直方图
    hist = R.TH1D(name, name, nbins, x_edge_arr)

    # 设置每个bin的内容和误差
    for i in range(nbins):
        hist.SetBinContent(i+1, content[i])
        hist.SetBinError(i+1, err[i])

    return hist

def Numpy2TH2(x_edge, y_edge, content, err, name=""):
    """
    [兼容模式] 从二维Numpy矩阵创建一个可变分箱的ROOT TH2D。
    此版本使用经典的 SetBinContent 循环，以确保与不支持现代Numpy接口的旧版ROOT完全兼容。

    Args:
        x_edge (np.ndarray): X轴的bin边界数组。
        y_edge (np.ndarray): Y轴的bin边界数组。
        content (np.ndarray): 2D的bin内容矩阵，形状为 (n_ybins, n_xbins)。
        err (np.ndarray): 2D的bin误差矩阵，形状为 (n_ybins, n_xbins)。
        name (str, optional): 直方图的名称和标题。

    Returns:
        R.TH2D: 创建并填充好的ROOT二维直方图。
    """
    import array
    # 确保ROOT不会将直方图与任何打开的文件关联
    R.gROOT.cd()

    x_nbins = len(x_edge) - 1
    y_nbins = len(y_edge) - 1

    # 校验输入的二维矩阵形状是否与bin的数量匹配
    if content.shape != (y_nbins, x_nbins) or err.shape != (y_nbins, x_nbins):
        raise ValueError(f"内容(content)或误差(err)矩阵的形状应为 ({y_nbins}, {x_nbins})，但当前为 {content.shape}")

    # 使用 'array' 模块创建C-style数组，这是与旧版ROOT C++交互最可靠的方式
    x_edge_arr = array.array('d', x_edge)
    y_edge_arr = array.array('d', y_edge)

    # 创建空的TH2D
    hist = R.TH2D(name, name,
                  x_nbins, x_edge_arr,
                  y_nbins, y_edge_arr)
    
    # 显式激活误差存储。对于SetBinError是必需的。
    hist.Sumw2()

    # 使用嵌套循环填充，这是最兼容、最可靠的方法
    for iy in range(y_nbins):
        for ix in range(x_nbins):
            # 直接使用2D索引访问矩阵
            # 注意：ROOT的bin索引从1开始，而Numpy索引从0开始
            hist.SetBinContent(ix + 1, iy + 1, content[iy, ix])
            hist.SetBinError(ix + 1, iy + 1, err[iy, ix])

    return hist

def TH22Numpy(hist):
    """
    将ROOT的TH2D直方图转换为numpy数组格式。

    参数:
        hist (TH2D): 输入的ROOT二维直方图对象

    返回:
        tuple: 包含三个numpy数组的元组 (x轴坐标数组, y轴坐标数组, z轴数值数组)

    说明:
        暂时没有处理二维直方图的误差
    """
    import ROOT as R
    import numpy as np

    x = np.array([hist.GetXaxis().GetBinLowEdge(i)
                 for i in range(1, hist.GetXaxis().GetNbins() + 1)])
    y = np.array([hist.GetYaxis().GetBinLowEdge(i)
                 for i in range(1, hist.GetYaxis().GetNbins() + 1)])
    z = np.array([[hist.GetBinContent(hist.FindBin(xi, yi)) for xi in x]
                  for yi in y])
    err = np.array([[hist.GetBinError(hist.FindBin(xi, yi)) for xi in x]
                  for yi in y])

    x = np.append(x, hist.GetXaxis().GetBinLowEdge(hist.GetXaxis(
    ).GetNbins()) + hist.GetXaxis().GetBinWidth(hist.GetXaxis().GetNbins()))
    y = np.append(y, hist.GetYaxis().GetBinLowEdge(hist.GetYaxis(
    ).GetNbins()) + hist.GetYaxis().GetBinWidth(hist.GetYaxis().GetNbins()))

    return x, y, z, err


@dataclass
class HistStaff(Staff):
    name: str
    type: StaffType = StaffType.other
    path: str = field(default=None, repr=False)
    histogram: Any = field(default=None)
    
    # xedge, y, yerr
    numpy_tuple: Tuple[List, List, List] = field(default=None, repr=False)

    # 因为 weight 和 multiply 相互冲突
    # 暂时不放 weight 属性，所有 histogram 都通过 multiply 来实现权重
    # weight: float = field(default=1.0, init=False, repr=False)
    dimension: int = field(default=1, init=False, repr=False)

    def __post_init__(self):
        self.load()

    def load(self):
        if self.histogram is not None:
            self.dimension = self.histogram.GetDimension()
            pass
        elif self.path is not None:
            if ":" in self.path:
                path, obj_name = self.path.split(":", 1)
            else:
                path = self.path
                obj_name = self.name
            with R.TFile.Open(path, "read") as f:
                temp_hist = f.Get(obj_name)
                if temp_hist:
                    temp_hist.SetDirectory(R.nullptr)
                else:
                    print(f"Warning: there is no {obj_name} object in {path}")
                    temp_hist = None
                
            self.histogram = temp_hist    
            if self.histogram is not None:
                self.dimension = self.histogram.GetDimension()
                self.histogram.Sumw2()
            else:
                self.dimension = 0

        elif self.numpy_tuple is not None:
            self.histogram = Numpy2TH1(*self.numpy_tuple)
            self.dimension = 1
            self.histogram.Sumw2()
        else:
            raise Exception("HistStaff: path and histogram are both None.")



    def save(self, path: str):
        with R.TFile(path, "update") as f:
            f.WriteObject(self.histogram.Clone(), self.name)

        # ToDO: save weight information; mkdir directory to separate different distributions

    def _get_value(self, obj: Self):
        try: 
            obj.histogram = obj.histogram.GetValue()
        except Exception as e:
            pass
        
    def __add__(self, other: Self):

        if self.dimension != other.dimension:
            raise Exception("Invalid dimension for division.")

        self._get_value(self)
        self._get_value(other)

        res = copy(self)
        res.histogram.Add(other.histogram)
        return res

    def __sub__(self, other: Self):
        if self.dimension != other.dimension:
            raise Exception("Invalid dimension for division.")

        self._get_value(self)
        other._get_value(other)

        res = copy(self)
        res.histogram.Add(other.histogram, -1)
        return res

    def __mul__(self, other):
        # TODO: consider variance propogation
        self._get_value(self)
        self._get_value(other)
        
        res = copy(self)
        if isinstance(other, Number):
            # print(res.histogram.Integral())
            res.histogram.Scale(other)
            # print(res.histogram.Integral())
        elif isinstance(other, HistStaff):
            if self.dimension != other.dimension:
                raise Exception("Invalid dimension for division.")
            res.histogram.Multiply(other.histogram)
        else:
            raise Exception(f"Invalid type for other: {type(other)}")
        return res

    def __truediv__(self, other):
        # TODO: consider variance propogation
        self._get_value(self)
        self._get_value(other)
        
        if self.dimension != other.dimension:
            raise Exception("Invalid dimension for division.")

        if isinstance(other, Number):
            res = copy(self)
            res.histogram.Scale(1/other)
        elif isinstance(other, type(self)):
            res = copy(self)
            res.histogram.Divide(other.histogram)
        else:
            raise Exception(f"Invalid type for other: {type(other)}")
        return res
    
    def __copy__(self):
        self._get_value(self)
        return HistStaff(name=self.name, 
                         histogram=self.histogram.Clone(),
                         type = self.type)
    
    def __deepcopy__(self, memo):
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        self._get_value(self)
        return HistStaff(name=self.name, 
                         histogram=self.histogram.Clone(),
                         type = self.type)
    
    def get_eff(self, other: Self):
        self._get_value(self)
        self._get_value(other)

        eff_hist = R.TGraphAsymmErrors(self.histogram)
        eff_hist.Divide(self.histogram, other.histogram,
                        "cl=0.683 b(1,1) mode")
        return eff_hist
    
    def get_numpy(self):
        import numpy as np
        if self.dimension == 1:
            return TH12Numpy(self.histogram)
        elif self.dimension == 2:
            return TH22Numpy(self.histogram)
        else:
            pass

    # --- UHI conversion ----------------------------------------
    def get_uhi(self):
        """
        Convert a HistStaff (TH1/TH2) into a UHI-compatible boost_histogram.Histogram.

        Returns
        -------
        bh.Histogram
            A histogram with Variable axes and Weight storage (values + variances).
        """
        import hist as hist
        import numpy as np

        if self.histogram is None or self.dimension == 0:
            raise ValueError("HistStaff2UHI: empty histogram in self.")

        if self.dimension == 1:
            # x, content, err, x_edge
            _, content, err, x_edge = self.get_numpy()
            h = hist.Hist(
                hist.axis.Variable(np.asarray(x_edge, dtype=float),
                                name = 'name'),
                storage=hist.storage.Weight()
            )
            
            view = h.view()
            view['value'][...] = content
            view['variance'][...] = err
            return h

        elif self.dimension == 2:
            # x_edge, y_edge, z (y,x), err (y,x)
            x_edge, y_edge, z, err = self.get_numpy()
            # boost-histogram uses axis order [x, y]; our TH22Numpy returns shape (y, x)
            z_xy = np.asarray(z, dtype=float).T
            v_xy = (np.asarray(err, dtype=float)**2).T

            h = hist.Hist(
                hist.Variable(np.asarray(x_edge, dtype=float)),
                hist.Variable(np.asarray(y_edge, dtype=float)),
                storage=hist.storage.Weight()
            )
            view = h.view()
            view.value[:, :] = z_xy
            view.variance[:, :] = v_xy
            return h

        else:
            raise NotImplementedError("HistStaff2UHI currently supports 1D and 2D histograms only.")


    def concatenate(self, other: Self) -> Self:
        import numpy as np

        if self.dimension != 1 or other.dimension != 1:
            raise Exception("Only 1D histograms can be concatenated.")

        x1, y1, yerr1, xedge1 = self.get_numpy()
        x2, y2, yerr2, xedge2 = other.get_numpy()
        concatenate_y = np.concatenate([y1, y2])
        concatenate_yerr = np.concatenate([yerr1, yerr2])
        concatenate_edge = np.concatenate([xedge1,
                                           xedge2[1:] - xedge2[0] + xedge1[-1]])
        res = HistStaff(name=self.name,
                        numpy_tuple=(concatenate_edge,
                                     concatenate_y,
                                     concatenate_yerr),
                        type=self.type)
        return res

    def norm_to(self, count: float):
        if self.histogram.Integral() != 0:
            weight = count / self.histogram.Integral()
            self.histogram.Scale(weight)
        else:
            pass
        
    def get_norm_factor(self, count: float):
        return count / self.histogram.Integral()

    def plot(self, xlabel, ax = None, stair_args = {}):
        import matplotlib.pyplot as plt


        if ax is None:
            ax = plt.figure().subplots(1)
        
        if self.dimension == 1:
            x, y, yerr, edge = self.get_numpy()
            ax.stairs(y, edge, label = "$" + self.name + "$", **stair_args)
        elif self.dimension == 2:
            x, y, z, zerr = self.get_numpy()
            c = ax.pcolormesh(x,y,z)
            plt.colorbar(c)
        ax.set(xlabel = xlabel)
        return ax


@dataclass
class HistFactory(Factory):
    staff_dict: Dict[str, HistStaff] = field(default=None, repr=True)
    path_dict: Dict[str, str] = field(default=None, repr=False)
    type_dict: Dict[str, StaffType] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.load()

    def _check_root_object(self, spec: str):
        """
        输入格式: 'path.root:objectName'
        返回: (exists, message)
        """
        if ":" not in spec:
            return os.path.isfile(spec)

        path, obj_name = spec.split(":", 1)

        # 检查文件存在
        if not os.path.isfile(path):
            return False

        f = R.TFile.Open(path, "READ")
        if not f or f.IsZombie():
            return False

        obj = f.Get(obj_name)

        if not obj:
            return False
        else:
            f.Close()
            return True


    def load(self):
        if self.staff_dict is not None:
            # print(self.staff_dict)
            pass
        elif self.path_dict is not None:
            self.staff_dict = {}
            for name, path in self.path_dict.items():
                if self._check_root_object(path) == False:
                    print(f"HistFactory.load(): {path} not exist.")
                    continue

                temp = HistStaff(name=name,
                                 path=path,
                                 type=self.type_dict.get(name, StaffType.background))
                if temp.histogram is not None:
                    self.staff_dict[name] = temp
                else:
                    pass
        else:
            raise Exception("HistFactory: path and histogram are both None.")

    def save(self, path: str):
        # with R.TFile(path, "recreate") as f:
        for name, staff in self.staff_dict.items():
            staff.save(path)

    def get_numpy(self):
        return {key: val.get_numpy() for key, val in self.staff_dict.items()}
    def get_uhi(self):
        return {key: val.get_uhi() for key, val in self.staff_dict.items()}

    def sum(self, type_list: List[StaffType] = [StaffType.signal, StaffType.background]) -> HistStaff:
        self._get_value()
        res = None

        for name, staff in self.staff_dict.items():
            if staff.type not in type_list:
                continue
            if res is None:
                res = deepcopy(staff)
            else:
                res += staff
        
        return res

    def concatenate(self, other: Self) -> Self:
        res = deepcopy(self)
        for name, staff in self.staff_dict.items():
            if name in other.staff_dict.keys():
                res.staff_dict[name] = staff.concatenate(other.staff_dict[name])
            else:
                # print(f"Warning: no histogram for {name}, concatenate a zero histogram.")
                fake_hist = list(other.staff_dict.values())[0].histogram.Clone()
                fake_hist.Reset()
                
                res.staff_dict[name] = staff.concatenate(
                    HistStaff(name = name, histogram = fake_hist, type=self.type_dict[name])
                )

        return res

    def _get_value(self):
        for name, staff in self.staff_dict.items():
            staff._get_value(staff)

    def __copy__(self):
        
        return HistFactory(staff_dict={key: copy(val) for key, val in self.staff_dict.items()},
                           type_dict = deepcopy(self.type_dict))

    def __deepcopy__(self, memo):
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        return HistFactory(staff_dict={key: deepcopy(val) for key, val in self.staff_dict.items()},
                           type_dict = deepcopy(self.type_dict))

    def __sub__(self, other: Dict[str, Any] | float) -> Self:
        res = copy(self)
        
        for name, staff in res.staff_dict.items():
            res.staff_dict[name] -= other.staff_dict[name]
        return res

    def __truediv__(self, other: Dict[str, Any] | float) -> Self:
        res = copy(self)
        if isinstance(other, Staff):
            for name, staff in res.staff_dict.items():
                res.staff_dict[name] /= other
        elif isinstance(other, Factory):
            for name, staff in res.staff_dict.items():
                res.staff_dict[name] /= other.staff_dict[name]
        else:
            raise ValueError("__truediv__: Undefined division")

        return res

    def __mul__(self, other: Dict[str, Any] | float) -> Self:
        res = copy(self)
        if isinstance(other, Number):
            for name, staff in res.staff_dict.items():
                res.staff_dict[name] *= other
        else:
            # check if other's key are identical to self's
            if set(other.keys()) in set(self.staff_dict.keys()):
                raise ValueError("The two element are different in keys.")
            
            for name, staff in res.staff_dict.items():
                if name in other:
                    res.staff_dict[name] *= other[name]
                else:
                    print(f"HistFacotry.__mul__: no key {name} in weights")

        return res

    def norm_to(self, count_dict: int | float):
        for name, staff in self.staff_dict.items():
            if name in count_dict.keys():
                # print(name, count_dict[name])
                staff.norm_to(count_dict[name])

    def get_norm_factor(self, count_dict: int | float):
        res = {}
        for name, staff in self.staff_dict.items():
            if name in count_dict.keys():
                res[name] = staff.get_norm_factor(count_dict[name])

        return res
    
    def append(self, histstaff: HistStaff):
        self.staff_dict[histstaff.name] = copy(histstaff)
        self.type_dict[histstaff.name] = histstaff.type

    def pop(self, name):
        if name in self.staff_dict:
            self.staff_dict.pop(name)
        if name in self.type_dict:
            self.type_dict.pop(name)

    def remove_empty(self):
        empty_keys = []
        for name, staff in self.staff_dict.items():
            if staff.histogram.Integral() < 1e-16:
                empty_keys.append(name)
        for name in empty_keys:
            if name in self.staff_dict:
                self.staff_dict.pop(name)
            if name in self.type_dict:
                self.type_dict.pop(name)
                
    def plot(self,
                 *,
                 xlabel: str = "",
                 ylabel: str = r"$\mathrm{Count}$",
                 norm_by_width: bool = False,
                 normalize_to_one: bool = False,
                 yscale: str = "linear",
                 xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 sharey: bool = True,
                 cols: int = 3,
                 figsize: Tuple[float, float] = (4,3)):
        """
        将所有 HistStaff 分面绘制（每个 staff 一个子图），共享坐标轴，子图间距为 0。
        不做任何 Data/MC 对比。

        参数
        ----
        xlabel, ylabel : 轴标签（仅在整体下方/左侧各显示一次）
        norm_by_width : 是否按 bin 宽做归一（y/Δx 和 err/Δx）
        normalize_to_one : 是否将每个分布面积归一为 1（与 norm_by_width 可同时开启）
        yscale : 'linear' 或 'log'
        xlim, ylim : 可选的统一坐标范围
        cols : 分面列数
        figsize : 画布大小
        save : 若提供，形如 {'path':'plots','name':'staffs','fmt':'pdf','prefix':'v1'}

        返回
        ----
        (fig, axes)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        from matplotlib.ticker import MaxNLocator, LogLocator

        self._get_value()
        def _prep(y, err, edges, *, by_width: bool, to_one: bool):
            widths = np.diff(edges)
            y_ = y.astype(float).copy()
            e_ = err.astype(float).copy()
            if by_width:
                y_ = np.divide(y_, widths, out=np.zeros_like(y_), where=widths != 0)
                e_ = np.divide(e_, widths, out=np.zeros_like(e_), where=widths != 0)
            if to_one:
                total = np.sum(y)  # 面积用“原始计数”定义；与 by_width 独立
                if total > 0:
                    y_ /= total
                    e_ /= total
            return y_, e_, widths

        # 仅保留有直方图的 staff（保持插入顺序）
        names = [k for k, v in self.staff_dict.items() if v.histogram is not None]
        if not names:
            raise ValueError("No HistStaff with valid histogram to plot.")

        n = len(names)
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=figsize, 
                                 sharex=True, sharey=sharey)
        if isinstance(axes, np.ndarray):
            axes = axes.reshape(rows, cols)
        else:
            axes = np.array([[axes]])

        # 子图之间空隙
        plt.subplots_adjust(wspace=0.0, hspace=0.15)

        # 逐 staff 作图
        for idx, name in enumerate(names):
            r, c = divmod(idx, cols)
            ax = axes[r, c]

            xcent, y, err, edges = self.staff_dict[name].get_numpy()
            y_show, e_show, widths = _prep(y, err, edges,
                                           by_width=norm_by_width,
                                           to_one=normalize_to_one)

            # 用 stairs 画直方图（边为 edges）
            ax.stairs(y_show, edges, color = 'k')
            # 右上角标注 staff 名称
            ax.text(0.97, 0.97, f"${name}$", transform=ax.transAxes,
                    ha="right", va="top", fontsize="xx-small")

        # # 统一坐标设置与刻度防重叠策略
        # for r in range(rows):
        #     for c in range(cols):
        #         ax = axes[r, c]
        #         if (r, c) == (0, 0):
        #             # 仅设置一次全局属性（其余共享）
        #             if xlim is not None:
        #                 ax.set_xlim(*xlim)
        #             if ylim is not None:
        #                 ax.set_ylim(*ylim)
        #             ax.set_yscale(yscale)
        #         # 只有最左列显示 ytick；其余隐藏以避免重叠
        #         if c != 0:
        #             ax.tick_params(labelleft=False)
        #         # 只有最后一行显示 xtick；其余隐藏以避免重叠
        #         if r != rows - 1:
        #             ax.tick_params(labelbottom=False)

        for ax in axes.ravel():
            # 控制刻度密度，减小重叠概率
            ax.tick_params(axis="both", which="major", labelsize="x-small")
            
            if xlim is not None:
                ax.set(xlim = xlim)
            else:
                ax.set(xmargin=0)
                
            if ylim is not None:
                ax.set(ylim = ylim)
            ax.set_yscale(yscale)
            # ax.yaxis.get_offset_text().set_visible(False)
            if 'log' not in yscale:
                ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="upper"))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4,prune="upper"))
                    
        axes[0,0].yaxis.get_offset_text().set_visible(True)
        # # 总体轴标签（仅显示一次）
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)

        return fig, axes
