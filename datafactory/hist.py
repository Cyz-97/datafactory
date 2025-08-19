from .core import Staff, StaffType, Factory
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Self
from copy import copy, deepcopy

import ROOT as R


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

    x = np.append(x, hist.GetXaxis().GetBinLowEdge(hist.GetXaxis(
    ).GetNbins()) + hist.GetXaxis().GetBinWidth(hist.GetXaxis().GetNbins()))
    y = np.append(y, hist.GetYaxis().GetBinLowEdge(hist.GetYaxis(
    ).GetNbins()) + hist.GetYaxis().GetBinWidth(hist.GetYaxis().GetNbins()))

    return x, y, z


@dataclass
class HistStaff(Staff):
    name: str
    type: StaffType = StaffType.other
    path: str = field(default=None, repr=False)
    histogram: Any = field(default=None)

    numpy_tuple: Tuple[List, List, List] = field(default=None, repr=False)

    # 因为 weight 和 multiply 相互冲突
    # 暂时不放 weight 属性，所有 histogram 都通过 multiply 来实现权重
    # weight: float = field(default=1.0, init=False, repr=False)
    dimension: int = field(default=1, init=False, repr=False)

    def __post_init__(self):
        self.load()

    def load(self):
        if self.histogram is not None:
            pass
        elif self.path is not None:
            with R.TFile.Open(self.path, "read") as f:
                temp_hist = f.Get(self.name)
                if temp_hist:
                    temp_hist.SetDirectory(R.nullptr)
                else:
                    print(f"Warning: there is no {self.name} object in {self.path}")
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
            self.dimension = self.histogram.GetDimension()
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
        if isinstance(other, float):
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

        if isinstance(other, float):
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
        res = HistStaff(name="concatenate",
                        numpy_tuple=(concatenate_edge,
                                     concatenate_y,
                                     concatenate_yerr),
                        type=self.type)
        return res

    def norm_to(self, count: float):
        self.histogram.Scale(count / self.histogram.Integral())
        
    def get_norm_factor(self, count: float):
        return count / self.histogram.Integral()


@dataclass
class HistFactory(Factory):
    staff_dict: Dict[str, HistStaff] = field(default=None, repr=True)
    path_dict: Dict[str, str] = field(default=None, repr=False)
    type_dict: Dict[str, StaffType] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.load()

    def load(self):
        if self.staff_dict is not None:
            # print(self.staff_dict)
            pass
        elif self.path_dict is not None:
            self.staff_dict = {}
            for name, path in self.path_dict.items():
                # print(name)
                temp = HistStaff(name=name,
                                 path=path,
                                 type=self.type_dict.get(name, StaffType.other))
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
            res.staff_dict[name] = staff.concatenate(other.staff_dict[name])

        return res

    def _get_value(self):
        for name, staff in self.staff_dict.items():
            staff._get_value(staff)

    def __copy__(self):
        return HistFactory(staff_dict={key: copy(val) for key, val in self.staff_dict.items()},
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
        if isinstance(other, float):
            for name, staff in res.staff_dict.items():
                res.staff_dict[name] *= other
        else:
            # check if other's key are identical to self's
            if set(other.keys()) in set(self.staff_dict.keys()):
                raise ValueError("The two element are different in keys.")
            
            for name, staff in res.staff_dict.items():
                res.staff_dict[name] *= other[name]

        return res

    def norm_to(self, count_dict: int | float):
        for name, staff in self.staff_dict.items():
            if name in count_dict.keys():
                staff.norm_to(count_dict[name])

    def get_norm_factor(self, count_dict: int | float):
        res = {}
        for name, staff in self.staff_dict.items():
            if name in count_dict.keys():
                res[name] = staff.get_norm_factor(count_dict[name])

        return res