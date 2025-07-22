import abc
from enum import Enum, auto

class StaffType(Enum):
    """用于标记直方图的类型：信号、背景或数据。"""
    signal = auto()
    background = auto()
    data = auto()
    other = auto() # 用于默认值或process与其他process重复。

class Staff(abc.ABC):
    """
    【抽象基类】单个数据或MC样本的抽象表示。

    这个类的作用是定义所有具体样本类（无论是基于RDataFrame还是Histogram）
    都必须遵循的基本接口规范。它代表了一个独立的分析对象，例如“数据”、“tau tau信号”等。
    """
    def __init__(self, name: str, type: StaffType):
        """
        初始化一个样本。

        参数:
        - name (str): 样本的唯一标识名称，例如 "Data", "DY", r"\\tau\\tau"。
        """
        self.name = name
        self.type = type

    @abc.abstractmethod
    def load(self, source):
        """
        【必须被重写】从指定的源加载数据。
        """
        pass

    @abc.abstractmethod
    def save(self, destination):
        """
        【必须被重写】将处理后的数据保存到指定位置。
        """
        pass

class Factory(abc.ABC):
    """
    【抽象基类】一组Staff样本的集合管理器。

    它的作用是管理一组相关的Staff对象，例如，一个Factory可以包含
    一个数据Staff和多个MC背景Staff。它负责对这些样本进行统一的操作。
    """
    def __init__(self):
        self.staff_collection = {}

    def add_staff(self, staff: Staff):
        """
        向工厂中添加一个样本。

        参数:
        - staff (Staff): 一个Staff类的实例。
        """
        if staff.name in self.staff_collection:
            print(f"警告：样本 '{staff.name}' 已存在，将被覆盖。")
        self.staff_collection[staff.name] = staff

    def get_staff(self, name: str) -> Staff:
        """
        根据名称获取一个样本。

        参数:
        - name (str): 样本的名称。

        返回:
        - Staff: 找到的Staff实例，如果不存在则返回None。
        """
        return self.staff_collection.get(name)

    def __iter__(self):
        """使Factory对象可以被迭代，方便地遍历所有样本。"""
        return iter(self.staff_collection.values())

    def __getitem__(self, key):
        """使Factory对象可以通过像字典一样的中括号语法访问样本。"""
        return self.staff_collection[key]
    
class DataInfo:
    from hepunits import MeV, GeV, invpb, invnb, invfb, nb, pb, fb

    def __init__(self, cms_energy, luminosity: float, path = None, mc_process = None):
        """
        @cms_energy: str formate value with unit of MeV
        @luminosity: data luminosity with unit of pb^{-1}
        """
        self.CMSEnergy = float(cms_energy) * self.MeV
        self.Luminosity = luminosity * self.invpb
        self.Path = path
        self.MCProcess = mc_process
        self.Cut = []

    def __str__(self):
        return self.print_label()

    def __repr__(self):
        return self.print_label()

    def print_label(self, en_unit="MeV", lumi_unit="invpb"):
        if en_unit == "MeV":
            str_en = f"{self.CMSEnergy / self.MeV:.0f} \\mathrm{{~MeV}}"
        elif en_unit == "GeV":
            str_en = f"{self.CMSEnergy / self.GeV:.3f} \\mathrm{{~GeV}}"
        else:
            str_en = f"{self.CMSEnergy:.0f} \\mathrm{{\\textcolor{{red}}{{Bad unit}}}}"

        if lumi_unit == "invpb":
            str_lumi = f"{self.Luminosity / self.invpb:.1f} \\mathrm{{~pb^{{-1}}}}"
        elif lumi_unit == "invnb":
            str_lumi = f"{self.Luminosity / self.invnb:.4f} \\mathrm{{~nb^{{-1}}}}"
        else:
            str_lumi = f"{self.Luminosity / self.invpb:.1f} \\mathrm{{\\textcolor{{red}}{{Bad unit}}}}"

        return str_en + "~(" + str_lumi + ")"
