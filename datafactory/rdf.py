from .core import Staff, StaffType, Factory
from .hist import HistFactory, HistStaff
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from hepunits import MeV, GeV, invpb, invnb, invfb, nb, pb, fb

import ROOT as R

def ensure_parent_directory(path):
    """
    确保指定路径的父目录存在，如果不存在则创建
    
    Args:
        path (str): 需要检查的文件或目录路径
    
    Prints:
        输出父目录是否已存在或已被创建的信息
    """
    import os
    parent_dir = os.path.dirname(path)  # 获取父级目录路径
    if not os.path.exists(parent_dir):  # 检查目录是否存在
        os.makedirs(parent_dir)  # 创建目录
        # print(f"Directory '{parent_dir}' has been created.")
    else:
        pass
        # print(f"Directory '{parent_dir}' already exists.")

@dataclass
class CutFlow:
    name: str
    list_bystander_: Dict[str, Tuple]
    formular_: str
    latex_: str = field(default_factory=lambda: r"\mathrm{name}")

    # Runtime attributes (not in __init__)
    sample_init: Any = field(init=False, repr=False)
    sample_final: Any = field(init=False, repr=False)
    dir_bystander_hist_init: Dict[str, Any] = field(init=False, repr=False)
    dir_bystander_hist_final: Dict[str, Any] = field(init=False, repr=False)
    count_init: Any = field(init=False, repr=False)
    count_final: Any = field(init=False, repr=False)

    def __post_init__(self):
        """
        CutFlow 数据类，用于管理 ROOT RDataFrame 分析中的切选流程。
        
        属性:
            name: 切选名称 (str)
            list_bystander_: 旁观者分支名称与分箱配置的映射字典 (Dict[str, Tuple])
            formular_: ROOT RDataFrame 过滤公式 (str)
            latex_: 可选的LaTeX格式显示名称 (str)
        
        运行时属性 (不在__init__中初始化):
            sample_init: 初始样本
            sample_final: 切选后样本  
            dir_bystander_hist_init: 初始旁观者直方图字典
            dir_bystander_hist_final: 切选后旁观者直方图字典
            count_init: 初始计数
            count_final: 切选后计数
        
        方法:
            __post_init__: 自动生成默认latex名称（如果未提供）
            copy: 创建CutFlow实例的深拷贝
            apply_on_rdf: 在RDataFrame样本上应用切选流程
            get_latex: 获取LaTeX格式名称
        """    
        if self.latex_ == r"\mathrm{name}":
            self.latex_ = fr"\mathrm{{{self.name}}}"

    def copy(self) -> 'CutFlow':
        """Create a deep copy of the CutFlow instance"""
        return CutFlow(
            name=self.name,
            list_bystander_=self.list_bystander_.copy(),
            formular_=self.formular_,
            latex_=self.latex_
        )

    def apply_on_rdf(self, sample: Any) -> None:
        """
        Initialize sample processing with RDataFrame.

        Args:
            sample: Input ROOT RDataFrame sample
        """
        self.sample_init = R.RDF.AsRNode(sample)
        self.sample_final = self.sample_init.Filter(self.formular_, self.name)
        
        # Initialize 1D/2D histograms
        self.dir_bystander_hist_init = {
            name: (self.sample_init.Histo1D(("", "", *binning), name) 
                   if isinstance(name, str) 
                   else self.sample_init.Histo2D(("", "", *binning), *name))
            for name, binning in self.list_bystander_.items()
        }
        
        self.dir_bystander_hist_final = {
            name: (self.sample_final.Histo1D(("", "", *binning), name) 
                   if isinstance(name, str) 
                   else self.sample_final.Histo2D(("", "", *binning), *name))
            for name, binning in self.list_bystander_.items()
        }
        
        self.count_init = self.sample_init.Count()
        self.count_final = self.sample_final.Count()

        
    def get_latex(self):
        return self.latex_


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
    cms_energy: str
    luminosity: float
    path: Optional[str] = None
    mc_process: Optional[str] = None
    cut: list = field(default_factory=list)
    
    def __post_init__(self):
        """Convert string energy to float and apply units after initialization."""
        self.CMSEnergy = float(self.cms_energy) * MeV
        self.Luminosity = self.luminosity * invpb

    def set_cuts(self, cut_flow):
        """Set the cuts for this data.
        
        Args:
            cut_flow: CutFlow object containing the cuts
        """
        self.cut = cut_flow

    def __str__(self):
        return self.print_label()

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


@dataclass
class RDFStaff(Staff):
    """A class for handling ROOT RDataFrame operations with dataclass support."""

    # process name
    name: str
    # path of root file
    path: str          
    # name of the tree in the root file
    tree_name: str = "evt"            
    # name of the tree recording the pre selection cut chain
    pre_cut_tree_name: str = "cut"             
    # type of the process: signal, background, data, or other
    type: StaffType = StaffType.other  
    # name of pre selection criteria
    pre_cut_names: Optional[List[str]] = field(default=None, init=True)
    # range of the RDF: used for testing
    range: Optional[int] = field(default=None, init=True)
    # cross section of the process, in nb
    xsec: Optional[float] = field(default=1, init=True)

    # weight
    weight: Optional[float] = field(default=1, init=False)
    # pre-selection cut chain
    pre_cut_chain: List[float] = field(default_factory=list, init=False, repr=False)
    # pre-selection cut tree
    pre_cut_tree: R.RDataFrame = field(init=False, repr=False)
    # cut applied on the sample: CutFlow
    cuts: List[CutFlow] = field(default_factory=list[CutFlow], init=False)
    # core RDataFrame
    rdf: Any = field(init=False, repr=False)
    
    # initial dictionary of rdf, used to avoid reuse,
    # thereby improving the efficiency of the code
    __REUSE_DF__: Dict[Any, Any] = field(
        default_factory=dict, init=False, repr=False)
    # cache of column names
    _column_names: List[str] = field(init=False, repr=False)

    def __post_init__(self):
        """在dataclass自动生成的__init__后执行初始化操作。
        加载ROOT文件、设置截面并注册预选切割条件。
        
        注意:
        - 继承父类的初始化逻辑
        - 依次调用load()和pre_selection()方法完成初始化
        """
        """Initialize after dataclass auto-generated __init__."""

        # Load ROOT file, set cross section, and register pre-selection cuts
        self.load()
        self.pre_selection()
        # self._column_names = self.rdf.GetColumnNames()

    def load(self):
        # Create RDataFrame from ROOT file(s)
        if self.path in self.__REUSE_DF__:
            self.rdf = self.__REUSE_DF__[self.path]
        else:
            try:
                if self.range is None:
                    self.__REUSE_DF__[self.path] = R.RDF.AsRNode(
                        R.RDataFrame(self.tree_name, self.path))
                else:
                    self.__REUSE_DF__[self.path] = R.RDF.AsRNode(
                        R.RDataFrame(self.tree_name, self.path)).Range(self.range)

                self.rdf = self.__REUSE_DF__[self.path]
                self.rdf.GetColumnNames()
            except Exception as e:
                self.__REUSE_DF__[self.path] = None
                self.rdf = self.__REUSE_DF__[self.path]
            # root_file.Close()

    def save(self, tree_name: str, path: str, var: list[str]):
        """
        保存处理后的数据到指定路径。

        参数
        ----------
        tree_name : str
            要保存的 Tree name
        path : str
            保存文件的完整路径
        var : list[str]
            要保存的变量列表

        功能
        ----------
        1. 将经过筛选的数据保存为ROOT Tree
        2. 保存 cut chain 的计数信息
        3. 保存 spectator 变量的直方图
        """
        import os

        ensure_parent_directory(path)

        opts_cut = R.RDF.RSnapshotOptions()
        opts_cut.fLazy = False
        opts_cut.fMode = "Update"
        opts_cut.fOverwriteIfExists = True

        self.cuts[-1].sample_final.Snapshot(tree_name, path, var, opts_cut)

        # save cut chain
        cut_df = R.RDataFrame(1)
        cut_df = cut_df.Define("N0", f"{self.pre_cut_chain['N0'].GetValue()}")
        for idx, name in enumerate(self.pre_cut_chain.keys()):
            if idx > 0:
                cut_df = cut_df.Define(
                    f"N{idx}", f"{self.pre_cut_chain[name].GetValue()}")
        for cut in self.cuts:
            cut_df = cut_df.Define(cut.name, f"{cut.count_final.GetValue()}")

        # save histograms of spectators at all levels of cuts.
        cut_df.Snapshot(f"cut_{tree_name}", path,
                        cut_df.GetColumnNames(), opts_cut)
        with R.TFile(path, "update") as outfile:
            for cut in self.cuts:
                for bystander_name in cut.list_bystander_:
                    if isinstance(bystander_name, str):
                        hist_name = bystander_name + "_" + cut.name
                    elif len(bystander_name) == 2:  # 2D plot
                        hist_name = bystander_name[0] + "_vs_" + \
                            bystander_name[1] + "_2D_" + cut.name
                    else:
                        hist_name = "UNKOWN"
                    outfile.WriteObject(cut.dir_bystander_hist_final[bystander_name].GetValue(
                    ), hist_name)

    def pre_selection(self):
        """
        Apply pre-selection cuts to RDataFrame.

        Parameters
        ----------
        tree_name : (str), optional
            Name of the ROOT tree containing the pre-selection cuts. Default is "cut".
        cut_names : (list, str), optional
            List of names for each pre-selection cut. Default is [].
        """
        # Create RDataFrame from pre-selection tree and calculate pre-selection cut results
        self.pre_cut_tree = R.RDF.AsRNode(
            R.RDataFrame(
                self.pre_cut_tree_name,
                self.path
            ))
        self.pre_cut_chain = {}
        if self.pre_cut_names is None:
            self.pre_cut_names = [str(i) for i in self.pre_cut_tree.GetColumnNames()]
        for idx, name in enumerate(self.pre_cut_names):
            if idx > 14: # Generally, the number of cut layers for the files output by BOSS will not be greater than 7.
                break
            else:
                self.pre_cut_chain[name] = self.pre_cut_tree.Sum(f"{name}")
                
    def set_cuts(self, cuts: List[CutFlow]):
        """
        Register filter list.

        Parameters
        ----------
        cuts : list of functions
            List of functions to apply as filters to the RDataFrame.
        """
        import copy
        if self.rdf != None:
            self.cuts = copy.deepcopy(cuts)
            iter_rdf = R.RDF.AsRNode(self.rdf)
            for cut in self.cuts:
                cut.apply_on_rdf(iter_rdf)
                iter_rdf = R.RDF.AsRNode(cut.sample_final)
        else:
            print("set_cuts(): There is no RDF.")
        
    def get_hist(self, func):
        """
        Apply filter list and calculate histogram using given function.

        Parameters
        ----------
        func : function
            Function calculating histogram, RNode -> RResultPtr
        """

        if (len(self.cuts) > 0):
            hist = func(self.cuts[-1].sample_final)
        else:
            hist = func(self.rdf)
        return hist

    def get_histstaff(self, func):
        """
        Apply filter list and calculate histogram using given function.

        Parameters
        ----------
        func : function
            Function calculating histogram, RNode -> RResultPtr
        """

        if (len(self.cuts) > 0):
            hist = func(self.cuts[-1].sample_final)
        else:
            hist = func(self.rdf)

        return HistStaff(name = self.name, histogram=hist, type = self.type)

    def get_cut_chain_table(self):
        """
        获取cut chain统计表
        
        从原始ROOT文件中提取预定义的cut chain数据，并追加后期定义的cut chain数据，
        组合成一个pandas Series对象。
        
        返回:
            pd.Series: 包含所有切割链名称及其对应计数值的Series对象，
                        Series名称为"$" + self.name + "$"格式
        """   
        import pandas as pd

        cut_chain_names = list(self.pre_cut_chain.keys()) + [elem.name for elem in self.cuts]
        cut_counts = [i.GetValue() for i in self.pre_cut_chain.values()] + [elem.count_final.GetValue() for elem in self.cuts]
        res = pd.Series(dict(zip(cut_chain_names, cut_counts)), name = "$" + self.name + "$")
        return res

    def empty(self):
        return self.rdf is None
    
    def define(self, branch_name: str, func_str: str):
        if branch_name in self.rdf.GetColumnNames():
            self.rdf = self.rdf.Redefine(branch_name, func_str)
        else: 
            self.rdf = self.rdf.Define(branch_name, func_str)
            # self._column_names = self.rdf.GetColumnNames()
    
    
@dataclass
class RDFFactory(Factory):
    """
    A factory class for loading and processing Monte Carlo (MC) samples using ROOT's RDataFrame.
    Manages a collection of RDFStaff instances corresponding to different physics processes.

    Attributes:
        path_dict (Dict[str, str]): Dictionary of file paths for each sample
        xsec_dict (Dict[str, float]): Dictionary of cross sections (in nb) for each sample
        luminosity (float): Integrated luminosity (in nb⁻¹)
        tree_name (str): Name of the event TTree (default: "evt")
        pre_cut_tree_name (str): Name of the preselection cut TTree (default: "cut")
        cuts (List[str]): Additional event selection cuts
        classify_dict (Dict[str, str]): Truth-level selections to divide samples
        pre_cut_names (Optional[List[str]]): Names of preselection cuts
        range (Optional[Any]): Range restriction for testing
    """
    path_dict: Dict[str, str]
    xsec_dict: Dict[str, float]
    luminosity: float
    
    # Configuration options with defaults
    type_dict: Dict[str, StaffType] = field(default_factory=dict)
    tree_name: str = "evt"
    tree_name: str = "evt"
    pre_cut_tree_name: str = "cut"
    cuts: List[str] = field(default_factory=list)
    classify_dict: Dict[str, str] = field(default_factory=dict)
    pre_cut_names: Optional[List[str]] = None
    range: Optional[Any] = None
    
    # Internal state
    staff_dict: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Validate inputs and load samples after initialization"""
        self._validate_inputs()
        self.load()

    def _validate_inputs(self):
        """Check consistency between path and cross section dictionaries"""
        missing_xsec = set(self.path_dict.keys()) - set(self.xsec_dict.keys())
        if missing_xsec:
            raise ValueError(
                f"Missing cross sections for samples: {missing_xsec}\n"
                f"Path keys: {set(self.path_dict.keys())}\n"
                f"Xsec keys: {set(self.xsec_dict.keys())}"
            )
        
    def load(self):
        """
        Loads the MC samples using the RDFStaff class.
        """
        import os
        self.staff_dict = {}
        for key, value in self.path_dict.items():
            #if not os.path.isfile(value):
            #    print("not a file", key)
            #    continue            
            evt_chain = R.TChain(self.tree_name, "Read")
            evt_chain.Add(value)
            cut_chain = R.TChain(self.pre_cut_tree_name, "Read")
            cut_chain.Add(value)          
            if (evt_chain.GetEntries() <= 0) and (cut_chain.GetEntries() <= 0 ):
                print("no cut and evt trees, skip", key)
                continue
            else:
                self.staff_dict[key] = RDFStaff(path = value,
                                            name = key,
                                            xsec = self.xsec_dict[key], 
                                            pre_cut_tree_name = self.pre_cut_tree_name,
                                            pre_cut_names = self.pre_cut_names,
                                            range = self.range,
                                            type = self.type_dict.get(key, StaffType.other))
            
        # Divide sample into components basing on a set of cuts
        for key, value in self.classify_dict.items():
            if (len(value) != 0):
                self.staff_dict[key].rdf = self.staff_dict[
                    key].rdf.Filter(value, "preprocess: ")

    def define(self, branch_name: str, formula_str: str):
        """
        Define a new branch for all RDataFrame
        """
        for key, value in self.staff_dict.items():
            value.define(branch_name, formula_str)

    def set_cuts(self, cuts: List[CutFlow]):
        """
        Register the event selection cuts to each sample.

        Parameters:
        -----------
        cuts: (list)
            A list of event selection cuts.
        """
        for key, value in self.staff_dict.items():
            value.set_cuts(cuts)

    def get_hist(self, func, log=True):
        """
        Returns a list of histograms of the final selected events for each MC sample.

        Parameters:
        -----------
        func: function
            A function that takes in a ROOT.TH1 object and returns a modified ROOT.TH1 object.
        log: bool, optional
            If True, prints the weights for each sample. Default is True.

        Returns:
        --------
        ROOT.THStack object
            The stacked histogram of the final selected events.
        """
        hists = {i: self.staff_dict[i].get_hist(
            func) for i in self.staff_dict.keys() if not self.staff_dict[i].empty()}
#         names = list(self.staff_dict.keys())
        [hists[i].SetTitle(i) for i in self.staff_dict.keys() if not self.staff_dict[i].empty()]
        return hists

    def get_histfactory(self, func, log=True):
        """
        Returns a list of histograms of the final selected events for each MC sample.

        Parameters:
        -----------
        func: function
            A function that takes in a ROOT.TH1 object and returns a modified ROOT.TH1 object.
        log: bool, optional
            If True, prints the weights for each sample. Default is True.

        Returns:
        --------
        ROOT.THStack object
            The stacked histogram of the final selected events.
        """
        hists_dict = {i: self.staff_dict[i].get_histstaff(
            func) for i in self.staff_dict.keys() if not self.staff_dict[i].empty()}
        res = HistFactory(staff_dict=hists_dict, type_dict=self.type_dict)
        return res
    
    
    def get_weights(self, virtual_xsec=None):
        if virtual_xsec == None:
            virtual_xsec = self.xsec_dict

        weights = {key:
                   self.luminosity *
                   virtual_xsec[key] /
                   self.staff_dict[key].pre_cut_chain[list(self.staff_dict[key].pre_cut_chain.keys())[0]].GetValue() if self.staff_dict[key].pre_cut_chain[list(self.staff_dict[key].pre_cut_chain.keys())[0]].GetValue() > 0 else 1
                   for key in self.staff_dict.keys()
                   }
        
        return weights

    def get_cut_chain_table(self, weight = None):
        """
        从每个 RDFStaff 中获得 Cut Chain 的 pandas.Series，归一化到 weight 之后生成pandas.DataFrame
        """
        import pandas as pd
        if weight == None:
            weight = self.get_weights()
            
        res = []
        # print(self.staff_dict)

        for rdf in self.staff_dict.values():
            # print(rdf.name)
            res.append(rdf.get_cut_chain_table() * weight[rdf.name])
        res = pd.DataFrame(res)
        res = res.fillna(0)
        res.loc["Sum",:] = res.sum()
        return res
    
    def save(self, tree_name: str, path_dir: Dict[str, str], var: List[str]):
        for key, val in self.staff_dict.items():
            print(f"Writing file: {path_dir[key]}")
            val.save(tree_name, path_dir[key], var)