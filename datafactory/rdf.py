import os, glob
from .core import Staff, StaffType, Factory
from .hist import HistFactory, HistStaff
from dataclasses import dataclass, field
import dataclasses as dc
from typing import Optional, List, Dict, Any, Tuple
from hepunits import MeV, GeV, invpb, invnb, invfb, nb, pb, fb
from copy import copy, deepcopy

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

def _copy_init_fields(src, dst, memo):
    # Copy only dataclass fields with init=True
    for f in dc.fields(src.__class__):
        if f.init:
            setattr(dst, f.name, deepcopy(getattr(src, f.name), memo))

@dataclass
class CutFlow:
    name: str
    list_bystander: Dict[str, Tuple]
    formular: str = field(init=True, repr = False)
    latex: str = field(default_factory=lambda: r"\mathrm{name}")

    # Runtime attributes (not in __init__)
    sample_init: Any = field(init=False, repr=False)
    sample_final: Any = field(init=False, repr=False)
    dir_bystander_hist_init: Dict[str, Any] = field(init=False, repr=False)
    dir_bystander_hist_final: Dict[str, Any] = field(init=False, repr=False)
    count_init: Any = field(init=False, repr=False)
    count_final: Any = field(init=False, repr=False)

    def __post_init__(self):
        """
        CutFlow 数据类，用于管理 ROOT RDataFrame 分析中的cut流程。
        
        属性:
            name: 选择条件名称 (str)
            list_bystander: 旁观者分支名称与分箱配置的映射字典 (Dict[str, Tuple])
            formular: ROOT RDataFrame 过滤公式 (str)
            latex: 可选的LaTeX格式显示名称 (str)
        
        运行时属性 (不在__init__中初始化):
            sample_init: 初始样本
            sample_final: cut后样本  
            dir_bystander_hist_init: 初始旁观者直方图字典
            dir_bystander_hist_final: cut后旁观者直方图字典
            count_init: 初始计数
            count_final: cut后计数
        
        方法:
            __post_init__: 自动生成默认latex名称（如果未提供）
            copy: 创建CutFlow实例的深拷贝
            apply_on_rdf: 在RDataFrame样本上应用cut流程
            get_latex: 获取LaTeX格式名称
        """    
        if self.latex == r"\mathrm{name}":
            self.latex = fr"\mathrm{{{self.name}}}"

    def __copy__(self,) -> 'CutFlow':
        """Create a deep copy of the CutFlow instance"""
        return CutFlow(
            name=self.name,
            list_bystander=self.list_bystander.copy(),
            formular=self.formular,
            latex=self.latex
        )
    
    def __deepcopy__(self, memo):
        from copy import deepcopy
        return CutFlow(
            name=self.name,
            list_bystander=deepcopy(self.list_bystander, memo),
            formular=self.formular,
            latex=self.latex
        )
    
    def apply_on_rdf(self, sample: Any) -> None:
        """
        Initialize sample processing with RDataFrame.

        Args:
            sample: Input ROOT RDataFrame sample
        """
        self.sample_init = R.RDF.AsRNode(sample)
        self.sample_final = self.sample_init.Filter(self.formular, self.name)
        
        # Initialize 1D/2D histograms
        self.dir_bystander_hist_init = {
            name: (self.sample_init.Histo1D(("", "", *binning), name) 
                   if isinstance(name, str) 
                   else self.sample_init.Histo2D(("", "", *binning), *name))
            for name, binning in self.list_bystander.items()
        }
        
        self.dir_bystander_hist_final = {
            name: (self.sample_final.Histo1D(("", "", *binning), name) 
                   if isinstance(name, str) 
                   else self.sample_final.Histo2D(("", "", *binning), *name))
            for name, binning in self.list_bystander.items()
        }
        
        self.count_init = self.sample_init.Count()
        self.count_final = self.sample_final.Count()

        
    def get_latex(self):
        return self.latex


@dataclass
class RDFStaff(Staff):
    """A class for handling ROOT RDataFrame operations with dataclass support."""

    # process name
    name: str
    # path of root file
    path: str          
    # necessary columns for analysis
    # - this is defined to tolerate the empty event trees.
    # When a file has events originially but event selection wrapped all
    # events, the file does not include an event tree.
    # In this case, RDFStaff fake one RDF, with necessary columns but no entries.
    # then the empty histograms can be created, making sure the process
    # being passed through smoothly.
    necessary_columns: List[str] = field(init=True, default_factory=list, repr=False) 

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
    pre_cut_chain: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    # pre-selection cut tree
    pre_cut_tree: R.RDataFrame = field(init=False, repr=False)
    # cut applied on the sample: CutFlow
    cuts: List[CutFlow] = field(default_factory=list, init=False)
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
        self.set_cuts([])
        # self._column_names = self.rdf.GetColumnNames()




    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)         # don't run __init__/__post_init__
        memo[id(self)] = new

        # 1) copy only init=True fields (plain data)
        _copy_init_fields(self, new, memo)

        # 2) simple non-init data you want to keep
        new.weight = copy(self.weight)

        # 3) share (or shallow-clone) caches; sharing is typical
        new.__REUSE_DF__ = self.__REUSE_DF__

        # 4) rewrap ROOT nodes to the same underlying graphs (no reload)
        # new.rdf = R.RDF.AsRNode(self.rdf) if getattr(self, "rdf", None) is not None else None
        # new.pre_cut_tree = (R.RDF.AsRNode(self.pre_cut_tree)
        #                     if getattr(self, "pre_cut_tree", None) is not None else None)
        new.rdf = self.rdf if getattr(self, "rdf", None) is not None else None
        new.pre_cut_tree = self.pre_cut_tree if getattr(self, "pre_cut_tree", None) is not None else None
        # 5) rebuild pre_cut_chain from the copied pre_cut_tree (fresh RResultPtr)
        new.pre_cut_chain = {}
        # ensure we have names
        if new.pre_cut_tree is not None:
            if new.pre_cut_names is None:
                # copy the discovered columns from the source tree to keep behavior stable
                src_names = list(self.pre_cut_chain.keys()) or [str(c) for c in self.pre_cut_tree.GetColumnNames()]
                new.pre_cut_names = src_names
            for idx, name in enumerate(new.pre_cut_names):
                if idx > 14:
                    break
                new.pre_cut_chain[name] = new.pre_cut_tree.Sum(f"{name}").GetValue()

        # 6) rebuild CutFlow results against the copied rdf
        #    (copy only declarative parts; then apply_on_rdf to create fresh ROOT handles)
        new.cuts = []
        if new.rdf is not None and getattr(self, "cuts", None):
            for cf in self.cuts:
                cf_copy = deepcopy(cf, memo)   # uses the CutFlow deepcopy above
                cf_copy.apply_on_rdf(new.rdf if len(new.cuts) == 0 else new.cuts[-1].sample_final)
                new.cuts.append(cf_copy)

        # 7) optional cosmetic cache
        new._column_names = list(getattr(self, "_column_names", []))

        return new


    def load(self):
        # Create RDataFrame from ROOT file(s)
        if self.path in self.__REUSE_DF__:
            print("?")
            self.rdf = self.__REUSE_DF__[self.path]
        else:

            try: ## Not TTree, maybe RNTuple
                R.gErrorIgnoreLevel = R.kFatal
                self.__REUSE_DF__[self.path] = R.RDF.AsRNode(
                    R.RDataFrame(self.tree_name, self.path)
                )
                self.rdf = self.__REUSE_DF__[self.path]
            except Exception as e:
                try: ## Assuming the data structure is TTree
                    R.gErrorIgnoreLevel = R.kFatal
                    chain = R.TChain(self.tree_name)

                    good_files = []
                    for f in glob.glob(os.path.expanduser(self.path)):
                        tf = R.TFile.Open(f)
                        print(self.tree_name,  self.tree_name in tf.GetListOfKeys())
                        if not tf or tf.IsZombie():
                            continue
                        if self.tree_name in tf.GetListOfKeys():
                            good_files.append(f)
                        tf.Close()

                    if len(good_files) == 0:
                        print(f"Warning: Tree '{self.tree_name}' not found or is empty in '{self.path}'. Creating a fake RDataFrame.")
                        self.__REUSE_DF__[self.path] = self._create_fake_rdf()
                        self.rdf = self.__REUSE_DF__[self.path]

                    for f in good_files:
                        chain.Add(f)

                    if self.range is None:
                        self.__REUSE_DF__[self.path] = R.RDF.AsRNode(
                            R.RDataFrame(chain.Clone()))
                    else:
                        self.__REUSE_DF__[self.path] = R.RDF.AsRNode(
                            R.RDataFrame(chain.Clone())).Range(self.range)

                    self.rdf = self.__REUSE_DF__[self.path]
                    self.rdf.GetColumnNames()
                except Exception as e2:
                        # If tree or evt is not found, create a fake RDataFrame
                        print(e)
                        print(f"Warning: Could not find tree '{self.tree_name}' in file '{self.path}'. Creating fake RDataFrame.")
                        self.__REUSE_DF__[self.path] = self._create_fake_rdf()
                        self.rdf = self.__REUSE_DF__[self.path]
            # root_file.Close()

            # else:
            #     print(f"Warning: Tree '{self.tree_name}' has no entries. Creating a fake RDataFrame.")
            #     self.__REUSE_DF__[self.path] = self._create_fake_rdf()
            #     self.rdf = self.__REUSE_DF__[self.path]


    def _create_fake_rdf(self):
        """
        Create a fake RDataFrame when the original tree/evt cannot be found.
        This allows the package to run normally and return empty histograms.
        """
        # Create a fake RDataFrame with a dummy column to avoid errors when calling Histo1D/Histo2D
        fake_rdf = R.RDataFrame(1)  # DataFrame with 1 entry
        fake_rdf = fake_rdf.Define("fake_var", "-1")  # Add a dummy column
        for column in self.necessary_columns:
            if "[]" in column:
                fake_rdf = fake_rdf.Define(column.replace("[]",""), "ROOT::VecOps::RVec<double>({})")  # Add a dummy column
            else:
                fake_rdf = fake_rdf.Define(column, "-1.e99")  # Add a dummy column
        # Filter out all entries to make it effectively empty
        fake_rdf = fake_rdf.Filter("fake_var > 0")  # This will result in 0 entries
        return R.RDF.AsRNode(fake_rdf)

    def save(self, tree_name: str, path: str, var: list[str], 
             truth_classification_count: float | None = None):
    
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

        # 重新 apply cuts，确保新定义的变量传递到 sample final
        self.set_cuts(self.cuts)

        opts_cut = R.RDF.RSnapshotOptions()
        opts_cut.fLazy = False
        opts_cut.fMode = "Update"
        opts_cut.fOverwriteIfExists = True

        self.cuts[-1].sample_final.Snapshot(tree_name, path, var, opts_cut)

        # =============================================================================
        # CHANGELOG
        # -----------------------------------------------------------------------------
        # Date      : 2026-01-27 (Asia/Singapore)
        # Author    : Coo
        # Module    : RDFStaff.save
        # Summary   : Persist truth classification effective count into cut tree.
        #
        # Motivation:
        #   - When MC is truth-split into components, the effective base
        #     statistics is the post-classification count, not file-level N0.
        #   - Persisting "TruthClassification" makes downstream QA reproducible and
        #     consistent with RDFFactory.get_cut_chain_table() and get_weights().
        #
        # Compatibility:
        #   - New argument is optional; existing calls remain valid.
        # =============================================================================

        # save cut chain
        cut_df = R.RDataFrame(1)
        cut_df = cut_df.Define("N0", f"{self.pre_cut_chain['N0']}")

        # NEW: truth classification effective count (constant column)
        if truth_classification_count is not None:
            cut_df = cut_df.Define("TruthClassification", f"{float(truth_classification_count)}")

        for idx, name in enumerate(self.pre_cut_chain.keys()):
            if idx > 0:
                cut_df = cut_df.Define(f"N{idx}", f"{self.pre_cut_chain[name]}")

        for cut in self.cuts:
            cut_df = cut_df.Define(cut.name, f"{cut.count_final.GetValue()}")

        # save histograms of spectators at all levels of cuts.
        cut_df.Snapshot(f"cut_{tree_name}", path,
                        cut_df.GetColumnNames(), opts_cut)
        with R.TFile(path, "update") as outfile:
            for cut in self.cuts:
                for bystander_name in cut.list_bystander:
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
        
        self.pre_cut_chain = {}
        # def make_rdf_or_empty(path, treename, n0):
        #     import ROOT as R, os
        #     if not os.path.exists(path):
        #         return R.RDataFrame(1).Define("N0", str(n0))

        #     f = R.TFile.Open(path)
        #     if not f or f.IsZombie():
        #         return R.RDataFrame(1).Define("N0", str(n0))

        #     obj = f.Get(treename)
        #     if obj and obj.InheritsFrom("TTree"):
        #         return R.RDataFrame(obj)

        #     return R.RDataFrame(1).Define("N0", str(n0))
        
        # self.pre_cut_tree = make_rdf_or_empty(
        #     self.path, 
        #     self.pre_cut_tree_name, 
        #     n0 = self.rdf.Count().GetValue()
        # )
        
        # if self.pre_cut_names is None:
        #     self.pre_cut_names = [str(i) for i in self.pre_cut_tree.GetColumnNames()]
        #     for idx, name in enumerate(self.pre_cut_names):
        #         if idx > 14: # Generally, the number of cut layers for the files output by BOSS will not be greater than 7.
        #             break
        #         else:
        #             temp = self.pre_cut_tree.Sum(f"{name}")
        #             if hasattr(temp, "GetValue"):
        #                 self.pre_cut_chain[name] = temp.GetValue()
        #             else:
        #                 self.pre_cut_chain[name] = temp
        if self.pre_cut_names is not None:
            self.pre_cut_tree = R.RDF.AsRNode(
                R.RDataFrame(
                    self.pre_cut_tree_name,
                    self.path
                ))
            for idx, name in enumerate(self.pre_cut_names):
                if idx > 14: # Generally, the number of cut layers for the files output by BOSS will not be greater than 7.
                    break
                else:
                    temp = self.pre_cut_tree.Sum(f"{name}")
                    if hasattr(temp, "GetValue"):
                        self.pre_cut_chain[name] = temp.GetValue()
                    else:
                        self.pre_cut_chain[name] = temp
        else:
            self.pre_cut_tree = R.RDataFrame(1)
            if self.pre_cut_names is None:
                self.pre_cut_names = ["N0"]
                self.pre_cut_chain["N0"] = self.rdf.Count().GetValue()
                self.pre_cut_tree = self.pre_cut_tree.Define("N0", f"{self.pre_cut_chain['N0']}")

            
                
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

            if len(self.cuts) == 0:
                init_cut = CutFlow(name = "Init", list_bystander={},
                            formular = "true", latex = r"\text{Init}")
                self.cuts = [init_cut]
                
            iter_rdf = R.RDF.AsRNode(self.rdf)
            for cut in self.cuts:
                cut.apply_on_rdf(iter_rdf)
                iter_rdf = R.RDF.AsRNode(cut.sample_final)
        else:
            print("set_cuts(): There is no RDF.")

    def append_cuts(self, cuts: CutFlow | List[CutFlow]):
        """
        Append a cut flow or a list of cut flows.

        Parameters
        ----------
        cuts : list of functions
            List of functions to apply as filters to the RDataFrame.
        """
        import copy
        for key, staff in self.staff_dict.items():
            head = self._normalize_classify(self.classify_dict.get(key, []))
            staff.set_cuts(head + self.cuts)
        
        
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
        cut_counts = [i for i in self.pre_cut_chain.values()] + [elem.count_final.GetValue() for elem in self.cuts]
        res = pd.Series(dict(zip(cut_chain_names, cut_counts)), name = "$" + self.name + "$")
        return res

    def empty(self):
        if self.rdf is None:
            return True
        elif len(self.cuts) > 0:
            if self.cuts[-1].sample_final.Count().GetValue() == 0:
                return True
            else:
                return False
        elif list(self.pre_cut_chain.values())[-1] == 0:
            return True
        # elif self.rdf.Count().GetValue() == 0:
        #     return True
        else:
            return False
    
    def define(self, branch_name: str, func_str: str):
        if branch_name in self.rdf.GetColumnNames():
            self.rdf = self.rdf.Redefine(branch_name, func_str)
            if (len(self.cuts) > 0):
                self.cuts[-1].sample_final = self.cuts[-1].sample_final.Redefine(branch_name, func_str)
        else: 
            self.rdf = self.rdf.Define(branch_name, func_str)
            if (len(self.cuts) > 0):
                self.cuts[-1].sample_final = self.cuts[-1].sample_final.Define(branch_name, func_str)
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
        classify_dict (Dict[str, List[CutFlow]]): Truth-level selections to divide samples
        pre_cut_names (Optional[List[str]]): Names of preselection cuts
        range (Optional[Any]): Range restriction for testing
    """
    path_dict: Dict[str, str]
    xsec_dict: Dict[str, float]
    luminosity: float
    
    # Configuration options with defaults
    type_dict: Dict[str, StaffType] = field(default_factory=dict)
    tree_name: str = "evt"
    pre_cut_tree_name: str = "cut"
    cuts: List[str] = field(default_factory=list)
    classify_dict: Dict[str, List[CutFlow]] = field(default_factory=dict)
    pre_cut_names: Optional[List[str]] = None
    range: Optional[Any] = None
    necessary_columns: List[str] = field(init=True, default_factory=list, repr=False) 

    
    # Internal state
    staff_dict: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Validate inputs and load samples after initialization"""
        self._validate_inputs()
        self.load()
        self.set_cuts([])

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
            # 即使 RDFStaff.load 中做了相关的保护，这段保护代码仍然对那些连cut 
            # chain 都没有的文件起作用。
            # 未来需要有一个自检功能，提前检查所有文件的格式，显式给出warning
            #evt_chain = R.TChain(self.tree_name, "Read")
            #evt_chain.Add(value)
            #cut_chain = R.TChain(self.pre_cut_tree_name, "Read")
            #cut_chain.Add(value)
            #if (evt_chain.GetEntries() <= 0) and (cut_chain.GetEntries() <= 0 ):
            #    continue
            #else:
            self.staff_dict[key] = RDFStaff(path = value,
                                            name = key,
                                            xsec = self.xsec_dict[key], 
                                            tree_name = self.tree_name,
                                            pre_cut_tree_name = self.pre_cut_tree_name,
                                            pre_cut_names = self.pre_cut_names,
                                            range = self.range,
                                            type = self.type_dict.get(key, StaffType.other),
                                            necessary_columns = self.necessary_columns)
            
        # Divide sample into components basing on a set of cuts
        #for key, value in self.classify_dict.items():
        #    print(value)
        #    if (len(value) != 0):
        #        self.staff_dict[key].rdf = self.staff_dict[
        #            key].rdf.Filter(value, "preprocess: ")

    def define(self, branch_name: str, formula_str: str):
        """
        Define a new branch for all RDataFrame
        """
        for key, value in self.staff_dict.items():
            if value.empty():
                continue
            value.define(branch_name, formula_str)

    def set_cuts(self, cuts: List[CutFlow]):
        """
        Register the event selection cuts to each sample.

        Parameters:
        -----------
        cuts: (list)
            A list of event selection cuts.
        """
        self.cuts = list(cuts)
        
        init_cut = CutFlow(name = "Init", list_bystander={},
                           formular = "true", latex = r"\text{Init}")
        if len(cuts) + len(self.classify_dict) == 0:
            self.cuts = [init_cut]

        for key, value in self.staff_dict.items():
            head = self._normalize_classify(self.classify_dict.get(key, []))
            value.set_cuts(head + self.cuts)
            # value.set_cuts( self.classify_dict.get(key, []) + self.cuts)

    def _normalize_cuts(self, cuts) -> List[CutFlow]:
        """Normalize analysis cuts to list[CutFlow]."""
        if cuts is None:
            return []
        if isinstance(cuts, CutFlow):
            return [cuts]
        if isinstance(cuts, list):
            return cuts
        return []

    def append_cuts(self, cuts: CutFlow | List[CutFlow]) -> None:
        """
        Staff-local: append cuts to current cut list, then rebuild.
        In factory-managed workflows, prefer RDFFactory.append_cuts().
        """
        import copy
        if self.rdf is None:
            return

        if cuts is None:
            tail = []
        elif isinstance(cuts, list):
            tail = [copy.copy(c) for c in cuts]
        else:
            tail = [copy.copy(cuts)]

        self.set_cuts(list(self.cuts) + tail)

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
        self.set_cuts(self.cuts)
        hists_dict = {}
        for i in self.staff_dict.keys():
            if self.staff_dict[i].empty():
                continue
            hists_dict[i] = self.staff_dict[i].get_histstaff(func) 

        res = HistFactory(staff_dict=hists_dict, type_dict=self.type_dict)
        return res
    
    
    def get_weights(self, virtual_xsec=None):
        """
        Weight definition:
          w = L * xsec / N_effective

        N_effective policy:
          - If classify_dict defines truth-split for this sample, use the
            post-classification count as N_effective.
          - Otherwise fallback to N0.

        This makes component normalization consistent when a single file is split
        into truth-defined components with different xsec (e.g. BR-weighted).
        """
        if virtual_xsec is None:
            virtual_xsec = self.xsec_dict

        weights = {}

        for key, staff in self.staff_dict.items():
            # fallback if staff is empty/uninitialized
            if staff.empty():
                weights[key] = 1.0
                continue

            denom = self._truth_count_after_classify(staff, key)

            # additional fallback safety
            if denom <= 0:
                denom = staff.pre_cut_chain.get("N0", 0.0)
            if denom <= 0:
                denom = 1.0

            weights[key] = float(self.luminosity) * float(virtual_xsec[key]) / float(denom)

        return weights


    def get_cut_chain_table(self, weight=None):
        """
        从每个 RDFStaff 中获得 Cut Chain 的 pandas.Series，归一化到 weight 之后生成pandas.DataFrame

        行为说明：
        - truth classification（classify_dict）不再逐个展开成普通 cut 列；
        - 折叠为单列 "Truth classification"，紧跟在 N0 后面；
        - 后续 analysis cuts 正常展开。
        """
        import pandas as pd
        if weight is None:
            weight = self.get_weights()

        rows = []
        for staff in self.staff_dict.values():
            key = staff.name
            cls_names = set(self._classify_names_for(key))

            od = {}

            # 1) pre-cut chain first (keeps N0 semantics intact: file-level initial)
            for n, v in staff.pre_cut_chain.items():
                od[n] = float(v)

            # 2) collapsed truth classification column
            if cls_names:
                od["TruthClassification"] = self._truth_count_after_classify(staff, key)

            # 3) append post cuts, excluding classify cuts
            for c in getattr(staff, "cuts", []):
                if c.name in cls_names:
                    continue
                od[c.name] = float(c.count_final.GetValue())
                
            rows.append(pd.Series(od, name="$" + key + "$") * weight.get(key, 1.0))

        df = pd.DataFrame(rows).fillna(0.0)
        df.loc["Sum", :] = df.sum()
        return df

    def save(self, tree_name: str, path_dir: Dict[str, str], var: List[str]):
        """
        Save selected events and cutflow metadata.

        Notes:
          - Pass truth-classification effective count to staff.save so the persisted
            cut tree contains "TruthClassification" consistent with weights/table.
        """
        for key, staff in self.staff_dict.items():
            print(f"Writing file: {path_dir[key]}")
            truth_n = self._truth_count_after_classify(staff, key)
            staff.save(tree_name, path_dir[key], var, truth_classification_count=truth_n)


    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        # init fields
        new.path_dict = deepcopy(self.path_dict, memo)
        new.xsec_dict = deepcopy(self.xsec_dict, memo)
        new.luminosity = self.luminosity

        new.type_dict = deepcopy(self.type_dict, memo)
        new.tree_name = self.tree_name
        new.pre_cut_tree_name = self.pre_cut_tree_name
        new.classify_dict = deepcopy(self.classify_dict, memo)
        new.pre_cut_names = deepcopy(self.pre_cut_names, memo)
        new.range = deepcopy(self.range, memo)
        new.necessary_columns = deepcopy(self.necessary_columns, memo)

        # staff_dict is rebuilt from existing objects via RDFStaff.__deepcopy__
        new.staff_dict = {k: deepcopy(v, memo) for k, v in self.staff_dict.items()}

        # reapply current cuts declaratively (this rebuilds ROOT results on copied RDFs)
        new.cuts = deepcopy(self.cuts, memo)
        for k, staff in new.staff_dict.items():
            head = self.classify_dict.get(k, [])
            if isinstance(head, CutFlow):  # normalize
                head = [head]
            staff.set_cuts(head + new.cuts)

        return new

    def pop(self, key: str):
        """
        从工厂中移除并返回名为 `key` 的 RDFStaff。
        
        参数
        ----
        key : str
            要剔除的 staff 名称（通常就是样本名）
        返回
        ----
        RDFStaff
            被移除的 RDFStaff；若 key 不存在且提供了 default，则返回 default
        """
        if key in self.staff_dict:
            removed = self.staff_dict.pop(key)
            self.type_dict.pop(key, None)
            self.xsec_dict.pop(key, None)
            return removed
        else:
            return None 
    
    def append(self, staff: RDFStaff) -> bool:
        """
        Append a single RDFStaff into this factory.

        - Ensures the incoming staff is an RDFStaff and its name is unused.
        - Synchronizes factory-level configs (tree names, pre-cut names, range, necessary columns).
        - Makes sure RDF/pre-cut tree are loaded (in case the staff was constructed without __post_init__).
        - Updates path/xsec/type dictionaries so weights & saving continue to work.
        - Applies classify cuts (if any) followed by the factory's current cuts.
        """
        # basic checks
        if not isinstance(staff, RDFStaff):
            return False
        key = staff.name
        if key in self.staff_dict:
            # don't overwrite existing
            return False

        # sync config from factory -> staff (keeps behavior consistent)
        staff.pre_cut_names = self.pre_cut_names
        staff.range = self.range
        staff.necessary_columns = list(self.necessary_columns)  # copy for safety

        # ensure the staff is ready (if constructed bypassing __post_init__)
        if getattr(staff, "rdf", None) is None:
            staff.load()
            staff.pre_selection()

        # register into the factory
        self.staff_dict[key] = staff
        # keep these dicts in sync so get_weights()/save() keep working
        self.path_dict[key] = staff.path
        self.xsec_dict[key] = float(staff.xsec) if staff.xsec is not None else 1.0
        self.type_dict[key] = staff.type

        # apply classify head (if any) + the factory's current cuts
        head = self.classify_dict.get(key, [])
        if isinstance(head, CutFlow):
            head = [head]
        staff.set_cuts(head + self.cuts)

        return True


    # =============================================================================
    # CHANGELOG
    # -----------------------------------------------------------------------------
    # Date      : 2026-01-27 (Asia/Singapore)
    # Author    : Coo
    # Module    : RDFFactory
    # Summary   : Treat truth classification as "pre-selection semantics" for
    #             reporting and normalization.
    #
    # Motivation:
    #   - classify_dict is used to split a shared ROOT file into mutually-exclusive
    #     MC components using truth. It should NOT appear as regular analysis cut
    #     steps in cutflow tables, and weights must be normalized by the post-
    #     classification effective statistics.
    #
    # Changes:
    #   1) Add helpers to resolve classify cut names and obtain post-classification
    #      count (Count_final after the last classify CutFlow).
    #   2) get_cut_chain_table(): collapse classify cut(s) into a single column
    #      "Truth classification" right after N0.
    #   3) get_weights(): use post-classification count as denominator by default.
    #   4) save(): pass truth-classification count into RDFStaff.save so the cut tree
    #      persisted on disk is consistent with the table/weights semantics.
    #
    # Compatibility:
    #   - Does not change the execution graph (RDataFrame filtering order is the same).
    #   - Only changes *reporting* and *normalization* semantics.
    # =============================================================================

    def _normalize_classify(self, cs) -> list:
        """Normalize classify_dict entry to a list[CutFlow]."""
        if cs is None:
            return []
        if isinstance(cs, CutFlow):
            return [cs]
        if isinstance(cs, list):
            return cs
        # fallback: unknown type
        return []

    def _classify_names_for(self, key: str) -> list[str]:
        """Return list of CutFlow.name used for truth classification for this sample."""
        cs = self._normalize_classify(self.classify_dict.get(key, []))
        return [c.name for c in cs if hasattr(c, "name")]

    def _truth_count_after_classify(self, staff: RDFStaff, key: str) -> float:
        """
        Count AFTER truth classification (i.e. after the last classify CutFlow).
        If no classify defined / not found in staff.cuts, fallback to N0.

        在分类后的真实计数（即在最后一个分类CutFlow之后）。
        如果没有定义分类或在staff.cuts中找不到，则回退到N0计数。
        
        Args:
            staff (RDFStaff): 包含RDF数据和切割信息的对象
            key (str): 用于识别分类的名称键
        
        Returns:
            float: 分类后的计数结果，如果找不到分类则返回N0或总计数
        
        Note:
            如果找不到指定的分类名称，会尝试回退到预切割链中的N0计数，
            如果N0也不存在，则返回RDF的总计数
        """
        cls_names = self._classify_names_for(key)
        if not cls_names:
            # fallback to N0 if available, else rdf.Count()
            if getattr(staff, "pre_cut_chain", None) and "N0" in staff.pre_cut_chain:
                return float(staff.pre_cut_chain["N0"])
            return float(staff.rdf.Count().GetValue())

        idxs = [i for i, c in enumerate(getattr(staff, "cuts", [])) if c.name in cls_names]
        if not idxs:
            if getattr(staff, "pre_cut_chain", None) and "N0" in staff.pre_cut_chain:
                return float(staff.pre_cut_chain["N0"])
            return float(staff.rdf.Count().GetValue())

        return float(staff.cuts[max(idxs)].count_final.GetValue())
