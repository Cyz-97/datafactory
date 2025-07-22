# datafactory: a simple framework handling baisc HEP data-MC analysis

本项目是一个 Python 工具集，针对一套数据和多个蒙卡样本的联合分析。
基于`ROOT::RDataFrame`提供事例筛选、一维-二维直方图统计分析和可视化相关的功能模块，
主要包括：

- 核心功能（core.py）：基础工具类与通用方法
- 统计分析（statistic.py）：数据统计与分析方法
- 直方图工具（hist.py）：统计分布与直方图绘制
- 绘图功能（plot.py）：通用数据可视化工具
- RDF 处理（rdf.py）：资源描述框架（RDF）数据解析或操作
- 样式配置（style.mplstyle）：Matplotlib 图表风格定制

MC 样本被描述为`RDF`和`Histogram`的字典，根据过程截面、数据亮度与数据对齐。

```mermaid
classDiagram
    direction TB

    subgraph Core Components
        direction LR
        Factory o-- "*" Staff: contains
        Staff ..> StaffType: has a
        class Staff {
            <<Abstract>>
            # name: string
            +load(source)*
            +save(destination)*
        }

        class Factory {
            <<Abstract>>
            # staff_collection: dict~string, Staff~
            +add_staff(staff: Staff)
            +get_staff(name: string): Staff
            +__getitem__(key: str): Staff
        }
    end

    subgraph RDataFrame Implementation
        direction TB
        RDFStaff --|> Staff
        RDFFactory --|> Factory
        RDFFactory o-- "*" RDFStaff: contains
        RDFStaff <..> CutFlow : symbiosis

        class RDFStaff {
            +weight: float
            +rdf: RDataFrame
            +define(...)
            +pre_selection()
            +empty()
            +set_cut(cut: list[CutFlow])
            +get_rdf(): RDataFrame
            +get_hist(func: function)
            +get_cut_chain()
            +save(tree_name:str, path:str, var:list[str])
        }

        class RDFFactory {
            +luminosity: float
            +xsec: dict~name: float~
            +weight: dict~name: float~
            -staff_list: dict~name: RDFStaff~
            +apply_cuts_all(...)
            +get_hist(func: function)
            +define(name: str, func: function)
            +set_cut(cut: list[CutFlow])
            +get_weights(xsec: dict~name:float~)
            +get_cut_chain(weight: dict~name:float~)
            +save(tree_name: str, path: dict~name:path~, var: list[str])
        }
    end

    subgraph Histogram Implementation
        direction TB
        HistStaff --|> Staff
        HistFactory --|> Factory
        HistFactory o-- "*" HistStaff: contains

        class HistStaff {
            +histogram: THn
            +type: StaffType
            +get_numpy(style: str): tuple/THn/hist
            +__add__(other: HistStaff): plus
            +__sub__(other: HistStaff): minus
            +__mul__(other: HistStaff): multiply
            +__div__(other: HistStaff): divide
        }

        class HistFactory {
            +sum(type=[signal, background]: list[StaffType]): HistStaff
            +get_hist(): tuple()
            +get_type(name: list[str]): StaffType
            +get_chi2(data: HistStaff, weight: dict~name: float~): float
        }
    end

    subgraph Tools
        direction TB
        class Fitter {
            +template_fit(data_hist: HistStaff, mc_hists: dict): RooFitResult
        }
        class Plotter {
            +plot_data_mc(data_hist: HistStaff, mc_hists: dict, info: DataInfo)
        }
        Fitter ..> HistFactory : uses
        Plotter ..> HistFactory : uses
        Plotter ..> DataInfo : uses
    end


    subgraph Helpers
        direction TB
        class StaffType {
            <<enumeration>>
            SIGNAL
            BACKGROUND
            DATA
        }
        class CutFlow {
          +name: string
          +formula: string
          +speactator: dict~name:hist_model~
          +apply_on_rdf(rdf: RDFStaff)
          +get_latex()           
        }
        class DataInfo {
          +cms_en_MeV: str
          +cms_en: float     # MeV
          +luminosity: float # invpb
          +xsec: dict~name: xsec~ # nb
          +get_latex()
        }
    end
```