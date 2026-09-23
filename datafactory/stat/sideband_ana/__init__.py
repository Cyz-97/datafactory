"""sideband subtraction 分析包。

分阶段实现（见 docs/sideband_ana_architecture.md）：

- 区域契约与 transfer factor / coefficients（纯数学，只依赖 numpy）；
- SymbolFit 本底选择 + zfit 质量谱 / 质量平面拟合；
- sideband subtraction 的容斥式实现；
- 拟合与 transfer 报告输出。

公共 API 从本文件 re-export，内部模块名可能调整。
"""

from .transfer import (
    MassRegions1D,
    TransferFactor1D,
    TransferFactors2D,
    calculate_transfer_factor_1d,
    calculate_transfer_factors_2d,
    regions_from_offsets,
)
from .subtract import (
    SubtractionResult,
    subtract_sideband_1d,
    subtract_sideband_2d,
)

__all__ = [
    "MassRegions1D",
    "regions_from_offsets",
    "TransferFactor1D",
    "TransferFactors2D",
    "calculate_transfer_factor_1d",
    "calculate_transfer_factors_2d",
    "SubtractionResult",
    "subtract_sideband_1d",
    "subtract_sideband_2d",
]

# 拟合与报告依赖 TensorFlow/zfit/SymbolFit/matplotlib，按需导入，
# 不在包级别强制引入。
_LAZY_EXPORTS = {
    "fit_mass_spectrum_1d": (".fit", "fit_mass_spectrum_1d"),
    "fit_mass_plane_2d": (".fit", "fit_mass_plane_2d"),
    "FitResult1D": (".fit", "FitResult1D"),
    "FitResult2D": (".fit", "FitResult2D"),
    "BackgroundModel": (".fit", "BackgroundModel"),
    "NormalizedDensity1D": (".fit", "NormalizedDensity1D"),
    "ComponentModel2D": (".fit", "ComponentModel2D"),
    "write_fit_report_1d": (".report", "write_fit_report_1d"),
    "write_fit_report_2d": (".report", "write_fit_report_2d"),
    "write_transfer_summary": (".report", "write_transfer_summary"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attribute = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attribute)
    raise AttributeError(f"module {__name__!r} 没有属性 {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY_EXPORTS))
