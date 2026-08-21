"""DataFactory public API with ROOT-dependent classes loaded on first use."""

from importlib import import_module

from .core import DataInfo, Factory, Staff, StaffType

__all__ = [
    "Staff", "Factory", "StaffType", "DataInfo",
    "HistStaff", "HistFactory", "UnbinnedStaff", "UnbinnedFactory",
    "RDFStaff", "RDFFactory", "CutFlow",
]

_ROOT_EXPORTS = {
    "HistStaff": (".hist", "HistStaff"),
    "HistFactory": (".hist", "HistFactory"),
    "UnbinnedStaff": (".hist", "UnbinnedStaff"),
    "UnbinnedFactory": (".hist", "UnbinnedFactory"),
    "RDFStaff": (".rdf", "RDFStaff"),
    "RDFFactory": (".rdf", "RDFFactory"),
    "CutFlow": (".rdf", "CutFlow"),
}


def __getattr__(name):
    """Preserve existing top-level imports without loading PyROOT eagerly."""
    if name not in _ROOT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _ROOT_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
