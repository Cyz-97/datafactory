"""Physics-facing API for fitted sideband subtraction analyses."""

from .fit import FitResult1D, FitResult2D, fit_mass_plane_2d, fit_mass_spectrum_1d
from .report import ReportArtifacts, write_fit_report_1d, write_fit_report_2d, write_transfer_summary
from .subtract import SubtractionResult, subtract_sideband_1d, subtract_sideband_2d
from .transfer import (
    MassRegions1D,
    TransferFactor1D,
    TransferFactors2D,
    calculate_transfer_factor_1d,
    calculate_transfer_factors_2d,
    regions_from_offsets,
)

__all__ = [
    "FitResult1D",
    "FitResult2D",
    "MassRegions1D",
    "ReportArtifacts",
    "SubtractionResult",
    "TransferFactor1D",
    "TransferFactors2D",
    "calculate_transfer_factor_1d",
    "calculate_transfer_factors_2d",
    "fit_mass_plane_2d",
    "fit_mass_spectrum_1d",
    "regions_from_offsets",
    "subtract_sideband_1d",
    "subtract_sideband_2d",
    "write_fit_report_1d",
    "write_fit_report_2d",
    "write_transfer_summary",
]
