#!/usr/bin/env python3
"""Extract the small sideband_ana fixture from the published 09 ROOT product.

The extraction is mechanical: it copies bin edges, bin contents, and
``GetBinError()**2`` from real ``TH1D``/``TH2D`` objects.  No events are
regenerated, rebinned, rounded, or replaced by toy values.  Run this script in
the ``root6.34`` environment when the upstream product is refreshed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ROOT as R


def histogram_1d(root_file, object_name):
    """Return ``(edges, counts, variances)`` from one ROOT ``TH1`` object."""

    histogram = root_file.Get(object_name)
    if not histogram or not histogram.InheritsFrom("TH1"):
        raise RuntimeError(f"missing TH1 source object: {object_name}")
    n_bins = histogram.GetNbinsX()
    edges = np.array(
        [histogram.GetBinLowEdge(index) for index in range(1, n_bins + 1)]
        + [histogram.GetXaxis().GetBinUpEdge(n_bins)],
        dtype=np.float64,
    )
    counts = np.array(
        [histogram.GetBinContent(index) for index in range(1, n_bins + 1)],
        dtype=np.float64,
    )
    errors = np.array(
        [histogram.GetBinError(index) for index in range(1, n_bins + 1)],
        dtype=np.float64,
    )
    return edges, counts, np.square(errors)


def validate_arrays(array_map):
    """Check the serialized fixture contract before and after writing NPZ."""

    variance_names = {name for name in array_map if name.endswith("_variances")}
    edge_names = {name for name in array_map if name.endswith("_edges")}
    for name, values in array_map.items():
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {name}")
        if name in variance_names and np.any(values < 0.0):
            raise ValueError(f"negative variance in {name}")
    for name in edge_names:
        if array_map[name].size < 2 or np.any(np.diff(array_map[name]) <= 0.0):
            raise ValueError(f"non-increasing edges in {name}")
    for counts_name, variances_name in (
        ("mass_fit_counts", "mass_fit_variances"),
        ("mass_plane_counts", "mass_plane_variances"),
        ("mass_region_counts", "mass_region_variances"),
        ("delta_phi_thrust_counts", "delta_phi_thrust_variances"),
    ):
        if array_map[counts_name].shape != array_map[variances_name].shape:
            raise ValueError(f"shape mismatch: {counts_name}/{variances_name}")
    if array_map["mass_region_counts"].shape != (4, 30):
        raise ValueError("mass regions must have fixed shape (4, 30)")
    if array_map["delta_phi_thrust_counts"].shape != (4, 10):
        raise ValueError("delta-phi regions must have fixed shape (4, 10)")
    if array_map["mass_plane_counts"].shape != (100, 100):
        raise ValueError("mass plane must have fixed shape (100, 100)")


# %% -- command-line inputs and source metadata
R.gROOT.SetBatch(True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-product", type=Path, required=True)
parser.add_argument("--source-metadata", type=Path, default=None)
parser.add_argument(
    "--output-dir", type=Path, default=Path(__file__).resolve().parent
)
args = parser.parse_args()
source_product = args.source_product.expanduser().resolve()
source_metadata_path = (
    args.source_metadata.expanduser().resolve()
    if args.source_metadata is not None
    else source_product.with_name("pair_jet_distribution_metadata.json")
)
if not source_product.is_file():
    raise FileNotFoundError(source_product)
if not source_metadata_path.is_file():
    raise FileNotFoundError(source_metadata_path)
args.output_dir.mkdir(parents=True, exist_ok=True)
source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))

# %% -- ROOT extraction (objects are copied without rebinning)
root_file = R.TFile.Open(str(source_product), "READ")
if not root_file or root_file.IsZombie():
    raise RuntimeError(f"unable to open ROOT source: {source_product}")
try:
    fit_edges, fit_counts, fit_variances = histogram_1d(
        root_file, "mass_fit_ll_data"
    )
    plane_histogram = root_file.Get("mass_plane_ll_data_cat0_llbar")
    if not plane_histogram or not plane_histogram.InheritsFrom("TH2"):
        raise RuntimeError("missing TH2 source object: mass_plane_ll_data_cat0_llbar")
    nx, ny = plane_histogram.GetNbinsX(), plane_histogram.GetNbinsY()
    plane_x_edges = np.array(
        [plane_histogram.GetXaxis().GetBinLowEdge(index) for index in range(1, nx + 1)]
        + [plane_histogram.GetXaxis().GetBinUpEdge(nx)],
        dtype=np.float64,
    )
    plane_y_edges = np.array(
        [plane_histogram.GetYaxis().GetBinLowEdge(index) for index in range(1, ny + 1)]
        + [plane_histogram.GetYaxis().GetBinUpEdge(ny)],
        dtype=np.float64,
    )
    plane_counts = np.empty((nx, ny), dtype=np.float64)
    plane_errors = np.empty_like(plane_counts)
    for ix in range(1, nx + 1):
        for iy in range(1, ny + 1):
            plane_counts[ix - 1, iy - 1] = plane_histogram.GetBinContent(ix, iy)
            plane_errors[ix - 1, iy - 1] = plane_histogram.GetBinError(ix, iy)
    plane_variances = np.square(plane_errors)
    mass_region_arrays = [
        histogram_1d(
            root_file,
            f"regional_ll_data_ch1_cat0_reg{region}_mass_5gev",
        )
        for region in range(4)
    ]
    delta_phi_arrays = [
        histogram_1d(
            root_file,
            f"regional_ll_data_ch1_cat0_reg{region}_delta_phi_thrust",
        )
        for region in range(4)
    ]
finally:
    root_file.Close()

# %% -- assemble fixed arrays and validate source content
mass_region_edges = mass_region_arrays[0][0]
delta_phi_edges = delta_phi_arrays[0][0]
if not all(np.array_equal(mass_region_edges, values[0]) for values in mass_region_arrays):
    raise RuntimeError("mass region objects do not share bin edges")
if not all(np.array_equal(delta_phi_edges, values[0]) for values in delta_phi_arrays):
    raise RuntimeError("delta-phi region objects do not share bin edges")

arrays = {
    "mass_fit_edges": fit_edges,
    "mass_fit_counts": fit_counts,
    "mass_fit_variances": fit_variances,
    "mass_plane_x_edges": plane_x_edges,
    "mass_plane_y_edges": plane_y_edges,
    "mass_plane_counts": plane_counts,
    "mass_plane_variances": plane_variances,
    "mass_region_edges": mass_region_edges,
    "mass_region_counts": np.stack([values[1] for values in mass_region_arrays]),
    "mass_region_variances": np.stack([values[2] for values in mass_region_arrays]),
    "delta_phi_thrust_edges": delta_phi_edges,
    "delta_phi_thrust_counts": np.stack([values[1] for values in delta_phi_arrays]),
    "delta_phi_thrust_variances": np.stack([values[2] for values in delta_phi_arrays]),
}
validate_arrays(arrays)

# %% -- write NPZ and verify the serialized fixture
output_npz = args.output_dir / "sideband_ll_data_cat0_llbar.npz"
np.savez_compressed(output_npz, **arrays)
with np.load(output_npz, allow_pickle=False) as serialized:
    serialized_arrays = {name: serialized[name] for name in serialized.files}
    if set(serialized_arrays) != set(arrays):
        raise ValueError("serialized NPZ fields differ from source fields")
    validate_arrays(serialized_arrays)
    for name, values in arrays.items():
        if serialized_arrays[name].shape != values.shape:
            raise ValueError(f"serialized NPZ shape differs for {name}")

# %% -- write self-describing metadata
definitions = source_metadata["definitions"]
mass_fit = source_metadata["mass_fits"]["ll_data"]
event_counts = {
    key: value for key, value in source_metadata["event_counts"].items()
    if key.startswith("data_")
}
source_digest = hashlib.sha256()
with source_product.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        source_digest.update(block)
generated_utc = datetime.now(timezone.utc).isoformat()
array_meanings = {
    "mass_fit_edges": "single-Lambda mass fit bin edges [GeV]",
    "mass_fit_counts": "all-event LL data Lambda-mass fit counts",
    "mass_fit_variances": "Sumw2 variances for mass_fit_counts",
    "mass_plane_x_edges": "first swapped Lambda endpoint mass edges [GeV]",
    "mass_plane_y_edges": "second swapped Lambda endpoint mass edges [GeV]",
    "mass_plane_counts": "raw cat0/llbar 2-D Lambda-mass-plane counts",
    "mass_plane_variances": "Sumw2 variances for mass_plane_counts",
    "mass_region_edges": "pair invariant-mass observable edges [GeV]",
    "mass_region_counts": "pair-mass counts, axis 0 ordered SS/BS/SB/BB",
    "mass_region_variances": "Sumw2 variances for mass_region_counts",
    "delta_phi_thrust_edges": "delta_phi_thrust edges [rad]",
    "delta_phi_thrust_counts": "delta_phi_thrust counts, axis 0 ordered SS/BS/SB/BB",
    "delta_phi_thrust_variances": "Sumw2 variances for delta_phi_thrust_counts",
}
fixture_metadata = {
    "schema_version": "sideband_ana_fixture_v1",
    "fixture_file": output_npz.name,
    "sample": "data",
    "years": source_metadata["years"],
    "year_merge_weights": {year: event_counts[f"data_{year}"]["weight"] for year in source_metadata["years"]},
    "category": "all_event",
    "category_index": 0,
    "subset": "llbar",
    "channel": "lambda_lambdabar",
    "channel_index": 1,
    "selection": source_metadata["selection"],
    "tree": source_metadata["tree"],
    "observable": "delta_phi_thrust",
    "observable_units": "radians",
    "delta_phi_definition": definitions["delta_phi_thrust"],
    "mass_units": "GeV",
    "mass_fit": {
        "source_object": "mass_fit_ll_data",
        "range_gev": [float(fit_edges[0]), float(fit_edges[-1])],
        "peak_mean_gev": float(mass_fit["mean"]),
        "peak_mean_error_gev": float(mass_fit["mean_error"]),
        "chi2": float(mass_fit["chi2"]),
        "ndf": int(mass_fit["ndf"]),
        "background_formula": mass_fit["formula"],
    },
    "regions": {
        "order": ["SS", "BS", "SB", "BB"],
        "source_reg_indices": {"SS": 0, "BS": 1, "SB": 2, "BB": 3},
        "orientation": "09 uses the upper sideband only on each Lambda-mass endpoint; B is that upper sideband, not a low+high union.",
        "atomic_mapping": {
            "SS": ["S", "S"],
            "BS": ["B", "S"],
            "SB": ["S", "B"],
            "BB": ["B", "B"],
        },
        "source_objects": {
            label: {
                "mass": f"regional_ll_data_ch1_cat0_reg{index}_mass_5gev",
                "delta_phi_thrust": f"regional_ll_data_ch1_cat0_reg{index}_delta_phi_thrust",
            }
            for label, index in zip(["SS", "BS", "SB", "BB"], range(4))
        },
        "aggregation": "The fixture uses one atomic LL channel (ch1=Lambda-Lambdabar), so there is no channel aggregation; each axis-0 row is copied from one existing 09 two-endpoint region object.",
        "definition": definitions["region_definition_ll"],
        "signal_window_gev": float(definitions["signal_window_gev"]),
        "sideband_offset_gev": float(definitions["sideband_offset_gev"]),
        "peak_mean_gev": float(mass_fit["mean"]),
    },
    "cut_chain": source_metadata["cut_chain"],
    "source": {
        "analysis_script": "lambda-ana/scripts/pair_distributions/09_pair_jet_distribution_summary.py",
        "product": str(source_product),
        "metadata": str(source_metadata_path),
        "product_sha256": source_digest.hexdigest(),
        "source_metadata_utc": source_metadata["utc"],
        "fixture_generated_utc": generated_utc,
        "extraction": "ROOT TH1D/TH2D bin edges, contents and GetBinError()^2 (Sumw2 variances); no rebinning or toy generation.",
    },
    "arrays": {
        name: {"shape": list(values.shape), "dtype": str(values.dtype), "meaning": array_meanings[name]}
        for name, values in arrays.items()
    },
}
metadata_path = args.output_dir / "sideband_ll_data_cat0_llbar.json"
metadata_path.write_text(
    json.dumps(fixture_metadata, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(f"wrote {output_npz}")
print(f"wrote {metadata_path}")
