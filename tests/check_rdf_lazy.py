"""Manual ROOT regression checks; no files, TensorFlow or CI setup required.

Run with root6.34 active: python tests/check_rdf_lazy.py.  GetNRuns measures
event loops rather than wall time, so these checks do not depend on disk caches.
"""
# %% Definitions and unchanged cuts must reuse the booked selection count.
from copy import copy, deepcopy

import sys

import numpy as np
import ROOT as R

# Optional N verifies the same lazy contract with N ROOT worker slots.
if len(sys.argv) > 1:
    R.EnableImplicitMT(int(sys.argv[1]))

from datafactory.core import StaffType
from datafactory.hist import HistFactory, HistStaff
from datafactory.rdf import CutFlow, RDFFactory, RDFStaff

staff = RDFStaff.__new__(RDFStaff)
staff.name, staff.type = "toy", StaffType.signal
staff.rdf, staff.cuts, staff.pre_cut_chain = R.RDataFrame(100), [], {}
staff.set_cuts([])
factory = RDFFactory.__new__(RDFFactory)
factory.staff_dict, factory.type_dict = {"toy": staff}, {"toy": staff.type}
factory.cuts, factory.classify_dict = [], {}
factory.set_cuts([])
for index in range(3):
    factory.define(f"x{index}", "double(rdfentry_)")
assert staff.rdf.GetNRuns() == 1, "Define repeatedly scanned the same selection"
first = factory.get_histfactory(lambda node: node.Histo1D(("first", "", 10, 0., 100.), "x0"))
second = factory.get_histfactory(lambda node: node.Histo1D(("second", "", 10, 0., 100.), "x1"))
assert staff.rdf.GetNRuns() == 1, "Histogram booking triggered an event loop"
np.testing.assert_array_equal(first.sum().get_numpy()[1], np.full(10, 10.))
np.testing.assert_array_equal(second.sum().get_numpy()[1], np.full(10, 10.))
assert staff.rdf.GetNRuns() == 2, "Booked histograms did not run together"

# %% Replacing cuts, classification, a column or the input must refresh counts.
factory.set_cuts([CutFlow("upper", {}, "x0 >= 50")])
assert staff.cuts[-1].count_final.GetValue() == 50
factory.define("x0", "double(rdfentry_) + 100.")
assert staff.cuts[-1].count_final.GetValue() == 100, "Redefine left stale cut counts"
factory.classify_dict["toy"] = [CutFlow("class", {}, "x0 < 125")]
selected = factory.get_histfactory(lambda node: node.Histo1D(("class", "", 25, 100., 125.), "x0"))
assert selected.sum().histogram.Integral() == 25
factory.set_cuts([CutFlow("empty", {}, "x0 < 0")])
assert staff.empty()
factory.set_cuts([])
assert not staff.empty()
staff.rdf = R.RDataFrame(0).Define("x0", "0.")
assert staff.empty(), "Input replacement left a stale nonempty count"
staff.rdf = R.RDataFrame(4).Define("x0", "double(rdfentry_)")
assert not staff.empty(), "Input replacement left a stale empty count"
staff.set_cuts([CutFlow("monitor", {"x0": (4, 0., 4.)}, "true")])
staff.cuts[0].list_bystander["x0"] = (8, 0., 4.)
staff.set_cuts(staff.cuts)
assert staff.cuts[0].dir_bystander_hist_final["x0"].GetValue().GetNbinsX() == 8

# %% HistStaff/HistFactory copies stay lazy and own independent mutable TH1s.
node = R.RDataFrame(100).Define("x", "double(rdfentry_)")
hist = HistStaff("original", histogram=node.Histo1D(("h1", "", 10, 0., 100.), "x"))
plane = HistStaff("plane", histogram=node.Histo2D(("h2", "", 10, 0., 100., 10, 0., 100.), "x", "x"))
cube = HistStaff("cube", histogram=node.Histo3D(("h3", "", 10, 0., 100., 10, 0., 100., 10, 0., 100.), "x", "x", "x"))
copies = [copy(hist), deepcopy(hist)]
wrapped = HistFactory(staff_dict={"original": hist}, type_dict={"original": StaffType.signal})
assert node.GetNRuns() == 0, "Constructing/copying HistStaff or HistFactory was eager"
hist.histogram.Scale(2)
assert hist.histogram.Integral() == 200
for copied in [*copies, wrapped.staff_dict["original"]]:
    assert copied.histogram.Integral() == 100, "Histogram copies share mutable storage"
copies[0].histogram.Scale(3)
assert hist.histogram.Integral() == 200
assert copies[1].histogram.Integral() == 100
assert plane.dimension == 2 and cube.dimension == 3
assert plane.histogram.Integral() == cube.histogram.Integral() == 100
assert node.GetNRuns() == 1
physical_copy = deepcopy(hist)
physical_copy.histogram.Scale(4)
assert physical_copy.histogram.Integral() == 800 and hist.histogram.Integral() == 200
np.testing.assert_array_equal((hist * 0.5).get_numpy()[1], np.full(10, 10.))
print("PASS: count invalidation, joint lazy booking, 1D/2D/3D dimensions and independent copies")

# %% Weighted fills and Sumw2 merge correctly across IMT slots (roundoff only).
node = R.RDataFrame(1000).Define("x", "double(rdfentry_ % 10)")
node = node.Define("weight", "double((rdfentry_ % 7) + 1) / 13.")
weighted = HistStaff("weighted", histogram=node.Histo1D(("weighted", "", 10, 0., 10.), "x", "weight"))
assert node.GetNRuns() == 0
histogram = weighted.histogram
indices = np.arange(1000)
weights = ((indices % 7) + 1) / 13.
np.testing.assert_allclose([histogram.GetBinContent(index) for index in range(1, 11)],
                           np.bincount(indices % 10, weights=weights), rtol=1e-13)
np.testing.assert_allclose([histogram.GetBinError(index)**2 for index in range(1, 11)],
                           np.bincount(indices % 10, weights=weights**2), rtol=1e-13)
assert node.GetNRuns() == 1
assert node.GetNSlots() == (int(sys.argv[1]) if len(sys.argv) > 1 else 1)
print(f"PASS: weighted counts/Sumw2 with {node.GetNSlots()} slot(s)")
