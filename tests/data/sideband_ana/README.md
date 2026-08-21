# Fixed `sideband_ana` fixture

`sideband_ll_data_cat0_llbar.npz` is a small, committed extraction from the
real `09_pair_jet_distribution_summary.py` publication product.  It is not a
toy sample: every value is copied from the ROOT `TH1D`/`TH2D` object named in
`sideband_ll_data_cat0_llbar.json`; each `*_variances` array is
`GetBinError()**2` (the ROOT `Sumw2` variance).

The fixed analysis slice is:

- data, years 1992--1995, tight Lambda selection, all-event (`cat0`);
- LL `llbar` subset, one atomic channel (`ch1 = Lambda-Lambdabar`), so no
  channel aggregation is hidden in the fixture;
- region axis order, for both mass and `delta_phi_thrust`, is exactly
  `SS`, `BS`, `SB`, `BB` (ROOT `reg0`, `reg1`, `reg2`, `reg3`);
- the source uses one upper sideband per Lambda-mass endpoint (`B` means the
  09 upper sideband), so `BS=(B,S)`, `SB=(S,B)`, and `BB=(B,B)`.  These are
  the source's four two-endpoint products, not a fabricated nine-region
  low/high union; the JSON records this orientation explicitly;
- the mass plane is the merged four-year data plane (`n_periods=1` for a
  future 2-D fit check), while the JSON records the original per-year data
  merge weights (all one).

Array names, shapes, units, fit metadata, cuts, region orientation, source
object names, and generation time are recorded in the sidecar JSON.  The
observable is the script's `delta_phi_thrust`: the angle in radians after
projecting both pair momenta onto the plane perpendicular to the category
thrust axis (`Btag_thrustVector` for `all_event`).

## Refresh

Run in the ROOT environment after regenerating the upstream 09 product:

```sh
source /home/cheyuzhi/opt/miniconda3/etc/profile.d/conda.sh
conda activate root6.34
python tests/data/sideband_ana/extract_fixture.py \
  --source-product /path/to/pair_jet_distribution_products.root \
  --source-metadata /path/to/pair_jet_distribution_metadata.json
```

The extractor performs no rebinning, toy generation, or rounding.  It updates
the compressed NPZ and sidecar JSON together, including the upstream ROOT
SHA-256 and UTC generation timestamp.
