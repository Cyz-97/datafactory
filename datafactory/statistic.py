from typing import Optional, Tuple, List, Any

import ROOT as R #, RooFitResult, RooRealVar, RooDataHist, RoohistPdf 
from .hist import HistFactory, HistStaff

# def template_fit(data_hist: HistStaff, mc_hists: HistFactory) -> RooFitResult:
#     pass

def bayes_divide(y_pass, y_tot):
    """
    Calculates Bayesian efficiency and confidence intervals for binomial proportions.
    
    Args:
        y_pass (array-like): Array of successful event counts (numerator)
        y_tot (array-like): Array of total event counts (denominator)
    
    Returns:
        tuple: A tuple containing:
            - eff (ndarray): Efficiency values (y_pass/y_tot)
            - lower_error (ndarray): Lower 1-sigma confidence interval bounds
            - upper_error (ndarray): Upper 1-sigma confidence interval bounds
    
    Notes:
        - Uses beta distribution (Beta(1+y_pass, 1+y_tot-y_pass)) for Bayesian inference
        - Returns 1-sigma (68.27%) confidence intervals (16th and 84th percentiles)
        - Handles edge cases (zero efficiency and perfect efficiency)
    """
    import numpy as np
    from scipy.stats import beta
    
    # Assuming you have two histograms: `numerator` and `denominator`
    # with the same binning, and these histograms are given as arrays of bin contents.
    
    # Calculate efficiencies
    eff = np.divide(y_pass, y_tot, where = y_tot != 0, out = np.zeros_like(y_pass))
    
    # Calculate Bayesian errors
    alpha = 1 + y_pass
    beta_param = 1 + (y_tot - y_pass)
    lower_error = eff - beta.ppf(0.15865, alpha, beta_param)
    upper_error = beta.ppf(0.84135, alpha, beta_param) - eff
    lower_error[eff == 0] = 0
    upper_error[eff == 1] = 0
    return eff, lower_error, upper_error

def baker_cousins_chi2(observed, expected, n_params):
    """
    Computes the Baker–Cousins chi-squared goodness-of-fit statistic for
    binned Poisson data.

    Baker–Cousins χ² is defined as:

        χ²_BC = 2 Σᵢ [ nᵢ ln(nᵢ / μᵢ) + μᵢ − nᵢ ]

    where nᵢ and μᵢ are the observed and expected (model) counts in bin i.
    For bins with nᵢ = 0 the nᵢ ln(nᵢ/μᵢ) term vanishes, leaving a
    contribution of μᵢ.

    This statistic equals 2 × Poisson NLL plus the constant Σ nᵢ ln(nᵢ),
    so χ²_BC − χ²_BC(best) is exactly the likelihood-ratio test statistic
    used in MINUIT / zfit.  It has the same asymptotic χ²(ndf) distribution
    as Pearson χ² but is more accurate in low-count bins where the Poisson
    distribution is noticeably non-Gaussian.

    The number of degrees of freedom is ndf = N_valid_bins − n_params, and
    the reduced chi² is χ² / ndf (or NaN when ndf ≤ 0).

    Args:
        observed (array-like): Observed bin counts (e.g. integer histogram
            entries).  Shape ``(N_bins,)``.
        expected (array-like): Expected bin counts from the fitted model.
            Same shape as ``observed``.  Must be > 0 in every bin used.
        n_params (int): Number of free parameters in the fit (e.g. 3 for
            nsig, nbkg, and a slope parameter).

    Returns:
        dict: A dictionary with keys:
            - ``chi2`` (float):      Baker–Cousins χ² value.
            - ``ndf`` (int):         Number of degrees of freedom.
            - ``reduced_chi2`` (float): χ² / ndf, or ``float('nan')`` if
              ndf ≤ 0.
    """
    import numpy as np

    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)

    # Only include bins where the model predicts a positive count
    valid = exp > 0
    obs_v = obs[valid]
    exp_v = exp[valid]

    # Baker–Cousins: 2 * Σ [n ln(n/μ) + μ − n]
    # For n = 0: n ln(n/μ) → 0, contribution = μ
    ratio = np.where(obs_v > 0, obs_v * np.log(obs_v / exp_v), 0.0)
    chi2 = 2.0 * np.sum(ratio + exp_v - obs_v)
    ndf = len(obs_v) - n_params
    reduced_chi2 = chi2 / ndf if ndf > 0 else float("nan")

    return {"chi2": float(chi2), "ndf": int(ndf), "reduced_chi2": float(reduced_chi2)}

def fuck_roofit_param(fit_result):
    final_params = fit_result.floatParsFinal()
    # 在pyROOT中，通常使用迭代器来遍历RooArgList
    result_dict = {}
    for i in range(final_params.getSize()):
        param = final_params.at(i)
        result_dict[param.GetName()] = ( param.getVal(), param.getError())
    return result_dict

from typing import Optional, Tuple, List, Any
from datafactory.hist import TH12Numpy
# --- zfit template fit ----------------------------------------

def fit_mc_data(
    mc_factory: "HistFactory",
    data: "HistStaff",
    *,
    obs_name: str = "x",
    extended: bool = True,
    yield_init: Optional[float] = None,
    yield_bounds: Optional[Tuple[float, float]] = None,
    minimizer: Optional["zfit.minimize.Minimizer"] = None,
    constraints: Optional[List[Any]] = None,
    allow_negative_yields: bool = False,
):
    """Fit a (binned) data histogram using a sum of MC template components with zfit.

    This implements a classic template fit:
      data(bin)  ~  sum_i  N_i * template_i(bin)

    Parameters
    ----------
    mc_factory:
        HistFactory containing MC components (each value must be a 1D TH1-like histogram).
    data:
        HistStaff containing the observed data histogram (1D TH1-like).
    obs_name:
        Observable name used by zfit/UHI axis naming.
    extended:
        If True, each component is treated as an extended PDF with a floating yield parameter N_i.
        If False, the templates are normalized shapes and combined with floating fractions.
    yield_init:
        Initial value for each yield parameter. If None, it is set to (data integral)/(n_components).
    yield_bounds:
        Bounds for each yield parameter (min, max). If None, defaults to (0, 10 * data integral).
        If allow_negative_yields is True, the lower bound will be set to -max(|bound|, 10*sqrt(Ndata)).
    minimizer:
        zfit minimizer. If None, uses zfit.minimize.Minuit().
    constraints:
        Optional list of zfit constraints to be added to the loss.
    allow_negative_yields:
        If True, yields are allowed to go negative (useful for background-subtraction tests).

    Returns
    -------
    dict
        A dictionary with keys:
          - 'result': zfit fit result
          - 'yields': dict[name -> fitted yield]
          - 'yield_params': dict[name -> zfit.Parameter]
          - 'model': combined zfit PDF
          - 'loss': zfit loss object
          - 'data_binned': zfit.data.BinnedData
          - 'component_pdfs': dict[name -> zfit.pdf.HistogramPDF]

    Notes
    -----
    - This function currently supports **1D** histograms.
    - It assumes all MC components and data are binned identically. If not, it will raise.
    - zfit binned PDFs are UHI-compatible; we convert ROOT TH1 to a `hist.Hist` (UHI).
    """

    # Local imports to keep the module import-light unless fitting is used.
    import numpy as np
    import zfit
    import hist as uhi_hist

    # Make sure ROOT objects are materialized (in case they are RResultPtr etc.)
    mc_factory._get_value()
    data._get_value(data)

    if data.dimension != 1:
        raise NotImplementedError("fit_histfactory_to_data_zfit currently supports 1D histograms only.")

    # --- helper: ROOT TH1 -> UHI hist.Hist (with named axis) -----------------
    def _th1_to_uhi(th1, *, axis_name: str):
        xcent, counts, err, edges = TH12Numpy(th1)
        edges = np.asarray(edges, dtype=float)
        counts = np.asarray(counts, dtype=float)
        err = np.asarray(err, dtype=float)

        h = uhi_hist.Hist(
            uhi_hist.axis.Variable(edges, name=axis_name),
            storage=uhi_hist.storage.Weight(),
        )
        view = h.view()
        view["value"][...] = counts
        view["variance"][...] = err**2
        return h

    # --- build data ----------------------------------------------------------
    if data.histogram is None:
        raise ValueError("Data histogram is empty.")

    data_uhi = _th1_to_uhi(data.histogram, axis_name=obs_name)
    data_binned = zfit.data.BinnedData.from_hist(data_uhi)
    n_data = float(np.sum(data_uhi.view().value))

    # --- build component PDFs ------------------------------------------------
    component_pdfs: dict[str, zfit.pdf.HistogramPDF] = {}
    yield_params: dict[str, zfit.Parameter] = {}

    # Determine initial and bounds
    n_components = len(mc_factory.staff_dict)
    if n_components == 0:
        raise ValueError("mc_factory has no components to fit.")

    if yield_init is None:
        yield_init_val = n_data / max(n_components, 1)
    else:
        yield_init_val = float(yield_init)

    if yield_bounds is None:
        y_min = 0.0
        y_max = max(1.0, 10.0 * max(n_data, 1.0))
    else:
        y_min, y_max = float(yield_bounds[0]), float(yield_bounds[1])

    if allow_negative_yields:
        # Heuristic: allow down to roughly a few sigma of Ndata if no explicit min given.
        if yield_bounds is None:
            y_min = -max(10.0 * np.sqrt(max(n_data, 1.0)), 1.0)

    # Reference binning: data
    ref_edges = data_uhi.axes[0].edges

    for name, staff in mc_factory.staff_dict.items():
        if staff.histogram is None:
            continue
        if staff.dimension != 1:
            raise NotImplementedError(f"Component '{name}' is not 1D; only 1D is supported.")

        comp_uhi = _th1_to_uhi(staff.histogram, axis_name=obs_name)
        comp_edges = comp_uhi.axes[0].edges
        if len(comp_edges) != len(ref_edges) or np.max(np.abs(comp_edges - ref_edges)) > 0:
            raise ValueError(
                f"Binning mismatch for component '{name}'. "
                "All components must have identical bin edges to data."
            )

        if extended:
            y = zfit.Parameter(f"N_{name}", staff.histogram.Integral(), y_min, y_max)
            print(y)
            pdf = zfit.pdf.HistogramPDF(comp_uhi, extended=y, label=name)
            yield_params[name] = y
        else:
            # Shape-only templates: normalize each component to 1 and fit fractions.
            # (Fractions are handled after we create all PDFs.)
            pdf = zfit.pdf.HistogramPDF(comp_uhi, extended=False, label=name)

        component_pdfs[name] = pdf

    if not component_pdfs:
        raise ValueError("No valid MC component histograms found in mc_factory.")

    # --- build combined model ------------------------------------------------
    pdfs = list(component_pdfs.values())

    if extended:
        # Sum of extended binned PDFs -> extended model with total yield = sum_i N_i
        model = zfit.pdf.BinnedSumPDF(pdfs, label="sum_model")
        loss = zfit.loss.ExtendedBinnedNLL(model=model, data=data_binned, constraints=constraints)
    else:
        # Non-extended: fit fractions (simplex parameterization)
        # Fix number of fracs = n_pdfs - 1
        fracs = [
            zfit.Parameter(f"frac_{i}", 1.0 / len(pdfs), 0.0, 1.0)
            for i in range(len(pdfs) - 1)
        ]
        model = zfit.pdf.BinnedSumPDF(pdfs, fracs=fracs, label="sum_model")
        loss = zfit.loss.BinnedNLL(model=model, data=data_binned, constraints=constraints)

    # --- minimize ------------------------------------------------------------
    if minimizer is None:
        minimizer = zfit.minimize.Minuit()

    result = minimizer.minimize(loss)
    result.hesse()

    # --- collect yields ------------------------------------------------------
    yields: dict[str, float] = {}
    if extended:
        for name, par in yield_params.items():
            yields[name] = float(result.params[par]["value"])
    else:
        # Convert fitted fractions into yields by scaling to total data count
        fitted_fracs = [float(result.params[p]["value"]) for p in model.params.values() if p.name.startswith("frac_")]
        # zfit SumPDF uses n-1 fracs; the last one is implicit
        if len(pdfs) == 1:
            frac_all = [1.0]
        else:
            last = 1.0 - float(np.sum(fitted_fracs))
            frac_all = fitted_fracs + [last]
        for (name, _), frac in zip(component_pdfs.items(), frac_all):
            yields[name] = float(frac) * n_data

    return {
        "result": result,
        "yields": yields,
        "yield_params": yield_params,
        "model": model,
        "loss": loss,
        "data_binned": data_binned,
        "component_pdfs": component_pdfs,
    }



def cut_chain_to_eff_pur(table, total_column="N0"):
    """
    将给定的表格转换为效率和纯度的表格。

    参数:
    table (pandas.DataFrame): 包含数据的输入表格。格式如下：
    ```
    |                        |            N0 |        Cut 1st |     Cut 2nd |     Cut 3rd |       Cut 4th |
    |: ----------------------|--------------:|---------------:|------------:|------------:|--------------:|
    | $\text{Signal}$        | 9.531562e+04  |   16955.7      |   16712.7   |     1.71829 |    0.763686   |
    | $\text{Background A}$  | 4.729948e+04  |   6143.45      |    5976.58  |    40.4405  |   22.73       |
    | $\text{Background B}$  | 5.164237e+03  |    483.16      |     461.819 |   110.259   |   66.4095     |
    | Sum                    | 3.845172e+08	 |   2.38722e+06  |  971540     |  3526.52    |  419.03       |
    ```

    返回:
    tuple: 包含效率和纯度（%）表格的元组。格式如下：
    ```
    |                         |   N0 |   Cut 1st |    Cut 2nd |      Cut 3rd |       Cut 4th |
    |:------------------------|-----:|----------:|-----------:|-------------:|--------------:|
    | $\text{Signal}$         |  100 | 17.789    | 17.5341    | 17.1499      |   6.95197     |
    | $\text{Background A}$   |  100 | 12.9884   | 12.6356    |  5.88961     |   0.231468    |
    | $\text{Background B}$   |  100 |  9.35589  |  8.94264   |  2.06024     |   0.0060007   |
    | Sum                     |  100 |  0.620836 |  0.252665  |  0.00634966  |   0.00180734  |
    ```
    """
    # 复制表格以避免修改原始数据
    eff = table.copy()
    # 计算效率，将每列除以第一列（通常是总计数），然后乘以100转换为百分比
    eff = (eff.T / eff.loc[:,total_column].T).T * 100
    # 复制表格以避免修改原始数据
    purity = table.copy()
    # 计算纯度，将每列除以"Sum"行（通常是总计数），然后乘以100转换为百分比
    purity = (purity / purity.loc["Sum",:]) * 100
    # 返回效率和纯度表格
    return eff, purity

def errors_corr_from_covariance_safe(cov):
    """
    Extract errors and correlation matrix from covariance, 
    safely handling zero-variance components.

    Parameters
    ----------
    cov : array_like
        Covariance matrix.

    Returns
    -------
    errors : ndarray
        Standard deviations (sigma_i).
    corr : ndarray
        Correlation matrix (with undefined entries set to 0).
    """
    cov = np.asarray(cov, dtype=float)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    n = cov.shape[0]
    errors = np.sqrt(np.diag(cov))
    corr = np.zeros_like(cov)

    for i in range(n):
        for j in range(n):
            if errors[i] > 0 and errors[j] > 0:
                corr[i, j] = cov[i, j] / (errors[i] * errors[j])
            else:
                corr[i, j] = 0.0

    # ensure diagonal = 1
    for i in range(n):
        corr[i, i] = 1.0

    return errors, corr