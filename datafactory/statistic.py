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

def fuck_roofit_param(fit_result):
    final_params = fit_result.floatParsFinal()
    # 在pyROOT中，通常使用迭代器来遍历RooArgList
    result_dict = {}
    for i in range(final_params.getSize()):
        param = final_params.at(i)
        result_dict[param.GetName()] = ( param.getVal(), param.getError())
    return result_dict

def fit_mc_data(mc_hist, data_hist, artificial_model = False):

    mc_hist._get_value()
    data_hist._get_value(data_hist)

    x_min = data_hist.histogram.GetXaxis().GetXmin()
    x_max = data_hist.histogram.GetXaxis().GetXmax()
    x = R.RooRealVar("x", "s", x_min, x_max)
    
    rdh_data = R.RooDataHist("data_rdh", "Data", R.RooArgList(x), data_hist.histogram)
    rdh_mc = {}
    for key, value in mc_hist.staff_dict.items():
        rdh_mc[key] = R.RooDataHist(f"rdh_{key}", f"rdh_{key}", R.RooArgList(x), value.histogram)
    # 3. Convert to PDFs
    pdf_mc = {key: R.RooHistPdf(f"pdf_{key}", f"pdf_{key}", R.RooArgList(x), value) for key, value in rdh_mc.items()}
    # 5. Fit fractions (or yields)
    n_mc = {key: R.RooRealVar(f"n_{key}", f"n_{key}", mc_hist.staff_dict[key].histogram.Integral(), 0, mc_hist.staff_dict[key].histogram.Integral()*1e4) for key, value in pdf_mc.items()}
    if artificial_model:
        a0 = R.RooRealVar("mean", "mean", 1.6854, 1.5, 1.8)
        a1 = R.RooRealVar("sigma", "sigma", 0.1, 1e-19, 0.2)
        poly_bkg = R.RooGaussian("pdf_artificial_bkg", "Polynomial background", x, a0, a1)
        n_poly = R.RooRealVar(r"n_\text{Artificial background}", "PolyBkg yield", 0, 0, 1e6)
        parameterize_model = [poly_bkg]
        param_model_yield = [n_poly]
    else:
        parameterize_model = []
        param_model_yield = []

    # 6. Total PDF
    model = R.RooAddPdf("model", "Model",
                        R.RooArgList(list(pdf_mc.values()) + parameterize_model),
                        R.RooArgList(list(n_mc.values()) + param_model_yield)
                        )
    fit_result = model.fitTo(rdh_data, R.RooFit.Save(), R.RooFit.PrintLevel(-1), R.RooFit.Verbose(False))
    # frame = x.frame(R.RooFit.Title("Fit to data"))
    # rdh_data.plotOn(frame)
    # model.plotOn(frame)
    # i = 0
    # for key, value in pdf_mc.items():
    #     model.plotOn(frame, R.RooFit.Components(f"pdf_{key}"), R.RooFit.LineStyle(R.kDashed), R.RooFit.LineColor(R.kRed + i))
    #     i+=1
    # model.plotOn(frame, R.RooFit.Components("pdf_artificial_bkg"), R.RooFit.LineStyle(R.kDashed), R.RooFit.LineColor(R.kBlue))
    # c1 = R.TCanvas()
    # frame.Draw()
    # c1.BuildLegend()
    # # c1.SetLogy()
    # c1.Draw()
    fit_param = fuck_roofit_param(fit_result)
    return fit_result, parameterize_model,fit_param 


def cut_chain_to_eff_pur(table):
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
    eff = (eff.T / eff.iloc[:,0].T).T * 100
    # 复制表格以避免修改原始数据
    purity = table.copy()
    # 计算纯度，将每列除以"Sum"行（通常是总计数），然后乘以100转换为百分比
    purity = (purity / purity.loc["Sum",:]) * 100
    # 返回效率和纯度表格
    return eff, purity
