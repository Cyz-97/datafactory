# Sideband transfer factors

- analysis: sideband_ana module test
- channel: lambda_lambdabar
- selection: tight
- peak_mean_gev: 1.1162307456725606

## llbar_fixed_bg
- scope: cat0/llbar/tight (background fixed)
- fallback: none
- x regions: signal [1.10623, 1.12623], high sideband [1.15053, 1.17053]
- 1D transfer:
  - I_S = 295123
  - I_H = 232739
  - I_B = 232739
  - r_H = 1.26804
  - r = 1.26804 ± 0.0066
  - background: `12615500.0*(a4 + (a3 + ((x0 - 1.0801583333333333) * 10.5615)*(a2 + a5*exp(a1*((x0 - 1.0801583333333333) * 10.5615))))**2)`

## llbar_profiled_bg
- scope: cat0/llbar/tight (background profiled)
- fallback: none
- x regions: signal [1.10623, 1.12623], high sideband [1.15053, 1.17053]
- 1D transfer:
  - I_S = 313712
  - I_H = 390869
  - I_B = 390869
  - r_H = 0.802602
  - r = 0.802602 ± 0.013
  - background: `12615500.0*(a4 + (a3 + ((x0 - 1.0801583333333333) * 10.5615)*(a2 + a5*exp(a1*((x0 - 1.0801583333333333) * 10.5615))))**2)`

## llbar_2d
- scope: cat0/llbar/tight (mass-plane 2D)
- fallback: none
- x regions: signal [1.10623, 1.12623], high sideband [1.15053, 1.17053]
- y regions: signal [1.10623, 1.12623], high sideband [1.15053, 1.17053]
- 2D transfer:
  - atomic region integrals:
    - SS: SxSy=0.9617, BxSy=0.2346, SxBy=0.2346, BxBy=0.05723
    - SH: SxSy=5.066e-12, BxSy=1.236e-12, SxBy=0.1819, BxBy=0.04437
    - HS: SxSy=5.066e-12, BxSy=0.1819, SxBy=1.236e-12, BxBy=0.04437
    - HH: SxSy=2.669e-23, BxSy=9.582e-13, SxBy=9.582e-13, BxBy=0.0344
  - aggregated integrals:
    - SxSy: SS=0.9617, BS=5.066e-12, SB=5.066e-12, BB=2.669e-23
    - BxSy: SS=0.2346, BS=0.1819, SB=1.236e-12, BB=9.582e-13
    - SxBy: SS=0.2346, BS=1.236e-12, SB=0.1819, BB=9.582e-13
    - BxBy: SS=0.05723, BS=0.04437, SB=0.04437, BB=0.0344
  - w_H = 1.28983 ± 0.012
  - w_V = 1.28983 ± 0.012
  - w_C = -1.66367 ± 0.03
  - closure w_C + w_H*w_V = 2.22e-16
  - signal leakage:
    - SH: 5.268e-12
    - HS: 5.268e-12
    - HH: 2.775e-23
