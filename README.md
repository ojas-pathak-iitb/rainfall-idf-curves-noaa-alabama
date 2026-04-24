# rainfall-idf-curves-noaa-alabama

**Construction and Statistical Analysis of Intensity-Duration-Frequency (IDF) Curves from High-Resolution Rainfall Data**

A fully from-scratch Python implementation of IDF curve derivation using 43 years of 15-minute NOAA precipitation records from Randolph, Alabama. All statistical methods — Gumbel and Log-Pearson Type III fitting, KS testing, Grubbs outlier detection, and bootstrap confidence intervals — are implemented without any statistical solver libraries. You can refer to the Master Report attached with the project for a detailed analysis.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset Setup](#dataset-setup)
- [Dependencies](#dependencies)
- [Running the Pipeline](#running-the-pipeline)
- [Pipeline Steps](#pipeline-steps)
- [Outputs](#outputs)
- [Methods Summary](#methods-summary)
- [Reference](#reference)

---

## Overview

IDF (Intensity-Duration-Frequency) curves describe the relationship between rainfall intensity, storm duration, and return period. They are the foundational design tool for stormwater drains, culverts, and flood control structures.

This project:
- Parses raw NOAA DSI-3260 `.dat` files (fixed-width format, compressed `.tar.Z` archives)
- Extracts the Annual Maximum Series (AMS) for 6 storm durations: 15 min, 30 min, 1 hr, 2 hr, 6 hr, 24 hr
- Fits Gumbel (EV-I) and Log-Pearson Type III (LP3) distributions using Method of Moments
- Selects the better-fitting distribution per duration using the Kolmogorov-Smirnov test
- Derives IDF curves for return periods T = 2, 5, 10, 25, 50, and 100 years
- Fits the empirical formula `i = a·T^b / d^e` via linear least squares
- Validates results against NOAA Atlas 14, Volume 9 official estimates
- Detects outliers via the Grubbs test and quantifies their impact on IDF estimates
- Performs a split-period temporal trend analysis (1972–1992 vs 1993–2013)
- Computes bootstrap confidence intervals for uncertainty quantification

**Station:** 01014000 / 01014007 — Randolph, Alabama, USA  
**Record period:** 1971–2013 (43 years)  
**Data source:** NOAA DSI-3260 15-Minute Precipitation Dataset

---

## Project Structure

```
project-root/
│
├── code/
│   ├── step0_extract.py          # Decompress .tar.Z archives
│   ├── step1_parse.py            # Parse DSI-3260 .dat files → clean_data.csv
│   ├── step2_ams.py              # Extract Annual Maximum Series
│   ├── step3_distributions.py   # Fit Gumbel and LP3 (Method of Moments)
│   ├── step4_ks_test.py          # KS goodness-of-fit test
│   ├── step5_return_periods.py   # Compute IDF table for T = 2–100 yr
│   ├── step6_idf_formula.py      # Fit empirical formula i = a·T^b / d^e
│   ├── step7_plot.py             # Plot IDF curves
│   ├── step8_atlas14_compare.py  # Compare against NOAA Atlas 14
│   ├── step9_sensitivity.py      # Gumbel vs LP3 sensitivity analysis
│   ├── step10_trend_analysis.py  # Split-period trend analysis
│   ├── step10b_trend_no_outlier.py # Trend analysis with outlier removed
│   ├── step11_outlier_detection.py # Grubbs outlier test
│   ├── step12_bootstrap_ci.py    # Bootstrap confidence intervals
│   └── step13_cdf_plot.py        # CDF comparison plots
│
├── Dataset/                      # ← See Dataset Setup below
│   ├── state_data/
│   │   ├── 01/                   # .tar.Z files for Alabama
│   │   ├── by_month2011/
│   │   ├── by_month2012/
│   │   └── by_month2013/
│   └── extracted/                # Created automatically by step0
│
└── output/                       # Created automatically; all results go here
```

---

## Dataset Setup

The raw NOAA dataset is too large to include in this repository. Please follow these steps:

1. **Download the Dataset folder** from Google Drive:  
   👉 [https://drive.google.com/drive/folders/1sbrJpBvuLgzLX1ZnzIStkeB-iCpUXdJT?usp=sharing](https://drive.google.com/drive/folders/1sbrJpBvuLgzLX1ZnzIStkeB-iCpUXdJT?usp=sharing)

2. **Place the downloaded `Dataset` folder** in the same directory as the `code` folder, so your structure looks like:
   ```
   project-root/
   ├── code/
   └── Dataset/
   ```

3. The `output/` folder will be created automatically when you run the scripts.

---

## Dependencies

Only the following Python packages are used:

```
numpy
matplotlib
pandas
unlzw3
```

Install them with:

```bash
pip install numpy matplotlib pandas unlzw3
```

Python 3.8 or later is recommended. No statistical solver libraries (scipy, statsmodels, etc.) are used — all methods are implemented from scratch.

---

## Running the Pipeline

Run the scripts **in order** from the project root directory:

```bash
python code/step0_extract.py
python code/step1_parse.py
python code/step2_ams.py
python code/step3_distributions.py
python code/step4_ks_test.py
python code/step5_return_periods.py
python code/step6_idf_formula.py
python code/step7_plot.py
python code/step8_atlas14_compare.py
python code/step9_sensitivity.py
python code/step10_trend_analysis.py
python code/step10b_trend_no_outlier.py
python code/step11_outlier_detection.py
python code/step12_bootstrap_ci.py
python code/step13_cdf_plot.py
```

Each script reads from and writes to the `output/` folder. Steps 0 and 1 only need to be run once to generate `clean_data.csv`.

---

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 0 | `step0_extract.py` | Decompresses `.tar.Z` archive files for Alabama using `unlzw3` |
| 1 | `step1_parse.py` | Parses fixed-width DSI-3260 `.dat` files; handles missing/deleted entries |
| 2 | `step2_ams.py` | Reconstructs 15-min time series; extracts AMS via rolling window for 6 durations |
| 3 | `step3_distributions.py` | Fits Gumbel (EV-I) and LP3 using Method of Moments from scratch |
| 4 | `step4_ks_test.py` | KS test at 5% significance; critical value = 1.36/√n |
| 5 | `step5_return_periods.py` | Computes intensities for T = 2–100 yr using best-fitting distribution |
| 6 | `step6_idf_formula.py` | Fits `i = a·T^b / d^e` via log-linearisation and least squares |
| 7 | `step7_plot.py` | Plots the full family of IDF curves |
| 8 | `step8_atlas14_compare.py` | Overlays derived IDF values against NOAA Atlas 14 Vol. 9 reference |
| 9 | `step9_sensitivity.py` | Side-by-side comparison of Gumbel vs LP3 estimates across all durations |
| 10 | `step10_trend_analysis.py` | Split-period IDF comparison: 1972–1992 vs 1993–2013 |
| 10b | `step10b_trend_no_outlier.py` | Same trend analysis with the confirmed outlier removed |
| 11 | `step11_outlier_detection.py` | Grubbs test (two-sided) on AMS; refits LP3 and quantifies % impact |
| 12 | `step12_bootstrap_ci.py` | Bootstrap confidence intervals for IDF estimates |
| 13 | `step13_cdf_plot.py` | CDF overlay plots comparing empirical, Gumbel, and LP3 fits |

---

## Outputs

All outputs are saved to the `output/` folder:

| File | Description |
|------|-------------|
| `clean_data.csv` | Parsed 43-year 15-min rainfall time series |
| `ams.csv` | Annual Maximum Series for all 6 durations |
| `params.csv` | Fitted Gumbel and LP3 parameters |
| `ks_results.csv` | KS test statistics and best-fit selection per duration |
| `idf_table.csv` | IDF intensities (mm/hr) for all durations and return periods |
| `idf_formula_params.csv` | Fitted parameters a, b, e for empirical IDF formula |
| `idf_curves.png` | IDF curve family plot |
| `idf_plot_curves.csv` | Data underlying the IDF curve plot |
| `atlas14_comparison.csv` | Derived vs Atlas 14 values with % difference |
| `atlas14_comparison.png` | Overlay plot |
| `sensitivity_comparison.csv` / `.png` | Gumbel vs LP3 intensity comparison |
| `trend_comparison.csv` / `_clean.csv` | Split-period trend results (with / without outlier) |
| `trend_analysis.png` / `_clean.png` | Trend analysis plots |
| `outlier_results.csv` | Grubbs test results per duration |
| `outlier_analysis.png` | Outlier visualisation |
| `idf_table_no_outlier.csv` | IDF table with outlier removed |
| `bootstrap_ci.csv` / `.png` | Bootstrap confidence intervals |
| `cdf_comparison.csv` / `.png` | CDF comparison for all distributions |

---

## Methods Summary

- **Distribution fitting:** Method of Moments for both Gumbel (EV-I) and Log-Pearson Type III; LP3 uses log₁₀-transformed data
- **Frequency factor:** Wilson-Hilferty (1931) cube-root approximation for LP3 quantiles
- **Goodness-of-fit:** Two-sided KS test; critical value at α = 0.05 is 1.36/√n
- **Empirical IDF formula:** `i = a·T^b / d^e` fitted via log-linearisation; solved as `β = (AᵀA)⁻¹Aᵀy` using NumPy
- **Outlier detection:** Grubbs test, `G = max|xᵢ − x̄| / s`; critical value via t-distribution approximation
- **Uncertainty:** Non-parametric bootstrap with 1000 resamples

---

## Reference

Koutsoyiannis, D., Kozonis, D., & Manetas, A. (1998). A mathematical framework for studying rainfall intensity-duration-frequency relationships. *Journal of Hydrology, 206*(1–2), 118–135. https://doi.org/10.1016/S0022-1694(98)00097-3
