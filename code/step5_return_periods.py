"""
IDF Curve Project
Step 5: Compute Return Period Rainfall Intensities

What this does:
    - Reads fitted distribution parameters and KS test results
    - For each duration, uses the best-fitting distribution (from KS test)
      to estimate rainfall intensity for return periods T = 2, 5, 10, 25, 50, 100 years
    - Gumbel quantile: x_T = mu - sigma * ln(-ln(1 - 1/T))
    - LP3 quantile: x_T = 10^(ybar + K_T * sy)
      where K_T is the Wilson-Hilferty frequency factor
    - All methods coded from scratch — no scipy used

Input  : output/params.csv, output/ks_results.csv
Output : output/idf_table.csv
"""

import os
import pandas as pd
import numpy as np

# --- Paths (relative to project root) ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

PARAMS_CSV = os.path.join(OUTPUT_DIR, "params.csv")
KS_CSV     = os.path.join(OUTPUT_DIR, "ks_results.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "idf_table.csv")

RETURN_PERIODS = [2, 5, 10, 25, 50, 100]


def gumbel_quantile(T, mu, sigma):
    """
    Gumbel (EV-I) quantile for return period T.
    x_T = mu - sigma * ln(-ln(1 - 1/T))
    """
    return mu - sigma * np.log(-np.log(1.0 - 1.0 / T))


def inv_normal_cdf(p):
    """
    Rational approximation for the inverse standard normal CDF.
    Based on Abramowitz and Stegun (1964).
    """
    if p <= 0.0: return -8.0
    if p >= 1.0: return  8.0
    if p < 0.5:  return -inv_normal_cdf(1.0 - p)
    t = np.sqrt(-2.0 * np.log(1.0 - p))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - (c[0] + c[1]*t + c[2]*t**2) / \
               (1.0 + d[0]*t + d[1]*t**2 + d[2]*t**3)


def kt_wilson_hilferty(g, T):
    """
    Frequency factor K_T for LP3 using Wilson-Hilferty approximation.
    K_T = (2/g) * [(1 + g*z/6 - g^2/36)^3 - 1]
    """
    p = 1.0 - 1.0 / T
    z = inv_normal_cdf(p)
    if abs(g) < 1e-6:
        return z
    return (2.0 / g) * ((1.0 + g*z/6.0 - g**2/36.0)**3 - 1.0)


def lp3_quantile(T, ybar, sy, g):
    """
    LP3 quantile for return period T.
    x_T = 10^(ybar + K_T * sy)
    """
    return 10.0 ** (ybar + kt_wilson_hilferty(g, T) * sy)


# --- Main ---
params = pd.read_csv(PARAMS_CSV)
ks     = pd.read_csv(KS_CSV)

rows = []

for _, p_row in params.iterrows():
    dur      = p_row["duration"]
    best_fit = ks[ks["duration"] == dur]["best_fit"].values[0]

    for T in RETURN_PERIODS:
        if best_fit == "Gumbel":
            intensity = gumbel_quantile(T, p_row["gumbel_mu"],
                                           p_row["gumbel_sigma"])
        else:
            intensity = lp3_quantile(T, p_row["lp3_ybar"],
                                        p_row["lp3_sy"],
                                        p_row["lp3_g"])
        rows.append({
            "duration":            dur,
            "T_years":             T,
            "best_fit_dist":       best_fit,
            "intensity_mm_per_hr": round(intensity, 3)
        })

idf_df = pd.DataFrame(rows)
idf_df.to_csv(OUTPUT_CSV, index=False)

print("IDF Table (intensity in mm/hr):")
print("-" * 60)
pivot = idf_df.pivot(index="duration", columns="T_years",
                     values="intensity_mm_per_hr")
print(pivot.to_string())
print(f"\nIDF table saved to: {OUTPUT_CSV}")