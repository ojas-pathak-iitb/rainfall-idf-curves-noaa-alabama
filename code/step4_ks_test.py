"""
IDF Curve Project
Step 4: Kolmogorov-Smirnov Goodness-of-Fit Test

What this does:
    - Reads AMS data and fitted distribution parameters
    - Computes the KS statistic for both Gumbel and LP3 for each duration
    - KS statistic = max absolute difference between empirical and
      theoretical CDF
    - Critical value at 5% significance = 1.36 / sqrt(n)
    - Identifies which distribution fits better per duration
    - All methods coded from scratch — no scipy used

Input  : output/ams.csv, output/params.csv
Output : output/ks_results.csv
"""

import os
import pandas as pd
import numpy as np

# --- Paths (relative to project root) ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

AMS_CSV    = os.path.join(OUTPUT_DIR, "ams.csv")
PARAMS_CSV = os.path.join(OUTPUT_DIR, "params.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ks_results.csv")


def gumbel_cdf(x, mu, sigma):
    """
    Cumulative Distribution Function of the Gumbel (EV-I) distribution.
    F(x) = exp(-exp(-(x - mu) / sigma))
    """
    return np.exp(-np.exp(-(x - mu) / sigma))


def inv_normal_cdf(p):
    """
    Rational approximation for the inverse standard normal CDF.
    Accurate to about 4 decimal places.
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
    Frequency factor K_T for the Pearson Type III distribution,
    computed using the Wilson-Hilferty (1931) cube-root approximation.
    K_T = (2/g) * [(1 + g*z/6 - g^2/36)^3 - 1]
    """
    p = 1.0 - 1.0 / T
    z = inv_normal_cdf(p)
    if abs(g) < 1e-6:
        return z
    return (2.0 / g) * ((1.0 + g*z/6.0 - g**2/36.0)**3 - 1.0)


def lp3_quantile(T, ybar, sy, g):
    """Return LP3 intensity estimate for return period T."""
    return 10.0 ** (ybar + kt_wilson_hilferty(g, T) * sy)


def lp3_cdf(x_vals, ybar, sy, g):
    """
    Compute LP3 CDF at each value in x_vals.
    Uses a vectorised probability grid and interpolation.
    """
    p_grid = np.linspace(0.0001, 0.9999, 50000)
    z_grid = np.array([inv_normal_cdf(p) for p in p_grid])

    if abs(g) < 1e-6:
        kt_grid = z_grid
    else:
        kt_grid = (2.0/g) * ((1.0 + g*z_grid/6.0 - g**2/36.0)**3 - 1.0)

    x_grid   = 10.0 ** (ybar + kt_grid * sy)
    sort_idx = np.argsort(x_grid)
    x_sorted = x_grid[sort_idx]
    p_sorted = p_grid[sort_idx]

    return np.interp(x_vals, x_sorted, p_sorted)


def ks_statistic(data_sorted, cdf_values):
    """
    Compute the two-sided KS statistic.
    D = max |F_empirical(x_i) - F_theoretical(x_i)|
    Critical value at alpha=0.05: D_crit = 1.36 / sqrt(n)
    """
    n       = len(data_sorted)
    emp_cdf = np.arange(1, n+1) / n
    D       = np.max(np.abs(emp_cdf - cdf_values))
    D_crit  = 1.36 / np.sqrt(n)
    return D, D_crit, bool(D < D_crit)


# --- Main ---
ams    = pd.read_csv(AMS_CSV,    index_col="year")
params = pd.read_csv(PARAMS_CSV)

results = []

for _, row in params.iterrows():
    dur  = row["duration"]
    data = np.sort(ams[dur].dropna().values)

    # Gumbel KS
    g_cdf_vals      = gumbel_cdf(data, row["gumbel_mu"], row["gumbel_sigma"])
    D_g, Dc, pass_g = ks_statistic(data, g_cdf_vals)

    # LP3 KS
    l_cdf_vals      = lp3_cdf(data, row["lp3_ybar"], row["lp3_sy"], row["lp3_g"])
    D_l, Dc, pass_l = ks_statistic(data, l_cdf_vals)

    best = "LP3" if D_l <= D_g else "Gumbel"

    print(f"{dur}: "
          f"Gumbel D = {D_g:.4f} ({'PASS' if pass_g else 'FAIL'}), "
          f"LP3 D = {D_l:.4f} ({'PASS' if pass_l else 'FAIL'}), "
          f"D_crit = {Dc:.4f} | Best fit: {best}")

    results.append({
        "duration":      dur,
        "D_gumbel":      round(D_g, 4),
        "D_lp3":         round(D_l, 4),
        "D_critical":    round(Dc,  4),
        "gumbel_passes": pass_g,
        "lp3_passes":    pass_l,
        "best_fit":      best
    })

ks_df = pd.DataFrame(results)
ks_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nKS results saved to: {OUTPUT_CSV}")