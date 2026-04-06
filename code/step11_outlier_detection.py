"""
IDF Curve Project
Step 11: Outlier Detection using Grubbs Test

What this does:
    - Applies the Grubbs test (two-sided) to the AMS for each duration
      to formally identify statistically significant outliers
    - Grubbs statistic: G = max|xi - xbar| / s
    - Critical value approximated using t-distribution
    - For each duration where an outlier is detected:
        - Refits LP3 with the outlier removed
        - Quantifies the impact on IDF estimates (% change)
    - All methods coded from scratch — no scipy used

Input  : output/ams.csv, output/params.csv
Output : output/outlier_analysis.png
         output/outlier_results.csv
         output/idf_table_no_outlier.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Paths (relative to project root) ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

AMS_CSV        = os.path.join(OUTPUT_DIR, "ams.csv")
PARAMS_CSV     = os.path.join(OUTPUT_DIR, "params.csv")
OUTPUT_PNG     = os.path.join(OUTPUT_DIR, "outlier_analysis.png")
OUTPUT_CSV     = os.path.join(OUTPUT_DIR, "outlier_results.csv")
OUTPUT_IDF_CSV = os.path.join(OUTPUT_DIR, "idf_table_no_outlier.csv")

RETURN_PERIODS = [2, 5, 10, 25, 50, 100]


def inv_normal_cdf(p):
    """Rational approximation for inverse standard normal CDF."""
    if p <= 0.0: return -8.0
    if p >= 1.0: return  8.0
    if p < 0.5:  return -inv_normal_cdf(1.0 - p)
    t = np.sqrt(-2.0 * np.log(1.0 - p))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - (c[0] + c[1]*t + c[2]*t**2) / \
               (1.0 + d[0]*t + d[1]*t**2 + d[2]*t**3)


def t_inv_approx(p, df):
    """
    Approximate inverse t-distribution CDF using Cornish-Fisher expansion.
    Used for computing Grubbs critical value.
    """
    z = inv_normal_cdf(p)
    t_approx = z + (z**3 + z) / (4*df) + \
               (5*z**5 + 16*z**3 + 3*z) / (96*df**2)
    return t_approx


def grubbs_test(data, alpha=0.05):
    """
    Two-sided Grubbs test for a single outlier at significance level alpha.

    G_stat = max|xi - xbar| / s
    G_crit = ((n-1)/sqrt(n)) * sqrt(t^2 / (n-2+t^2))
    where t is the t critical value at alpha/(2n), df=n-2

    Returns
    -------
    G_stat      : Grubbs test statistic
    G_crit      : Critical value at alpha=0.05
    outlier_idx : Index of the suspected outlier in data array
    is_outlier  : True if G_stat > G_crit
    """
    n    = len(data)
    xbar = np.mean(data)
    s    = np.std(data, ddof=1)

    deviations  = np.abs(data - xbar)
    G_stat      = np.max(deviations) / s
    outlier_idx = int(np.argmax(deviations))

    p_t    = alpha / (2 * n)
    t_crit = t_inv_approx(1.0 - p_t, df=n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * \
             np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

    return G_stat, G_crit, outlier_idx, bool(G_stat > G_crit)


def lp3_params(data):
    """LP3 parameters via Method of Moments on log10-transformed data."""
    y    = np.log10(data)
    ybar = np.mean(y)
    sy   = np.std(y, ddof=1)
    n    = len(y)
    g    = (n * np.sum((y - ybar)**3)) / ((n - 1) * (n - 2) * sy**3)
    return ybar, sy, g


def kt_wilson_hilferty(g, T):
    """Wilson-Hilferty frequency factor K_T for LP3."""
    p = 1.0 - 1.0 / T
    z = inv_normal_cdf(p)
    if abs(g) < 1e-6:
        return z
    return (2.0 / g) * ((1.0 + g*z/6.0 - g**2/36.0)**3 - 1.0)


def lp3_quantile(T, ybar, sy, g):
    """LP3 quantile: x_T = 10^(ybar + K_T * sy)"""
    return 10.0 ** (ybar + kt_wilson_hilferty(g, T) * sy)


# --- Main ---
ams    = pd.read_csv(AMS_CSV, index_col="year")
params = pd.read_csv(PARAMS_CSV)

durations  = ams.columns.tolist()
out_rows   = []
idf_rows   = []

print(f"{'Duration':<10} {'G_stat':>8} {'G_crit':>8} "
      f"{'Outlier?':>10} {'Outlier Year':>14} {'Value (mm/hr)':>15}")
print("-" * 70)

for dur in durations:
    data  = ams[dur].dropna()
    years = data.index.values
    vals  = data.values

    G_stat, G_crit, idx, is_outlier = grubbs_test(vals)

    out_year = int(years[idx]) if is_outlier else None
    out_val  = float(vals[idx]) if is_outlier else None

    print(f"{dur:<10} {G_stat:>8.4f} {G_crit:>8.4f} "
          f"{'YES' if is_outlier else 'no':>10} "
          f"{str(out_year) if out_year else '-':>14} "
          f"{round(out_val, 3) if out_val else '-':>15}")

    out_rows.append({
        "duration":     dur,
        "G_stat":       round(G_stat, 4),
        "G_crit":       round(G_crit, 4),
        "is_outlier":   is_outlier,
        "outlier_year": out_year,
        "outlier_value":round(out_val, 3) if out_val else None
    })

    # Fit LP3 with and without outlier
    ybar_full, sy_full, g_full = lp3_params(vals)
    clean = np.delete(vals, idx) if is_outlier else vals
    ybar_clean, sy_clean, g_clean = lp3_params(clean)

    for T in RETURN_PERIODS:
        i_full  = lp3_quantile(T, ybar_full,  sy_full,  g_full)
        i_clean = lp3_quantile(T, ybar_clean, sy_clean, g_clean)
        pct     = (i_clean - i_full) / i_full * 100
        idf_rows.append({
            "duration":                  dur,
            "T_years":                   T,
            "intensity_with_outlier":    round(i_full,  3),
            "intensity_without_outlier": round(i_clean, 3),
            "pct_change":                round(pct, 2)
        })

# Save CSVs
outlier_df = pd.DataFrame(out_rows)
outlier_df.to_csv(OUTPUT_CSV, index=False)

idf_df = pd.DataFrame(idf_rows)
idf_df.to_csv(OUTPUT_IDF_CSV, index=False)

# Print impact table
print("\nImpact of outlier removal on IDF intensities (% change):")
print("-" * 65)
print(f"{'Duration':<10} {'T=2':>8} {'T=5':>8} {'T=10':>8} "
      f"{'T=25':>8} {'T=50':>8} {'T=100':>8}")
print("-" * 65)
for dur in durations:
    sub  = idf_df[idf_df["duration"] == dur].sort_values("T_years")
    pcts = sub["pct_change"].values
    print(f"{dur:<10} " + " ".join(f"{v:>+7.1f}%" for v in pcts))

# --- Plot ---
DUR_ORDER  = ["15min", "30min", "1hr", "2hr", "6hr", "24hr"]
DUR_LABELS = ["15 min", "30 min", "1 hr", "2 hr", "6 hr", "24 hr"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
axes = axes.flatten()

for ax, dur, dlabel in zip(axes, DUR_ORDER, DUR_LABELS):
    sub     = idf_df[idf_df["duration"] == dur].sort_values("T_years")
    out_row = outlier_df[outlier_df["duration"] == dur].iloc[0]
    T       = sub["T_years"].values
    i_w     = sub["intensity_with_outlier"].values
    i_c     = sub["intensity_without_outlier"].values
    is_out  = out_row["is_outlier"]

    ax.plot(T, i_w, "o-",  color="#d62728", linewidth=2.0,
            markersize=6, label="With outlier")
    ax.plot(T, i_c, "s--", color="#1f77b4", linewidth=2.0,
            markersize=6, label="Without outlier")
    ax.fill_between(T, i_w, i_c, alpha=0.12, color="gray")

    ax.set_xscale("log")
    ax.set_xticks(RETURN_PERIODS)
    ax.set_xticklabels([str(t) for t in RETURN_PERIODS])
    ax.set_xlabel("Return Period (years)", fontsize=9)
    ax.set_ylabel("Intensity (mm/hr)", fontsize=9)

    if is_out:
        title_str = (f"Duration: {dlabel}\n"
                     f"Outlier: year {int(out_row['outlier_year'])} "
                     f"({out_row['outlier_value']:.1f} mm/hr)")
    else:
        title_str = f"Duration: {dlabel}\nNo outlier detected"

    ax.set_title(title_str, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

fig.suptitle(
    "Outlier Detection (Grubbs Test) — Impact on IDF Curves\n"
    "Alabama Station 01014000/01014007 (1971–2013)",
    fontsize=12, y=1.01)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved to:   {OUTPUT_PNG}")
print(f"Outlier results: {OUTPUT_CSV}")
print(f"IDF comparison:  {OUTPUT_IDF_CSV}")