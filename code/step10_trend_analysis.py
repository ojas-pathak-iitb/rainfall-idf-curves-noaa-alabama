"""
IDF Curve Project
Step 10: Temporal Trend Analysis — Split Period Comparison

What this does:
    - Splits the 43-year AMS into two equal periods:
        Period 1: 1971-1992 (22 years)
        Period 2: 1993-2013 (21 years)
    - Fits LP3 distribution separately to each period
    - Computes IDF intensities for both periods
    - Shows percentage change from Period 1 to Period 2
    - NOTE: Results include the 1971 outlier. See step10b for
      the clean version with the outlier removed.

Input  : output/ams.csv
Output : output/trend_analysis.png
         output/trend_comparison.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Paths (relative to project root) ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

AMS_CSV    = os.path.join(OUTPUT_DIR, "ams.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "trend_analysis.png")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "trend_comparison.csv")

SPLIT_YEAR     = 1992
RETURN_PERIODS = [2, 5, 10, 25, 50, 100]


def lp3_params(data):
    """LP3 parameters via Method of Moments on log10-transformed data."""
    y    = np.log10(data)
    ybar = np.mean(y)
    sy   = np.std(y, ddof=1)
    n    = len(y)
    g    = (n * np.sum((y - ybar)**3)) / ((n - 1) * (n - 2) * sy**3)
    return ybar, sy, g


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
ams = pd.read_csv(AMS_CSV, index_col="year")

period1 = ams[ams.index <= SPLIT_YEAR]
period2 = ams[ams.index >  SPLIT_YEAR]

print(f"Period 1: {period1.index.min()} -- {period1.index.max()} "
      f"({len(period1)} years)")
print(f"Period 2: {period2.index.min()} -- {period2.index.max()} "
      f"({len(period2)} years)")

durations = ams.columns.tolist()
rows      = []

for dur in durations:
    for label, period in [("1971-1992", period1), ("1993-2013", period2)]:
        data        = period[dur].dropna().values
        ybar, sy, g = lp3_params(data)
        for T in RETURN_PERIODS:
            rows.append({
                "period":              label,
                "duration":            dur,
                "T_years":             T,
                "intensity_mm_per_hr": round(lp3_quantile(T, ybar, sy, g), 3)
            })

trend_df = pd.DataFrame(rows)
trend_df.to_csv(OUTPUT_CSV, index=False)

# Print percentage change table
print("\nPercentage change from Period 1 to Period 2:")
print("-" * 65)
print(f"{'Duration':<10} {'T=2':>8} {'T=5':>8} {'T=10':>8} "
      f"{'T=25':>8} {'T=50':>8} {'T=100':>8}")
print("-" * 65)
for dur in durations:
    p1  = trend_df[(trend_df["period"] == "1971-1992") &
                   (trend_df["duration"] == dur)].set_index("T_years")
    p2  = trend_df[(trend_df["period"] == "1993-2013") &
                   (trend_df["duration"] == dur)].set_index("T_years")
    pct = [(p2.loc[T, "intensity_mm_per_hr"] -
            p1.loc[T, "intensity_mm_per_hr"]) /
            p1.loc[T, "intensity_mm_per_hr"] * 100
           for T in RETURN_PERIODS]
    print(f"{dur:<10} " + " ".join(f"{v:>+7.1f}%" for v in pct))

# --- Plot ---
DUR_ORDER  = ["15min", "30min", "1hr", "2hr", "6hr", "24hr"]
DUR_LABELS = ["15 min", "30 min", "1 hr", "2 hr", "6 hr", "24 hr"]
COLOR1     = "#1f77b4"
COLOR2     = "#d62728"

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
axes = axes.flatten()

for ax, dur, dlabel in zip(axes, DUR_ORDER, DUR_LABELS):
    p1 = trend_df[(trend_df["period"] == "1971-1992") &
                  (trend_df["duration"] == dur)].sort_values("T_years")
    p2 = trend_df[(trend_df["period"] == "1993-2013") &
                  (trend_df["duration"] == dur)].sort_values("T_years")
    T  = p1["T_years"].values
    i1 = p1["intensity_mm_per_hr"].values
    i2 = p2["intensity_mm_per_hr"].values

    ax.plot(T, i1, "o-", color=COLOR1, linewidth=2.0,
            markersize=6, label="1971–1992")
    ax.plot(T, i2, "s-", color=COLOR2, linewidth=2.0,
            markersize=6, label="1993–2013")
    ax.fill_between(T, i1, i2, where=(i2 >= i1),
                    alpha=0.15, color=COLOR2, label="Increase")
    ax.fill_between(T, i1, i2, where=(i2 <  i1),
                    alpha=0.15, color=COLOR1, label="Decrease")

    ax.set_xscale("log")
    ax.set_xticks(RETURN_PERIODS)
    ax.set_xticklabels([str(t) for t in RETURN_PERIODS])
    ax.set_xlabel("Return Period (years)", fontsize=9)
    ax.set_ylabel("Intensity (mm/hr)", fontsize=9)
    ax.set_title(f"Duration: {dlabel}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

fig.suptitle(
    "Temporal Trend Analysis — Split Period Comparison (LP3)\n"
    "Alabama Station 01014000/01014007 | "
    "Period 1: 1971–1992 vs Period 2: 1993–2013\n"
    "(Note: includes 1971 outlier — see step10b for clean version)",
    fontsize=11, y=1.01)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved to: {OUTPUT_PNG}")
print(f"CSV  saved to: {OUTPUT_CSV}")