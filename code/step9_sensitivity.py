"""
IDF Curve Project
Step 9: Sensitivity Analysis — Gumbel vs LP3

What this does:
    - Computes IDF intensity estimates using BOTH Gumbel and LP3
      for all durations and return periods, regardless of which
      distribution was selected as best fit
    - Shows the percentage difference between LP3 and Gumbel estimates
    - Highlights where distribution choice has significant impact
    - Produces a 6-panel subplot, one per duration

Input  : output/params.csv, output/ks_results.csv
Output : output/sensitivity_comparison.png
         output/sensitivity_comparison.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Paths (relative to project root) ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

PARAMS_CSV = os.path.join(OUTPUT_DIR, "params.csv")
KS_CSV     = os.path.join(OUTPUT_DIR, "ks_results.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "sensitivity_comparison.png")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "sensitivity_comparison.csv")

RETURN_PERIODS = [2, 5, 10, 25, 50, 100]


def gumbel_quantile(T, mu, sigma):
    """Gumbel quantile: x_T = mu - sigma * ln(-ln(1 - 1/T))"""
    return mu - sigma * np.log(-np.log(1.0 - 1.0 / T))


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
params = pd.read_csv(PARAMS_CSV)
ks     = pd.read_csv(KS_CSV)

rows = []

for _, p_row in params.iterrows():
    dur    = p_row["duration"]
    ks_row = ks[ks["duration"] == dur].iloc[0]

    for T in RETURN_PERIODS:
        i_gumbel = gumbel_quantile(T, p_row["gumbel_mu"],
                                      p_row["gumbel_sigma"])
        i_lp3    = lp3_quantile(T, p_row["lp3_ybar"],
                                   p_row["lp3_sy"],
                                   p_row["lp3_g"])
        pct_diff = round((i_lp3 - i_gumbel) / i_gumbel * 100, 2)

        rows.append({
            "duration":                dur,
            "T_years":                 T,
            "gumbel_intensity":        round(i_gumbel, 3),
            "lp3_intensity":           round(i_lp3,    3),
            "pct_diff_lp3_vs_gumbel":  pct_diff,
            "gumbel_passes_ks":        ks_row["gumbel_passes"],
            "lp3_passes_ks":           ks_row["lp3_passes"],
            "best_fit":                ks_row["best_fit"]
        })

sens_df = pd.DataFrame(rows)
sens_df.to_csv(OUTPUT_CSV, index=False)

# Print summary table
print("Sensitivity table (% difference LP3 vs Gumbel):")
pivot = sens_df.pivot(index="duration", columns="T_years",
                      values="pct_diff_lp3_vs_gumbel")
print(pivot.to_string())

# --- Plot ---
DUR_ORDER  = ["15min", "30min", "1hr", "2hr", "6hr", "24hr"]
DUR_LABELS = ["15 min", "30 min", "1 hr", "2 hr", "6 hr", "24 hr"]
COLOR_G    = "#1f77b4"   # blue  — Gumbel
COLOR_L    = "#d62728"   # red   — LP3

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
axes = axes.flatten()

for ax, dur, dlabel in zip(axes, DUR_ORDER, DUR_LABELS):
    sub    = sens_df[sens_df["duration"] == dur].sort_values("T_years")
    ks_row = ks[ks["duration"] == dur].iloc[0]
    T      = sub["T_years"].values
    i_g    = sub["gumbel_intensity"].values
    i_l    = sub["lp3_intensity"].values

    pass_g = ks_row["gumbel_passes"]
    pass_l = ks_row["lp3_passes"]
    best   = ks_row["best_fit"]

    ax.plot(T, i_g, "o--", color=COLOR_G, linewidth=1.8,
            markersize=6, label=f"Gumbel {'✓' if pass_g else '✗'}")
    ax.plot(T, i_l, "s-",  color=COLOR_L, linewidth=1.8,
            markersize=6, label=f"LP3 {'✓' if pass_l else '✗'}")
    ax.fill_between(T, i_g, i_l, alpha=0.12, color="gray")

    ax.set_xscale("log")
    ax.set_xticks(RETURN_PERIODS)
    ax.set_xticklabels([str(t) for t in RETURN_PERIODS])
    ax.set_xlabel("Return Period (years)", fontsize=9)
    ax.set_ylabel("Intensity (mm/hr)", fontsize=9)
    ax.set_title(f"Duration: {dlabel}\n(Best fit: {best})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

fig.suptitle(
    "Sensitivity Analysis — Gumbel vs LP3\n"
    "Alabama Station 01014000/01014007 (1971–2013)",
    fontsize=12, y=1.01)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved to: {OUTPUT_PNG}")
print(f"CSV  saved to: {OUTPUT_CSV}")