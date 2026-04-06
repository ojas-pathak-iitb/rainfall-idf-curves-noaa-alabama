"""
IDF Curve Project
Step 12: Bootstrap 95% Confidence Intervals on IDF Estimates

What this does:
    - Resamples the AMS with replacement 1000 times for each duration
    - Refits LP3 distribution to each resample
    - Computes the 2.5th and 97.5th percentile of bootstrap estimates
      to form 95% confidence intervals around each IDF point
    - Wide CI = high uncertainty (driven by outliers or short record)
    - Narrow CI = reliable estimate
    - Bootstrap procedure coded from scratch using NumPy only

Input  : output/ams.csv
Output : output/bootstrap_ci.png
         output/bootstrap_ci.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Paths (relative to project root) ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

AMS_CSV    = os.path.join(OUTPUT_DIR, "ams.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "bootstrap_ci.png")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "bootstrap_ci.csv")

RETURN_PERIODS = [2, 5, 10, 25, 50, 100]
N_BOOTSTRAP    = 1000
SEED           = 42
np.random.seed(SEED)


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
ams       = pd.read_csv(AMS_CSV, index_col="year")
DUR_ORDER = ["15min", "30min", "1hr", "2hr", "6hr", "24hr"]
DUR_LABELS = ["15 min", "30 min", "1 hr", "2 hr", "6 hr", "24 hr"]

all_rows = []
results  = {}

print(f"Running {N_BOOTSTRAP} bootstrap resamples per duration...\n")

for dur in DUR_ORDER:
    data = ams[dur].dropna().values
    n    = len(data)
    results[dur] = {T: np.zeros(N_BOOTSTRAP) for T in RETURN_PERIODS}

    for b in range(N_BOOTSTRAP):
        # Resample with replacement
        sample      = data[np.random.randint(0, n, size=n)]
        ybar, sy, g = lp3_params(sample)
        for T in RETURN_PERIODS:
            results[dur][T][b] = lp3_quantile(T, ybar, sy, g)

    # Point estimate from original data
    ybar0, sy0, g0 = lp3_params(data)

    for T in RETURN_PERIODS:
        boot_vals = results[dur][T]
        point_est = lp3_quantile(T, ybar0, sy0, g0)
        ci_lo     = float(np.percentile(boot_vals, 2.5))
        ci_hi     = float(np.percentile(boot_vals, 97.5))
        all_rows.append({
            "duration":   dur,
            "T_years":    T,
            "point_est":  round(point_est, 3),
            "ci_lo_2p5":  round(ci_lo,     3),
            "ci_hi_97p5": round(ci_hi,     3),
            "ci_width":   round(ci_hi - ci_lo, 3)
        })

    print(f"  {dur}: done. "
          f"CI width at T=100yr = "
          f"{results[dur][100].max() - results[dur][100].min():.1f} mm/hr range")

ci_df = pd.DataFrame(all_rows)
ci_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nCSV saved to: {OUTPUT_CSV}")

# --- Plot ---
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
axes = axes.flatten()

for ax, dur, dlabel in zip(axes, DUR_ORDER, DUR_LABELS):
    sub = ci_df[ci_df["duration"] == dur].sort_values("T_years")
    T   = sub["T_years"].values
    pt  = sub["point_est"].values
    lo  = sub["ci_lo_2p5"].values
    hi  = sub["ci_hi_97p5"].values

    ax.plot(T, pt, "o-", color="#d62728", linewidth=2.0,
            markersize=6, zorder=3, label="LP3 estimate")
    ax.fill_between(T, lo, hi, alpha=0.25, color="#1f77b4",
                    label="95% CI (bootstrap)")
    ax.plot(T, lo, "--", color="#1f77b4", linewidth=1.0)
    ax.plot(T, hi, "--", color="#1f77b4", linewidth=1.0)

    ax.set_xscale("log")
    ax.set_xticks(RETURN_PERIODS)
    ax.set_xticklabels([str(t) for t in RETURN_PERIODS])
    ax.set_xlabel("Return Period (years)", fontsize=9)
    ax.set_ylabel("Intensity (mm/hr)", fontsize=9)
    ax.set_title(f"Duration: {dlabel}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

fig.suptitle(
    f"Bootstrap 95% Confidence Intervals on IDF Estimates "
    f"(n = {N_BOOTSTRAP} resamples)\n"
    "Alabama Station 01014000/01014007 (1971–2013) | LP3 Distribution",
    fontsize=12, y=1.01)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved to: {OUTPUT_PNG}")