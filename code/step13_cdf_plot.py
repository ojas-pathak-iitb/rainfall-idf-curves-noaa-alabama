"""
IDF Curve Project
Step 13: Empirical vs Fitted CDF Plot

What this does:
    - For each duration, plots the empirical CDF of the AMS data
      against the theoretical CDFs of the fitted Gumbel and LP3 distributions
    - Empirical CDF uses the Weibull plotting position: F(x_i) = i / (n+1)
    - This is the visual equivalent of the KS test — shows where each
      distribution fits well and where it deviates from the data
    - The isolated outlier point (1971 event) is clearly visible as
      a dot far to the right of the main data cloud in short durations

Input  : output/ams.csv, output/params.csv
Output : output/cdf_comparison.png
         output/cdf_comparison.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Paths (relative to project root) ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

AMS_CSV    = os.path.join(OUTPUT_DIR, "ams.csv")
PARAMS_CSV = os.path.join(OUTPUT_DIR, "params.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "cdf_comparison.png")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "cdf_comparison.csv")


def gumbel_cdf(x, mu, sigma):
    """
    Gumbel (EV-I) CDF: F(x) = exp(-exp(-(x - mu) / sigma))
    """
    return np.exp(-np.exp(-(x - mu) / sigma))


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


def lp3_cdf_vectorised(x_vals, ybar, sy, g):
    """
    Compute LP3 CDF at each value in x_vals using a vectorised
    probability grid and interpolation.
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


# --- Main ---
ams    = pd.read_csv(AMS_CSV, index_col="year")
params = pd.read_csv(PARAMS_CSV)

DUR_ORDER  = ["15min", "30min", "1hr", "2hr", "6hr", "24hr"]
DUR_LABELS = ["15 min", "30 min", "1 hr", "2 hr", "6 hr", "24 hr"]

csv_rows = []

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for ax, dur, dlabel in zip(axes, DUR_ORDER, DUR_LABELS):
    data  = np.sort(ams[dur].dropna().values)
    n     = len(data)
    p_row = params[params["duration"] == dur].iloc[0]

    # Empirical CDF using Weibull plotting position: i / (n+1)
    emp_cdf = np.arange(1, n+1) / (n + 1)

    # Theoretical CDFs over a smooth x range
    x_fine     = np.linspace(data.min() * 0.8, data.max() * 1.2, 500)
    gumbel_fit = gumbel_cdf(x_fine, p_row["gumbel_mu"],
                                     p_row["gumbel_sigma"])
    lp3_fit    = lp3_cdf_vectorised(x_fine, p_row["lp3_ybar"],
                                             p_row["lp3_sy"],
                                             p_row["lp3_g"])

    # Plot
    ax.plot(x_fine, gumbel_fit, "-",  color="#1f77b4",
            linewidth=1.8, label="Gumbel (fitted)")
    ax.plot(x_fine, lp3_fit,   "--", color="#d62728",
            linewidth=1.8, label="LP3 (fitted)")
    ax.plot(data, emp_cdf, "o", color="black",
            markersize=5, zorder=5, label="Empirical (Weibull)")

    ax.set_xlabel("Rainfall Intensity (mm/hr)", fontsize=9)
    ax.set_ylabel("Cumulative Probability", fontsize=9)
    ax.set_title(f"Duration: {dlabel}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.02)

    # Save to CSV rows
    for xi, pi in zip(data, emp_cdf):
        g_val = float(gumbel_cdf(xi, p_row["gumbel_mu"],
                                     p_row["gumbel_sigma"]))
        l_val = float(lp3_cdf_vectorised(
                      np.array([xi]),
                      p_row["lp3_ybar"],
                      p_row["lp3_sy"],
                      p_row["lp3_g"])[0])
        csv_rows.append({
            "duration":        dur,
            "intensity_mm_hr": round(xi,    3),
            "empirical_cdf":   round(pi,    4),
            "gumbel_cdf":      round(g_val, 4),
            "lp3_cdf":         round(l_val, 4),
            "gumbel_error":    round(abs(pi - g_val), 4),
            "lp3_error":       round(abs(pi - l_val), 4)
        })

fig.suptitle(
    "Empirical vs Fitted CDF — Gumbel and LP3\n"
    "Alabama Station 01014000/01014007 (1971–2013)",
    fontsize=12, y=1.01)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved to: {OUTPUT_PNG}")

cdf_df = pd.DataFrame(csv_rows)
cdf_df.to_csv(OUTPUT_CSV, index=False)
print(f"CSV  saved to: {OUTPUT_CSV}")