"""
IDF Curve Project
Step 7: Plot the Final IDF Curves

What this does:
    - Reads computed IDF intensity values and fitted formula parameters
    - Plots the complete family of IDF curves on a log-log scale
    - Shows computed points (dots) and fitted formula curves (lines)
      for all 6 return periods and 6 durations
    - Saves the plot and underlying data as CSV

Input  : output/idf_table.csv, output/idf_formula_params.csv
Output : output/idf_curves.png
         output/idf_plot_points.csv
         output/idf_plot_curves.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Paths (relative to project root) ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

IDF_CSV        = os.path.join(OUTPUT_DIR, "idf_table.csv")
PARAMS_CSV     = os.path.join(OUTPUT_DIR, "idf_formula_params.csv")
OUTPUT_PNG     = os.path.join(OUTPUT_DIR, "idf_curves.png")
OUTPUT_PTS_CSV = os.path.join(OUTPUT_DIR, "idf_plot_points.csv")
OUTPUT_CRV_CSV = os.path.join(OUTPUT_DIR, "idf_plot_curves.csv")

DUR_MAP   = {"15min": 0.25, "30min": 0.5, "1hr": 1.0,
             "2hr": 2.0,   "6hr": 6.0,   "24hr": 24.0}
DUR_ORDER = ["15min", "30min", "1hr", "2hr", "6hr", "24hr"]
T_VALUES  = [2, 5, 10, 25, 50, 100]
COLORS    = ["#1f77b4", "#ff7f0e", "#2ca02c",
             "#d62728", "#9467bd", "#8c564b"]

idf    = pd.read_csv(IDF_CSV)
params = pd.read_csv(PARAMS_CSV).iloc[0]
a, b, e = params["a"], params["b"], params["e"]

# Save computed points CSV
idf["d_hours"] = idf["duration"].map(DUR_MAP)
idf.to_csv(OUTPUT_PTS_CSV, index=False)
print(f"Computed IDF points saved to: {OUTPUT_PTS_CSV}")

# Save fitted curve CSV
d_fine     = np.logspace(np.log10(0.25), np.log10(24), 200)
curve_rows = []
for T in T_VALUES:
    i_fine = a * T**b / d_fine**e
    for d_val, i_val in zip(d_fine, i_fine):
        curve_rows.append({
            "T_years":             T,
            "duration_hours":      round(d_val, 6),
            "intensity_mm_per_hr": round(i_val, 4)
        })
curves_df = pd.DataFrame(curve_rows)
curves_df.to_csv(OUTPUT_CRV_CSV, index=False)
print(f"Fitted curve data saved to:   {OUTPUT_CRV_CSV}")

# Plot
fig, ax = plt.subplots(figsize=(10, 7))

for T, color in zip(T_VALUES, COLORS):
    x_pts = [DUR_MAP[d] for d in DUR_ORDER]
    y_pts = []
    for d in DUR_ORDER:
        val = idf[(idf["duration"] == d) &
                  (idf["T_years"] == T)]["intensity_mm_per_hr"].values
        y_pts.append(val[0] if len(val) > 0 else np.nan)
    ax.plot(x_pts, y_pts, "o", color=color, markersize=6, zorder=3)

    i_fine = a * T**b / d_fine**e
    ax.plot(d_fine, i_fine, "-", color=color,
            linewidth=1.8, label=f"T = {T} years")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Duration", fontsize=12)
ax.set_ylabel("Rainfall Intensity (mm/hr)", fontsize=12)
ax.set_title(
    "IDF Curves — Alabama Station 01014000/01014007\n"
    f"NOAA DSI-3260 (1971–2013) | "
    f"i = {a:.2f} × T$^{{{b:.3f}}}$ / d$^{{{e:.3f}}}$",
    fontsize=12)
ax.legend(title="Return Period", fontsize=10)
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.set_xticks([0.25, 0.5, 1.0, 2.0, 6.0, 24.0])
ax.set_xticklabels(["15 min", "30 min", "1 hr",
                    "2 hr",   "6 hr",   "24 hr"],
                   rotation=30, ha="right")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
plt.show()
print(f"Plot saved to: {OUTPUT_PNG}")