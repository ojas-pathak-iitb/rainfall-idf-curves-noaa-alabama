"""
IDF Curve Project
Step 8: Compare Derived IDF Values Against NOAA Atlas 14

What this does:
    - Compares the IDF intensities derived in this study against
      officially published NOAA Atlas 14 values for the same location
    - Atlas 14 source: NOAA Atlas 14, Volume 9, Version 2
      Location: Randolph, Alabama (Lat 32.8879, Lon -86.8772)
      URL: https://hdsc.nws.noaa.gov/pfds/pfds_printpage.html?lat=32.8879&lon=-86.8772
    - Atlas 14 uses Partial Duration Series (PDS); this study uses AMS.
      For T >= 10yr the two methods give nearly identical results.
      For T = 2yr, PDS gives slightly higher values — expected difference.
    - Computes percentage difference per duration and return period
    - Produces overlay plot and comparison table

Input  : output/idf_table.csv
Output : output/atlas14_comparison.png
         output/atlas14_comparison.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# --- Paths (relative to project root) ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

IDF_CSV    = os.path.join(OUTPUT_DIR, "idf_table.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "atlas14_comparison.png")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "atlas14_comparison.csv")

# ---------------------------------------------------------------
# NOAA Atlas 14, Volume 9, Version 2
# Location: Randolph, Alabama (Lat 32.8879, Lon -86.8772)
# Retrieved: 17 March 2026
# Values are rainfall DEPTHS in inches per duration
# Source: https://hdsc.nws.noaa.gov/pfds/
# ---------------------------------------------------------------
INCH_TO_MM = 25.4

atlas14_depth_inches = {
    "15min": {2: 0.911, 5: 1.10,  10: 1.25,  25: 1.46,  50: 1.61,  100: 1.77},
    "30min": {2: 1.32,  5: 1.60,  10: 1.82,  25: 2.12,  50: 2.58,  100: 2.81},
    "1hr":   {2: 1.72,  5: 2.07,  10: 2.36,  25: 2.78,  50: 3.10,  100: 3.44},
    "2hr":   {2: 2.12,  5: 2.54,  10: 2.90,  25: 3.43,  50: 3.85,  100: 4.30},
    "6hr":   {2: 2.88,  5: 3.46,  10: 4.00,  25: 4.83,  50: 5.54,  100: 6.31},
    "24hr":  {2: 4.16,  5: 5.13,  10: 6.03,  25: 7.41,  50: 8.60,  100: 9.88},
}

# Duration in hours (to convert depth to intensity)
DUR_HOURS = {
    "15min": 0.25,
    "30min": 0.50,
    "1hr":   1.00,
    "2hr":   2.00,
    "6hr":   6.00,
    "24hr":  24.0,
}

T_VALS = [2, 5, 10, 25, 50, 100]

# --- Build Atlas 14 intensity table (mm/hr) ---
atlas_rows = []
for dur, t_dict in atlas14_depth_inches.items():
    for T, depth_in in t_dict.items():
        intensity_mm_hr = (depth_in / DUR_HOURS[dur]) * INCH_TO_MM
        atlas_rows.append({
            "duration":          dur,
            "T_years":           T,
            "atlas14_intensity": round(intensity_mm_hr, 3)
        })

atlas_df = pd.DataFrame(atlas_rows)

# --- Load our computed IDF ---
our_idf = pd.read_csv(IDF_CSV)

# --- Merge and compute differences ---
merged = pd.merge(our_idf, atlas_df, on=["duration", "T_years"])
merged["diff_mm_hr"]     = round(
    merged["intensity_mm_per_hr"] - merged["atlas14_intensity"], 3)
merged["pct_difference"] = round(
    (merged["diff_mm_hr"] / merged["atlas14_intensity"]) * 100, 1)

merged.to_csv(OUTPUT_CSV, index=False)

# --- Print comparison table ---
print("Comparison Table (intensity in mm/hr):")
print("-" * 75)
print(f"{'Duration':<10} {'T (yr)':<8} {'This Study':>12} "
      f"{'Atlas 14':>12} {'Diff %':>10}")
print("-" * 75)
for _, row in merged.sort_values(["duration", "T_years"]).iterrows():
    print(f"{row['duration']:<10} {int(row['T_years']):<8} "
          f"{row['intensity_mm_per_hr']:>12.3f} "
          f"{row['atlas14_intensity']:>12.3f} "
          f"{row['pct_difference']:>9.1f}%")

print(f"\nMean absolute % difference: "
      f"{merged['pct_difference'].abs().mean():.1f}%")

# --- Plot ---
DUR_MAP   = {"15min": 0.25, "30min": 0.5, "1hr": 1.0,
             "2hr": 2.0,   "6hr": 6.0,   "24hr": 24.0}
DUR_ORDER = ["15min", "30min", "1hr", "2hr", "6hr", "24hr"]
COLORS    = ["#1f77b4", "#ff7f0e", "#2ca02c",
             "#d62728", "#9467bd", "#8c564b"]

fig, ax = plt.subplots(figsize=(11, 7))

for T, color in zip(T_VALS, COLORS):
    sub = merged[merged["T_years"] == T].copy()
    sub["d_hours"] = sub["duration"].map(DUR_MAP)
    sub = sub.sort_values("d_hours")

    # This study — solid line, filled circles
    ax.plot(sub["d_hours"], sub["intensity_mm_per_hr"],
            "o-", color=color, linewidth=2.0, markersize=7, zorder=3)

    # Atlas 14 — dashed line, open squares
    ax.plot(sub["d_hours"], sub["atlas14_intensity"],
            "s--", color=color, linewidth=1.5, markersize=7,
            markerfacecolor="white", zorder=3)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Duration", fontsize=12)
ax.set_ylabel("Rainfall Intensity (mm/hr)", fontsize=12)
ax.set_title(
    "IDF Curves — This Study vs NOAA Atlas 14\n"
    "Randolph, Alabama (Lat 32.8879°, Lon −86.8772°) | "
    "NOAA Atlas 14 Vol. 9 Ver. 2",
    fontsize=11)

ax.set_xticks([0.25, 0.5, 1.0, 2.0, 6.0, 24.0])
ax.set_xticklabels(["15 min", "30 min", "1 hr",
                    "2 hr",   "6 hr",   "24 hr"],
                   rotation=30, ha="right")
ax.grid(True, which="both", linestyle="--", alpha=0.4)

# Legend
style_legend = [
    Line2D([0], [0], color="gray", linewidth=2.0, marker="o",
           label="This study (LP3, AMS-based)"),
    Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--",
           marker="s", markerfacecolor="white",
           label="NOAA Atlas 14 (PDS-based, official)"),
]
color_legend = [
    mpatches.Patch(color=c, label=f"T = {T} years")
    for T, c in zip(T_VALS, COLORS)
]
ax.legend(handles=style_legend + color_legend,
          fontsize=9, loc="upper right", ncol=2)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
plt.show()
print(f"\nPlot saved to: {OUTPUT_PNG}")
print(f"CSV  saved to: {OUTPUT_CSV}")