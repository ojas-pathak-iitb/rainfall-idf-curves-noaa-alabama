"""
IDF Curve Project
Step 2: Extract Annual Maximum Series (AMS)

What this does:
    - Reads clean_data.csv
    - Reconstructs a full 15-minute time series with zeroes for dry periods
    - Uses a rolling window to find maximum rainfall depth for each duration
    - Converts depth to intensity (mm/hr)
    - Picks the single maximum intensity per year (Annual Maximum Series)
    - Saves AMS for all 6 durations to output/ams.csv

Input : output/clean_data.csv
Output: output/ams.csv
"""

import os
import pandas as pd
import numpy as np

# --- Paths (relative to project root) ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

INPUT_CSV  = os.path.join(OUTPUT_DIR, "clean_data.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ams.csv")

# --- Load data ---
df = pd.read_csv(INPUT_CSV)

# Build a proper datetime index
df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]])
df = df.set_index("datetime").sort_index()

# Resample to 15-min grid, filling missing (dry) periods with 0
ts = df["precip_mm"].resample("15min").sum().fillna(0)

# --- Durations to compute ---
# Key   = label used in output CSV
# Value = number of 15-min steps in that duration
durations = {
    "15min": 1,
    "30min": 2,
    "1hr":   4,
    "2hr":   8,
    "6hr":   24,
    "24hr":  96
}

ams_dict = {}

for label, steps in durations.items():
    print(f"Computing AMS for duration: {label}")

    # Rolling sum over 'steps' consecutive 15-min periods
    rolling = ts.rolling(window=steps).sum()

    # Convert rainfall depth (mm) to intensity (mm/hr)
    duration_hours = steps * 15 / 60.0
    intensity      = rolling / duration_hours

    # Take the single maximum per calendar year
    annual_max = intensity.groupby(intensity.index.year).max()
    ams_dict[label] = annual_max

# --- Build and save AMS DataFrame ---
ams_df            = pd.DataFrame(ams_dict)
ams_df.index.name = "year"

# Drop years where any duration has no valid data
ams_df = ams_df.dropna()

ams_df.to_csv(OUTPUT_CSV)

print(f"\nAMS saved to: {OUTPUT_CSV}")
print(f"Years in AMS: {ams_df.index.min()} -- {ams_df.index.max()} "
      f"({len(ams_df)} years)")
print("\nDescriptive statistics (intensity in mm/hr):")
print(ams_df.describe().round(3))