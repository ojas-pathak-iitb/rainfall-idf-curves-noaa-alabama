"""
IDF Curve Project
Step 6: Fit Empirical IDF Formula using Linear Least Squares

What this does:
    - Reads the IDF table (intensity values for all durations and return periods)
    - Fits the empirical IDF formula: i = a * T^b / d^e
      where i = intensity (mm/hr), T = return period (years), d = duration (hours)
    - Linearises by taking natural log: ln(i) = ln(a) + b*ln(T) - e*ln(d)
    - Solves as a linear least squares problem: beta = (A^T A)^{-1} A^T y
    - Matrix operations coded using NumPy — no scipy or optimisation library used
    - Computes RMSE as a measure of formula fit quality

Input  : output/idf_table.csv
Output : output/idf_formula_params.csv
"""

import os
import pandas as pd
import numpy as np

# --- Paths (relative to project root) ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

INPUT_CSV  = os.path.join(OUTPUT_DIR, "idf_table.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "idf_formula_params.csv")

DUR_MAP = {
    "15min": 0.25,
    "30min": 0.50,
    "1hr":   1.00,
    "2hr":   2.00,
    "6hr":   6.00,
    "24hr":  24.0
}

# --- Load data ---
idf = pd.read_csv(INPUT_CSV)
idf["d_hours"] = idf["duration"].map(DUR_MAP)

# --- Build linear system in log space ---
# i = a * T^b / d^e
# ln(i) = ln(a) + b*ln(T) - e*ln(d)
# A matrix rows: [1, ln(T), -ln(d)]

ln_i = np.log(idf["intensity_mm_per_hr"].values)
ln_T = np.log(idf["T_years"].values)
ln_d = np.log(idf["d_hours"].values)

A = np.column_stack([
    np.ones(len(ln_i)),
    ln_T,
    -ln_d
])
y = ln_i

# --- Least squares solution: beta = (A^T A)^{-1} A^T y ---
ATA  = A.T @ A
ATy  = A.T @ y
beta = np.linalg.solve(ATA, ATy)

ln_a, b, e = beta
a = float(np.exp(ln_a))
b = float(b)
e = float(e)

print(f"Fitted IDF Formula: i = {a:.4f} * T^{b:.4f} / d^{e:.4f}")

# --- Residuals and RMSE ---
i_pred    = a * idf["T_years"].values**b / idf["d_hours"].values**e
residuals = idf["intensity_mm_per_hr"].values - i_pred
rmse      = float(np.sqrt(np.mean(residuals**2)))

print(f"RMSE: {rmse:.4f} mm/hr")

# --- Save ---
result = pd.DataFrame([{
    "a":    round(a,    4),
    "b":    round(b,    4),
    "e":    round(e,    4),
    "RMSE": round(rmse, 4)
}])
result.to_csv(OUTPUT_CSV, index=False)
print(f"Formula parameters saved to: {OUTPUT_CSV}")