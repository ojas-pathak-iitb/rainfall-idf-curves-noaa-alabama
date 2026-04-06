"""
IDF Curve Project
Step 3: Fit Gumbel and LP3 frequency distributions to the AMS

What this does:
    - Reads the Annual Maximum Series from output/ams.csv
    - Fits Gumbel (EV-I) distribution using Method of Moments
    - Fits Log-Pearson Type III (LP3) distribution using Method of Moments
      on log-transformed data
    - All parameter estimation is coded from scratch — no scipy or
      statistical solver libraries used
    - Saves fitted parameters to output/params.csv

Input  : output/ams.csv
Output : output/params.csv
"""

import os
import pandas as pd
import numpy as np

# --- Paths (relative to project root) ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

INPUT_CSV  = os.path.join(OUTPUT_DIR, "ams.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "params.csv")


def gumbel_params(data):
    """
    Estimate Gumbel (EV-I) distribution parameters using Method of Moments.

    For the Gumbel distribution:
        sigma (scale)    = (sqrt(6) / pi) * s
        mu    (location) = x_bar - 0.5772 * sigma

    where x_bar is the sample mean and s is the sample standard deviation.

    Parameters
    ----------
    data : numpy array of AMS values

    Returns
    -------
    mu    : location parameter
    sigma : scale parameter
    """
    xbar  = np.mean(data)
    s     = np.std(data, ddof=1)
    sigma = (np.sqrt(6) / np.pi) * s
    mu    = xbar - 0.5772 * sigma
    return mu, sigma


def lp3_params(data):
    """
    Estimate Log-Pearson Type III (LP3) distribution parameters
    using Method of Moments applied to log10-transformed data.

    Parameters
    ----------
    data : numpy array of AMS values

    Returns
    -------
    ybar : mean of log10(data)
    sy   : standard deviation of log10(data)
    g    : skewness coefficient of log10(data)
    """
    y    = np.log10(data)
    n    = len(y)
    ybar = np.mean(y)
    sy   = np.std(y, ddof=1)
    g    = (n * np.sum((y - ybar)**3)) / ((n - 1) * (n - 2) * sy**3)
    return ybar, sy, g


# --- Main ---
ams       = pd.read_csv(INPUT_CSV, index_col="year")
durations = ams.columns.tolist()

results = []

for dur in durations:
    data = ams[dur].dropna().values

    mu, sigma      = gumbel_params(data)
    ybar, sy, g    = lp3_params(data)

    results.append({
        "duration":     dur,
        "n":            len(data),
        "gumbel_mu":    round(mu,    4),
        "gumbel_sigma": round(sigma, 4),
        "lp3_ybar":     round(ybar,  4),
        "lp3_sy":       round(sy,    4),
        "lp3_g":        round(g,     4)
    })

    print(f"{dur}:")
    print(f"  Gumbel  -> mu = {mu:.3f},   sigma = {sigma:.3f}")
    print(f"  LP3     -> ybar = {ybar:.3f}, sy = {sy:.3f}, g = {g:.3f}")

params_df = pd.DataFrame(results)
params_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nParameters saved to: {OUTPUT_CSV}")