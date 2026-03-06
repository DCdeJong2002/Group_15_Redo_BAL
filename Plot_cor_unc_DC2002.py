from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# File path
# -------------------------------------------------------
base_dir = Path(__file__).resolve().parent
csv_file = base_dir / "processed_outputs_modeloff" / "rudder_0_elevator_0.csv"

# -------------------------------------------------------
# Settings
# -------------------------------------------------------
target_aoa = 2.5
target_js = [2.0, 1.6, 2.4, 2.8]

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
df = pd.read_csv(csv_file)

# -------------------------------------------------------
# Basic checks
# -------------------------------------------------------
required_cols = ["AoA_round", "AoS", "CMyaw", "J"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -------------------------------------------------------
# Filter AoA = 2.5
# -------------------------------------------------------
df_aoa = df[df["AoA_round"] == target_aoa].copy()

if df_aoa.empty:
    raise ValueError(f"No rows found with AoA_round = {target_aoa}")

# -------------------------------------------------------
# Plot
# Use nearest available J group for each requested target J
# but label with both requested and actual mean J
# -------------------------------------------------------
plt.figure(figsize=(8, 5))

for target_j in target_js:
    # find rows closest to requested J
    j_values = df_aoa["J_round"].dropna().unique()
    if len(j_values) == 0:
        raise ValueError("No valid J values found in filtered data.")

    actual_j = j_values[np.argmin(np.abs(j_values - target_j))]

    # use a tolerance to capture the corresponding group
    # adapt if needed
    tol = 0.08
    df_j = df_aoa[np.abs(df_aoa["J"] - actual_j) <= tol].copy()

    # fallback: if nothing found, use exact nearest value rows
    if df_j.empty:
        df_j = df_aoa[np.isclose(df_aoa["J"], actual_j)].copy()

    if df_j.empty:
        print(f"Skipping target J={target_j}: no matching rows found.")
        continue

    # sort by AoS for plotting
    df_j = df_j.sort_values("AoS")

    # representative actual J for legend
    actual_j_mean = df_j["J"].mean()

    plt.plot(
        df_j["AoS"],
        df_j["CMyaw"],
        marker="o",
        label=f"target J={target_j:.1f}, actual J≈{actual_j_mean:.2f}"
    )

# -------------------------------------------------------
# Layout
# -------------------------------------------------------
plt.xlabel("AoS [deg]")
plt.ylabel("CMyaw [-]")
plt.title("CMyaw vs AoS at AoA = 2.5°")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()