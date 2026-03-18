import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------------
# File locations
# -------------------------------------------------------
rudder_angle_data = 4  # 1 for rudder -5, 2 for rudder 0


if rudder_angle_data == 1:
    matlab_file = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\matlab_processed_results\matlab_rudder_0_elevator_0.csv"
    python_file = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\processed_outputs\rudder_0_elevator_0.csv"
elif rudder_angle_data == 2:
    matlab_file = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\matlab_processed_results\matlab_rudder_m5_elevator_0.csv"
    python_file = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\processed_outputs\rudder_m5_elevator_0.csv"
elif rudder_angle_data == 3:
    matlab_file = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\matlab_processed_results\matlab_rudder_m10_elevator_0.csv"
    python_file = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\processed_outputs\rudder_m10_elevator_0.csv"
elif rudder_angle_data == 4:
    matlab_file = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\matlab_processed_results\matlab_rudder_m20_elevator_0.csv"
    python_file = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\processed_outputs\rudder_m20_elevator_0.csv"



# -------------------------------------------------------
# Load files
# -------------------------------------------------------
matlab_df = pd.read_csv(matlab_file)
python_df = pd.read_csv(python_file)

# -------------------------------------------------------
# All MATLAB result columns you listed
# Only columns existing in BOTH files will be compared
# -------------------------------------------------------
matlab_result_columns = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "rpmWT",
    "rho",
    "q",
    "V",
    "Re",
    "rpsM1",
    "rpsM2",
    "iM1",
    "iM2",
    "dPtQ",
    "tM1",
    "tM2",
    "vM1",
    "vM2",
    "pInf",
    "nu",
    "J_M1",
    "J_M2",
    "FX",
    "FY",
    "FZ",
    "MX",
    "MY",
    "MZ",
    "CFX",
    "CFY",
    "CFZ",
    "CMX",
    "CMY",
    "CMZ",
    "CN",
    "CT",
    "CY",
    "CL",
    "CD",
    "CYaw",
    "CMroll",
    "CMpitch",
    "CMpitch25c",
    "CMyaw",
    "b",
    "c",
    "S",
]

common_cols = [c for c in matlab_result_columns if c in matlab_df.columns and c in python_df.columns]

print("\nColumns compared:")
print(common_cols)

# -------------------------------------------------------
# Ensure same row order
# Prefer sorting by run; if not enough, also sort by AoA/AoS
# -------------------------------------------------------
sort_cols = [c for c in ["run", "AoA", "AoS"] if c in matlab_df.columns and c in python_df.columns]

if sort_cols:
    matlab_df = matlab_df.sort_values(sort_cols).reset_index(drop=True)
    python_df = python_df.sort_values(sort_cols).reset_index(drop=True)

# -------------------------------------------------------
# Check row count
# -------------------------------------------------------
if len(matlab_df) != len(python_df):
    raise ValueError(
        f"Row count mismatch: MATLAB has {len(matlab_df)} rows, "
        f"Python has {len(python_df)} rows."
    )

# -------------------------------------------------------
# Compare columns
# -------------------------------------------------------
print("\nDifferences between MATLAB and Python\n")
summary_rows = []

for col in common_cols:
    matlab_vals = pd.to_numeric(matlab_df[col], errors="coerce").to_numpy()
    python_vals = pd.to_numeric(python_df[col], errors="coerce").to_numpy()

    diff = python_vals - matlab_vals

    max_err = np.nanmax(np.abs(diff))
    mean_err = np.nanmean(np.abs(diff))
    rmse = np.sqrt(np.nanmean(diff**2))

    summary_rows.append({
        "column": col,
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "rmse": rmse,
    })

    print(f"{col:12s}  max={max_err:.3e}   mean={mean_err:.3e}   rmse={rmse:.3e}")

# -------------------------------------------------------
# Put summary in a dataframe and sort by largest error
# -------------------------------------------------------
summary_df = pd.DataFrame(summary_rows).sort_values("max_abs_error", ascending=False)

print("\nColumns with largest differences:\n")
print(summary_df.to_string(index=False))

# -------------------------------------------------------
# Optional: inspect worst rows for selected columns
# -------------------------------------------------------
cols_to_inspect = ["CY", "CMyaw", "CFY", "CMZ"]

for col in cols_to_inspect:
    if col in common_cols:
        matlab_vals = pd.to_numeric(matlab_df[col], errors="coerce").to_numpy()
        python_vals = pd.to_numeric(python_df[col], errors="coerce").to_numpy()
        diff = python_vals - matlab_vals

        idx = np.nanargmax(np.abs(diff))
        print(f"\nWorst row for {col}:")
        print(f"Row index      : {idx}")
        print(f"MATLAB {col:6s}: {matlab_vals[idx]:.12e}")
        print(f"Python {col:6s}: {python_vals[idx]:.12e}")
        print(f"Difference     : {diff[idx]:.12e}")