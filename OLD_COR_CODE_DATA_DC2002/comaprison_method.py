import pandas as pd
import numpy as np


# ==========================================================
# 1. Load data
# ==========================================================
type_data= 3

if type_data == 1:
    file1 = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\AERODYNAMIC_DATA_propoff\propOff_wake_blockage_corrected.csv"
    file2 = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\CORRECTIONS_V2\X1_propOff_wake_blockage_corrected.csv"
    merge_keys = ["AoA", "AoS", "V", "dE", "dR"]
elif type_data == 2:
    file1 = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\AERODYNAMIC_DATA_propoff\propOff_solid_blockage_corrected.csv"
    file2 = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\CORRECTIONS_V2\X1_propOff_solid_blockage_corrected.csv"
    merge_keys = ["AoA", "AoS", "V", "dE", "dR"]
elif type_data == 3:
    file1 = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\propOn_CORRECTIONS_FOLDER\propOn_wake_blockage_corrected.csv"
    file2 = r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\CORRECTIONS_V2\results_propOn_noModelOff\propOn_wake_blockage_corrected.csv"
    merge_keys = ["AoA", "AoS", "V", "dE", "dR"]


df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)


# ==========================================================
# 2. Define matching keys
# ==========================================================



# ==========================================================
# 3. Find common columns (excluding merge keys)
# ==========================================================

common_cols = list(
    (set(df1.columns) & set(df2.columns)) - set(merge_keys)
)

print(f"Comparing {len(common_cols)} columns:")
print(common_cols)


# ==========================================================
# 4. Merge dataframes
# ==========================================================

df_merged = df1.merge(
    df2,
    on=merge_keys,
    suffixes=("_df1", "_df2")
)


# ==========================================================
# 5. Compute percent differences
# ==========================================================

percent_diffs = {}

for col in common_cols:

    col1 = f"{col}_df1"
    col2 = f"{col}_df2"

    pct_diff = (df_merged[col1] - df_merged[col2]) / df_merged[col2] * 100

    percent_diffs[col] = pct_diff

    print(f"\nColumn: {col}")
    print(f"Mean % diff: {pct_diff.mean():.4f}%")
    print(f"Max  % diff: {pct_diff.max():.4f}%")
    print(f"Min  % diff: {pct_diff.min():.4f}%")


# ==========================================================
# 6. Optional: create dataframe with all percent differences
# ==========================================================

pct_df = pd.DataFrame(percent_diffs)

pct_df.to_csv("percent_differences.csv", index=False)

print("\nSaved percent differences to percent_differences.csv")