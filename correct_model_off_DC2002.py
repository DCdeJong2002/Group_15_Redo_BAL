from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Paths
# ============================================================

base_dir = Path(__file__).resolve().parent

input_dir = base_dir / "processed_outputs"
output_dir = base_dir / "processed_outputs_modeloff"
output_dir.mkdir(parents=True, exist_ok=True)

correction_file = base_dir / "MODEL_OFF_DATA" / "model_off_corrections_grid.csv"

# ============================================================
# Load correction grid
# ============================================================

corr_df = pd.read_csv(correction_file)

# Make sure key columns exist
required_corr_cols = ["AoA_round", "AoS_round", "CD", "CY", "CL", "CMroll", "CMpitch", "CMyaw"]
missing_corr = [c for c in required_corr_cols if c not in corr_df.columns]
if missing_corr:
    raise ValueError(f"Missing columns in correction file: {missing_corr}")

# Keep only needed correction columns and rename them
corr_df = corr_df[required_corr_cols].copy()
corr_df = corr_df.rename(columns={
    "CD": "CD_modeloff",
    "CY": "CY_modeloff",
    "CL": "CL_modeloff",
    "CMroll": "CMroll_modeloff",
    "CMpitch": "CMpitch_modeloff",
    "CMyaw": "CMyaw_modeloff",
})

# Deduplicate just in case
corr_df = corr_df.drop_duplicates(subset=["AoA_round", "AoS_round"]).reset_index(drop=True)


# ============================================================
# Correction application
# ============================================================

# Columns to correct
correction_map = {
    "CD": "CD_modeloff",
    "CY": "CY_modeloff",
    "CL": "CL_modeloff",
    "CMroll": "CMroll_modeloff",
    "CMpitch": "CMpitch_modeloff",
    "CMyaw": "CMyaw_modeloff",
}

# Optional: also correct CMpitch25c using CMpitch support correction
# since the model-off file only contains CMpitch, not CMpitch25c
apply_cmpitch_to_25c = True

all_corrected_dfs = []

csv_files = sorted(input_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {input_dir}")

for csv_file in csv_files:
    print(f"Processing {csv_file.name}")

    df = pd.read_csv(csv_file)

    # Skip files that do not look like processed outputs with rounded columns
    if "AoA_round" not in df.columns or "AoS_round" not in df.columns:
        print(f"  Skipped {csv_file.name}: missing AoA_round/AoS_round")
        continue

    # Merge correction grid
    merged = df.merge(
        corr_df,
        on=["AoA_round", "AoS_round"],
        how="left"
    )

    # Track whether correction was found
    merged["modeloff_correction_found"] = (
        merged[[v for v in correction_map.values()]].notna().all(axis=1)
    )

    # Apply corrections
    for data_col, corr_col in correction_map.items():
        if data_col in merged.columns:
            merged[f"{data_col}_uncorrected"] = merged[data_col]
            merged[data_col] = merged[data_col] - merged[corr_col]

    # Optional CMpitch25c correction
    if apply_cmpitch_to_25c and "CMpitch25c" in merged.columns and "CMpitch_modeloff" in merged.columns:
        merged["CMpitch25c_uncorrected"] = merged["CMpitch25c"]
        merged["CMpitch25c"] = merged["CMpitch25c"] - merged["CMpitch_modeloff"]

    # Save corrected CSV
    out_csv = output_dir / csv_file.name
    merged.to_csv(out_csv, index=False)

    all_corrected_dfs.append(merged)

    n_total = len(merged)
    n_found = int(merged["modeloff_correction_found"].sum())
    print(f"  Saved {out_csv.name} | matched corrections: {n_found}/{n_total}")

# ============================================================
# Save combined corrected outputs
# ============================================================

if all_corrected_dfs:
    combined_corrected = pd.concat(all_corrected_dfs, ignore_index=True)

    # Save combined CSV
    combined_csv = output_dir / "all_rudder_cases_combined_modeloff.csv"
    combined_corrected.to_csv(combined_csv, index=False)

    # Save combined Excel
    combined_xlsx = output_dir / "all_rudder_cases_modeloff.xlsx"
    with pd.ExcelWriter(combined_xlsx, engine="openpyxl") as writer:
        for df_case in all_corrected_dfs:
            if "config" in df_case.columns:
                sheet_name = str(df_case["config"].iloc[0])[:31]
            else:
                sheet_name = "case"
            df_case.to_excel(writer, sheet_name=sheet_name, index=False)

        combined_corrected.to_excel(writer, sheet_name="combined_all_cases", index=False)

    print("\nDone.")
    print(f"Saved combined CSV  : {combined_csv}")
    print(f"Saved combined Excel: {combined_xlsx}")
else:
    print("No corrected dataframes were produced.")