import os
import pandas as pd


def parse_windtunnel_table(source, *, parse_time=False):
    """
    Parse TU Delft / wind-tunnel style whitespace tables with:
      - a header line starting with 'Run_nr'
      - a units line right after the header
      - data lines after that

    Parameters
    ----------
    source : str
        Either a filepath OR the full text contents of the file.
    parse_time : bool
        If True and a 'Time' column exists (H:M:S), convert to pandas Timedelta.

    Returns
    -------
    df : pandas.DataFrame
    """

    # --- read lines from file or treat as raw text ---
    if "\n" in source or "\r" in source:
        lines = source.splitlines()
    else:
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()

    # --- find header line (the one containing Run_nr as first token) ---
    header_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if parts and parts[0] == "Run_nr":
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find a header line starting with 'Run_nr'.")

    # Next line is assumed to be the units line
    data_start = header_idx + 2  # header + units

    columns = lines[header_idx].strip().split()
    ncols = len(columns)

    rows = []
    for line in lines[data_start:]:
        s = line.strip()
        if not s:
            continue

        parts = s.split()

        # stop if we hit another header block
        if parts and parts[0] == "Run_nr":
            break

        # must have at least ncols tokens
        if len(parts) < ncols:
            continue

        rows.append(parts[:ncols])

    df = pd.DataFrame(rows, columns=columns)

    # --- convert columns to numeric where possible ---
    for col in df.columns:
        if col == "Time":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- parse Time if requested ---
    if parse_time and "Time" in df.columns:
        df["Time"] = pd.to_timedelta(df["Time"], errors="coerce")

    return df


def extract_rudder_setting(filepath):
    """
    Extract rudder setting from filename.

    Examples
    --------
    corr_rudder_0_elevator_0.txt   -> 0
    corr_rudder_m5_elevator_0.txt  -> -5
    corr_rudder_m10_elevator_0.txt -> -10
    corr_rudder_m20_elevator_0.txt -> -20
    """
    filename = os.path.basename(filepath)

    if "rudder_m20" in filename:
        return -20
    elif "rudder_m10" in filename:
        return -10
    elif "rudder_m5" in filename:
        return -5
    elif "rudder_0" in filename:
        return 0
    else:
        return None


# --------------------------------------------------
# 1. Store file paths in a dictionary
# --------------------------------------------------

base_dir = "Redo_BAL"

file_paths = {
    "corr": [
        os.path.join(base_dir, "corr_rudder_0_elevator_0.txt"),
        os.path.join(base_dir, "corr_rudder_m5_elevator_0.txt"),
        os.path.join(base_dir, "corr_rudder_m10_elevator_0.txt"),
        os.path.join(base_dir, "corr_rudder_m20_elevator_0.txt"),
    ],
    "raw": [
        os.path.join(base_dir, "raw_rudder_0_elevator_0.txt"),
        os.path.join(base_dir, "raw_rudder_m5_elevator_0.txt"),
        os.path.join(base_dir, "raw_rudder_m10_elevator_0.txt"),
        os.path.join(base_dir, "raw_rudder_m20_elevator_0.txt"),
    ],
    "unc": [
        os.path.join(base_dir, "unc_rudder_0_elevator_0.txt"),
        os.path.join(base_dir, "unc_rudder_m5_elevator_0.txt"),
        os.path.join(base_dir, "unc_rudder_m10_elevator_0.txt"),
        os.path.join(base_dir, "unc_rudder_m20_elevator_0.txt"),
    ]
}


# --------------------------------------------------
# 2. Create one combined DataFrame per file type
# --------------------------------------------------

combined_dfs = {}

for file_type, paths in file_paths.items():
    df_list = []

    for path in paths:
        df = parse_windtunnel_table(path, parse_time=False)

        # Add metadata columns
        df["source_file"] = os.path.basename(path)
        df["file_type"] = file_type
        df["rudder_setting"] = extract_rudder_setting(path)

        df_list.append(df)

    combined_dfs[file_type] = pd.concat(df_list, ignore_index=True)


# --------------------------------------------------
# 3. Easy access to the combined DataFrames
# --------------------------------------------------

df_corr = combined_dfs["corr"]
df_raw  = combined_dfs["raw"]
df_unc  = combined_dfs["unc"]


# --------------------------------------------------
# 5. Save combined DataFrames
# --------------------------------------------------

output_dir = "Redo_BAL_corrected"

# Save as CSV
df_corr.to_csv(os.path.join(output_dir, "corr_combined.csv"), index=False)
df_raw.to_csv(os.path.join(output_dir, "raw_combined.csv"), index=False)
df_unc.to_csv(os.path.join(output_dir, "unc_combined.csv"), index=False)

excel_path = os.path.join(output_dir, "windtunnel_data_combined.xlsx")

with pd.ExcelWriter(excel_path) as writer:
    df_corr.to_excel(writer, sheet_name="corr", index=False)
    df_raw.to_excel(writer, sheet_name="raw", index=False)
    df_unc.to_excel(writer, sheet_name="unc", index=False)

print("Files saved to:", output_dir)