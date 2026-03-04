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

    # Next line is assumed to be the units line (skip it if present)
    units_idx = header_idx + 1 if header_idx + 1 < len(lines) else None
    data_start = header_idx + 2  # header + units

    columns = lines[header_idx].strip().split()
    ncols = len(columns)

    rows = []
    for line in lines[data_start:]:
        s = line.strip()
        if not s:
            continue

        parts = s.split()

        # stop if we hit another header block (rare but happens in concatenated files)
        if parts and parts[0] == "Run_nr":
            break

        # must have at least ncols tokens; if longer, ignore trailing garbage
        if len(parts) < ncols:
            continue

        rows.append(parts[:ncols])

    df = pd.DataFrame(rows, columns=columns)

    # --- convert columns to numeric where possible ---
    # keep Time as string first (then optionally parse)
    for col in df.columns:
        if col == "Time":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- parse Time if present ---
    if parse_time and "Time" in df.columns:
        # format is H:M:S; convert to Timedelta
        df["Time"] = pd.to_timedelta(df["Time"], errors="coerce")

    return df


# 1) From a file path
df1 = parse_windtunnel_table(r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\Redo_BAL\corr_rudder_0_elevator_0.txt")
df2 = parse_windtunnel_table(r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\Redo_BAL\raw_rudder_0_elevator_0.txt")
df3 = parse_windtunnel_table(r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\Redo_BAL\unc_rudder_0_elevator_0.txt")


print(df1)
print(df2)
print(df3)