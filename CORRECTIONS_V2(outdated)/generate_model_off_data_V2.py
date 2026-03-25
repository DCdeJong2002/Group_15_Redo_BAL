from pathlib import Path
import numpy as np
import pandas as pd

# Optional: use smooth shape-preserving interpolation if SciPy is available
try:
    from scipy.interpolate import PchipInterpolator
    USE_PCHIP = True
except Exception:
    USE_PCHIP = False


# ============================================================
# Settings
# ============================================================

# Absolute path you gave
input_file = Path(
    r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\CORRECTIONS_V2\MODEL_OFF_DATA\modelOffData.xlsx"
)

# If you prefer repo-relative instead, comment the line above and use:
# input_file = Path(__file__).resolve().parent / "MODEL_OFF_DATA" / "modelOffData.xlsx"

output_dir = input_file.parent
csv_file = output_dir / "model_off_corrections_grid.csv"
xlsx_file = output_dir / "model_off_corrections_grid.xlsx"


# ============================================================
# Read the two blocks from the workbook
# ============================================================

# Left block: AoA sweep at AoS = 0
aoa_df = pd.read_excel(
    input_file,
    sheet_name="Sheet1",
    header=10,        # row 11 in Excel
    usecols="A:H"
)

# Right block: AoS sweep at AoA = 0
aos_df = pd.read_excel(
    input_file,
    sheet_name="Sheet1",
    header=10,
    usecols="J:Q"
)

# Rename right block columns to match left block
aos_df.columns = ["AoA [deg]", "AoS [deg]", "CD", "CYaw", "CL", "CMroll", "CMpitch", "CMyaw"]

# Left block uses "Cy" in the sheet; normalize to "CYaw"
aoa_df = aoa_df.rename(columns={"Cy": "CYaw"})
aos_df = aos_df.rename(columns={"Cy": "CYaw"})

# Drop empty rows
aoa_df = aoa_df.dropna(subset=["AoA [deg]", "AoS [deg]"]).copy()
aos_df = aos_df.dropna(subset=["AoA [deg]", "AoS [deg]"]).copy()

# Ensure numeric
for df in [aoa_df, aos_df]:
    for col in ["AoA [deg]", "AoS [deg]", "CD", "CYaw", "CL", "CMroll", "CMpitch", "CMyaw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["AoA [deg]", "AoS [deg]"], inplace=True)

# Sort
aoa_df = aoa_df.sort_values("AoA [deg]").reset_index(drop=True)
aos_df = aos_df.sort_values("AoS [deg]").reset_index(drop=True)


# ============================================================
# Interpolation helper
# ============================================================

def interp_1d(x_data, y_data, x_new):
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    # Remove duplicates if any
    tmp = pd.DataFrame({"x": x_data, "y": y_data}).groupby("x", as_index=False)["y"].mean()
    x_data = tmp["x"].to_numpy()
    y_data = tmp["y"].to_numpy()

    if USE_PCHIP and len(x_data) >= 2:
        f = PchipInterpolator(x_data, y_data, extrapolate=False)
        y_new = f(x_new)
    else:
        # linear interpolation fallback
        y_new = np.interp(x_new, x_data, y_data, left=np.nan, right=np.nan)

    return y_new


# ============================================================
# Build target grid
# ============================================================

aoa_min = np.ceil(aoa_df["AoA [deg]"].min() * 2) / 2
aoa_max = np.floor(aoa_df["AoA [deg]"].max() * 2) / 2
aos_min = int(np.ceil(aos_df["AoS [deg]"].min()))
aos_max = int(np.floor(aos_df["AoS [deg]"].max()))

aoa_grid = np.arange(aoa_min, aoa_max + 0.25, 0.5)   # 0.5 deg increments
aos_grid = np.arange(aos_min, aos_max + 1, 1.0)      # 1 deg increments

grid = pd.MultiIndex.from_product(
    [aoa_grid, aos_grid],
    names=["AoA [deg]", "AoS [deg]"]
).to_frame(index=False)


# ============================================================
# Create correction map using additive cross approximation
# C(a,b) ≈ C(a,0) + C(0,b) - C(0,0)
# ============================================================

coeff_cols = ["CD", "CYaw", "CL", "CMroll", "CMpitch", "CMyaw"]

# Interpolate AoA sweep on target AoA grid
aoa_interp = pd.DataFrame({"AoA [deg]": aoa_grid})
for col in coeff_cols:
    aoa_interp[col] = interp_1d(aoa_df["AoA [deg]"], aoa_df[col], aoa_grid)

# Interpolate AoS sweep on target AoS grid
aos_interp = pd.DataFrame({"AoS [deg]": aos_grid})
for col in coeff_cols:
    aos_interp[col] = interp_1d(aos_df["AoS [deg]"], aos_df[col], aos_grid)

# Get baseline at (0,0)
baseline = {}
for col in coeff_cols:
    c_a00 = interp_1d(aoa_df["AoA [deg]"], aoa_df[col], np.array([0.0]))[0]
    c_0b0 = interp_1d(aos_df["AoS [deg]"], aos_df[col], np.array([0.0]))[0]

    # average the two available estimates of C(0,0)
    vals = [v for v in [c_a00, c_0b0] if not pd.isna(v)]
    baseline[col] = np.mean(vals) if len(vals) > 0 else np.nan

# Merge into full 2D grid
corr_df = grid.merge(aoa_interp, on="AoA [deg]", how="left", suffixes=("", "_aoa"))
corr_df = corr_df.merge(aos_interp, on="AoS [deg]", how="left", suffixes=("_aoa", "_aos"))

# Combine with additive cross approximation
out = corr_df[["AoA [deg]", "AoS [deg]"]].copy()

for col in coeff_cols:
    out[col] = corr_df[f"{col}_aoa"] + corr_df[f"{col}_aos"] - baseline[col]

# Nice rounded coordinate columns
out["AoA_round"] = np.round(out["AoA [deg]"] * 2) / 2
out["AoS_round"] = np.round(out["AoS [deg]"]).astype(int)

# Optional: reorder columns
out = out[
    [
        "AoA [deg]",
        "AoS [deg]",
        "AoA_round",
        "AoS_round",
        "CD",
        "CYaw",
        "CL",
        "CMroll",
        "CMpitch",
        "CMyaw",
    ]
].sort_values(["AoS [deg]", "AoA [deg]"]).reset_index(drop=True)


# ============================================================
# Save outputs
# ============================================================

# CSV
out.to_csv(csv_file, index=False)

# Excel with multiple sheets
with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
    out.to_excel(writer, sheet_name="correction_grid", index=False)
    aoa_df.to_excel(writer, sheet_name="raw_AoA_sweep", index=False)
    aos_df.to_excel(writer, sheet_name="raw_AoS_sweep", index=False)

print(f"Saved CSV  : {csv_file}")
print(f"Saved XLSX : {xlsx_file}")
print("\nPreview:")
print(out.head(15))