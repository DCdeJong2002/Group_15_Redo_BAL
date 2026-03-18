import numpy as np
import pandas as pd
from pathlib import Path


def attach_nearest_velocity_fit(
    data_df,
    fit_df,
    vel_col_data="V_round",
    vel_col_fit="V_round",
    exact_cols=("AoS_round", "dE", "dR"),
    fit_value_cols=("CD0_fit", "k_fit"),
    velocity_tolerance=0.5
):
    """
    Attach fit parameters to each row in data_df by:
      - exact match on exact_cols
      - nearest match on velocity within velocity_tolerance

    Returns a copy of data_df with fit columns added:
      - matched_fit_velocity
      - velocity_match_error
      - fit_found
      - fit columns such as CD0_fit, k_fit
    """

    data_df = data_df.copy()
    fit_df = fit_df.copy()

    # initialize output columns
    data_df["matched_fit_velocity"] = np.nan
    data_df["velocity_match_error"] = np.nan
    data_df["fit_found"] = False

    for col in fit_value_cols:
        data_df[col] = np.nan

    # process per exact-match subgroup
    grouped_fits = {
        key: g.copy()
        for key, g in fit_df.groupby(list(exact_cols))
    }

    for idx, row in data_df.iterrows():
        key = tuple(row[col] for col in exact_cols)

        if key not in grouped_fits:
            continue

        gfit = grouped_fits[key]

        diffs = np.abs(gfit[vel_col_fit] - row[vel_col_data])
        best_idx = diffs.idxmin()
        best_diff = diffs.loc[best_idx]

        if best_diff <= velocity_tolerance:
            data_df.at[idx, "matched_fit_velocity"] = gfit.at[best_idx, vel_col_fit]
            data_df.at[idx, "velocity_match_error"] = best_diff
            data_df.at[idx, "fit_found"] = True

            for col in fit_value_cols:
                data_df.at[idx, col] = gfit.at[best_idx, col]

    return data_df
# ============================================================
# Resolve repository paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Input: prop-on data that already contains solid blockage correction
PROPON_INPUT_CSV = REPO_ROOT / "propOn_CORRECTIONS_FOLDER" / "propOn_solid_blockage_corrected.csv"

# Input: reusable fit parameters obtained from prop-off data
FIT_PARAMS_CSV = REPO_ROOT / "AERODYNAMIC_DATA_propoff" / "propOff_CD_fit_parameters.csv"

OUTPUT_DIR = REPO_ROOT / "propOn_CORRECTIONS_FOLDER"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "propOn_wake_blockage_corrected.csv"


# ============================================================
# User settings
# ============================================================

# Reference area S and test-section area C
# Use consistent units, e.g. both in m^2
S_REF = 0.2172               # <-- replace with your actual reference area
TEST_SECTION_AREA = 2.0700   # <-- replace with your actual test-section area

# Corrected columns used as input for wake blockage
V_COL = "V_solid_blockage_corr"
CL_COL = "CL_solid_blockage_corr"
CD_COL = "CD_solid_blockage_corr"

# Whether to clip negative separated-drag remainder
CLIP_NEGATIVE_CDSEP = True


# ============================================================
# Utility functions
# ============================================================

def round_to_half(series):
    return np.round(series * 2.0) / 2.0


# ============================================================
# Load files
# ============================================================

df = pd.read_csv(PROPON_INPUT_CSV)
fit_df = pd.read_csv(FIT_PARAMS_CSV)


# ============================================================
# Validate required columns
# ============================================================

required_data_cols = [
    "AoA", "AoS", "dE", "dR",
    "V", V_COL,
    "CL", CL_COL,
    "CD", CD_COL
]
missing_data = [c for c in required_data_cols if c not in df.columns]
if missing_data:
    raise ValueError(f"Missing required columns in prop-on dataset: {missing_data}")

required_fit_cols = ["V_round", "AoS_round", "dE", "dR", "CD0_fit", "k_fit"]
missing_fit = [c for c in required_fit_cols if c not in fit_df.columns]
if missing_fit:
    raise ValueError(f"Missing required columns in fit CSV: {missing_fit}")


# ============================================================
# Create rounding/indexing columns on prop-on data
# Must match the prop-off fit indexing
# ============================================================

df["V_round"] = round_to_half(df[V_COL])
df["AoA_round"] = round_to_half(df["AoA"])
df["AoS_round"] = round_to_half(df["AoS"])


# ============================================================
# Merge prop-off fits into prop-on dataset
# ============================================================

df = attach_nearest_velocity_fit(
    data_df=df,
    fit_df=fit_df,
    vel_col_data="V_round",
    vel_col_fit="V_round",
    exact_cols=("AoS_round", "dE", "dR"),
    fit_value_cols=("CD0_fit", "k_fit"),
    velocity_tolerance=1
)


# ============================================================
# Check fit coverage
# ============================================================

df["fit_found"] = df["CD0_fit"].notna() & df["k_fit"].notna()

n_total = len(df)
n_fit = int(df["fit_found"].sum())
n_missing = n_total - n_fit

print(f"Total rows                 : {n_total}")
print(f"Rows with matched fit      : {n_fit}")
print(f"Rows without matched fit   : {n_missing}")

if n_fit == 0:
    raise ValueError(
        "No prop-off fits matched the prop-on data. "
        "Check the grouping keys and rounding convention."
    )


# ============================================================
# Compute wake blockage from imported fits
# ============================================================

# CL^2 from solid-blockage-corrected CL
df["CL2_fit"] = df[CL_COL] ** 2

# Induced drag from imported prop-off fit
df["CDi_fit"] = df["k_fit"] * df["CL2_fit"]

# Separated-drag remainder
df["CDsep_fit"] = df[CD_COL] - df["CD0_fit"] - df["CDi_fit"]

if CLIP_NEGATIVE_CDSEP:
    df["CDsep_fit"] = df["CDsep_fit"].clip(lower=0)

# Wake blockage factor from the book formula
df["ewb_t_fit"] = (
    (S_REF / (4.0 * TEST_SECTION_AREA)) * df["CD0_fit"]
    + (5.0 * S_REF / (4.0 * TEST_SECTION_AREA)) * df["CDsep_fit"]
)

# Dynamic pressure ratio qc/qu
df["q_ratio_wake"] = (
    1.0
    + (S_REF / (2.0 * TEST_SECTION_AREA)) * df["CD0_fit"]
    + (5.0 * S_REF / (2.0 * TEST_SECTION_AREA)) * df["CDsep_fit"]
)

# For unmatched rows, keep these as NaN
df.loc[~df["fit_found"], ["ewb_t_fit", "q_ratio_wake"]] = np.nan

# Safety check
bad_mask = df["fit_found"] & (df["q_ratio_wake"] <= 0)
if bad_mask.any():
    raise ValueError(
        "Found non-positive q_ratio_wake values. "
        "Check S_REF, TEST_SECTION_AREA, and merged fit values."
    )


# ============================================================
# Apply wake blockage correction
# ============================================================

# Velocity correction
df["V_wake_corr"] = np.nan
df.loc[df["fit_found"], "V_wake_corr"] = (
    df.loc[df["fit_found"], V_COL] * np.sqrt(df.loc[df["fit_found"], "q_ratio_wake"])
)

# Apply to all available coefficient columns that already have
# solid-blockage-corrected versions
coeff_candidates = [
    "CL_solid_blockage_corr",
    "CD_solid_blockage_corr",
    "CY_solid_blockage_corr",
    "CYaw_solid_blockage_corr",
    "CMx_solid_blockage_corr",
    "CMy_solid_blockage_corr",
    "CMz_solid_blockage_corr",
    "CMroll_solid_blockage_corr",
    "CMpitch_solid_blockage_corr",
    "CMyaw_solid_blockage_corr",
]

for col in coeff_candidates:
    if col in df.columns:
        out_col = f"{col}_wake_corr"
        df[out_col] = np.nan
        df.loc[df["fit_found"], out_col] = (
            df.loc[df["fit_found"], col] / df.loc[df["fit_found"], "q_ratio_wake"]
        )


# ============================================================
# Build clean output dataset
# ============================================================

final_columns = [
    "AoA",
    "AoS",
    "V",
    "V_solid_blockage_corr",
    "dE",
    "dR",
    "CL",
    "CL_solid_blockage_corr",
    "CD",
    "CD_solid_blockage_corr",
    "CYaw",
    "CYaw_solid_blockage_corr",
    "CMroll",
    "CMroll_solid_blockage_corr",
    "CMpitch",
    "CMpitch_solid_blockage_corr",
    "CMyaw",
    "CMyaw_solid_blockage_corr",
    "solid_blockage_e",
    "V_round",
    "AoA_round",
    "AoS_round",
    "ewb_t_fit",
    "q_ratio_wake",
    "V_wake_corr",
    "CL_solid_blockage_corr_wake_corr",
    "CD_solid_blockage_corr_wake_corr",
    "CMroll_solid_blockage_corr_wake_corr",
    "CMpitch_solid_blockage_corr_wake_corr",
    "CMyaw_solid_blockage_corr_wake_corr",
]

existing_cols = [c for c in final_columns if c in df.columns]
clean_df = df[existing_cols].copy()


# ============================================================
# Save output
# ============================================================

clean_df.to_csv(OUTPUT_CSV, index=False)

print("Finished applying prop-off fits to prop-on data for wake blockage.")
print("Fit file used :", FIT_PARAMS_CSV)
print("Output file   :", OUTPUT_CSV)