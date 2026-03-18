import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Resolve repository paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

PROPON_INPUT_CSV = REPO_ROOT / "propOn_CORRECTIONS_FOLDER" / "propOn_solid_blockage_corrected.csv"
FIT_PARAMS_CSV = REPO_ROOT / "AERODYNAMIC_DATA_propoff" / "propOff_CD_fit_parameters.csv"

OUTPUT_DIR = REPO_ROOT / "propOn_CORRECTIONS_FOLDER"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "propOn_wake_blockage_corrected.csv"
OUTPUT_CSV_2 = OUTPUT_DIR / "propOn_wake_blockage_corrected_full.csv"

# ============================================================
# User settings
# ============================================================

S_REF = 0.2172
TEST_SECTION_AREA = 2.0700

V_COL = "V_solid_blockage_corr"
CL_COL = "CL_solid_blockage_corr"
CD_COL = "CD_solid_blockage_corr"

VELOCITY_TOLERANCE = 1
CLIP_NEGATIVE_CDSEP = True


# ============================================================
# Utility functions
# ============================================================

def round_to_half(series):
    return np.round(series * 2.0) / 2.0


def attach_nearest_velocity_fit_with_sign_fallbacks(
    data_df,
    fit_df,
    vel_col_data="V_round",
    vel_col_fit="V_round",
    aos_col="AoS_round",
    de_col="dE",
    dr_col="dR",
    fit_value_cols=("CD0_fit", "k_fit"),
    velocity_tolerance=0.5
):
    data_df = data_df.copy()
    fit_df = fit_df.copy()

    data_df["matched_fit_velocity"] = np.nan
    data_df["velocity_match_error"] = np.nan
    data_df["fit_found"] = False
    data_df["fit_match_type"] = pd.Series([None] * len(data_df), dtype="object")
    data_df["matched_fit_AoS"] = np.nan
    data_df["matched_fit_dR"] = np.nan
    data_df["matched_fit_source_row"] = np.nan

    for col in fit_value_cols:
        data_df[col] = np.nan

    # optional but recommended: normalize key columns
    for frame in [data_df, fit_df]:
        frame[aos_col] = frame[aos_col].astype(float).round(3).replace(-0.0, 0.0)
        frame[de_col] = frame[de_col].astype(float).replace(-0.0, 0.0)
        frame[dr_col] = frame[dr_col].astype(float).replace(-0.0, 0.0)
        frame[vel_col_data if frame is data_df else vel_col_fit] = (
            frame[vel_col_data if frame is data_df else vel_col_fit].astype(float)
        )

    for idx, row in data_df.iterrows():
        aos_val = row[aos_col]
        de_val = row[de_col]
        dr_val = row[dr_col]
        v_val = row[vel_col_data]

        candidate_keys = [
            ((aos_val,  de_val,  dr_val),  "exact_AoS_exact_dR"),
            ((aos_val,  de_val, -dr_val),  "exact_AoS_flipped_dR"),
            ((aos_val,  de_val,  0.0),     "exact_AoS_zero_dR"),
            ((-aos_val, de_val,  dr_val),  "flipped_AoS_exact_dR"),
            ((-aos_val, de_val, -dr_val),  "flipped_AoS_flipped_dR"),
            ((-aos_val, de_val,  0.0),     "flipped_AoS_zero_dR"),
        ]

        matched = False

        for (aos_key, de_key, dr_key), label in candidate_keys:
            candidate = fit_df[
                np.isclose(fit_df[aos_col], aos_key) &
                np.isclose(fit_df[de_col], de_key) &
                np.isclose(fit_df[dr_col], dr_key)
            ].copy()

            if candidate.empty:
                continue

            candidate["vel_diff"] = np.abs(candidate[vel_col_fit] - v_val)
            best_idx = candidate["vel_diff"].idxmin()
            best_diff = candidate.loc[best_idx, "vel_diff"]

            if best_diff <= velocity_tolerance:
                data_df.at[idx, "matched_fit_velocity"] = candidate.at[best_idx, vel_col_fit]
                data_df.at[idx, "velocity_match_error"] = best_diff
                data_df.at[idx, "fit_found"] = True
                data_df.at[idx, "fit_match_type"] = label
                data_df.at[idx, "matched_fit_AoS"] = candidate.at[best_idx, aos_col]
                data_df.at[idx, "matched_fit_dR"] = candidate.at[best_idx, dr_col]
                data_df.at[idx, "matched_fit_source_row"] = best_idx

                for col in fit_value_cols:
                    data_df.at[idx, col] = candidate.at[best_idx, col]

                matched = True
                break

        if not matched:
            continue

    return data_df


# ============================================================
# Load files
# ============================================================

df = pd.read_csv(PROPON_INPUT_CSV)
fit_df = pd.read_csv(FIT_PARAMS_CSV)

for frame in [df, fit_df]:
    frame["AoS_round"] = np.round(frame["AoS_round"].astype(float), 3)
    frame["dE"] = frame["dE"].astype(float)
    frame["dR"] = frame["dR"].astype(float)

    # remove negative zero
    frame["AoS_round"] = frame["AoS_round"].replace(-0.0, 0.0)
    frame["dR"] = frame["dR"].replace(-0.0, 0.0)

# ============================================================
# Validate columns
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
# Create indexing columns
# ============================================================

df["V_round"] = round_to_half(df[V_COL])
df["AoA_round"] = round_to_half(df["AoA"])
df["AoS_round"] = round_to_half(df["AoS"])


# ============================================================
# Attach reusable fits from prop-off data
# ============================================================

df = attach_nearest_velocity_fit_with_sign_fallbacks(
    data_df=df,
    fit_df=fit_df,
    vel_col_data="V_round",
    vel_col_fit="V_round",
    aos_col="AoS_round",
    de_col="dE",
    dr_col="dR",
    fit_value_cols=("CD0_fit", "k_fit"),
    velocity_tolerance=VELOCITY_TOLERANCE
)


# ============================================================
# Coverage report
# ============================================================

n_total = len(df)
n_fit = int(df["fit_found"].sum())
n_missing = n_total - n_fit

print(f"Total rows               : {n_total}")
print(f"Rows with matched fit    : {n_fit}")
print(f"Rows without matched fit : {n_missing}")

if n_fit == 0:
    raise ValueError("No fits matched the prop-on data.")


# ============================================================
# Compute wake blockage from imported fits
# ============================================================

df["CL2_fit"] = df[CL_COL] ** 2
df["CDi_fit"] = df["k_fit"] * df["CL2_fit"]
df["CDsep_fit"] = df[CD_COL] - df["CD0_fit"] - df["CDi_fit"]

if CLIP_NEGATIVE_CDSEP:
    df["CDsep_fit"] = df["CDsep_fit"].clip(lower=0)

df["ewb_t_fit"] = (
    (S_REF / (4.0 * TEST_SECTION_AREA)) * df["CD0_fit"]
    + (5.0 * S_REF / (4.0 * TEST_SECTION_AREA)) * df["CDsep_fit"]
)

df["q_ratio_wake"] = (
    1.0
    + (S_REF / (2.0 * TEST_SECTION_AREA)) * df["CD0_fit"]
    + (5.0 * S_REF / (2.0 * TEST_SECTION_AREA)) * df["CDsep_fit"]
)

df.loc[~df["fit_found"], ["ewb_t_fit", "q_ratio_wake"]] = np.nan

bad_mask = df["fit_found"] & (df["q_ratio_wake"] <= 0)
if bad_mask.any():
    raise ValueError("Found non-positive q_ratio_wake values.")


# ============================================================
# Apply wake blockage correction
# ============================================================

df["V_wake_corr"] = np.nan
df.loc[df["fit_found"], "V_wake_corr"] = (
    df.loc[df["fit_found"], V_COL] * np.sqrt(df.loc[df["fit_found"], "q_ratio_wake"])
)

coeff_candidates = [
    "CL_solid_blockage_corr",
    "CD_solid_blockage_corr",
    "CYaw_solid_blockage_corr",
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


df.to_csv(OUTPUT_CSV_2, index=False)
# ============================================================
# Clean output
# ============================================================

final_columns = [
    "AoA", "AoS", "V", "V_solid_blockage_corr", "dE", "dR",
    "CL", "CL_solid_blockage_corr",
    "CD", "CD_solid_blockage_corr",
    "CYaw", "CYaw_solid_blockage_corr",
    "CMroll", "CMroll_solid_blockage_corr",
    "CMpitch", "CMpitch_solid_blockage_corr",
    "CMyaw", "CMyaw_solid_blockage_corr",
    "solid_blockage_e",
    "V_round", "AoA_round", "AoS_round",
    "ewb_t_fit", "q_ratio_wake",
    "V_wake_corr",
    "CL_solid_blockage_corr_wake_corr",
    "CD_solid_blockage_corr_wake_corr",
    "CMroll_solid_blockage_corr_wake_corr",
    "CMpitch_solid_blockage_corr_wake_corr",
    "CMyaw_solid_blockage_corr_wake_corr",
]

existing_cols = [c for c in final_columns if c in df.columns]
clean_df = df[existing_cols].copy()

clean_df.to_csv(OUTPUT_CSV, index=False)

print("Finished applying wake blockage with velocity matching and sign fallbacks.")
print("Output file:", OUTPUT_CSV)

# ============================================================
# Print rows without a matched fit
# ============================================================

missing_fit_rows = df.loc[~df["fit_found"], ["AoS", "AoA", "V", "dE", "dR"]]

if not missing_fit_rows.empty:
    print("\nRows without available wake-blockage fit:")
    print(missing_fit_rows.to_string(index=False))
else:
    print("\nAll rows have an available wake-blockage fit.")

