import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Resolve repository paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

INPUT_CSV = REPO_ROOT / "AERODYNAMIC_DATA_propoff" / "propOff_solid_blockage_corrected.csv"

OUTPUT_DIR = REPO_ROOT / "AERODYNAMIC_DATA_propoff"
OUTPUT_DIR.mkdir(exist_ok=True)

FIT_PARAMS_CSV = OUTPUT_DIR / "propOff_CD_fit_parameters.csv"
FIT_ROWLEVEL_CSV = OUTPUT_DIR / "propOff_with_CD_fit_values.csv"
WAKE_CORRECTED_CSV = OUTPUT_DIR / "propOff_wake_blockage_corrected.csv"


# ============================================================
# Tunnel / model parameters
# ============================================================

S_REF = 0.2172        # model reference area
TEST_SECTION_AREA = 2.07   # tunnel cross-section


# ============================================================
# Columns used
# ============================================================

V_COL = "V_solid_blockage_corr"
CL_COL = "CL_solid_blockage_corr"
CD_COL = "CD_solid_blockage_corr"


# ============================================================
# Utility functions
# ============================================================

def round_to_half(series):
    return np.round(series * 2) / 2


def linear_fit(x, y):

    slope, intercept = np.polyfit(x, y, 1)

    y_hat = intercept + slope * x

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return intercept, slope, y_hat, r2


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)

df["V_round"] = round_to_half(df[V_COL])
df["AoA_round"] = round_to_half(df["AoA"])
df["AoS_round"] = round_to_half(df["AoS"])

df["CL2"] = df[CL_COL] ** 2

group_cols = ["V_round", "AoS_round", "dE", "dR"]


summary = []
groups_out = []

MIN_AOA_POINTS = 4


# ============================================================
# Perform CD fits
# ============================================================

for keys, g in df.groupby(group_cols):

    g = g.copy().sort_values("AoA_round")

    if g["AoA_round"].nunique() < MIN_AOA_POINTS:

        g["fit_used"] = False

        groups_out.append(g)

        summary.append({
            "V_round": keys[0],
            "AoS_round": keys[1],
            "dE": keys[2],
            "dR": keys[3],
            "fit_used": False
        })

        continue

    x = g["CL2"].values
    y = g[CD_COL].values

    cd0, k, y_hat, r2 = linear_fit(x, y)

    g["fit_used"] = True
    g["CD0_fit"] = cd0
    g["k_fit"] = k
    g["R2_fit"] = r2
    g["CD_fit_pred"] = y_hat

    g["CDi_fit"] = k * g["CL2"]

    summary.append({
        "V_round": keys[0],
        "AoS_round": keys[1],
        "dE": keys[2],
        "dR": keys[3],
        "CD0_fit": cd0,
        "k_fit": k,
        "R2_fit": r2
    })

    groups_out.append(g)

summary_df = []
summary_df = pd.DataFrame(summary)
row_df = pd.concat(groups_out)


# ============================================================
# Save fit parameters (reusable)
# ============================================================

summary_df.to_csv(FIT_PARAMS_CSV, index=False)


# ============================================================
# Save row-level data with fit results
# ============================================================

row_df.to_csv(FIT_ROWLEVEL_CSV, index=False)


# ============================================================
# Apply wake blockage correction
# ============================================================

row_df["CDsep_fit"] = row_df[CD_COL] - row_df["CD0_fit"] - row_df["CDi_fit"]
row_df["CDsep_fit"] = row_df["CDsep_fit"].clip(lower=0)

row_df["ewb_t_fit"] = (
        (S_REF / (4 * TEST_SECTION_AREA)) * row_df["CD0_fit"]
        + (5 * S_REF / (4 * TEST_SECTION_AREA)) * row_df["CDsep_fit"]
)

row_df["q_ratio_wake"] = (
        1
        + (S_REF / (2 * TEST_SECTION_AREA)) * row_df["CD0_fit"]
        + (5 * S_REF / (2 * TEST_SECTION_AREA)) * row_df["CDsep_fit"]
)


# ============================================================
# Apply corrections
# ============================================================

row_df["V_wake_corr"] = row_df[V_COL] * np.sqrt(row_df["q_ratio_wake"])

coeff_cols = [
    "CL_solid_blockage_corr",
    "CD_solid_blockage_corr",
    "CY_solid_blockage_corr",
    "CMroll_solid_blockage_corr",
    "CMpitch_solid_blockage_corr",
    "CMyaw_solid_blockage_corr"
]

for col in coeff_cols:

    if col in row_df.columns:

        row_df[f"{col}_wake_corr"] = row_df[col] / row_df["q_ratio_wake"]


# ============================================================
# Create clean wake-blockage corrected dataset
# ============================================================

final_columns = [
'AoA', 'AoS', 'V', 'V_solid_blockage_corr', 'dE', 'dR',
'CL', 'CL_solid_blockage_corr',
'CD', 'CD_solid_blockage_corr',
'CYaw', 'CYaw_solid_blockage_corr',
'CMroll', 'CMroll_solid_blockage_corr',
'CMpitch', 'CMpitch_solid_blockage_corr',
'CMyaw', 'CMyaw_solid_blockage_corr',
'solid_blockage_e',
'V_round', 'AoA_round', 'AoS_round',
'ewb_t_fit', 'q_ratio_wake',
'V_wake_corr',
'CL_solid_blockage_corr_wake_corr',
'CD_solid_blockage_corr_wake_corr',
'CMroll_solid_blockage_corr_wake_corr',
'CMpitch_solid_blockage_corr_wake_corr',
'CMyaw_solid_blockage_corr_wake_corr'
]

# keep only columns that actually exist
existing_cols = [c for c in final_columns if c in row_df.columns]

clean_df = row_df[existing_cols].copy()


# ============================================================
# Save clean dataset
# ============================================================

clean_df.to_csv(WAKE_CORRECTED_CSV, index=False)


print("Finished processing")

print("Fit parameters saved to:")
print(FIT_PARAMS_CSV)

print("Row-level fit diagnostics saved to:")
print(FIT_ROWLEVEL_CSV)

print("Clean wake-blockage corrected dataset saved to:")
print(WAKE_CORRECTED_CSV)