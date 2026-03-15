import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Resolve repository paths (machine independent)
# ============================================================

# directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent

# assume repo root is one level above scripts/
REPO_ROOT = SCRIPT_DIR.parent

INPUT_CSV = REPO_ROOT / "AERODYNAMIC_DATA_propoff" / "propOff.csv"

OUTPUT_DIR = REPO_ROOT / "AERODYNAMIC_DATA_propoff"
OUTPUT_DIR.mkdir(exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "propOff_CD_fit_summary.csv"
ROWLEVEL_CSV = OUTPUT_DIR / "propOff_with_CD_fit_values.csv"


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

required = ["AoA", "AoS", "V", "dE", "dR", "CL", "CD"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# Create rounded indexing columns
# ============================================================

df["V_round"] = round_to_half(df["V"])
df["AoA_round"] = round_to_half(df["AoA"])
df["AoS_round"] = round_to_half(df["AoS"])

df["CL2"] = df["CL"] ** 2


group_cols = ["V_round", "AoS_round", "dE", "dR"]

summary = []
groups_out = []

MIN_AOA_POINTS = 4


# ============================================================
# Perform fits
# ============================================================

for keys, g in df.groupby(group_cols):

    g = g.copy().sort_values("AoA_round")

    if g["AoA_round"].nunique() < MIN_AOA_POINTS:

        g["fit_used"] = False
        g["CD0_fit"] = np.nan
        g["k_fit"] = np.nan
        g["CDi"] = np.nan
        g["CDs"] = np.nan
        g["R2_fit"] = np.nan

        summary.append({
            "V_round": keys[0],
            "AoS_round": keys[1],
            "dE": keys[2],
            "dR": keys[3],
            "fit_used": False
        })

        groups_out.append(g)
        continue

    x = g["CL2"].values
    y = g["CD"].values

    cd0, k, y_hat, r2 = linear_fit(x, y)

    g["fit_used"] = True
    g["CD0_fit"] = cd0
    g["k_fit"] = k
    g["R2_fit"] = r2

    g["CDi"] = k * g["CL2"]

    g["CDs"] = g["CD"] - g["CD0_fit"] - g["CDi"]
    g["CDs"] = g["CDs"].clip(lower=0)

    summary.append({
        "V_round": keys[0],
        "AoS_round": keys[1],
        "dE": keys[2],
        "dR": keys[3],
        "fit_used": True,
        "CD0_fit": cd0,
        "k_fit": k,
        "R2_fit": r2
    })

    groups_out.append(g)


# ============================================================
# Save results
# ============================================================
summary_df = []
summary_df = pd.DataFrame(summary)
row_df = pd.concat(groups_out)

summary_df.to_csv(SUMMARY_CSV, index=False)
row_df.to_csv(ROWLEVEL_CSV, index=False)

print("Finished CD fits")
print("Summary file:", SUMMARY_CSV)
print("Row-level file:", ROWLEVEL_CSV)