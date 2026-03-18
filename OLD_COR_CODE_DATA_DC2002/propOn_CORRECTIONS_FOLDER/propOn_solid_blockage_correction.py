from pathlib import Path
import pandas as pd


# ============================================================
# Paths
# ============================================================

# directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent

# assume repo root is one level above scripts/
REPO_ROOT = SCRIPT_DIR.parent

INPUT_CSV = REPO_ROOT / "processed_outputs_modeloff" / "all_rudder_cases_combined_modeloff.csv"

OUTPUT_DIR = REPO_ROOT / "propOn_CORRECTIONS_FOLDER"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "propOn_solid_blockage_corrected.csv"


# ============================================================
# User input: solid blockage factor e
# ============================================================

# Option 1: one constant e for the whole dataset
USE_CONSTANT_E = True
E_CONSTANT = 0.007229438   # <-- replace with your actual blockage factor

# Option 2: e is already present as a column in the CSV
USE_E_COLUMN = False
E_COLUMN = "e"        # <-- change if your column has another name

if USE_CONSTANT_E and USE_E_COLUMN:
    raise ValueError("Choose either USE_CONSTANT_E=True or USE_E_COLUMN=True, not both.")

if not USE_CONSTANT_E and not USE_E_COLUMN:
    raise ValueError("Choose one source for e: constant or CSV column.")


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)


# ============================================================
# Get blockage factor e
# ============================================================

if USE_CONSTANT_E:
    e = E_CONSTANT
    df["solid_blockage_e"] = E_CONSTANT
else:
    if E_COLUMN not in df.columns:
        raise ValueError(f"Column '{E_COLUMN}' not found in {INPUT_CSV.name}")
    e = df[E_COLUMN]


# ============================================================
# Solid blockage correction
# ============================================================
# Standard form:
#   V_corrected = V / (1 + e)
#   coefficient_corrected = coefficient / (1 + e)^2
# ============================================================

velocity_factor = 1.0 / (1.0 + e)
coefficient_factor = 1.0 / (1.0 + e) ** 2


# ============================================================
# Columns to correct
# ============================================================

# velocity columns present in your dataset
velocity_columns = [
    "V",
]

# aerodynamic coefficient columns commonly present
coefficient_columns = [
    "CL",
    "CD",
    "CY",
    "CMx",
    "CMy",
    "CMz",
    "CYaw",
    "CMroll",
    "CMpitch",
    "CMyaw",
]

# only keep columns that actually exist
velocity_columns = [col for col in velocity_columns if col in df.columns]
coefficient_columns = [col for col in coefficient_columns if col in df.columns]


# ============================================================
# Add corrected columns
# ============================================================

for col in velocity_columns:
    df[f"{col}_solid_blockage_corr"] = df[col] * velocity_factor

for col in coefficient_columns:
    df[f"{col}_solid_blockage_corr"] = df[col] * coefficient_factor


# ============================================================
# Reorder columns so corrected columns are next to originals
# ============================================================

new_order = []
already_added = set()

for col in df.columns:
    if col in already_added:
        continue

    # original column
    new_order.append(col)
    already_added.add(col)

    # matching corrected column directly after it
    corr_col = f"{col}_solid_blockage_corr"
    if corr_col in df.columns and corr_col not in already_added:
        new_order.append(corr_col)
        already_added.add(corr_col)

# append anything missed
for col in df.columns:
    if col not in already_added:
        new_order.append(col)
        already_added.add(col)

df = df[new_order]


# ============================================================
# Save output
# ============================================================

df.to_csv(OUTPUT_CSV, index=False)

print(f"Input file : {INPUT_CSV}")
print(f"Output file: {OUTPUT_CSV}")
print(f"Rows saved : {len(df)}")

created_cols = [col for col in df.columns if col.endswith("_solid_blockage_corr")]
print("Created corrected columns:")
for col in created_cols:
    print(f"  - {col}")