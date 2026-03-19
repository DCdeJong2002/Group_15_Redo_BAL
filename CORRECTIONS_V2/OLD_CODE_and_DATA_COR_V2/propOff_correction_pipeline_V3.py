from pathlib import Path
import pandas as pd

from correction_classes import ModelOffCorrector, PropOffData, PropOnData, TailOffData
from TAILOFF_correction_pipeline_FINAL import run_tailoff_workflow

save_outputs = True  # Set to False if you want to skip saving intermediate CSVs

BASE_DIR = Path(__file__).resolve().parent

modeloff = ModelOffCorrector(
    correction_csv=BASE_DIR / "INPUT_BALANCE_DATA" / "model_off_corrections_grid.csv",
    save_dir=BASE_DIR / "results_propOff"
)

# ------------------------------------------------------------
# Example 2: prop-off workflow
# raw prop-off -> optional model-off -> solid blockage -> fit -> wake blockage
# ------------------------------------------------------------
# only do this if you truly want to apply model-off to prop-off as well
prop_off_raw = pd.read_csv(BASE_DIR / "INPUT_BALANCE_DATA" / "propOff.csv")

df_prop_off_modeloff = modeloff.apply(
    prop_off_raw,
    save_csv=save_outputs,
    filename="propOff_modeloff_corrected.csv"
)

propoff = PropOffData(df_prop_off_modeloff)
propoff.set_save_directory(BASE_DIR / "results_propOff")

df_prop_off_solid = propoff.apply_solid_blockage(
    save_csv=save_outputs,
    filename="propOff_solid_blockage_corrected.csv"
)

df_prop_off_fit = propoff.fit_cd_polar(
    save_csv=save_outputs,
    filename="propOff_with_CD_fit_values.csv",
    fit_params_filename="propOff_CD_fit_parameters.csv"
)

df_prop_off_wake = propoff.apply_wake_blockage(
    save_csv=save_outputs,
    filename="propOff_wake_blockage_corrected.csv"
)

BASE_DIR = Path(__file__).resolve().parent

modeloff = ModelOffCorrector(
    correction_csv=BASE_DIR / "INPUT_BALANCE_DATA" / "model_off_corrections_grid.csv",
    save_dir=BASE_DIR / "results_TAILOFF"
)

# ------------------------------------------------------------
# Create tailoff instance
# ------------------------------------------------------------
tailoff,_,_,_ = run_tailoff_workflow(save_outputs=False)

# ------------------------------------------------------------
# Streamline curvature correction for prop-off using tail-off data
# ------------------------------------------------------------
df_prop_off_sc = propoff.apply_streamline_curvature_correction(
    tailoff=tailoff,
    tau=0.045,
    delta=0.1065,
    geom_factor=0.2172/2.07,   # replace with your actual geometric factor
    cl_source_col="CL_solid_blockage_corr_wake_corr",
    cm_source_col="CMpitch_solid_blockage_corr_wake_corr",
    save_csv=save_outputs,
    filename="propOff_streamline_curvature_corrected.csv"
)

df_prop_off_dw = propoff.apply_downwash_correction(
    tailoff=tailoff,
    delta=0.1065,
    geom_factor=0.2172 / 2.07,
    aoa_source_col="AoA_streamline_curvature_corr",
    cd_source_col="CD_solid_blockage_corr_wake_corr",
    save_csv=save_outputs,
    filename="propOff_downwash_corrected.csv"
)

df_prop_off_tail = propoff.apply_tail_correction(
    tailoff=tailoff,
    delta=0.1085,
    geom_factor=0.2172 / 2.07,
    tau2_lt=0.8*0.535,
    dcmpitch_dalpha=-0.15676,
    dcmpitch_dalpha_unit="per_rad",
    aoa_source_col="AoA_downwash_corr",
    cmpitch_source_col="CMpitch_solid_blockage_corr_wake_corr_sc_corr",
    save_csv=save_outputs,
    filename="propOff_tail_corrected.csv"
)