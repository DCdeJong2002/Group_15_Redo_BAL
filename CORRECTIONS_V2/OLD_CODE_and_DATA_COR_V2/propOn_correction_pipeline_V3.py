from pathlib import Path
import pandas as pd

from correction_classes import ModelOffCorrector, PropOffData, PropOnData, TailOffData
from TAILOFF_correction_pipeline_FINAL import run_tailoff_workflow

save_outputs = True  # Set to False if you want to skip saving intermediate CSVs

BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Example 1: prop-on workflow
# raw prop-on -> model-off -> solid blockage -> wake blockage
# ------------------------------------------------------------

prop_on_raw = pd.read_csv(BASE_DIR / "INPUT_BALANCE_DATA" / "all_rudder_cases_combined.csv")

modeloff = ModelOffCorrector(
    correction_csv=BASE_DIR / "INPUT_BALANCE_DATA" / "model_off_corrections_grid.csv",
    save_dir=BASE_DIR / "results_propOn"
)

df_prop_on_modeloff = modeloff.apply(
    prop_on_raw,
    save_csv=save_outputs,
    filename="propOn_modeloff_corrected.csv"
)

propon = PropOnData(df_prop_on_modeloff, velocity_tolerance=1.0)
propon.set_save_directory(BASE_DIR / "results_propOn")

df_prop_on_solid = propon.apply_solid_blockage(
    save_csv=save_outputs,
    filename="propOn_solid_blockage_corrected.csv"
)

df_prop_on_wake = propon.apply_wake_blockage(
    fit_csv=BASE_DIR / "results_propOff" / "propOff_CD_fit_parameters.csv",
    save_csv=save_outputs,
    filename="propOn_wake_blockage_corrected.csv"
)

# ------------------------------------------------------------
# Create tailoff instance
# ------------------------------------------------------------
tailoff,_,_,_ = run_tailoff_workflow(save_outputs=False)

df_prop_on_sc = propon.apply_streamline_curvature_correction(
    tailoff=tailoff,
    tau=0.045,
    delta=0.1065,
    geom_factor=0.2172/2.07,   # replace with your actual geometric factor
    cl_source_col="CL_solid_blockage_corr_wake_corr",
    cm_source_col="CMpitch_solid_blockage_corr_wake_corr",
    save_csv=save_outputs,
    filename="propOn_streamline_curvature_corrected.csv"
)

df_prop_on_dw = propon.apply_downwash_correction(
    tailoff=tailoff,
    delta=0.1065,
    geom_factor=0.2172 / 2.07,
    aoa_source_col="AoA_streamline_curvature_corr",
    cd_source_col="CD_solid_blockage_corr_wake_corr",
    save_csv=save_outputs,
    filename="propOn_downwash_corrected.csv"
)

df_prop_on_tail = propon.apply_tail_correction(
    tailoff=tailoff,
    delta=0.1085,
    geom_factor=0.2172 / 2.07,
    tau2_lt=0.8*0.535,
    dcmpitch_dalpha=-0.15676,
    dcmpitch_dalpha_unit="per_rad",
    aoa_source_col="AoA_downwash_corr",
    cmpitch_source_col="CMpitch_solid_blockage_corr_wake_corr_sc_corr",
    save_csv=True,
    filename="propOn_tail_corrected.csv"
)