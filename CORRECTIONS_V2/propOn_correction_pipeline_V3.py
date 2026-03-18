from pathlib import Path
import pandas as pd

from correction_classes import ModelOffCorrector, PropOffData, PropOnData, TailOffData


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
    save_csv=True,
    filename="propOn_modeloff_corrected.csv"
)

propon = PropOnData(df_prop_on_modeloff, velocity_tolerance=1.0)
propon.set_save_directory(BASE_DIR / "results_propOn")

df_prop_on_solid = propon.apply_solid_blockage(
    save_csv=True,
    filename="propOn_solid_blockage_corrected.csv"
)

df_prop_on_wake = propon.apply_wake_blockage(
    fit_csv=BASE_DIR / "results_propOff" / "propOff_CD_fit_parameters.csv",
    save_csv=True,
    filename="propOn_wake_blockage_corrected.csv"
)

# ------------------------------------------------------------
# TAILOFF workflow
# ------------------------------------------------------------

tail_off_raw = pd.read_csv(BASE_DIR / "INPUT_BALANCE_DATA" / "all_TAILOFF_cases_combined.csv")

df_tail_off_modeloff = modeloff.apply(
    tail_off_raw,
    save_csv=False,
    filename="TAILOFF_modeloff_corrected.csv"
)

tailoff = TailOffData(df_tail_off_modeloff)
tailoff.set_save_directory(BASE_DIR / "results_TAILOFF")

df_tail_off_solid = tailoff.apply_solid_blockage(
    save_csv=False,
    filename="TAILOFF_solid_blockage_corrected.csv"
)

df_grid = tailoff.build_alpha_slice_grid_by_velocity(
    coeff_cols=[
        "CL_solid_blockage_corr",
        "CD_solid_blockage_corr",
        "CY_solid_blockage_corr",
        "CMroll_solid_blockage_corr",
        "CMpitch_solid_blockage_corr",
        "CMyaw_solid_blockage_corr",
    ],
    anchor_aoa_vals=(0.0, 5.0, 10.0),
    save_csv=False,
    filename="TAILOFF_grid_by_velocity_alpha_slice.csv"
)

cl_alpha_df = tailoff.compute_cl_alpha_slope_by_case(
    aoa_min=-4.0,
    aoa_max=8.0,
    cl_col="CL_solid_blockage_corr",
    save_csv=False,
    filename="TAILOFF_cl_alpha_slopes.csv"
)

propon.apply_streamline_curvature_correction(
    tailoff=tailoff,
    tau=0.045,
    delta=0.1065,
    geom_factor=0.2172/2.07,   # replace with your actual geometric factor
    cl_source_col="CL_solid_blockage_corr_wake_corr",
    cm_source_col="CMpitch_solid_blockage_corr_wake_corr",
    save_csv=True,
    filename="propOn_streamline_curvature_corrected.csv"
)


