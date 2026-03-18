from pathlib import Path
import pandas as pd

from correction_classes import ModelOffCorrector, PropOffData, PropOnData, TailOffData


BASE_DIR = Path(__file__).resolve().parent

modeloff = ModelOffCorrector(
    correction_csv=BASE_DIR / "INPUT_BALANCE_DATA" / "model_off_corrections_grid.csv",
    save_dir=BASE_DIR / "results_TAILOFF"
)

# ------------------------------------------------------------
# Example 2: prop-off workflow
# raw prop-off -> optional model-off -> solid blockage -> fit -> wake blockage
# ------------------------------------------------------------
# only do this if you truly want to apply model-off to prop-off as well
tail_off_raw = pd.read_csv(BASE_DIR / "INPUT_BALANCE_DATA" / "all_TAILOFF_cases_combined.csv")

df_tail_off_modeloff = modeloff.apply(
    tail_off_raw,
    save_csv=True,
    filename="TAILOFF_modeloff_corrected.csv"
)

tailoff = TailOffData(df_tail_off_modeloff)
tailoff.set_save_directory(BASE_DIR / "results_TAILOFF")

df_tail_off_solid = tailoff.apply_solid_blockage(
    save_csv=True,
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
    save_csv=True,
    filename="TAILOFF_grid_by_velocity_alpha_slice.csv"
)

cl_alpha_df = tailoff.compute_cl_alpha_slope_by_case(
    aoa_min=-4.0,
    aoa_max=8.0,
    cl_col="CL_solid_blockage_corr",
    save_csv=True,
    filename="TAILOFF_cl_alpha_slopes.csv"
)

print(cl_alpha_df)