from pathlib import Path
import pandas as pd

from correction_classes import ModelOffCorrector, TailOffData

BASE_DIR = Path(__file__).resolve().parent


def run_tailoff_workflow(save_outputs=True):

    modeloff = ModelOffCorrector(
        correction_csv=BASE_DIR / "INPUT_BALANCE_DATA" / "model_off_corrections_grid.csv",
        save_dir=BASE_DIR / "results_TAILOFF"
    )

    tail_off_raw = pd.read_csv(
        BASE_DIR / "INPUT_BALANCE_DATA" / "all_TAILOFF_cases_combined.csv"
    )

    df_tail_off_modeloff = modeloff.apply(
        tail_off_raw,
        save_csv=save_outputs,
        filename="TAILOFF_modeloff_corrected.csv"
    )

    tailoff = TailOffData(df_tail_off_modeloff)

    if save_outputs:
        tailoff.set_save_directory(BASE_DIR / "results_TAILOFF")

    df_tail_off_solid = tailoff.apply_solid_blockage(
        save_csv=save_outputs,
        filename="TAILOFF_solid_blockage_corrected.csv"
    )

    extra_aos_vals = [-10.0, -8.0, -5.0, -4.0, 0.0, 4.0, 5.0, 7.0, 8.0, 10.0]
    extra_aoa_vals = [-5.0, -2.5, 0.0, 2.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]

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
        extra_aoa_vals=extra_aoa_vals,
        extra_aos_vals=extra_aos_vals,
        save_csv=save_outputs,
        filename="TAILOFF_grid_by_velocity_alpha_slice_extended.csv"
    )

    cl_alpha_df = tailoff.compute_cl_alpha_slope_by_case(
        aoa_min=-4.0,
        aoa_max=8.0,
        cl_col="CL_solid_blockage_corr",
        save_csv=save_outputs,
        filename="TAILOFF_cl_alpha_slopes.csv"
    )

    return tailoff, df_tail_off_solid, df_grid, cl_alpha_df


if __name__ == "__main__":
    tailoff, df_tail_off_solid, df_grid, cl_alpha_df = run_tailoff_workflow(save_outputs=True)