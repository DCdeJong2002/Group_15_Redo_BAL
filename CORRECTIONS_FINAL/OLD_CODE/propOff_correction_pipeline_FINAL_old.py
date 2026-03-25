from pathlib import Path
import pandas as pd

from correction_classes_FINAL_old import ModelOffCorrector, PropOffData
from TAILOFF_correction_pipeline_FINAL_old import run_tailoff_workflow


def run_propoff_workflow(
    save_outputs: bool = True,
    apply_modeloff: bool = True,
    apply_solid_blockage: bool = True,
    apply_wake_blockage: bool = True,
    apply_streamline_curvature: bool = True,
    apply_downwash: bool = True,
    apply_tail_correction: bool = True,
    save_directory: str = "results_propOff"
):
    """
    Run the prop-off correction workflow with optional correction steps.

    The function automatically chooses the correct source columns depending
    on which earlier corrections were applied, tracked via `active_cols`.

    Returns
    -------
    propoff : PropOffData
        Final PropOffData object with updated self.df
    df_final : pd.DataFrame
        Final corrected dataframe
    outputs : dict
        Dictionary with intermediate dataframes
    """
    BASE_DIR = Path(__file__).resolve().parent

    outputs = {}

    # ------------------------------------------------------------
    # Active column tracker
    # Holds the name of the most recently corrected version of each
    # logical quantity. Each correction step updates the relevant keys.
    # ------------------------------------------------------------
    active_cols = {
        "CL":      "CL",
        "CD":      "CD",
        "AoA":     "AoA",
        "CMpitch": "CMpitch",
    }

    # ------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------
    prop_off_raw = pd.read_csv(BASE_DIR / "INPUT_BALANCE_DATA" / "propOff.csv")
    outputs["raw"] = prop_off_raw.copy()

    current_df = prop_off_raw.copy()

    # ------------------------------------------------------------
    # Optional model-off correction
    # ------------------------------------------------------------
    if apply_modeloff:
        modeloff = ModelOffCorrector(
            correction_csv=BASE_DIR / "INPUT_BALANCE_DATA" / "model_off_corrections_grid.csv",
            save_dir=BASE_DIR / save_directory
        )

        current_df = modeloff.apply(
            current_df,
            save_csv=save_outputs,
            filename="propOff_modeloff_corrected.csv"
        )
        outputs["modeloff"] = current_df.copy()
        # model-off correction overwrites into the same column names, so active_cols unchanged

    # ------------------------------------------------------------
    # Create PropOffData instance
    # ------------------------------------------------------------
    propoff = PropOffData(current_df)
    propoff.set_save_directory(BASE_DIR / save_directory)

    # Fit CD polar
    current_df = propoff.fit_cd_polar(
        save_csv=save_outputs,
        filename="propOff_with_CD_fit_values.csv",
        fit_params_filename="propOff_CD_fit_parameters.csv"
    )
    outputs["fit_cd_polar"] = current_df.copy()

    # Compute solid blockage factor
    current_df = propoff.compute_solid_blockage_e(
        save_csv=save_outputs,
        filename="propOff_solid_blockage_corrected.csv"
    )
    outputs["solid_blockage"] = current_df.copy()

    # Compute wake blockage factor
    current_df = propoff.compute_wake_blockage_e(
        cd0_col="CD0_fit",
        cd_col=active_cols["CD"],
        cl_col=active_cols["CL"],
        k_col="k_fit",
        save_csv=save_outputs,
        filename="propOff_wake_blockage_e.csv"
    )
    outputs["wake_blockage_e"] = current_df.copy()

    # Apply combined blockage correction
    current_df = propoff.apply_blockage_correction(
        apply_esb=apply_solid_blockage,
        apply_ewb=apply_wake_blockage,
        suffix="blockage_corr",
        save_csv=save_outputs,
        filename="propOff_blockage_corrected.csv"
    )
    outputs["blockage_corrected"] = current_df.copy()

    if apply_solid_blockage or apply_wake_blockage:
        active_cols["CL"]      = f"{active_cols['CL']}_blockage_corr"
        active_cols["CD"]      = f"{active_cols['CD']}_blockage_corr"
        active_cols["CMpitch"] = f"{active_cols['CMpitch']}_blockage_corr"

    # ------------------------------------------------------------
    # Tail-off data needed for SC / downwash / tail correction
    # ------------------------------------------------------------
    tailoff, _, _, _ = run_tailoff_workflow(save_outputs=False)

    # ------------------------------------------------------------
    # Optional streamline-curvature correction
    # ------------------------------------------------------------
    if apply_streamline_curvature:
        current_df = propoff.apply_streamline_curvature_correction(
            tailoff=tailoff,
            tau=0.045,
            delta=0.1065,
            geom_factor=0.2172 / 2.07,
            cl_source_col=active_cols["CL"],
            cm_source_col=active_cols["CMpitch"],
            save_csv=save_outputs,
            filename="propOff_streamline_curvature_corrected.csv"
        )
        outputs["streamline_curvature_corrected"] = current_df.copy()
        active_cols["AoA"]     = "AoA_streamline_curvature_corr"
        active_cols["CMpitch"] = f"{active_cols['CMpitch']}_sc_corr"

    # ------------------------------------------------------------
    # Optional downwash correction
    # ------------------------------------------------------------
    if apply_downwash:
        current_df = propoff.apply_downwash_correction(
            tailoff=tailoff,
            delta=0.1065,
            geom_factor=0.2172 / 2.07,
            aoa_source_col=active_cols["AoA"],
            cd_source_col=active_cols["CD"],
            save_csv=save_outputs,
            filename="propOff_downwash_corrected.csv"
        )
        outputs["downwash_corrected"] = current_df.copy()
        active_cols["AoA"] = "AoA_downwash_corr"

    # ------------------------------------------------------------
    # Optional tail correction
    # ------------------------------------------------------------
    if apply_tail_correction:
        current_df = propoff.apply_tail_correction(
            tailoff=tailoff,
            delta=0.1085,
            geom_factor=0.2172 / 2.07,
            tau2_lt=0.8 * 0.535,
            dcmpitch_dalpha=-0.15676,
            dcmpitch_dalpha_unit="per_rad",
            aoa_source_col=active_cols["AoA"],
            cmpitch_source_col=active_cols["CMpitch"],
            save_csv=save_outputs,
            filename="propOff_tail_corrected.csv"
        )
        outputs["tail_corrected"] = current_df.copy()

    current_df = propoff.rename_detected_final_force_moment_columns(
        save_csv=save_outputs,
        filename="propOff_final.csv",
        verbose=True
    )

    return propoff, current_df, outputs


if __name__ == "__main__":
    propoff, df_final, outputs = run_propoff_workflow(
        save_outputs=True,
        apply_modeloff=True,
        apply_solid_blockage=True,
        apply_wake_blockage=True,
        apply_streamline_curvature=True,
        apply_downwash=True,
        apply_tail_correction=True,
        save_directory="results_propOff_FINAL_old"
    )

    