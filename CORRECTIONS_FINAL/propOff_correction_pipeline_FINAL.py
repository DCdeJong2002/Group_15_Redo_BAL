from pathlib import Path
import pandas as pd

from correction_classes_FINAL_V import ModelOffCorrector, PropOffData
from TAILOFF_correction_pipeline_FINAL import run_tailoff_workflow


def run_propoff_workflow(
    save_outputs: bool = True,
    save_final_output: bool = True,
    verbose_flag: bool = True,
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
        "CYaw":    "CYaw",
        "CMpitch": "CMpitch",
        "CMroll":  "CMroll",
        "CMyaw":   "CMyaw",
        "AoA":     "AoA",
        "V":       "V",
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
        active_cols["CYaw"]    = f"{active_cols['CYaw']}_blockage_corr"
        active_cols["CMpitch"] = f"{active_cols['CMpitch']}_blockage_corr"
        active_cols["CMroll"]  = f"{active_cols['CMroll']}_blockage_corr"
        active_cols["CMyaw"]   = f"{active_cols['CMyaw']}_blockage_corr"
        active_cols["V"]       = f"{active_cols['V']}_blockage_corr"

    # ------------------------------------------------------------
    # Tail-off data needed for SC / downwash / tail correction
    # ------------------------------------------------------------
    tailoff,_,_,_ = run_tailoff_workflow(save_outputs=False)

    # ------------------------------------------------------------
    # Optional streamline-curvature correction
    # ------------------------------------------------------------
    if apply_streamline_curvature:
        current_df = propoff.apply_streamline_curvature_correction(
            tailoff=tailoff,
            cl_source_col=active_cols["CL"],
            cm_source_col=active_cols["CMpitch"],
            save_csv=save_outputs,
            filename="propOff_streamline_curvature_corrected.csv"
        )
        outputs["streamline_curvature_corrected"] = current_df.copy()
        active_cols["CL"]     = f"{active_cols['CL']}_sc_corr"
        active_cols["AoA"]     = f"{active_cols['AoA']}_sc_corr"
        active_cols["CMpitch"] = f"{active_cols['CMpitch']}_sc_corr"

    # ------------------------------------------------------------
    # Optional downwash correction
    # ------------------------------------------------------------
    if apply_downwash:
        current_df = propoff.apply_downwash_correction(
            tailoff=tailoff,
            aoa_source_col=active_cols["AoA"],
            cd_source_col=active_cols["CD"],
            save_csv=save_outputs,
            filename="propOff_downwash_corrected.csv"
        )
        outputs["downwash_corrected"] = current_df.copy()
        active_cols["AoA"] = f"{active_cols['AoA']}_dw_corr"
        active_cols["CD"]  = f"{active_cols['CD']}_dw_corr"

    # ------------------------------------------------------------
    # Optional tail correction
    # ------------------------------------------------------------
    if apply_tail_correction:
        current_df = propoff.apply_tail_correction(
            tailoff=tailoff,
            aoa_source_col=active_cols["AoA"],
            cmpitch_source_col=active_cols["CMpitch"],
            dcmpitch_dalpha_unit="per_deg",
            save_csv=save_outputs,
            filename="propOff_tail_corrected.csv"
        )
        outputs["tail_corrected"] = current_df.copy()
        active_cols["CMpitch"] = f"{active_cols['CMpitch']}_tail_corr"

    current_df = propoff.create_final_output_df(
        active_columns={
            "CL":      active_cols["CL"],
            "CD":      active_cols["CD"],
            "CYaw":    active_cols["CYaw"],
            "CMpitch": active_cols["CMpitch"],
            "CMroll":  active_cols["CMroll"],
            "CMyaw":   active_cols["CMyaw"],
            "AoA":     active_cols["AoA"],
            "V":       active_cols["V"],
        },
        save_csv=save_final_output,
        filename="propOff_final.csv",
        save_slim=save_final_output,
        slim_filename="propOff_final_slim.csv",
        verbose=verbose_flag,
        print_corrections=verbose_flag,
    )

    return propoff, current_df, outputs


if __name__ == "__main__":
    propoff, df_final, outputs = run_propoff_workflow(
        save_outputs=True,
        save_final_output=True,
        verbose_flag=True,
        apply_modeloff=True,
        apply_solid_blockage=True,
        apply_wake_blockage=True,
        apply_streamline_curvature=True,
        apply_downwash=True,
        apply_tail_correction=True,
        save_directory="results_propOff_FINAL"
    )

    