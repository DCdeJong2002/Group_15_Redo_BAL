from pathlib import Path
import pandas as pd

from correction_classes_FINAL import ModelOffCorrector, PropOffData
from TAILOFF_correction_pipeline_FINAL_V2 import run_tailoff_workflow


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
    on which earlier corrections were applied.

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

    # ------------------------------------------------------------
    # Create PropOffData instance
    # ------------------------------------------------------------
    propoff = PropOffData(current_df)
    propoff.set_save_directory(BASE_DIR / save_directory)

    #Fit CD polar
    current_df = propoff.fit_cd_polar(
        save_csv=save_outputs,
        filename="propOff_with_CD_fit_values.csv",
        fit_params_filename="propOff_CD_fit_parameters.csv"
    )
    outputs["fit_cd_polar"] = current_df.copy()

    #Find solid blockage correction
    current_df = propoff.compute_solid_blockage_e(
        save_csv=save_outputs,  
        filename="propOff_solid_blockage_corrected.csv"
    )
    outputs["solid_blockage"] = current_df.copy()

    #Find wake blockage e correction
    current_df = propoff.compute_wake_blockage_e(
        cd0_col="CD0_fit",
        cd_col="CD",
        cl_col="CL",
        k_col="k_fit",
        save_csv=save_outputs,
        filename="propOff_wake_blockage_e.csv"
    )
    outputs["wake_blockage_e"] = current_df.copy()

    #apply blockage corrections
    current_df = propoff.apply_blockage_correction(
        apply_esb=apply_solid_blockage,
        apply_ewb=apply_wake_blockage,
        suffix="blockage_corr",
        save_csv=save_outputs,
        filename="propOff_blockage_corrected.csv"
    )
    outputs["blockage_corrected"] = current_df.copy()


    # ------------------------------------------------------------
    # Tail-off data needed for SC / downwash / tail correction
    # ------------------------------------------------------------
    tailoff, _, _, _ = run_tailoff_workflow(save_outputs=False)

    # ------------------------------------------------------------
    # Determine source columns after wake / no wake
    # ------------------------------------------------------------
    if apply_solid_blockage or apply_wake_blockage:
        cl_for_sc = "CL_blockage_corr"
        cd_for_dw = "CD_blockage_corr"
        cmpitch_base = "CMpitch_blockage_corr"
    else:
        cl_for_sc = "CL"
        cd_for_dw = "CD"
        cmpitch_base = "CMpitch"

    # ------------------------------------------------------------
    # Optional streamline-curvature correction
    # ------------------------------------------------------------
    if apply_streamline_curvature:
        current_df = propoff.apply_streamline_curvature_correction(
            tailoff=tailoff,
            tau=0.045,
            delta=0.1065,
            geom_factor=0.2172 / 2.07,
            cl_source_col=cl_for_sc,
            cm_source_col=cmpitch_base,
            save_csv=save_outputs,
            filename="propOff_streamline_curvature_corrected.csv"
        )
        outputs["streamline_curvature"] = current_df.copy()
        aoa_for_downwash = "AoA_streamline_curvature_corr"
        cmpitch_for_tail = f"{cmpitch_base}_sc_corr"
    else:
        if "AoA" in current_df.columns:
            aoa_for_downwash = "AoA"
        else:
            aoa_for_downwash = "AoA_round"
        cmpitch_for_tail = cmpitch_base

    # ------------------------------------------------------------
    # Optional downwash correction
    # ------------------------------------------------------------
    if apply_downwash:
        current_df = propoff.apply_downwash_correction(
            tailoff=tailoff,
            delta=0.1065,
            geom_factor=0.2172 / 2.07,
            aoa_source_col=aoa_for_downwash,
            cd_source_col=cd_for_dw,
            save_csv=save_outputs,
            filename="propOff_downwash_corrected.csv"
        )
        outputs["downwash"] = current_df.copy()
        aoa_for_tail = "AoA_downwash_corr"
    else:
        aoa_for_tail = aoa_for_downwash

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
            aoa_source_col=aoa_for_tail,
            cmpitch_source_col=cmpitch_for_tail,
            save_csv=save_outputs,
            filename="propOff_tail_corrected.csv"
        )
        outputs["tail_correction"] = current_df.copy()

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
        save_directory="results_propOff_FINAL"
    )

    