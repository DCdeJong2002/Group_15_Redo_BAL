from pathlib import Path
import pandas as pd

from correction_classes_FINAL import ModelOffCorrector, PropOnData
from TAILOFF_correction_pipeline_FINAL_V2 import run_tailoff_workflow
from propOff_correction_pipeline_FINAL_V2 import run_propoff_workflow

def run_propon_workflow(
    save_outputs: bool = True,
    apply_modeloff: bool = True,
    apply_solid_blockage: bool = True,
    apply_wake_blockage: bool = True,
    apply_slipstream_blockage: bool = True,
    apply_streamline_curvature: bool = True,
    apply_downwash: bool = True,
    apply_tail_correction: bool = True,
    save_directory: str = "results_propOn"
):
    """
    Run the prop-on correction workflow with optional correction steps.

    The function automatically chooses the correct source columns depending
    on which earlier corrections were applied.

    Returns
    -------
    propon : PropOnData
        Final PropOnData object with updated self.df
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
    prop_on_raw = pd.read_csv(BASE_DIR / "INPUT_BALANCE_DATA" / "all_rudder_cases_combined.csv")
    prop_off_raw = pd.read_csv(BASE_DIR / "INPUT_BALANCE_DATA" / "propOff.csv")
    outputs["raw"] = prop_on_raw.copy()

    current_df = prop_on_raw.copy()

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
            filename="propOn_modeloff_corrected.csv"
        )
        outputs["modeloff"] = current_df.copy()

    # ------------------------------------------------------------
    # Create PropOnData instance
    # ------------------------------------------------------------
    propon = PropOnData(current_df)
    propon.set_save_directory(BASE_DIR / save_directory)

    # ------------------------------------------------------------
    # Get prop-off workflow results (needed for fit parameters)
    # ------------------------------------------------------------
    propoff, df_propoff_final, propoff_outputs = run_propoff_workflow(
        save_outputs=False,
        apply_modeloff=apply_modeloff,
        apply_solid_blockage=False,
        apply_wake_blockage=False,
        apply_streamline_curvature=False,
        apply_downwash=False,
        apply_tail_correction=False,
        save_directory="results_propOff_temp"
    )

    # ------------------------------------------------------------
    # Compute CT_prop and Tc_star
    # ------------------------------------------------------------
    current_df = propon.compute_ct_from_propon_propoff(
        propoff_df=prop_off_raw
    )
    outputs["ct_computed"] = current_df.copy()

    cols = [
        "AoA_round",
        "AoS_round",
        "V_round",
        "dR",
        "dE",
        "J_round",
        "CD",
        "CT",
        "CT_off",
        "CD_off",
        "CT_prop",
        "Tc_star",
        "dCD_net_pre_corr",
        "dCD_from_dCT_pre_corr"
    ]

    current_df[cols].to_csv(r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\test.csv", index=False)
    current_df.to_csv(r"C:\Users\douwe\AE4115-23_EXPERIMENTAL SIMULATIONS\Group_15_Redo_BAL\test_full.csv", index=False)
    
    
    # ------------------------------------------------------------
    # Attach prop-off CD polar fit parameters
    # ------------------------------------------------------------

    if propoff.fit_df is None:
        raise ValueError("Prop-off fit table is not available from run_propoff_workflow().")

    propon.fit_df = propoff.fit_df.copy()

    current_df = propon.attach_fits(
        input_fit_df=propoff.fit_df,
        save_csv=False,
        filename="propOn_with_attached_fits.csv"
    )
    outputs["attached_fits"] = current_df.copy()

    # ------------------------------------------------------------
    # Compute blockage factors
    # ------------------------------------------------------------
    current_df = propon.compute_solid_blockage_e(
        save_csv=False,
        filename="propOn_solid_blockage_e.csv"
    )
    outputs["solid_blockage_e"] = current_df.copy()

    #compute wake blockage e correction
    current_df = propon.compute_wake_blockage_e(
        cd0_col="CD0_fit",
        cd_col="CD",
        cl_col="CL",
        k_col="k_fit",
        save_csv=False,
        filename="propOn_wake_blockage_e.csv"
    )
    outputs["wake_blockage_e"] = current_df.copy()

    #Compute slipstream blockage e if CT_prop is available
    current_df = propon.compute_slipstream_blockage_e(
        tc_col="Tc_star",
        dp_value=0.2032,
        tunnel_area=2.07,
        output_col="ess",
        save_csv=False,
        filename="propOn_slipstream_blockage_e.csv"
    )
    outputs["slipstream_blockage_e"] = current_df.copy()

    # ------------------------------------------------------------
    # Apply combined blockage correction
    # ------------------------------------------------------------
    current_df = propon.apply_blockage_correction(
        apply_esb=apply_solid_blockage,
        apply_ewb=apply_wake_blockage,
        apply_ess=apply_slipstream_blockage,
        suffix="blockage_corr",
        save_csv=save_outputs,
        filename="propOn_blockage_corrected.csv"
    )
    outputs["blockage_corrected"] = current_df.copy()

    # ------------------------------------------------------------
    # Tail-off data needed for SC / downwash / tail correction
    # ------------------------------------------------------------
    tailoff, _, _, _ = run_tailoff_workflow(save_outputs=False)

    # ------------------------------------------------------------
    # Determine source columns after blockage / no blockage
    # ------------------------------------------------------------
    if apply_solid_blockage or apply_wake_blockage or apply_slipstream_blockage:
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
        current_df = propon.apply_streamline_curvature_correction(
            tailoff=tailoff,
            tau=0.045,
            delta=0.1065,
            geom_factor=0.2172 / 2.07,
            cl_source_col=cl_for_sc,
            cm_source_col=cmpitch_base,
            save_csv=save_outputs,
            filename="propOn_streamline_curvature_corrected.csv"
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
        current_df = propon.apply_downwash_correction(
            tailoff=tailoff,
            delta=0.1065,
            geom_factor=0.2172 / 2.07,
            aoa_source_col=aoa_for_downwash,
            cd_source_col=cd_for_dw,
            save_csv=save_outputs,
            filename="propOn_downwash_corrected.csv"
        )
        outputs["downwash"] = current_df.copy()
        aoa_for_tail = "AoA_downwash_corr"
    else:
        aoa_for_tail = aoa_for_downwash

    # ------------------------------------------------------------
    # Optional tail correction
    # ------------------------------------------------------------
    if apply_tail_correction:
        current_df = propon.apply_tail_correction(
            tailoff=tailoff,
            delta=0.1085,
            geom_factor=0.2172 / 2.07,
            tau2_lt=0.8 * 0.535,
            dcmpitch_dalpha=-0.15676,
            dcmpitch_dalpha_unit="per_rad",
            aoa_source_col=aoa_for_tail,
            cmpitch_source_col=cmpitch_for_tail,
            save_csv=save_outputs,
            filename="propOn_tail_corrected.csv"
        )
        outputs["tail_correction"] = current_df.copy()

    current_df = propon.rename_detected_final_force_moment_columns(
        save_csv=save_outputs,
        filename="propOn_final.csv",
        verbose=True
    )

    return propon, current_df, outputs


if __name__ == "__main__":
    propon, df_final, outputs = run_propon_workflow(
        save_outputs=True,
        apply_modeloff=True,
        apply_solid_blockage=True,
        apply_wake_blockage=True,
        apply_slipstream_blockage=True,
        apply_streamline_curvature=True,
        apply_downwash=True,
        apply_tail_correction=True,
        save_directory="results_propOn_FINAL"
    )