from pathlib import Path
import pandas as pd

from correction_classes_FINAL_V import PropOnData, ModelOffCorrector
from TAILOFF_correction_pipeline_FINAL import run_tailoff_workflow
from propOff_correction_pipeline_FINAL import run_propoff_workflow


def run_propon_workflow(
    save_outputs: bool = True,
    save_final_output: bool = True,
    verbose_flag: bool = True,
    recompute_thrust_separation: bool = True,
    ct_corr_type: str = "EXP", #choose from "EXP" or "BEM"
    recompute_cd_for_thrust_sep: bool = True,
    recompute_cl_for_thrust_sep: bool = True,
    recompute_cyaw_for_thrust_sep: bool = True,
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
    on which earlier corrections were applied, tracked via `active_cols`.

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
    # Active column tracker
    # Holds the name of the most recently corrected version of each
    # logical quantity. Each correction step updates the relevant keys.
    #
    # CYaw starts pointing at the raw column. It will be overwritten
    # to "CYaw_aero_BEM" if BEM thrust separation is run with
    # recompute_cyaw_for_thrust_sep=True.
    #
    # CFt starts as None because it only exists after BEM separation.
    # Any step that needs it should guard with:
    #   if active_cols["CFt"] is not None
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
        "CFt":     None,
    }

    # ------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------
    prop_on_raw = pd.read_csv(BASE_DIR / "INPUT_BALANCE_DATA" / "all_rudder_cases_combined.csv")
    outputs["raw"] = prop_on_raw.copy()

    current_df = prop_on_raw.copy()

    # ------------------------------------------------------------
    # Initialize PropOnData instance
    # ------------------------------------------------------------
    propon = PropOnData(current_df)
    propon.set_save_directory(BASE_DIR / save_directory)

    # ------------------------------------------------------------
    # Compute BEM thrust separation BEFORE model-off correction
    # ------------------------------------------------------------
    current_df = propon.compute_thrust_separation_BEM(
        recompute_cd=recompute_cd_for_thrust_sep,
        recompute_cl=recompute_cl_for_thrust_sep,
        recompute_cyaw=recompute_cyaw_for_thrust_sep,
    )
    outputs["thrust_separation_BEM"] = current_df.copy()

    current_df = propon.compute_thrust_separation_EXP(
        recompute_cd=recompute_cd_for_thrust_sep,
        recompute_cl=recompute_cl_for_thrust_sep,
        recompute_cyaw=recompute_cyaw_for_thrust_sep,
        exp_ct_path=BASE_DIR / "INPUT_BALANCE_DATA" / "Ct_V_exp_data.csv",
    )

    outputs["thrust_separation_EXP"] = current_df.copy()

    #Update column names to use after this
    if recompute_thrust_separation and ct_corr_type=="BEM":
        active_cols["CFt"] = "CFt_thrust_BEM"
        if recompute_cd_for_thrust_sep:
            active_cols["CD"] = "CD_aero_BEM"
        if recompute_cl_for_thrust_sep:
            active_cols["CL"] = "CL_aero_BEM"
        if recompute_cyaw_for_thrust_sep:
            active_cols["CYaw"] = "CYaw_aero_BEM"
    elif recompute_thrust_separation and ct_corr_type=="EXP":
        active_cols["CFt"] = "CFt_thrust_EXP"
        if recompute_cd_for_thrust_sep:
            active_cols["CD"] = "CD_aero_EXP"
        if recompute_cl_for_thrust_sep:
            active_cols["CL"] = "CL_aero_EXP"
        if recompute_cyaw_for_thrust_sep:
            active_cols["CYaw"] = "CYaw_aero_EXP"
    elif recompute_thrust_separation:
        raise ValueError("recompute_thrust_separation was set to True but no resulting data column were selected")
    
    # ------------------------------------------------------------
    # After initializing propoff compute the delta_CT
    # ------------------------------------------------------------
    propoff_raw_df = pd.read_csv(BASE_DIR / "INPUT_BALANCE_DATA" / "propOff.csv")

    current_df = propon.compute_delta_CT_from_propoff(
        df_propoff=propoff_raw_df,          
    )

    # ------------------------------------------------------------
    # Optional model-off correction
    # ------------------------------------------------------------
    if apply_modeloff:
        modeloff = ModelOffCorrector(
            correction_csv=BASE_DIR / "INPUT_BALANCE_DATA" / "model_off_corrections_grid.csv",
            save_dir=BASE_DIR / save_directory
        )

        source_columns = {
            "CD":      active_cols["CD"],
            "CL":      active_cols["CL"],
            "CYaw":    active_cols["CYaw"],
            "CMroll":  active_cols["CMroll"],
            "CMpitch": active_cols["CMpitch"],
            "CMyaw":   active_cols["CMyaw"],
        }

        current_df = propon.apply_modeloff_correction(
            modeloff_corrector=modeloff,
            source_columns=source_columns,
            cmpitch25c_column="CMpitch25c",
            save_csv=save_outputs,
            filename="propOn_modeloff_corrected.csv",
        )
        outputs["modeloff"] = current_df.copy()
        # model-off correction overwrites into the same column names, so active_cols unchanged

    # ------------------------------------------------------------
    # Get prop-off workflow results (needed for CD polar fit parameters)
    # ------------------------------------------------------------
    propoff,_,_ = run_propoff_workflow(
        save_outputs=False,
        save_final_output=False,
        verbose_flag=False,
        apply_modeloff=True,
        apply_solid_blockage=True,
        apply_wake_blockage=True,
        apply_streamline_curvature=False,
        apply_downwash=False,
        apply_tail_correction=False,
        save_directory="results_propOff_temp"
    )

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
        save_csv=save_outputs,
        filename="propOn_solid_blockage_e.csv"
    )
    outputs["solid_blockage_e"] = current_df.copy()

    current_df = propon.compute_wake_blockage_e(
        cd0_col="CD0_fit",
        cd_col=active_cols["CD"],
        cl_col=active_cols["CL"],
        k_col="k_fit",
        save_csv=save_outputs,
        filename="propOn_wake_blockage_e.csv"
    )
    outputs["wake_blockage_e"] = current_df.copy()

    current_df = propon.compute_slipstream_blockage_e(
        tc_col="Tc_star_BEM",
        output_col="ess",
        save_csv=save_outputs,
        filename="propOn_slipstream_blockage_e.csv"
    )
    outputs["slipstream_blockage_e"] = current_df.copy()

    # ------------------------------------------------------------
    # Apply combined blockage correction
    #
    # The standard force/moment coefficients (CL, CD, CYaw, CMpitch,
    # CMroll, CMyaw) receive all requested blockage components
    # (esb + ewb + ess).
    #
    # CFt_thrust_BEM receives solid + wake blockage only (never
    # slipstream), because it is the thrust coefficient that *causes*
    # slipstream blockage rather than a measurement contaminated by it.
    # The function handles this internally via the cft_thrust_col arg.
    # ------------------------------------------------------------
    current_df = propon.apply_blockage_correction(
        apply_esb=apply_solid_blockage,
        apply_ewb=apply_wake_blockage,
        apply_ess=apply_slipstream_blockage,
        coefficient_cols=(
            active_cols["CL"],
            active_cols["CD"],
            active_cols["CYaw"],
            active_cols["CMpitch"],
            active_cols["CMroll"],
            active_cols["CMyaw"],
        ),
        cft_thrust_col=active_cols["CFt"] if active_cols["CFt"] is not None else "",
        suffix="blockage_corr",
        save_csv=save_outputs,
        filename="propOn_blockage_corrected.csv"
    )
    outputs["blockage_corrected"] = current_df.copy()


    # CORRECT — builds on whatever active_cols currently holds:
    if apply_solid_blockage or apply_wake_blockage or apply_slipstream_blockage:
        active_cols["CL"]      = f"{active_cols['CL']}_blockage_corr"
        active_cols["CD"]      = f"{active_cols['CD']}_blockage_corr"
        active_cols["CYaw"]    = f"{active_cols['CYaw']}_blockage_corr"
        active_cols["CMpitch"] = f"{active_cols['CMpitch']}_blockage_corr"
        active_cols["CMroll"]  = f"{active_cols['CMroll']}_blockage_corr"
        active_cols["CMyaw"]   = f"{active_cols['CMyaw']}_blockage_corr"
        active_cols["V"]       = f"{active_cols['V']}_blockage_corr"

    if active_cols["CFt"] is not None and (apply_solid_blockage or apply_wake_blockage):
        active_cols["CFt"] = f"{active_cols['CFt']}_blockage_corr"

    # ------------------------------------------------------------
    # Tail-off data needed for SC / downwash / tail correction
    # ------------------------------------------------------------
    tailoff, _, _, _ = run_tailoff_workflow(save_outputs=False)

    # ------------------------------------------------------------
    # Optional streamline-curvature correction
    # ------------------------------------------------------------
    if apply_streamline_curvature:
        current_df = propon.apply_streamline_curvature_correction(
            tailoff=tailoff,
            cl_source_col=active_cols["CL"],
            cm_source_col=active_cols["CMpitch"],
            save_csv=save_outputs,
            filename="propOn_streamline_curvature_corrected.csv"
        )
        outputs["streamline_curvature"] = current_df.copy()
        active_cols["AoA"]     = f"{active_cols['AoA']}_sc_corr"
        active_cols["CMpitch"] = f"{active_cols['CMpitch']}_sc_corr"
        active_cols["CL"]     = f"{active_cols['CL']}_sc_corr"

    # ------------------------------------------------------------
    # Optional downwash correction
    # ------------------------------------------------------------
    if apply_downwash:
        current_df = propon.apply_downwash_correction(
            tailoff=tailoff,
            aoa_source_col=active_cols["AoA"],
            cd_source_col=active_cols["CD"],
            save_csv=save_outputs,
            filename="propOn_downwash_corrected.csv"
        )
        outputs["downwash"] = current_df.copy()
        active_cols["AoA"] = f"{active_cols['AoA']}_dw_corr"
        active_cols["CD"]  = f"{active_cols['CD']}_dw_corr"

    # ------------------------------------------------------------
    # Optional tail correction
    # ------------------------------------------------------------
    if apply_tail_correction:
        current_df = propon.apply_tail_correction(
            tailoff=tailoff,
            aoa_source_col=active_cols["AoA"],
            cmpitch_source_col=active_cols["CMpitch"],
            save_csv=save_outputs,
            filename="propOn_tail_corrected.csv"
        )
        outputs["tail_correction"] = current_df.copy()
        active_cols["AoA"]     = f"{active_cols['AoA']}_tail_corr"
        active_cols["CMpitch"] = f"{active_cols['CMpitch']}_tail_corr"


    active_columns_map = {
        "CL":      active_cols["CL"],
        "CD":      active_cols["CD"],
        "CYaw":    active_cols["CYaw"],
        "CMpitch": active_cols["CMpitch"],
        "CMroll":  active_cols["CMroll"],
        "CMyaw":   active_cols["CMyaw"],
        "AoA":     active_cols["AoA"],
        "V":       active_cols["V"],
    }
    if active_cols["CFt"] in propon.df.columns:
        active_columns_map["CFt"] = active_cols["CFt"]

    current_df = propon.create_final_output_df(
        active_columns=active_columns_map,
        save_csv=save_final_output,
        filename="propOn_final.csv",
        save_slim=save_final_output,
        slim_filename="propOn_final_slim.csv",
        verbose=verbose_flag,
        print_corrections=verbose_flag,
    )

    return propon, current_df, outputs


if __name__ == "__main__":
    propon, df_final, outputs = run_propon_workflow(
        save_outputs=True,
        save_final_output=True,
        verbose_flag=True,
        recompute_thrust_separation=False,
        ct_corr_type="EXP",
        recompute_cd_for_thrust_sep=True,
        recompute_cl_for_thrust_sep=True,
        recompute_cyaw_for_thrust_sep=True,
        apply_modeloff=True,
        apply_solid_blockage=True,
        apply_wake_blockage=True,
        apply_slipstream_blockage=True,
        apply_streamline_curvature=True,
        apply_downwash=True,
        apply_tail_correction=True,
        save_directory="results_propOn_FINAL"
    )
    

    if True:
        from generate_comparison_html_extended import load_and_build, generate_html
        from pathlib import Path

        BASE_DIR = Path(__file__).resolve().parent

        (cmp_rows, off_rows, meta, j_colors, v_colors,
        cn_beta_on, cn_beta_off, cn_dr_on, cn_dr_off,
        polar_on, polar_off, trim_rows, sc_meta) = load_and_build(
            propon_path  = BASE_DIR / "results_propOn_FINAL" / "propOn_final.csv",
            propoff_path = BASE_DIR / "results_propOff_FINAL" / "propOff_final.csv",
        )

        generate_html(
            cmp_rows, off_rows, meta, j_colors, v_colors,
            cn_beta_on, cn_beta_off, cn_dr_on, cn_dr_off,
            polar_on, polar_off, trim_rows, sc_meta,
            out_path = BASE_DIR / "results_propOn_FINAL" / "comparison_extended.html",
        )