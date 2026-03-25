from pathlib import Path
import pandas as pd

from correction_classes_FINAL import ModelOffCorrector, PropOnData
from TAILOFF_correction_pipeline_FINAL import run_tailoff_workflow
from propOff_correction_pipeline_FINAL import run_propoff_workflow



def run_propon_workflow(
    save_outputs: bool = True,
    recompute_thrust_separation_BEM: bool = True,
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
        "CFt":     None,         # produced by BEM separation, None until then
        "AoA":     "AoA",
        "CMpitch": "CMpitch",
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
    if recompute_thrust_separation_BEM:
        current_df = propon.compute_thrust_separation_BEM(
            recompute_cd=recompute_cd_for_thrust_sep,
            recompute_cl=recompute_cl_for_thrust_sep,
            recompute_cyaw=recompute_cyaw_for_thrust_sep,
        )
        outputs["thrust_separation_BEM"] = current_df.copy()

        # BEM separation produces aerodynamic-only columns; update active_cols
        # to point at those so all downstream steps use the correct inputs.
        # CFt_thrust_BEM is always produced by BEM separation.
        active_cols["CFt"] = "CFt_thrust_BEM"

        if recompute_cd_for_thrust_sep:
            active_cols["CD"] = "CD_aero_BEM"
        if recompute_cl_for_thrust_sep:
            active_cols["CL"] = "CL_aero_BEM"
        if recompute_cyaw_for_thrust_sep:
            active_cols["CYaw"] = "CYaw_aero_BEM"

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
            "CMroll":  "CMroll",
            "CMpitch": active_cols["CMpitch"],
            "CMyaw":   "CMyaw",
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
    propoff, df_propoff_final, propoff_outputs = run_propoff_workflow(
        save_outputs=False,
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
        save_csv=False,
        filename="propOn_solid_blockage_e.csv"
    )
    outputs["solid_blockage_e"] = current_df.copy()

    current_df = propon.compute_wake_blockage_e(
        cd0_col="CD0_fit",
        cd_col=active_cols["CD"],
        cl_col=active_cols["CL"],
        k_col="k_fit",
        save_csv=False,
        filename="propOn_wake_blockage_e.csv"
    )
    outputs["wake_blockage_e"] = current_df.copy()

    current_df = propon.compute_slipstream_blockage_e(
        tc_col="Tc_star_BEM",
        output_col="ess",
        save_csv=False,
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
            "CMroll",
            "CMyaw",
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

    current_df = propon.rename_detected_final_force_moment_columns(
        save_csv=True,
        filename="propOn_final.csv",
        verbose=True
    )

    return propon, current_df, outputs


if __name__ == "__main__":
    propon, df_final, outputs = run_propon_workflow(
        save_outputs=True,
        recompute_thrust_separation_BEM=True,
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
    
    import sys
    import subprocess
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    subprocess.run([
        sys.executable,                                                          # same Python that ran this script
        str(BASE_DIR / "generate_comparison_html.py"),                          # absolute path to the generator
        "--propon",  str(BASE_DIR / "results_propOn_FINAL" / "propOn_final.csv"),
        "--propoff", str(BASE_DIR / "results_propOff_FINAL"    / "propOff_final.csv"),
        "--out",     str(BASE_DIR / "results_propOn_FINAL" / "comparison.html"),
    ], check=True)