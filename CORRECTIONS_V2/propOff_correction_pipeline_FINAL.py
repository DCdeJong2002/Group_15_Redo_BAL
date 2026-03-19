from pathlib import Path
import pandas as pd

from correction_classes import ModelOffCorrector, PropOffData
from TAILOFF_correction_pipeline_FINAL import run_tailoff_workflow


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

    # ------------------------------------------------------------
    # Optional solid blockage correction
    # ------------------------------------------------------------
    if apply_solid_blockage:
        current_df = propoff.apply_solid_blockage(
            save_csv=save_outputs,
            filename="propOff_solid_blockage_corrected.csv"
        )
        outputs["solid_blockage"] = current_df.copy()

    # ------------------------------------------------------------
    # Fit CD polar
    # ------------------------------------------------------------
    # The current PropOffData wake-blockage implementation requires
    # fit parameters CD0_fit and k_fit. Those come from fit_cd_polar().
    # So if wake blockage is requested, the fit must be run first.
    if apply_wake_blockage and apply_solid_blockage:
        current_df = propoff.fit_cd_polar(
            save_csv=save_outputs,
            filename="propOff_with_CD_fit_values.csv",
            fit_params_filename="propOff_CD_fit_parameters.csv"
        )
        outputs["fit_cd_polar"] = current_df.copy()

        current_df = propoff.apply_wake_blockage(
            save_csv=save_outputs,
            filename="propOff_wake_blockage_corrected.csv"
        )
        outputs["wake_blockage"] = current_df.copy()

    elif apply_wake_blockage and not apply_solid_blockage:
        current_df = propoff.fit_cd_polar(
            save_csv=save_outputs,
            filename="propOff_with_CD_fit_values.csv",
            fit_params_filename="propOff_CD_fit_parameters.csv",
            v_col="V",
            cl_col="CL",
            cd_col="CD"
        )
        outputs["fit_cd_polar"] = current_df.copy()

        current_df = propoff.apply_wake_blockage(
            save_csv=save_outputs,
            filename="propOff_wake_blockage_corrected.csv",
            v_col="V",
            cl_col="CL",
            cd_col="CD",
            mode="raw"
        )
        outputs["wake_blockage"] = current_df.copy()

    # ------------------------------------------------------------
    # Tail-off data needed for SC / downwash / tail correction
    # ------------------------------------------------------------
    need_tailoff = apply_streamline_curvature or apply_downwash or apply_tail_correction
    tailoff = None

    if need_tailoff:
        tailoff, _, _, _ = run_tailoff_workflow(save_outputs=False)

    # ------------------------------------------------------------
    # Determine source columns after wake / no wake
    # ------------------------------------------------------------
    if apply_solid_blockage and apply_wake_blockage:
        cl_for_sc = "CL_solid_blockage_corr_wake_corr"
        cd_for_dw = "CD_solid_blockage_corr_wake_corr"
        cmpitch_base = "CMpitch_solid_blockage_corr_wake_corr"
    elif apply_solid_blockage and not apply_wake_blockage:
        cl_for_sc = "CL_solid_blockage_corr"
        cd_for_dw = "CD_solid_blockage_corr"
        cmpitch_base = "CMpitch_solid_blockage_corr"
    elif not apply_solid_blockage and apply_wake_blockage:
        cl_for_sc = "CL_wake_blockage_corr"
        cd_for_dw = "CD_wake_blockage_corr"
        cmpitch_base = "CMpitch_wake_blockage_corr"
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

    # ------------------------------------------------------------
    # Final export
    # Keep only operating variables, metadata, and
    # uncorrected + final corrected force/moment coefficients
    # ------------------------------------------------------------

    export_cols = [
        # operating variables first
        "AoA",
        "AoS",
        "dR",
        "dE",
        "V",

        # corrected angles / velocity indexing
        "AoA_round",
        "AoS_round",
        "V_round",
        "J",
        "J_round",
        "AoA_tail_corr",   # most final corrected AoA

        # measurement / flow metadata
        "dPb",
        "pBar",
        "temp",
        "rho",
        "q",
        "Re",
        "J_M1",
        "J_M2",
        "CT",

        # uncorrected + final corrected aerodynamic coefficients
        "CL_uncorrected",
        "CL_solid_blockage_corr_wake_corr_sc_corr",

        "CD_uncorrected",
        "CD_solid_blockage_corr_wake_corr_dw_corr",

        "CYaw_uncorrected",
        "CYaw_solid_blockage_corr_wake_corr",

        "CMroll_uncorrected",
        "CMroll_solid_blockage_corr_wake_corr",

        "CMpitch_uncorrected",
        "CMpitch_solid_blockage_corr_wake_corr_sc_corr_tail_corr",

        "CMyaw_uncorrected",
        "CMyaw_solid_blockage_corr_wake_corr",

        # optional extra uncorrected pitch reference
        "CMpitch25c_uncorrected",
    ]

    df_export = propoff.save_selected_columns(
        df=df_final,
        columns_to_keep=export_cols,
        filename="propOff_final_columns_rawnames.csv",
        allow_missing=True,
    )

    # ------------------------------------------------------------
    # Rename final corrected columns to clean canonical names
    # ------------------------------------------------------------
    rename_map = {
        "AoA_tail_corr": "AoA_corr",

        "CL_solid_blockage_corr_wake_corr_sc_corr": "CL",
        "CD_solid_blockage_corr_wake_corr_dw_corr": "CD",
        "CYaw_solid_blockage_corr_wake_corr": "CYaw",
        "CMroll_solid_blockage_corr_wake_corr": "CMroll",
        "CMpitch_solid_blockage_corr_wake_corr_sc_corr_tail_corr": "CMpitch",
        "CMyaw_solid_blockage_corr_wake_corr": "CMyaw",
    }

    df_export = df_export.rename(columns=rename_map)

    # ------------------------------------------------------------
    # Final clean order: each corrected coefficient next to its
    # uncorrected version
    # ------------------------------------------------------------
    final_order = [
        # operating variables first
        "AoA",
        "AoA_corr",
        "AoS",
        "dR",
        "dE",
        "V",

        # rounded / indexing
        "AoA_round",
        "AoS_round",
        "V_round",
        "J",
        "J_round",

        # coefficient pairs
        "CL_uncorrected",
        "CL",

        "CD_uncorrected",
        "CD",

        "CYaw_uncorrected",
        "CYaw",

        "CMroll_uncorrected",
        "CMroll",

        "CMpitch_uncorrected",
        "CMpitch",

        "CMyaw_uncorrected",
        "CMyaw",

        # optional extra reference
        "CMpitch25c_uncorrected",

        # metadata
        "dPb",
        "pBar",
        "temp",
        "rho",
        "q",
        "Re",
        "J_M1",
        "J_M2",
        "CT",
    ]

    df_export = df_export[[c for c in final_order if c in df_export.columns]]

    # ------------------------------------------------------------
    # Save final cleaned file
    # ------------------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent
    output_path = BASE_DIR / "FINAL_RESULTS" / "propOff_final_cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_export.to_csv(output_path, index=False)
    print(f"Saved final cleaned dataset: {output_path}")