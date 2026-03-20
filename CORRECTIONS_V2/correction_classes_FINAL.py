from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

class BaseCorrector:
    """
    Small utility base class.

    Provides:
    - configurable save directory
    - dataframe saving helper
    - required-column validation
    - round-to-half utility
    - shared solid blockage logic
    - shared streamline-curvature logic
    """

    def __init__(self, save_dir: str | Path | None = None) -> None:
        if save_dir is None:
            self.save_dir = Path(__file__).resolve().parent
        else:
            self.save_dir = Path(save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def round_to_half(series: pd.Series) -> pd.Series:
        return np.round(series * 2.0) / 2.0

    @staticmethod
    def require_columns(df: pd.DataFrame, required_cols: Sequence[str], context: str = "") -> None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            prefix = f"{context}: " if context else ""
            raise ValueError(f"{prefix}missing required columns: {missing}")

    def set_save_directory(self, directory: str | Path) -> None:
        """
        Change the output directory for saved files.
        """
        self.save_dir = Path(directory)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_df(self, df: pd.DataFrame, filename: str) -> Path:
        out_path = self.save_dir / filename
        df.to_csv(out_path, index=False)
        print(f"Saved file: {out_path}")
        return out_path

    # ============================================================
    # Shared helper: reorder corrected columns
    # ============================================================
    def _reorder_solid_blockage_columns(self) -> None:
        """
        Reorder columns so corrected columns appear next to originals.
        Assumes the working dataframe is stored in self.df.
        """
        df = self.df
        new_order = []
        already_added = set()

        for col in df.columns:
            if col in already_added:
                continue

            new_order.append(col)
            already_added.add(col)

            corr_col = f"{col}_solid_blockage_corr"
            if corr_col in df.columns and corr_col not in already_added:
                new_order.append(corr_col)
                already_added.add(corr_col)

        for col in df.columns:
            if col not in already_added:
                new_order.append(col)

        self.df = df[new_order]

    # ============================================================
    # Shared helper: streamline curvature correction
    # ============================================================
    def _apply_streamline_curvature_common(
        self,
        tailoff,
        tau: float,
        delta: float,
        geom_factor: float = 0.2172 / 2.07,
        cl_source_col: str = "CL_solid_blockage_corr",
        cm_source_col: str = "CMpitch_solid_blockage_corr",
        dataset_label: str = "dataset",
        save_csv: bool = False,
        filename: Optional[str] = None,
        default_filename: str = "streamline_curvature_corrected.csv",
    ) -> pd.DataFrame:
        """
        Shared streamline-curvature correction using tail-off data.
        """
        if tailoff.grid_df is None:
            raise ValueError(
                "tailoff.grid_df is not available. "
                "Run tailoff.build_alpha_slice_grid_by_velocity() first."
            )
        if tailoff.cl_a_df is None:
            raise ValueError(
                "tailoff.cl_a_df is not available. "
                "Run tailoff.compute_cl_alpha_slope_by_case() first."
            )

        df = self.df.copy()

        # --------------------------------------------------------
        # Ensure rounded matching columns exist
        # --------------------------------------------------------
        if "AoA_round" not in df.columns:
            if "AoA" not in df.columns:
                raise ValueError(f"Need either 'AoA_round' or 'AoA' in {dataset_label} dataframe.")
            df["AoA_round"] = self.round_to_half(df["AoA"])

        if "AoS_round" not in df.columns:
            if "AoS" not in df.columns:
                raise ValueError(f"Need either 'AoS_round' or 'AoS' in {dataset_label} dataframe.")
            df["AoS_round"] = self.round_to_half(df["AoS"])

        if "V_round" not in df.columns:
            if "V_solid_blockage_corr" in df.columns:
                df["V_round"] = self.round_to_half(df["V_solid_blockage_corr"])
            elif "V" in df.columns:
                df["V_round"] = self.round_to_half(df["V"])
            else:
                raise ValueError(
                    f"Need one of: 'V_round', 'V_solid_blockage_corr', or 'V' "
                    f"in {dataset_label} dataframe."
                )

        self.require_columns(
            df,
            ["V_round", "AoA_round", "AoS_round", cl_source_col, cm_source_col],
            context=f"{dataset_label} streamline curvature correction",
        )

        # --------------------------------------------------------
        # Tail-off CLw lookup
        # --------------------------------------------------------
        tail_grid = tailoff.grid_df.copy()
        self.require_columns(
            tail_grid,
            ["V_round", "AoA_round", "AoS_round", "CL_solid_blockage_corr"],
            context="TailOff grid for streamline curvature correction",
        )

        tail_grid_lookup = (
            tail_grid[["V_round", "AoA_round", "AoS_round", "CL_solid_blockage_corr"]]
            .drop_duplicates(subset=["V_round", "AoA_round", "AoS_round"])
            .rename(columns={"CL_solid_blockage_corr": "CLw_tailoff"})
        )

        key_cols = ["V_round", "AoA_round", "AoS_round"]

        for col in key_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)
            if col in tail_grid_lookup.columns:
                tail_grid_lookup[col] = tail_grid_lookup[col].astype(float)

        df = df.merge(
            tail_grid_lookup,
            on=["V_round", "AoA_round", "AoS_round"],
            how="left",
        )

        # --------------------------------------------------------
        # Tail-off CL-alpha lookup
        # --------------------------------------------------------
        cl_a_df = tailoff.cl_a_df.copy()
        self.require_columns(
            cl_a_df,
            ["V_round", "AoS_round", "cl_alpha_slope_per_deg"],
            context="TailOff CL-alpha table for streamline curvature correction",
        )

        cl_a_lookup = (
            cl_a_df[["V_round", "AoS_round", "cl_alpha_slope_per_deg"]]
            .drop_duplicates(subset=["V_round", "AoS_round"])
            .rename(columns={"cl_alpha_slope_per_deg": "CL_alpha_per_deg_tailoff"})
        )

        for col in key_cols:
            if col in cl_a_lookup.columns:
                cl_a_lookup[col] = cl_a_lookup[col].astype(float)
                
        df = df.merge(
            cl_a_lookup,
            on=["V_round", "AoS_round"],
            how="left",
        )

        # --------------------------------------------------------
        # Convert CL-alpha to per radian
        # --------------------------------------------------------
        df["CL_alpha_per_rad_tailoff"] = df["CL_alpha_per_deg_tailoff"] * (180.0 / np.pi)

        # --------------------------------------------------------
        # Compute streamline-curvature corrections
        # --------------------------------------------------------
        df["delta_alpha_sc_rad"] = tau * delta * geom_factor * df["CLw_tailoff"]
        df["delta_alpha_sc_deg"] = np.degrees(df["delta_alpha_sc_rad"])

        if "AoA" in df.columns:
            df["AoA_streamline_curvature_corr"] = df["AoA"] + df["delta_alpha_sc_deg"]
        else:
            df["AoA_streamline_curvature_corr"] = df["AoA_round"] + df["delta_alpha_sc_deg"]

        df["delta_CL_sc"] = -df["delta_alpha_sc_rad"] * df["CL_alpha_per_rad_tailoff"]
        df["delta_CMpitch_sc"] = -0.25 * df["delta_CL_sc"]

        # --------------------------------------------------------
        # Apply corrections
        # --------------------------------------------------------
        df[f"{cl_source_col}_sc_corr"] = df[cl_source_col] + df["delta_CL_sc"]
        df[f"{cm_source_col}_sc_corr"] = df[cm_source_col] + df["delta_CMpitch_sc"]

        df["streamline_curvature_data_found"] = (
            df["CLw_tailoff"].notna() & df["CL_alpha_per_rad_tailoff"].notna()
        )

        self.df = df

        if save_csv:
            save_name = filename or default_filename
            self.save_df(self.df, save_name)

        return self.df
    
    def save_selected_columns(
        self,
        df: pd.DataFrame,
        columns_to_keep: Sequence[str],
        filename: str,
        allow_missing: bool = False,
    ) -> pd.DataFrame:
        """
        Keep only selected columns from a dataframe and save the result as a CSV
        in the current save directory.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        columns_to_keep : Sequence[str]
            Columns to retain, in the exact order desired in the saved file.
        filename : str
            Output CSV filename.
        allow_missing : bool
            If False, raise an error when one or more requested columns are absent.
            If True, silently keep only the columns that exist.

        Returns
        -------
        pd.DataFrame
            Reduced dataframe containing only the selected columns.
        """
        if allow_missing:
            selected_cols = [c for c in columns_to_keep if c in df.columns]
        else:
            self.require_columns(df, columns_to_keep, context="save_selected_columns")
            selected_cols = list(columns_to_keep)

        df_out = df[selected_cols].copy()
        self.save_df(df_out, filename)
        return df_out
    
    #only used for tail off
    def _apply_solid_blockage_tailoff(
        self,
        save_csv: bool = False,
        filename: Optional[str] = None,
        default_filename: str = "solid_blockage_corrected.csv",
        use_constant_e: bool = True,
        use_e_column: bool = False,
        e_column: str = "e",
    ) -> pd.DataFrame:
        """
        Shared solid blockage correction:

            V_corr = V / (1 + e)
            coeff_corr = coeff / (1 + e)^2

        Uses:
        - self.df
        - self.e_constant
        """
        df = self.df.copy()

        if use_constant_e and use_e_column:
            raise ValueError("Choose either use_constant_e=True or use_e_column=True, not both.")
        if not use_constant_e and not use_e_column:
            raise ValueError("Choose one source for e.")

        if use_constant_e:
            e = self.e_constant
            df["solid_blockage_e"] = e
        else:
            if e_column not in df.columns:
                raise ValueError(f"Column '{e_column}' not found in dataframe.")
            e = df[e_column]

        velocity_factor = 1.0 / (1.0 + e)
        coefficient_factor = 1.0 / (1.0 + e) ** 2

        velocity_columns = [c for c in ["V"] if c in df.columns]
        coefficient_columns = [
            c for c in ["CL", "CD", "CY", "CMx", "CMy", "CMz", "CYaw", "CMroll", "CMpitch", "CMyaw"]
            if c in df.columns
        ]

        for col in velocity_columns:
            df[f"{col}_solid_blockage_corr"] = df[col] * velocity_factor

        for col in coefficient_columns:
            df[f"{col}_solid_blockage_corr"] = df[col] * coefficient_factor

        self.df = df
        self._reorder_solid_blockage_columns()

        if save_csv:
            save_name = filename or default_filename
            self.save_df(self.df, save_name)

        return self.df

    def _apply_downwash_correction_common(
        self,
        tailoff,
        delta: float,
        geom_factor: float = 0.2172 / 2.07,
        aoa_source_col: Optional[str] = None,
        cd_source_col: str = "CD_solid_blockage_corr",
        dataset_label: str = "dataset",
        save_csv: bool = False,
        filename: Optional[str] = None,
        default_filename: str = "downwash_corrected.csv",
    ) -> pd.DataFrame:
        """
        Shared downwash correction using tail-off wing lift data.

        Uses
        ----
        CLw from tailoff.grid_df matched on:
            (V_round, AoA_round, AoS_round)

        Formulae
        --------
        Delta_alpha_dw_deg = delta * geom_factor * CLw * 57.3
        Delta_CD_dw        = delta * geom_factor * CLw^2

        The correction is then applied to:
        - a selected AoA source column
        - a selected CD source column
        """
        if tailoff.grid_df is None:
            raise ValueError(
                "tailoff.grid_df is not available. "
                "Run tailoff.build_alpha_slice_grid_by_velocity() first."
            )

        df = self.df.copy()

        # --------------------------------------------------------
        # Remove existing columns that may collide with merge output
        # --------------------------------------------------------
        cols_to_drop_if_present = [
            "CLw_tailoff",
            "delta_alpha_dw_deg",
            "delta_alpha_dw_rad",
            "delta_CD_dw",
            "AoA_downwash_corr",
            "downwash_data_found",
        ]
        df = df.drop(columns=cols_to_drop_if_present, errors="ignore")

        # also remove any previous suffixed versions if they exist
        suffix_collision_cols = [c for c in df.columns if c.startswith("CLw_tailoff_")]
        if suffix_collision_cols:
            df = df.drop(columns=suffix_collision_cols, errors="ignore")

        # --------------------------------------------------------
        # Ensure rounded matching columns exist
        # --------------------------------------------------------
        if "AoA_round" not in df.columns:
            if "AoA" not in df.columns:
                raise ValueError(f"Need either 'AoA_round' or 'AoA' in {dataset_label} dataframe.")
            df["AoA_round"] = self.round_to_half(df["AoA"])

        if "AoS_round" not in df.columns:
            if "AoS" not in df.columns:
                raise ValueError(f"Need either 'AoS_round' or 'AoS' in {dataset_label} dataframe.")
            df["AoS_round"] = self.round_to_half(df["AoS"])

        if "V_round" not in df.columns:
            if "V_solid_blockage_corr" in df.columns:
                df["V_round"] = self.round_to_half(df["V_solid_blockage_corr"])
            elif "V" in df.columns:
                df["V_round"] = self.round_to_half(df["V"])
            else:
                raise ValueError(
                    f"Need one of: 'V_round', 'V_solid_blockage_corr', or 'V' "
                    f"in {dataset_label} dataframe."
                )

        # --------------------------------------------------------
        # Determine AoA source column
        # --------------------------------------------------------
        if aoa_source_col is None:
            if "AoA" in df.columns:
                aoa_source_col = "AoA"
            elif "AoA_round" in df.columns:
                aoa_source_col = "AoA_round"
            else:
                raise ValueError(
                    f"No AoA column found in {dataset_label} dataframe. "
                    "Provide aoa_source_col explicitly."
                )

        self.require_columns(
            df,
            ["V_round", "AoA_round", "AoS_round", aoa_source_col, cd_source_col],
            context=f"{dataset_label} downwash correction",
        )

        # --------------------------------------------------------
        # Tail-off CLw lookup
        # --------------------------------------------------------
        tail_grid = tailoff.grid_df.copy()
        self.require_columns(
            tail_grid,
            ["V_round", "AoA_round", "AoS_round", "CL_solid_blockage_corr"],
            context="TailOff grid for downwash correction",
        )

        tail_grid_lookup = (
            tail_grid[["V_round", "AoA_round", "AoS_round", "CL_solid_blockage_corr"]]
            .drop_duplicates(subset=["V_round", "AoA_round", "AoS_round"])
            .rename(columns={"CL_solid_blockage_corr": "CLw_tailoff"})
        )

        # --------------------------------------------------------
        # Make merge keys consistent dtype
        # --------------------------------------------------------
        key_cols = ["V_round", "AoA_round", "AoS_round"]

        for col in key_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)
            if col in tail_grid_lookup.columns:
                tail_grid_lookup[col] = tail_grid_lookup[col].astype(float)

        # --------------------------------------------------------
        # Merge CLw
        # --------------------------------------------------------
        df = df.merge(
            tail_grid_lookup,
            on=["V_round", "AoA_round", "AoS_round"],
            how="left",
        )

        # safety check
        if "CLw_tailoff" not in df.columns:
            raise ValueError(
                "Downwash correction merge did not produce 'CLw_tailoff'. "
                "This usually means a column name collision occurred before merge."
            )

        # --------------------------------------------------------
        # Compute downwash corrections
        # --------------------------------------------------------
        df["delta_alpha_dw_deg"] = delta * geom_factor * df["CLw_tailoff"] * 57.3
        df["delta_alpha_dw_rad"] = np.radians(df["delta_alpha_dw_deg"])
        df["delta_CD_dw"] = delta * geom_factor * (df["CLw_tailoff"] ** 2)

        # --------------------------------------------------------
        # Apply corrections
        # --------------------------------------------------------
        df["AoA_downwash_corr"] = df[aoa_source_col] + df["delta_alpha_dw_deg"]
        df[f"{cd_source_col}_dw_corr"] = df[cd_source_col] + df["delta_CD_dw"]

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------
        df["downwash_data_found"] = df["CLw_tailoff"].notna()

        self.df = df

        if save_csv:
            save_name = filename or default_filename
            self.save_df(self.df, save_name)

        return self.df
    
    def _apply_tail_correction_common(
        self,
        tailoff,
        delta: float,
        geom_factor: float,
        tau2_lt: float,
        dcmpitch_dalpha: float,
        dcmpitch_dalpha_unit: str = "per_deg",
        aoa_source_col: Optional[str] = None,
        cmpitch_source_col: str = "CMpitch_solid_blockage_corr",
        dataset_label: str = "dataset",
        save_csv: bool = False,
        filename: Optional[str] = None,
        default_filename: str = "tail_corrected.csv",
    ) -> pd.DataFrame:
        """
        Shared tail correction using tail-off wing lift data.

        Uses
        ----
        CLw from tailoff.grid_df matched on:
            (V_round, AoA_round, AoS_round)

        Formulae
        --------
        Delta_alpha_tail = delta * geom_factor * CLw * (1 + tau2_lt)

        Delta_CMpitch_tail =
            dcmpitch_dalpha * Delta_alpha_tail

        where dcmpitch_dalpha can be supplied either per degree or per radian.

        Parameters
        ----------
        tailoff : TailOffData
            Tail-off data object with grid_df already available.
        delta : float
            Correction factor delta.
        geom_factor : float
            Geometric prefactor, e.g. S/c or equivalent.
        tau2_lt : float
            Value of tau_2(l_t).
        dcmpitch_dalpha : float
            Tail pitching-moment sensitivity derivative.
        dcmpitch_dalpha_unit : str
            Either "per_deg" or "per_rad".
        aoa_source_col : str or None
            AoA column in self.df to correct. If None, uses 'AoA' if present,
            otherwise 'AoA_round'.
        cmpitch_source_col : str
            CMpitch column in self.df to correct.
        dataset_label : str
            Label used in error messages.
        save_csv : bool
            If True, save output CSV.
        filename : str or None
            Output CSV filename.
        default_filename : str
            Default output filename if filename is not provided.

        Returns
        -------
        pd.DataFrame
            Corrected dataframe.
        """
        if tailoff.grid_df is None:
            raise ValueError(
                "tailoff.grid_df is not available. "
                "Run tailoff.build_alpha_slice_grid_by_velocity() first."
            )

        if dcmpitch_dalpha_unit not in {"per_deg", "per_rad"}:
            raise ValueError("dcmpitch_dalpha_unit must be either 'per_deg' or 'per_rad'.")

        df = self.df.copy()

        # --------------------------------------------------------
        # Remove existing columns that may collide with merge output
        # --------------------------------------------------------
        cols_to_drop_if_present = [
            "CLw_tailoff",
            "delta_alpha_tail_rad",
            "delta_alpha_tail_deg",
            "delta_CMpitch_tail",
            "AoA_tail_corr",
            "tail_correction_data_found",
        ]
        df = df.drop(columns=cols_to_drop_if_present, errors="ignore")

        suffix_collision_cols = [c for c in df.columns if c.startswith("CLw_tailoff_")]
        if suffix_collision_cols:
            df = df.drop(columns=suffix_collision_cols, errors="ignore")

        # --------------------------------------------------------
        # Ensure rounded matching columns exist
        # --------------------------------------------------------
        if "AoA_round" not in df.columns:
            if "AoA" not in df.columns:
                raise ValueError(f"Need either 'AoA_round' or 'AoA' in {dataset_label} dataframe.")
            df["AoA_round"] = self.round_to_half(df["AoA"])

        if "AoS_round" not in df.columns:
            if "AoS" not in df.columns:
                raise ValueError(f"Need either 'AoS_round' or 'AoS' in {dataset_label} dataframe.")
            df["AoS_round"] = self.round_to_half(df["AoS"])

        if "V_round" not in df.columns:
            if "V_solid_blockage_corr" in df.columns:
                df["V_round"] = self.round_to_half(df["V_solid_blockage_corr"])
            elif "V" in df.columns:
                df["V_round"] = self.round_to_half(df["V"])
            else:
                raise ValueError(
                    f"Need one of: 'V_round', 'V_solid_blockage_corr', or 'V' "
                    f"in {dataset_label} dataframe."
                )

        # --------------------------------------------------------
        # Determine AoA source column
        # --------------------------------------------------------
        if aoa_source_col is None:
            if "AoA" in df.columns:
                aoa_source_col = "AoA"
            elif "AoA_round" in df.columns:
                aoa_source_col = "AoA_round"
            else:
                raise ValueError(
                    f"No AoA column found in {dataset_label} dataframe. "
                    "Provide aoa_source_col explicitly."
                )

        self.require_columns(
            df,
            ["V_round", "AoA_round", "AoS_round", aoa_source_col, cmpitch_source_col],
            context=f"{dataset_label} tail correction",
        )

        # --------------------------------------------------------
        # Tail-off CLw lookup
        # --------------------------------------------------------
        tail_grid = tailoff.grid_df.copy()
        self.require_columns(
            tail_grid,
            ["V_round", "AoA_round", "AoS_round", "CL_solid_blockage_corr"],
            context="TailOff grid for tail correction",
        )

        tail_grid_lookup = (
            tail_grid[["V_round", "AoA_round", "AoS_round", "CL_solid_blockage_corr"]]
            .drop_duplicates(subset=["V_round", "AoA_round", "AoS_round"])
            .rename(columns={"CL_solid_blockage_corr": "CLw_tailoff"})
        )

        # --------------------------------------------------------
        # Make merge keys consistent dtype
        # --------------------------------------------------------
        key_cols = ["V_round", "AoA_round", "AoS_round"]

        for col in key_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)
            if col in tail_grid_lookup.columns:
                tail_grid_lookup[col] = tail_grid_lookup[col].astype(float)

        # --------------------------------------------------------
        # Merge CLw
        # --------------------------------------------------------
        df = df.merge(
            tail_grid_lookup,
            on=["V_round", "AoA_round", "AoS_round"],
            how="left",
        )

        if "CLw_tailoff" not in df.columns:
            raise ValueError(
                "Tail correction merge did not produce 'CLw_tailoff'. "
                "This usually means a column name collision occurred before merge."
            )

        # --------------------------------------------------------
        # Compute tail alpha correction
        # --------------------------------------------------------
        df["delta_alpha_tail_rad"] = delta * geom_factor * df["CLw_tailoff"] * (1.0 + tau2_lt)
        df["delta_alpha_tail_deg"] = np.degrees(df["delta_alpha_tail_rad"])

        # --------------------------------------------------------
        # Compute tail pitching-moment correction
        # --------------------------------------------------------
        if dcmpitch_dalpha_unit == "per_deg":
            df["delta_CMpitch_tail"] = dcmpitch_dalpha * df["delta_alpha_tail_deg"]
        else:
            df["delta_CMpitch_tail"] = dcmpitch_dalpha * df["delta_alpha_tail_rad"]

        # --------------------------------------------------------
        # Apply corrections
        # --------------------------------------------------------
        df["AoA_tail_corr"] = df[aoa_source_col] + df["delta_alpha_tail_deg"]
        df[f"{cmpitch_source_col}_tail_corr"] = df[cmpitch_source_col] + df["delta_CMpitch_tail"]

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------
        df["tail_correction_data_found"] = df["CLw_tailoff"].notna()

        self.df = df

        if save_csv:
            save_name = filename or default_filename
            self.save_df(self.df, save_name)

        return self.df

    def compute_ct_from_propon_propoff(
        self,
        propoff_df,
        cd_col_off: str = "CD",
        cl_col_off: str = "CL",
        cyaw_col_off: str = "CYaw",
        aoa_col_off: str = "AoA",
        aos_col_off: str = "AoS",
        ct_col_on: str = "CT",
        cd_col_on: str = "CD",
        aoa_col_on: str = "AoA",
        aos_col_on: str = "AoS",
        dR_col: str = "dR",
        dE_col: str = "dE",
        V_col_off: str = "V",
        aoa_round_col: str = "AoA_round",
        aos_round_col: str = "AoS_round",
        V_round_col: str = "V_round",
        ct_off_col: str = "CT_off",
        cd_off_col: str = "CD_off",
        CT_prop_col: str = "CT_prop",
        Tc_star_col: str = "Tc_star",
        dCD_net_col: str = "dCD_net_pre_corr",
        dCD_from_dCT_col: str = "dCD_from_dCT_pre_corr",
        S_wing: float = 0.2172,
        S_prop: float = np.pi * 0.25 * (0.2032 ** 2),
    ):
        """
        Compute thrust-related prop-on minus prop-off quantities by matching prop-on
        and prop-off operating conditions.

        Workflow
        --------
        1. Copy the prop-on dataframe from self.df and the prop-off dataframe from
        propoff_df.
        2. In the prop-off dataframe create rounded matching columns:
        - V_round   : rounded to nearest integer
        - AoA_round : rounded to nearest 0.5 deg
        - AoS_round : rounded to nearest integer
        3. Compute CT_off from prop-off CD, CL, CYaw, AoA, and AoS using the inverse
        of the code's force transformation:

            CT = (CD*cos(beta) - CYaw*sin(beta))*cos(alpha) - CL*sin(alpha)

        where alpha = AoA [rad] and beta = AoS [rad].

        4. Build two prop-off lookup tables:
        - a full table that matches on AoS_round, AoA_round, dR, dE, V_round
        - a fallback table that ignores dR and averages over dR

        5. Match prop-on and prop-off as follows:
        - if dR == 0, require a full match including dR
        - if dR != 0, ignore dR during matching

        6. Compute:
        - CT_prop      = CT_off - CT_on
        - Tc_star      = CT_prop * (S_wing / S_prop)
        - dCD_net      = CD_off - CD_on
        - dCD_from_dCT = CT_prop * cos(AoA) * cos(AoS)

        Notes
        -----
        - CT_prop is defined here so that a positive value corresponds to a positive
        thrust-like contribution for the sign convention where CT_off is more
        positive than CT_on.
        - dCD_net is the direct net drag-axis difference between matched prop-off and
        prop-on cases.
        - dCD_from_dCT is the drag-axis difference implied by deltaCT only, assuming
        the force change acts only through the axial coefficient.
        - This preserves your currently implemented Tc_star definition. If you want
        the Barlow/Pope definition exactly, it is usually:

            Tc_star = 0.5 * CT_prop * (S_wing / S_prop)

        because CT is based on qS and q = 0.5 * rho * V^2.
        """

        df_on = self.df.copy()
        df_off = propoff_df.copy()

        # -----------------------------
        # Required columns
        # -----------------------------
        self.require_columns(
            df_on,
            [
                ct_col_on,
                cd_col_on,
                aoa_col_on,
                aos_col_on,
                aoa_round_col,
                aos_round_col,
                dR_col,
                dE_col,
                V_round_col,
            ],
            context="prop-on CT calculation",
        )

        self.require_columns(
            df_off,
            [
                cd_col_off,
                cl_col_off,
                cyaw_col_off,
                aoa_col_off,
                aos_col_off,
                dR_col,
                dE_col,
                V_col_off,
            ],
            context="prop-off CT calculation",
        )

        # -----------------------------
        # Build rounded matching columns in prop-off
        # -----------------------------
        df_off[V_round_col] = np.round(df_off[V_col_off]).astype("Int64")
        df_off[aoa_round_col] = np.round(df_off[aoa_col_off] * 2.0) / 2.0
        df_off[aos_round_col] = np.round(df_off[aos_col_off]).astype("Int64")

        # -----------------------------
        # Compute CT_off from CD, CL, CYaw, AoA, AoS
        # CT = (CD*cos(beta) - CYaw*sin(beta))*cos(alpha) - CL*sin(alpha)
        # Also keep CD_off for later merge
        # -----------------------------
        alpha_off = np.deg2rad(pd.to_numeric(df_off[aoa_col_off], errors="coerce"))
        beta_off = np.deg2rad(pd.to_numeric(df_off[aos_col_off], errors="coerce"))

        cd_off_numeric = pd.to_numeric(df_off[cd_col_off], errors="coerce")
        cl_off_numeric = pd.to_numeric(df_off[cl_col_off], errors="coerce")
        cyaw_off_numeric = pd.to_numeric(df_off[cyaw_col_off], errors="coerce")

        df_off[ct_off_col] = (
            (cd_off_numeric * np.cos(beta_off) - cyaw_off_numeric * np.sin(beta_off)) * np.cos(alpha_off)
            - cl_off_numeric * np.sin(alpha_off)
        )
        df_off[cd_off_col] = cd_off_numeric

        # -----------------------------
        # Build reduced prop-off tables
        # - full match table includes dR
        # - fallback table ignores dR and averages over dR
        # -----------------------------
        match_cols_full = [aos_round_col, aoa_round_col, dR_col, dE_col, V_round_col]
        match_cols_no_dR = [aos_round_col, aoa_round_col, dE_col, V_round_col]

        df_off_small_full = (
            df_off[match_cols_full + [ct_off_col, cd_off_col]]
            .groupby(match_cols_full, as_index=False, dropna=False)[[ct_off_col, cd_off_col]]
            .mean()
        )

        df_off_small_no_dR = (
            df_off[match_cols_no_dR + [ct_off_col, cd_off_col]]
            .groupby(match_cols_no_dR, as_index=False, dropna=False)[[ct_off_col, cd_off_col]]
            .mean()
        )

        # -----------------------------
        # Split prop-on data
        # - if dR == 0: match including dR
        # - if dR != 0: ignore dR in matching
        # -----------------------------
        df_on_dR0 = df_on[df_on[dR_col] == 0].copy()
        df_on_dRneq0 = df_on[df_on[dR_col] != 0].copy()

        # preserve original order
        df_on_dR0["_orig_order"] = df_on_dR0.index
        df_on_dRneq0["_orig_order"] = df_on_dRneq0.index

        # -----------------------------
        # Merge cases where dR = 0
        # -----------------------------
        df_match_dR0 = df_on_dR0.merge(
            df_off_small_full,
            on=match_cols_full,
            how="left",
        )

        # -----------------------------
        # Merge cases where dR != 0
        # -----------------------------
        df_match_dRneq0 = df_on_dRneq0.merge(
            df_off_small_no_dR,
            on=match_cols_no_dR,
            how="left",
        )

        # -----------------------------
        # Combine back and restore order
        # -----------------------------
        df = pd.concat([df_match_dR0, df_match_dRneq0], ignore_index=True)
        df = df.sort_values("_orig_order").drop(columns="_orig_order").reset_index(drop=True)

        # -----------------------------
        # Compute CT_prop and Tc_star
        # -----------------------------
        df[CT_prop_col] = (
            pd.to_numeric(df[ct_off_col], errors="coerce")
            - pd.to_numeric(df[ct_col_on], errors="coerce")
        )

        df[Tc_star_col] = df[CT_prop_col] * (S_wing / S_prop)

        # -----------------------------
        # Compute drag-difference diagnostics
        # dCD_net       = direct drag-axis difference
        # dCD_from_dCT  = drag-axis difference implied by deltaCT only
        # -----------------------------
        alpha_on = np.deg2rad(pd.to_numeric(df[aoa_col_on], errors="coerce"))
        beta_on = np.deg2rad(pd.to_numeric(df[aos_col_on], errors="coerce"))

        df[dCD_net_col] = (
            pd.to_numeric(df[cd_off_col], errors="coerce")
            - pd.to_numeric(df[cd_col_on], errors="coerce")
        )

        df[dCD_from_dCT_col] = (
            pd.to_numeric(df[CT_prop_col], errors="coerce")
            * np.cos(alpha_on)
            * np.cos(beta_on)
        )

        self.df = df
        return self.df
    
    # ============================================================
    # Shared helper: compute solid blockage factor only
    # ============================================================
    def _compute_solid_blockage_e_common(
        self,
        use_constant_e: bool = True,
        use_e_column: bool = False,
        e_column: str = "e",
        output_col: str = "esb",
        save_csv: bool = False,
        filename: Optional[str] = None,
        default_filename: str = "with_esb.csv",
    ) -> pd.DataFrame:
        """
        Compute and store the solid blockage correction factor only.

        Stores:
            output_col = e_sb

        Does not yet apply the correction to coefficients.
        """
        df = self.df.copy()

        if use_constant_e and use_e_column:
            raise ValueError("Choose either use_constant_e=True or use_e_column=True, not both.")
        if not use_constant_e and not use_e_column:
            raise ValueError("Choose one source for e.")

        if use_constant_e:
            df[output_col] = self.e_constant
        else:
            if e_column not in df.columns:
                raise ValueError(f"Column '{e_column}' not found in dataframe.")
            df[output_col] = df[e_column]

        self.df = df

        if save_csv:
            save_name = filename or default_filename
            self.save_df(self.df, save_name)

        return self.df

    # ============================================================
    # Shared helper: compute wake blockage factor only
    # ============================================================
    def _compute_wake_blockage_e_common(
        self,
        cd0_col: str = "CD0_fit",
        cd_col: str = "CD",
        cl_col: str = "CL",
        k_col: str = "k_fit",
        output_col: str = "ewb",
        clip_negative_cdsep: bool = True,
        save_csv: bool = False,
        filename: Optional[str] = None,
        default_filename: str = "with_ewb.csv",
    ) -> pd.DataFrame:
        """
        Compute and store the wake blockage correction factor only.

        Formula:
            CDi   = k * CL^2
            CDsep = CD - CD0 - CDi
            e_wb  = (S/(4C))*CD0 + (5S/(4C))*CDsep

        Stores:
            CL2_fit
            CDi_fit
            CDsep_fit
            output_col = ewb
        """
        df = self.df.copy()

        required = [cd0_col, cd_col, cl_col, k_col]
        self.require_columns(df, required, context="compute wake blockage factor")

        df["CL2_fit"] = df[cl_col] ** 2
        df["CDi_fit"] = df[k_col] * df["CL2_fit"]
        df["CDsep_fit"] = df[cd_col] - df[cd0_col] - df["CDi_fit"]

        if clip_negative_cdsep:
            df["CDsep_fit"] = df["CDsep_fit"].clip(lower=0)

        df[output_col] = (
            (self.s_ref / (4.0 * self.test_section_area)) * df[cd0_col]
            + (5.0 * self.s_ref / (4.0 * self.test_section_area)) * df["CDsep_fit"]
        )

        self.df = df

        if save_csv:
            save_name = filename or default_filename
            self.save_df(self.df, save_name)

        return self.df


    # ============================================================
    # Shared helper: apply combined blockage correction
    # ============================================================
    def _apply_combined_blockage_from_e_columns(
        self,
        apply_esb: bool = True,
        apply_ewb: bool = True,
        apply_ess: bool = False,
        esb_col: str = "esb",
        ewb_col: str = "ewb",
        ess_col: str = "ess",
        velocity_cols: Sequence[str] = ("V",),
        coefficient_cols: Sequence[str] = ("CL", "CD", "CYaw", "CMroll", "CMpitch", "CMyaw"),
        suffix: str = "blockage_combined_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
        default_filename: str = "combined_blockage_corrected.csv",
    ) -> pd.DataFrame:
        """
        Apply one-step blockage correction using selected e-columns.

        Velocity:
            V_corr = V / (1 + e_total)

        Coefficients:
            C_corr = C / (1 + e_total)^2

        where:
            e_total = e_sb + e_wb + e_ss
        depending on the requested flags.
        """
        df = self.df.copy()

        e_total = pd.Series(0.0, index=df.index, dtype=float)

        if apply_esb:
            self.require_columns(df, [esb_col], context="combined blockage correction")
            e_total = e_total + df[esb_col].fillna(0.0)

        if apply_ewb:
            self.require_columns(df, [ewb_col], context="combined blockage correction")
            e_total = e_total + df[ewb_col].fillna(0.0)

        if apply_ess:
            self.require_columns(df, [ess_col], context="combined blockage correction")
            e_total = e_total + df[ess_col].fillna(0.0)

        df["e_total_blockage"] = e_total

        velocity_factor = 1.0 / (1.0 + df["e_total_blockage"])
        coefficient_factor = 1.0 / (1.0 + df["e_total_blockage"]) ** 2

        for col in velocity_cols:
            if col in df.columns:
                df[f"{col}_{suffix}"] = df[col] * velocity_factor

        for col in coefficient_cols:
            if col in df.columns:
                df[f"{col}_{suffix}"] = df[col] * coefficient_factor

        self.df = df

        if save_csv:
            save_name = filename or default_filename
            self.save_df(self.df, save_name)

        return self.df

    def rename_detected_final_force_moment_columns(
        self,
        base_cols = ("CD", "CL", "CYaw", "CMroll", "CMpitch", "CMyaw", "AoA", "V"),
        final_suffix: str = "_FINAL",
        save_csv: bool = False,
        filename: str = "final_results.csv",
        save_directory=None,
        verbose: bool = True,
    ):
        """
        Detect the most fully corrected aerodynamic force and moment columns,
        rename them to *_FINAL, and optionally save the dataframe.

        Detection logic
        ---------------
        For each base coefficient (CD, CL, CY, CMroll, CMpitch, CMyaw):

        1. Find all columns starting with that base name.
        2. Keep only columns containing the substring "corr".
        3. Among those, select the column with the largest number of
        underscore-separated parts (i.e., most corrections applied).

        Example
        -------
        Columns:

            CD
            CD_solid_blockage_corr
            CD_solid_blockage_corr_wake_corr
            CD_solid_blockage_corr_wake_corr_prop_corr

        → detected final column:

            CD_solid_blockage_corr_wake_corr_prop_corr

        which will be renamed to:

            CD_FINAL
        """

        from pathlib import Path

        df = self.df.copy()
        rename_dict = {}

        # -----------------------------
        # Detect final corrected columns
        # -----------------------------
        for base in base_cols:

            candidates = [col for col in df.columns if col.startswith(base)]

            if not candidates:
                continue

            # prefer columns containing "corr"
            corr_candidates = [c for c in candidates if "corr" in c]

            if corr_candidates:
                final_col = max(corr_candidates, key=lambda c: len(c.split("_")))
            else:
                final_col = max(candidates, key=lambda c: len(c.split("_")))

            rename_dict[final_col] = f"{base}{final_suffix}"
        # -----------------------------
        # Rename detected columns
        # -----------------------------
        if rename_dict:
            df = df.rename(columns=rename_dict)

        if verbose and rename_dict:
            print("\nDetected final force/moment columns:")
            for old, new in rename_dict.items():
                print(f"  {old} -> {new}")

        self.df = df

        # -----------------------------
        # Optional saving
        # -----------------------------
        if save_csv:

            if save_directory is None:
                save_directory = getattr(self, "save_dir", None)

            if save_directory is None:
                raise ValueError("No save directory specified.")

            save_path = Path(save_directory) / filename
            df.to_csv(save_path, index=False)

            if verbose:
                print(f"\nSaved file: {save_path}")

        return self.df



#=============================================================================================
#Model-off data
#=============================================================================================
class ModelOffCorrector(BaseCorrector):
    """
    Applies model-off correction to a dataframe.
    """

    def __init__(
        self,
        correction_csv: str | Path,
        apply_cmpitch_to_25c: bool = True,
        save_dir: str | Path | None = None,
    ) -> None:
        super().__init__(save_dir=save_dir)
        self.correction_csv = Path(correction_csv)
        self.apply_cmpitch_to_25c = apply_cmpitch_to_25c
        self.corr_df = self._load_correction_grid()

    def _load_correction_grid(self) -> pd.DataFrame:
        corr_df = pd.read_csv(self.correction_csv)

        required_corr_cols = [
            "AoA_round", "AoS_round",
            "CD", "CY", "CL", "CMroll", "CMpitch", "CMyaw"
        ]
        self.require_columns(corr_df, required_corr_cols, context="ModelOff correction grid")

        corr_df = corr_df[required_corr_cols].copy()
        corr_df = corr_df.rename(columns={
            "CD": "CD_modeloff",
            "CY": "CY_modeloff",
            "CL": "CL_modeloff",
            "CMroll": "CMroll_modeloff",
            "CMpitch": "CMpitch_modeloff",
            "CMyaw": "CMyaw_modeloff",
        })

        corr_df = corr_df.drop_duplicates(subset=["AoA_round", "AoS_round"]).reset_index(drop=True)
        return corr_df

    def apply(
        self,
        df: pd.DataFrame,
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply model-off correction to the provided dataframe.

        Uses AoA_round / AoS_round if present.
        Otherwise uses AoA / AoS rounded internally to nearest 0.5.
        Temporary merge columns are not kept in the returned dataframe.
        """
        correction_map = {
            "CD": "CD_modeloff",
            "CY": "CY_modeloff",
            "CL": "CL_modeloff",
            "CMroll": "CMroll_modeloff",
            "CMpitch": "CMpitch_modeloff",
            "CMyaw": "CMyaw_modeloff",
        }

        df_work = df.copy()

        # --------------------------------------------------------
        # Determine which columns to use for matching
        # --------------------------------------------------------
        if "AoA_round" in df_work.columns and "AoS_round" in df_work.columns:
            df_work["_AoA_merge"] = df_work["AoA_round"]
            df_work["_AoS_merge"] = df_work["AoS_round"]
        else:
            self.require_columns(df_work, ["AoA", "AoS"], context="ModelOff apply")
            df_work["_AoA_merge"] = self.round_to_half(df_work["AoA"])
            df_work["_AoS_merge"] = self.round_to_half(df_work["AoS"])

        corr_work = self.corr_df.copy()
        corr_work = corr_work.rename(columns={
            "AoA_round": "_AoA_merge",
            "AoS_round": "_AoS_merge",
        })

        # --------------------------------------------------------
        # Merge correction table
        # --------------------------------------------------------
        merged = df_work.merge(
            corr_work,
            on=["_AoA_merge", "_AoS_merge"],
            how="left",
        )

        merged["modeloff_correction_found"] = (
            merged[list(correction_map.values())].notna().all(axis=1)
        )

        # --------------------------------------------------------
        # Apply corrections
        # --------------------------------------------------------
        for data_col, corr_col in correction_map.items():
            if data_col in merged.columns:
                merged[f"{data_col}_uncorrected"] = merged[data_col]
                merged[data_col] = merged[data_col] - merged[corr_col]

        if self.apply_cmpitch_to_25c and "CMpitch25c" in merged.columns and "CMpitch_modeloff" in merged.columns:
            merged["CMpitch25c_uncorrected"] = merged["CMpitch25c"]
            merged["CMpitch25c"] = merged["CMpitch25c"] - merged["CMpitch_modeloff"]

        # --------------------------------------------------------
        # Remove temporary merge keys
        # --------------------------------------------------------
        merged = merged.drop(columns=["_AoA_merge", "_AoS_merge"], errors="ignore")

        if save_csv:
            save_name = filename or "modeloff_corrected.csv"
            self.save_df(merged, save_name)

        return merged

#=============================================================================================
#Prop-off data
#=============================================================================================
class PropOffData(BaseCorrector):
    """
    Holds and corrects a prop-off dataframe.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        e_constant: float = 0.007229438,
        s_ref: float = 0.2172,
        test_section_area: float = 2.07,
        clip_negative_cdsep: bool = True,
        save_dir: str | Path | None = None,
    ) -> None:
        super().__init__(save_dir=save_dir)
        self.df = df.copy()
        self.e_constant = e_constant
        self.s_ref = s_ref
        self.test_section_area = test_section_area
        self.clip_negative_cdsep = clip_negative_cdsep
        self.fit_df: Optional[pd.DataFrame] = None

    @staticmethod
    def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
        """
        Fit y = intercept + slope * x.
        Returns intercept, slope, predicted y, R^2.
        """
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = intercept + slope * x

        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        return intercept, slope, y_hat, r2

    def apply_streamline_curvature_correction(
        self,
        tailoff,
        tau: float,
        delta: float,
        geom_factor: float = 0.2172 / 2.07,
        cl_source_col: str = "CL_solid_blockage_corr",
        cm_source_col: str = "CMpitch_solid_blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._apply_streamline_curvature_common(
            tailoff=tailoff,
            tau=tau,
            delta=delta,
            geom_factor=geom_factor,
            cl_source_col=cl_source_col,
            cm_source_col=cm_source_col,
            dataset_label="PropOff",
            save_csv=save_csv,
            filename=filename,
            default_filename="propOff_streamline_curvature_corrected.csv",
        )
    
    
    def apply_tail_correction(
        self,
        tailoff,
        delta: float  = 0.1085,
        geom_factor: float = 0.2172 / 2.07,
        tau2_lt: float = 0.8*0.535,
        dcmpitch_dalpha: float = -0.15676,
        dcmpitch_dalpha_unit: str = "per_rad",
        aoa_source_col: Optional[str] = None,
        cmpitch_source_col: str = "CMpitch_solid_blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply tail correction using tail-off CLw data.
        """
        return self._apply_tail_correction_common(
            tailoff=tailoff,
            delta=delta,
            geom_factor=geom_factor,
            tau2_lt=tau2_lt,
            dcmpitch_dalpha=dcmpitch_dalpha,
            dcmpitch_dalpha_unit=dcmpitch_dalpha_unit,
            aoa_source_col=aoa_source_col,
            cmpitch_source_col=cmpitch_source_col,
            dataset_label="PropOff",
            save_csv=save_csv,
            filename=filename,
            default_filename="propOff_tail_corrected.csv",
        )
    
    def apply_downwash_correction(
        self,
        tailoff,
        delta: float,
        geom_factor: float = 0.2172 / 2.07,
        aoa_source_col: Optional[str] = "AoA_streamline_curvature_corr",
        cd_source_col: str = "CD_solid_blockage_corr_wake_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:

        return self._apply_downwash_correction_common(
            tailoff=tailoff,
            delta=delta,
            geom_factor=geom_factor,
            aoa_source_col=aoa_source_col,
            cd_source_col=cd_source_col,
            dataset_label="PropOff",
            save_csv=save_csv,
            filename=filename,
            default_filename="propOff_downwash_corrected.csv",
        )

        
    def fit_cd_polar(
        self,
        save_csv: bool = False,
        filename: Optional[str] = None,
        fit_params_filename: Optional[str] = None,
        v_col: str = "V",
        cl_col: str = "CL",
        cd_col: str = "CD",
        min_aoa_points: int = 4,
    ) -> pd.DataFrame:
        """
        Fit CD = CD0 + k * CL^2 grouped by:
            V_round, AoS_round, dE, dR

        Returns row-level dataframe with fit values added.
        Stores summary fit table in self.fit_df.
        """
        df = self.df.copy()

        required = ["AoA", "AoS", "dE", "dR", v_col, cl_col, cd_col]
        self.require_columns(df, required, context="PropOff fit_cd_polar")

        df["V_round"] = self.round_to_half(df[v_col])
        df["AoA_round"] = self.round_to_half(df["AoA"])
        df["AoS_round"] = self.round_to_half(df["AoS"])
        df["CL2"] = df[cl_col] ** 2

        group_cols = ["V_round", "AoS_round", "dE", "dR"]

        summary_records = []
        groups_out = []

        for keys, g in df.groupby(group_cols):
            g = g.copy().sort_values("AoA_round")

            if g["AoA_round"].nunique() < min_aoa_points:
                g["fit_used"] = False
                groups_out.append(g)

                summary_records.append({
                    "V_round": keys[0],
                    "AoS_round": keys[1],
                    "dE": keys[2],
                    "dR": keys[3],
                    "fit_used": False,
                })
                continue

            x = g["CL2"].values
            y = g[cd_col].values

            cd0, k, y_hat, r2 = self._linear_fit(x, y)

            g["fit_used"] = True
            g["CD0_fit"] = cd0
            g["k_fit"] = k
            g["R2_fit"] = r2
            g["CD_fit_pred"] = y_hat
            g["CDi_fit"] = k * g["CL2"]

            groups_out.append(g)

            summary_records.append({
                "V_round": keys[0],
                "AoS_round": keys[1],
                "dE": keys[2],
                "dR": keys[3],
                "fit_used": True,
                "CD0_fit": cd0,
                "k_fit": k,
                "R2_fit": r2,
            })

        self.fit_df = pd.DataFrame(summary_records)
        self.df = pd.concat(groups_out, ignore_index=True)

        if save_csv:
            rowlevel_name = filename or "propOff_with_CD_fit_values.csv"
            self.save_df(self.df, rowlevel_name)

            if fit_params_filename is not None:
                self.save_df(self.fit_df, fit_params_filename)

        return self.df

    def compute_solid_blockage_e(
        self,
        save_csv: bool = False,
        filename: Optional[str] = None,
        use_constant_e: bool = True,
        use_e_column: bool = False,
        e_column: str = "e",
        output_col: str = "esb",
    ) -> pd.DataFrame:
        return self._compute_solid_blockage_e_common(
            use_constant_e=use_constant_e,
            use_e_column=use_e_column,
            e_column=e_column,
            output_col=output_col,
            save_csv=save_csv,
            filename=filename,
            default_filename="propOff_with_esb.csv",
        )

    def compute_wake_blockage_e(
        self,
        cd0_col: str = "CD0_fit",
        cd_col: str = "CD",
        cl_col: str = "CL",
        k_col: str = "k_fit",
        output_col: str = "ewb",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._compute_wake_blockage_e_common(
            cd0_col=cd0_col,
            cd_col=cd_col,
            cl_col=cl_col,
            k_col=k_col,
            output_col=output_col,
            clip_negative_cdsep=self.clip_negative_cdsep,
            save_csv=save_csv,
            filename=filename,
            default_filename="propOff_with_ewb.csv",
        )

    def apply_blockage_correction(
        self,
        apply_esb: bool = True,
        apply_ewb: bool = True,
        esb_col: str = "esb",
        ewb_col: str = "ewb",
        velocity_cols: Sequence[str] = ("V",),
        coefficient_cols: Sequence[str] = (
            "CL", "CD", "CYaw", "CMroll", "CMpitch", "CMyaw"
        ),
        suffix: str = "blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._apply_combined_blockage_from_e_columns(
            apply_esb=apply_esb,
            apply_ewb=apply_ewb,
            apply_ess=False,
            esb_col=esb_col,
            ewb_col=ewb_col,
            velocity_cols=velocity_cols,
            coefficient_cols=coefficient_cols,
            suffix=suffix,
            save_csv=save_csv,
            filename=filename,
            default_filename="propOff_blockage_corrected.csv",
        )


#=============================================================================================
#prop-on data class
#=============================================================================================
class PropOnData(BaseCorrector):
    """
    Holds and corrects a prop-on dataframe.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        e_constant: float = 0.007229438,
        s_ref: float = 0.2172,
        test_section_area: float = 2.07,
        clip_negative_cdsep: bool = True,
        velocity_tolerance: float = 1.0,
        save_dir: str | Path | None = None,
    ) -> None:
        super().__init__(save_dir=save_dir)
        self.df = df.copy()
        self.e_constant = e_constant
        self.s_ref = s_ref
        self.test_section_area = test_section_area
        self.clip_negative_cdsep = clip_negative_cdsep
        self.velocity_tolerance = velocity_tolerance
        self.fit_df: Optional[pd.DataFrame] = None

    def apply_streamline_curvature_correction(
        self,
        tailoff,
        tau: float,
        delta: float,
        geom_factor: float = 0.2172 / 2.07,
        cl_source_col: str = "CL_solid_blockage_corr",
        cm_source_col: str = "CMpitch_solid_blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._apply_streamline_curvature_common(
            tailoff=tailoff,
            tau=tau,
            delta=delta,
            geom_factor=geom_factor,
            cl_source_col=cl_source_col,
            cm_source_col=cm_source_col,
            dataset_label="PropOn",
            save_csv=save_csv,
            filename=filename,
            default_filename="propOn_streamline_curvature_corrected.csv",
        )
    
    def apply_downwash_correction(
        self,
        tailoff,
        delta: float,
        geom_factor: float = 0.2172 / 2.07,
        aoa_source_col: Optional[str] = "AoA_streamline_curvature_corr",
        cd_source_col: str = "CD_solid_blockage_corr_wake_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:

        return self._apply_downwash_correction_common(
            tailoff=tailoff,
            delta=delta,
            geom_factor=geom_factor,
            aoa_source_col=aoa_source_col,
            cd_source_col=cd_source_col,
            dataset_label="PropOn",
            save_csv=save_csv,
            filename=filename,
            default_filename="propOn_downwash_corrected.csv",
        )
    
    def apply_tail_correction(
        self,
        tailoff,
        delta: float  = 0.1085,
        geom_factor: float = 0.2172 / 2.07,
        tau2_lt: float = 0.8*0.535,
        dcmpitch_dalpha: float = -0.15676,
        dcmpitch_dalpha_unit: str = "per_rad",
        aoa_source_col: Optional[str] = None,
        cmpitch_source_col: str = "CMpitch_solid_blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply tail correction using tail-off CLw data.
        """
        return self._apply_tail_correction_common(
            tailoff=tailoff,
            delta=delta,
            geom_factor=geom_factor,
            tau2_lt=tau2_lt,
            dcmpitch_dalpha=dcmpitch_dalpha,
            dcmpitch_dalpha_unit=dcmpitch_dalpha_unit,
            aoa_source_col=aoa_source_col,
            cmpitch_source_col=cmpitch_source_col,
            dataset_label="PropOn",
            save_csv=save_csv,
            filename=filename,
            default_filename="propOn_tail_corrected.csv",
        )

    def load_fit_table(self, fit_csv: str | Path) -> pd.DataFrame:
        """
        Load the prop-off fit table and store it in self.fit_df.
        """
        self.fit_df = pd.read_csv(fit_csv)

        required_fit_cols = ["V_round", "AoS_round", "dE", "dR", "CD0_fit", "k_fit"]
        self.require_columns(self.fit_df, required_fit_cols, context="PropOn load_fit_table")

        return self.fit_df

    def attach_fits(
        self,
        input_fit_df: Optional[pd.DataFrame] = None,
        fit_csv: Optional[str | Path] = None,
        save_csv: bool = False,
        filename: Optional[str] = None,
        vel_col_data: str = "V_round",
        vel_col_fit: str = "V_round",
        aos_col: str = "AoS_round",
        de_col: str = "dE",
        dr_col: str = "dR",
        fit_value_cols: Tuple[str, str] = ("CD0_fit", "k_fit"),
    ) -> pd.DataFrame:
        """
        Attach prop-off fit values to the current prop-on dataframe using
        velocity matching and sign fallback logic.

        Fallback order:
        1. exact AoS, exact dR
        2. exact AoS, flipped dR
        3. exact AoS, dR = 0
        4. flipped AoS, exact dR
        5. flipped AoS, flipped dR
        6. flipped AoS, dR = 0
        """
        if self.fit_df is None and input_fit_df is not None:
            self.fit_df = input_fit_df.copy()
        elif self.fit_df is None and fit_csv is not None:
            self.load_fit_table(fit_csv)

        if self.fit_df is None:
            raise ValueError("No fit table loaded. Provide input_fit_df or fit_csv or call load_fit_table() first.")

        df = self.df.copy()
        fit_df = self.fit_df.copy()

        # build rounded columns if needed
        if "V_round" not in df.columns:
            if "V" not in df.columns:
                raise ValueError("Need either 'V_round' or 'V' in dataframe.")
            df["V_round"] = self.round_to_half(df["V"])

        if "AoA_round" not in df.columns and "AoA" in df.columns:
            df["AoA_round"] = self.round_to_half(df["AoA"])

        if "AoS_round" not in df.columns and "AoS" in df.columns:
            df["AoS_round"] = self.round_to_half(df["AoS"])

        required_data_cols = [vel_col_data, aos_col, de_col, dr_col]
        self.require_columns(df, required_data_cols, context="PropOn attach_fits")

        df["matched_fit_velocity"] = np.nan
        df["velocity_match_error"] = np.nan
        df["fit_found"] = False
        df["fit_match_type"] = pd.Series([None] * len(df), dtype="object")
        df["matched_fit_AoS"] = np.nan
        df["matched_fit_dR"] = np.nan
        df["matched_fit_source_row"] = np.nan

        for col in fit_value_cols:
            df[col] = np.nan

        for frame in [df, fit_df]:
            frame[aos_col] = frame[aos_col].astype(float).round(3).replace(-0.0, 0.0)
            frame[de_col] = frame[de_col].astype(float).replace(-0.0, 0.0)
            frame[dr_col] = frame[dr_col].astype(float).replace(-0.0, 0.0)

        for idx, row in df.iterrows():
            aos_val = row[aos_col]
            de_val = row[de_col]
            dr_val = row[dr_col]
            v_val = row[vel_col_data]

            candidate_keys = [
                ((aos_val,  de_val,  dr_val),  "exact_AoS_exact_dR"),
                ((aos_val,  de_val, -dr_val),  "exact_AoS_flipped_dR"),
                ((aos_val,  de_val,  0.0),     "exact_AoS_zero_dR"),
                ((-aos_val, de_val,  dr_val),  "flipped_AoS_exact_dR"),
                ((-aos_val, de_val, -dr_val),  "flipped_AoS_flipped_dR"),
                ((-aos_val, de_val,  0.0),     "flipped_AoS_zero_dR"),
            ]

            for (aos_key, de_key, dr_key), label in candidate_keys:
                candidate = fit_df[
                    np.isclose(fit_df[aos_col], aos_key) &
                    np.isclose(fit_df[de_col], de_key) &
                    np.isclose(fit_df[dr_col], dr_key)
                ].copy()

                if candidate.empty:
                    continue

                candidate["vel_diff"] = np.abs(candidate[vel_col_fit] - v_val)
                best_idx = candidate["vel_diff"].idxmin()
                best_diff = candidate.loc[best_idx, "vel_diff"]

                if best_diff <= self.velocity_tolerance:
                    df.at[idx, "matched_fit_velocity"] = candidate.at[best_idx, vel_col_fit]
                    df.at[idx, "velocity_match_error"] = best_diff
                    df.at[idx, "fit_found"] = True
                    df.at[idx, "fit_match_type"] = label
                    df.at[idx, "matched_fit_AoS"] = candidate.at[best_idx, aos_col]
                    df.at[idx, "matched_fit_dR"] = candidate.at[best_idx, dr_col]
                    df.at[idx, "matched_fit_source_row"] = best_idx

                    for col in fit_value_cols:
                        df.at[idx, col] = candidate.at[best_idx, col]
                    break

        self.df = df

        if save_csv:
            save_name = filename or "propOn_with_attached_fits.csv"
            self.save_df(self.df, save_name)

        return self.df


    def compute_slipstream_blockage_e(
        self,
        tc_col: str = "Tc_star",
        dp_value: Optional[float] = 0.2032,
        sp_value: Optional[float] = None,
        tunnel_area: Optional[float] = None,
        output_col: str = "ess",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        df = self.df.copy()

        if tunnel_area is None:
            tunnel_area = self.test_section_area

        self.require_columns(df, [tc_col], context="compute slipstream blockage factor")

        if sp_value is not None:
            sp = float(sp_value)
        elif dp_value is not None:
            sp = 0.25 * np.pi * (float(dp_value) ** 2)
        else:
            raise ValueError("Provide either sp_value or dp_value.")

        tc = df[tc_col].astype(float)

        if (1.0 + 2.0 * tc < 0).any():
            raise ValueError("Encountered Tc* values for which (1 + 2 Tc*) < 0, making ess invalid.")

        df[output_col] = -(tc / (2.0 * np.sqrt(1.0 + 2.0 * tc))) * (sp / tunnel_area)

        self.df = df

        if save_csv:
            save_name = filename or "propOn_with_ess.csv"
            self.save_df(self.df, save_name)

        return self.df
        
    def compute_solid_blockage_e(
        self,
        save_csv: bool = False,
        filename: Optional[str] = None,
        use_constant_e: bool = True,
        use_e_column: bool = False,
        e_column: str = "e",
        output_col: str = "esb",
    ) -> pd.DataFrame:
        return self._compute_solid_blockage_e_common(
            use_constant_e=use_constant_e,
            use_e_column=use_e_column,
            e_column=e_column,
            output_col=output_col,
            save_csv=save_csv,
            filename=filename,
            default_filename="propOn_with_esb.csv",
        )

    def compute_wake_blockage_e(
        self,
        cd0_col: str = "CD0_fit",
        cd_col: str = "CD",
        cl_col: str = "CL",
        k_col: str = "k_fit",
        output_col: str = "ewb",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._compute_wake_blockage_e_common(
            cd0_col=cd0_col,
            cd_col=cd_col,
            cl_col=cl_col,
            k_col=k_col,
            output_col=output_col,
            clip_negative_cdsep=self.clip_negative_cdsep,
            save_csv=save_csv,
            filename=filename,
            default_filename="propOn_with_ewb.csv",
        )

    def apply_blockage_correction(
        self,
        apply_esb: bool = True,
        apply_ewb: bool = True,
        apply_ess: bool = True,
        esb_col: str = "esb",
        ewb_col: str = "ewb",
        ess_col: str = "ess",
        velocity_cols: Sequence[str] = ("V",),
        coefficient_cols: Sequence[str] = (
            "CL", "CD", "CYaw", "CMroll", "CMpitch", "CMyaw"
        ),
        suffix: str = "blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._apply_combined_blockage_from_e_columns(
            apply_esb=apply_esb,
            apply_ewb=apply_ewb,
            apply_ess=apply_ess,
            esb_col=esb_col,
            ewb_col=ewb_col,
            ess_col=ess_col,
            velocity_cols=velocity_cols,
            coefficient_cols=coefficient_cols,
            suffix=suffix,
            save_csv=save_csv,
            filename=filename,
            default_filename="propOn_blockage_corrected.csv",
        )
    
    
#=============================================================================================
#Tail-off data
#=============================================================================================

class TailOffData(BaseCorrector):
    """
    Holds and corrects a tail-off dataframe.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        e_constant: float = 0.006406642,
        s_ref: float = 0.2172,
        test_section_area: float = 2.07,
        clip_negative_cdsep: bool = True,
        save_dir: str | Path | None = None,
    ) -> None:
        super().__init__(save_dir=save_dir)
        self.df = df.copy()
        self.e_constant = e_constant
        self.s_ref = s_ref
        self.test_section_area = test_section_area
        self.clip_negative_cdsep = clip_negative_cdsep
        self.grid_df: Optional[pd.DataFrame] = None
        self.cl_a_df: Optional[pd.DataFrame] = None

    def apply_solid_blockage(
        self,
        save_csv: bool = False,
        filename: Optional[str] = None,
        use_constant_e: bool = True,
        use_e_column: bool = False,
        e_column: str = "e",
    ) -> pd.DataFrame:
        return self._apply_solid_blockage_tailoff(
            save_csv=save_csv,
            filename=filename,
            default_filename="tailOff_solid_blockage_corrected.csv",
            use_constant_e=use_constant_e,
            use_e_column=use_e_column,
            e_column=e_column,
        )

    def build_alpha_slice_grid_by_velocity(
        self,
        coeff_cols=None,
        anchor_aoa_vals=(0.0, 5.0, 10.0),
        extra_aoa_vals=[-10.0, -8.0, -5.0, -4.0, 0.0, 4.0, 5.0, 7.0, 8.0, 10.0],
        extra_aos_vals=[-5.0, -2.5, 0.0, 2.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build a full AoA_round / AoS_round grid for each V_round using the available
        AoS sweeps at fixed AoA as anchor slices, while also forcing additional
        AoA/AoS values into the grid.

        Main idea
        ---------
        For each V_round and each coefficient:
        1. Use measured values where available.
        2. Use fixed-AoA AoS sweeps (e.g. AoA = 0, 5, 10) as anchor slices.
        3. For a target (AoA_round, AoS_round), interpolate in AoA between anchor
        slice values at that same AoS_round.
        4. If the target AoA lies outside the anchor range but at least two anchor
        slices exist, do linear extrapolation in AoA.
        5. If fewer than two usable anchor values are available for a target point,
        leave it unresolved (NaN).
        6. The grid axes are formed from:
        - values already present in the tail-off data
        - plus any extra AoA/AoS values passed in

        Parameters
        ----------
        coeff_cols : list[str] or None
            Coefficient columns to reconstruct. If None, chooses available corrected
            coefficient columns automatically.
        anchor_aoa_vals : tuple[float, ...]
            Preferred AoA slices to use as anchor AoS sweeps.
            Example: (0.0, 5.0, 10.0)
        extra_aoa_vals : sequence[float] or None
            Additional AoA_round values to force into the grid.
        extra_aos_vals : sequence[float] or None
            Additional AoS_round values to force into the grid.
        save_csv : bool
            If True, save the resulting grid.
        filename : str or None
            Output CSV filename.

        Returns
        -------
        pd.DataFrame
            Grid containing:
            - V_round
            - AoA_round
            - AoS_round
            - case_available
            - source_type
            - reconstructed coefficient columns
        """
        df = self.df.copy()

        # --------------------------------------------------------
        # Ensure rounded indexing columns exist
        # --------------------------------------------------------
        if "AoA_round" not in df.columns:
            if "AoA" not in df.columns:
                raise ValueError("Need either 'AoA_round' or 'AoA' in dataframe.")
            df["AoA_round"] = self.round_to_half(df["AoA"])

        if "AoS_round" not in df.columns:
            if "AoS" not in df.columns:
                raise ValueError("Need either 'AoS_round' or 'AoS' in dataframe.")
            df["AoS_round"] = self.round_to_half(df["AoS"])

        if "V_round" not in df.columns:
            if "V_solid_blockage_corr" in df.columns:
                df["V_round"] = self.round_to_half(df["V_solid_blockage_corr"])
            elif "V" in df.columns:
                df["V_round"] = self.round_to_half(df["V"])
            else:
                raise ValueError("Need one of: 'V_round', 'V_solid_blockage_corr', or 'V'.")

        # --------------------------------------------------------
        # Default coefficient columns
        # --------------------------------------------------------
        if coeff_cols is None:
            coeff_cols = [
                c for c in [
                    "CL_solid_blockage_corr",
                    "CD_solid_blockage_corr",
                    "CY_solid_blockage_corr",
                    "CMroll_solid_blockage_corr",
                    "CMpitch_solid_blockage_corr",
                    "CMyaw_solid_blockage_corr",
                    "CL",
                    "CD",
                    "CY",
                    "CMroll",
                    "CMpitch",
                    "CMyaw",
                ]
                if c in df.columns
            ]

        if not coeff_cols:
            raise ValueError("No coefficient columns found for grid construction.")

        # normalize extra values
        if extra_aoa_vals is None:
            extra_aoa_vals = []
        if extra_aos_vals is None:
            extra_aos_vals = []

        extra_aoa_vals = np.asarray(list(extra_aoa_vals), dtype=float)
        extra_aos_vals = np.asarray(list(extra_aos_vals), dtype=float)

        # --------------------------------------------------------
        # Average duplicate rows per rounded case
        # --------------------------------------------------------
        df_avg = (
            df.groupby(["V_round", "AoA_round", "AoS_round"], as_index=False)[coeff_cols]
            .mean()
        )

        all_velocity_grids = []

        # --------------------------------------------------------
        # Helper: 1D interpolation / extrapolation in AoA
        # --------------------------------------------------------
        def interp_or_extrap_alpha(x_target: float, x_known: np.ndarray, y_known: np.ndarray):
            """
            Returns (value, source_type)
            source_type in {'interp_alpha', 'extrap_alpha', 'unresolved'}
            """
            valid = ~(np.isnan(x_known) | np.isnan(y_known))
            x = x_known[valid]
            y = y_known[valid]

            if len(x) < 2:
                return np.nan, "unresolved"

            order = np.argsort(x)
            x = x[order]
            y = y[order]

            # exact hit
            exact_mask = np.isclose(x, x_target)
            if exact_mask.any():
                return y[exact_mask][0], "interp_alpha"

            # interpolation inside range
            if x.min() <= x_target <= x.max():
                return np.interp(x_target, x, y), "interp_alpha"

            # linear extrapolation using nearest two points
            if x_target < x.min():
                x0, x1 = x[0], x[1]
                y0, y1 = y[0], y[1]
            else:
                x0, x1 = x[-2], x[-1]
                y0, y1 = y[-2], y[-1]

            if np.isclose(x1, x0):
                return np.nan, "unresolved"

            slope = (y1 - y0) / (x1 - x0)
            y_target = y0 + slope * (x_target - x0)
            return y_target, "extrap_alpha"

        # --------------------------------------------------------
        # Build one grid per velocity
        # --------------------------------------------------------
        for v_val, g_v in df_avg.groupby("V_round"):
            # union of tail-off values and forced extra values
            aoa_vals = np.sort(np.unique(np.concatenate([
                g_v["AoA_round"].unique().astype(float),
                extra_aoa_vals
            ])))
            aos_vals = np.sort(np.unique(np.concatenate([
                g_v["AoS_round"].unique().astype(float),
                extra_aos_vals
            ])))

            # full AoA/AoS grid for this velocity
            grid = pd.MultiIndex.from_product(
                [aoa_vals, aos_vals],
                names=["AoA_round", "AoS_round"]
            ).to_frame(index=False)
            grid["V_round"] = v_val

            # mark measured points
            measured_keys = g_v[["AoA_round", "AoS_round"]].copy()
            measured_keys["case_available"] = True

            grid = grid.merge(
                measured_keys,
                on=["AoA_round", "AoS_round"],
                how="left"
            )
            grid["case_available"] = grid["case_available"].fillna(False).infer_objects(copy=False)

            source_type_series = pd.Series(index=grid.index, dtype="object")

            measured_lookup = g_v.set_index(["AoA_round", "AoS_round"])

            available_anchor_aoa = sorted(
                a for a in anchor_aoa_vals
                if np.isclose(g_v["AoA_round"].unique(), a).any()
            )

            if len(available_anchor_aoa) == 0:
                raise ValueError(
                    f"At V_round={v_val}, none of the requested anchor AoA slices "
                    f"{anchor_aoa_vals} are available."
                )

            for coeff in coeff_cols:
                values = []
                local_source = []

                # create lookup: for each anchor AoA, map AoS -> coeff
                anchor_maps = {}
                for a_anchor in available_anchor_aoa:
                    g_anchor = g_v[np.isclose(g_v["AoA_round"], a_anchor)][["AoS_round", coeff]].copy()
                    g_anchor = g_anchor.dropna(subset=[coeff])
                    anchor_maps[a_anchor] = dict(zip(g_anchor["AoS_round"], g_anchor[coeff]))

                for _, row in grid.iterrows():
                    aoa_t = row["AoA_round"]
                    aos_t = row["AoS_round"]

                    # 1. exact measured point takes priority
                    if (aoa_t, aos_t) in measured_lookup.index:
                        measured_val = measured_lookup.loc[(aoa_t, aos_t), coeff]
                        if not pd.isna(measured_val):
                            values.append(measured_val)
                            local_source.append("measured")
                            continue

                    # 2. gather anchor-slice values at this AoS
                    x_known = []
                    y_known = []

                    for a_anchor in available_anchor_aoa:
                        val = anchor_maps[a_anchor].get(aos_t, np.nan)
                        if not pd.isna(val):
                            x_known.append(a_anchor)
                            y_known.append(val)

                    x_known = np.asarray(x_known, dtype=float)
                    y_known = np.asarray(y_known, dtype=float)

                    val, src = interp_or_extrap_alpha(
                        x_target=float(aoa_t),
                        x_known=x_known,
                        y_known=y_known
                    )

                    values.append(val)
                    local_source.append(src)

                grid[coeff] = values

                # worst source wins across coefficients
                if source_type_series.isna().all():
                    source_type_series[:] = local_source
                else:
                    updated = []
                    for old, new in zip(source_type_series.tolist(), local_source):
                        priority = {
                            "unresolved": 4,
                            "extrap_alpha": 3,
                            "interp_alpha": 2,
                            "measured": 1,
                            None: 0,
                        }
                        updated.append(old if priority.get(old, 0) >= priority.get(new, 0) else new)
                    source_type_series[:] = updated

            grid["source_type"] = source_type_series.fillna("unresolved")

            ordered_cols = ["V_round", "AoA_round", "AoS_round", "case_available", "source_type"]
            ordered_cols += [c for c in coeff_cols if c in grid.columns]
            other_cols = [c for c in grid.columns if c not in ordered_cols]
            grid = grid[ordered_cols + other_cols]

            all_velocity_grids.append(grid)

        grid_df = pd.concat(all_velocity_grids, ignore_index=True)
        self.grid_df = grid_df

        if save_csv:
            save_name = filename or "TAILOFF_grid_by_velocity_alpha_slice.csv"
            self.save_df(grid_df, save_name)

        return grid_df
    

    def compute_cl_alpha_slope_by_case(
        self,
        aoa_min: float,
        aoa_max: float,
        cl_col: str = "CL_solid_blockage_corr",
        aoa_col: str = "AoA_round",
        aos_col: str = "AoS_round",
        v_col: str = "V_round",
        min_points: int = 2,
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute CL-alpha slope for each distinct (V_round, AoS_round) case using
        a linear fit over a specified AoA range.

        Parameters
        ----------
        aoa_min : float
            Lower bound of AoA range used in the fit.
        aoa_max : float
            Upper bound of AoA range used in the fit.
        cl_col : str
            Name of the CL column to fit.
        aoa_col : str
            Name of the AoA column to use. Usually 'AoA_round'.
        aos_col : str
            Name of the AoS column to group by. Usually 'AoS_round'.
        v_col : str
            Name of the velocity grouping column. Usually 'V_round'.
        min_points : int
            Minimum number of unique AoA points required to perform a fit.
        save_csv : bool
            If True, save the results as CSV.
        filename : str or None
            Output CSV filename.

        Returns
        -------
        pd.DataFrame
            One row per (V_round, AoS_round) case with fit diagnostics.
        """
        if self.grid_df is None:
            raise ValueError("grid_df is not available. Run build_alpha_slice_grid_by_velocity() first.")

        df = self.grid_df.copy()

        # --------------------------------------------------------
        # Ensure required columns exist
        # --------------------------------------------------------
        if aoa_col not in df.columns:
            if aoa_col == "AoA_round" and "AoA" in df.columns:
                df["AoA_round"] = self.round_to_half(df["AoA"])
            else:
                raise ValueError(f"Column '{aoa_col}' not found in dataframe.")

        if aos_col not in df.columns:
            if aos_col == "AoS_round" and "AoS" in df.columns:
                df["AoS_round"] = self.round_to_half(df["AoS"])
            else:
                raise ValueError(f"Column '{aos_col}' not found in dataframe.")

        if v_col not in df.columns:
            if v_col == "V_round":
                if "V_solid_blockage_corr" in df.columns:
                    df["V_round"] = self.round_to_half(df["V_solid_blockage_corr"])
                elif "V" in df.columns:
                    df["V_round"] = self.round_to_half(df["V"])
                else:
                    raise ValueError(f"Column '{v_col}' not found and no fallback V column available.")
            else:
                raise ValueError(f"Column '{v_col}' not found in dataframe.")

        self.require_columns(df, [v_col, aos_col, aoa_col, cl_col], context="compute_cl_alpha_slope_by_case")

        # --------------------------------------------------------
        # Restrict to requested AoA range
        # --------------------------------------------------------
        df_fit = df[(df[aoa_col] >= aoa_min) & (df[aoa_col] <= aoa_max)].copy()

        results = []

        # --------------------------------------------------------
        # Fit per (V_round, AoS_round)
        # --------------------------------------------------------
        for (v_val, aos_val), g in df_fit.groupby([v_col, aos_col]):
            g = g[[aoa_col, cl_col]].dropna().copy()

            # Average duplicates at same AoA if needed
            g = g.groupby(aoa_col, as_index=False)[cl_col].mean()

            n_points = len(g)
            n_unique_aoa = g[aoa_col].nunique()

            if n_unique_aoa < min_points:
                results.append({
                    v_col: v_val,
                    aos_col: aos_val,
                    "aoa_min_fit": aoa_min,
                    "aoa_max_fit": aoa_max,
                    "n_points": n_points,
                    "n_unique_aoa": n_unique_aoa,
                    "cl_alpha_slope_per_deg": np.nan,
                    "cl_at_aoa0": np.nan,
                    "r2": np.nan,
                    "fit_success": False,
                })
                continue

            x = g[aoa_col].to_numpy(dtype=float)
            y = g[cl_col].to_numpy(dtype=float)

            # Linear fit: CL = intercept + slope * AoA
            slope, intercept = np.polyfit(x, y, 1)
            y_hat = intercept + slope * x

            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

            results.append({
                v_col: v_val,
                aos_col: aos_val,
                "aoa_min_fit": aoa_min,
                "aoa_max_fit": aoa_max,
                "n_points": n_points,
                "n_unique_aoa": n_unique_aoa,
                "cl_alpha_slope_per_deg": slope,
                "cl_at_aoa0": intercept,
                "r2": r2,
                "fit_success": True,
            })

        result_df = pd.DataFrame(results).sort_values([v_col, aos_col]).reset_index(drop=True)

        self.cl_a_df = result_df

        if save_csv:
            save_name = filename or "tailoff_cl_alpha_slope_by_case.csv"
            self.save_df(result_df, save_name)

        return result_df