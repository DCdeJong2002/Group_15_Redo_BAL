from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
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
    # ============================================================
    # Shared tunnel / model constants
    # All subclasses (PropOffData, PropOnData, TailOffData) inherit
    # these automatically. Override at subclass level if needed.
    # ============================================================
    TUNNEL_AREA:     float = 2.07
    WING_AREA:       float = 0.2172
    GEOM_FACTOR:     float = WING_AREA / TUNNEL_AREA

    PROP_DIAMETER:   float = 0.2032
    PROP_AREA:       float = np.pi * 0.25 * (PROP_DIAMETER ** 2)
    N_PROPS:         int = 2

    DELTA_WING:      float = 0.1065
    DELTA_TAIL:      float = 0.1085
    TAU2_WING:       float = 0.045
    TAU2_TAIL:       float = 0.8

    DCMPITCH_DALPHA: float = -0.15676 #rad

    E_SOLID_BLOCKAGE: float = 0.007229438
    E_SOLID_BLOCKAGE_TAILOFF: float = 0.006406642

    # ============================================================
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
        geom_factor: float,
        cl_source_col: str = "CL_blockage_corr",
        cm_source_col: str = "CMpitch_blockage_corr",
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
        # Compute streamline-curvature corrections
        # --------------------------------------------------------
        df["delta_alpha_sc_rad"] = tau * delta * geom_factor * df["CLw_tailoff"]
        df["delta_alpha_sc_deg"] = np.degrees(df["delta_alpha_sc_rad"])

        if "AoA" in df.columns:
            df["AoA_streamline_curvature_corr"] = df["AoA"] + df["delta_alpha_sc_deg"]
        else:
            df["AoA_streamline_curvature_corr"] = df["AoA_round"] + df["delta_alpha_sc_deg"]

        df["delta_CL_sc"] = -df["delta_alpha_sc_deg"] * df["CL_alpha_per_deg_tailoff"]
        df["delta_CMpitch_sc"] = -0.25 * df["delta_CL_sc"]

        # --------------------------------------------------------
        # Apply corrections
        # --------------------------------------------------------
        df[f"{cl_source_col}_sc_corr"] = df[cl_source_col] + df["delta_CL_sc"]
        df[f"{cm_source_col}_sc_corr"] = df[cm_source_col] + df["delta_CMpitch_sc"]

        df["streamline_curvature_data_found"] = (
            df["CLw_tailoff"].notna() & df["CL_alpha_per_deg_tailoff"].notna()
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
        geom_factor: float,
        aoa_source_col: Optional[str] = None,
        cd_source_col: str = "CD_blockage_corr",
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
        dcmpitch_dalpha_unit: str = "per_rad",
        aoa_source_col: Optional[str] = None,
        cmpitch_source_col: str = "CMpitch_blockage_corr",
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
        df["delta_alpha_tail_rad"] = delta * geom_factor * df["CLw_tailoff"] * tau2_lt
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
        df[f"{cmpitch_source_col}_tail_corr"] = df[cmpitch_source_col] - df["delta_CMpitch_tail"] #- sign convention: tail downwash reduces effective AoA and thus reduces CMpitch (makes it more negative)

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------
        df["tail_correction_data_found"] = df["CLw_tailoff"].notna()

        self.df = df

        if save_csv:
            save_name = filename or default_filename
            self.save_df(self.df, save_name)

        return self.df


    def compute_ct_thrust_from_bem(
        self,
        df: pd.DataFrame,
        j_col: str = "J",
        v_col: str = "V",
        rho_col: str = "rho",
        q_col: str = "q",
        D: float = None,
        S_wing: float = None,
        S_prop: float = None,
        n_props: int = None,
        output_cft_col: str = "CFt_thrust_BEM",
        output_tcstar_col: str = "Tc_star_BEM",
    ) -> pd.DataFrame:
        """
        Compute the propeller thrust force coefficient from the BEM polynomial
        (Appendix B, Lab Manual AE4115) and non-dimensionalise it using the
        same convention as the balance data (q * S_wing).
 
        BEM polynomial (per propeller):
            CT_bem(J) = T / (rho * n^2 * D^4)
                      = -0.0051*J^4 + 0.0959*J^3 - 0.5888*J^2 + 1.0065*J - 0.1353
 
        Steps
        -----
        1. Evaluate CT_bem per row from J.
        2. Back-calculate n [rps] from V and J:  n = V / (J * D)
        3. Convert to dimensional thrust per prop: T = CT_bem * rho * n^2 * D^4
        4. Total thrust for both props:           T_total = n_props * T
        5. Non-dimensionalise with balance convention:
               CFt_thrust = T_total / (q * S_wing)
        6. Compute Tc_star_BEM = T_total / (q * S_prop)
 
        Parameters
        ----------
        df           : input dataframe (not modified in place; updated copy returned)
        j_col        : column with advance ratio
        v_col        : column with freestream velocity [m/s]
        rho_col      : column with air density [kg/m^3]
        q_col        : column with dynamic pressure [Pa]
        D            : propeller diameter [m]
        S_wing       : wing reference area [m^2]
        S_prop       : propeller disk area [m^2] (one prop)
        n_props      : number of propellers (default 2)
        output_cft_col   : name for the CFt_thrust output column
        output_tcstar_col: name for the Tc_star_BEM output column
 
        Returns
        -------
        df : dataframe with new columns added
        """
        df = df.copy()

        if D is None:
            D = self.PROP_DIAMETER
        if S_wing is None:
            S_wing = self.WING_AREA
        if S_prop is None:
            S_prop = self.PROP_AREA
        if n_props is None:
            n_props = self.N_PROPS
 
        self.require_columns(
            df,
            [j_col, v_col, rho_col, q_col],
            context="compute_ct_thrust_from_bem",
        )
 
        J     = pd.to_numeric(df[j_col],   errors="coerce")
        V     = pd.to_numeric(df[v_col],   errors="coerce")
        rho   = pd.to_numeric(df[rho_col], errors="coerce")
        q     = pd.to_numeric(df[q_col],   errors="coerce")
 
        # BEM polynomial: CT_bem = T / (rho * n^2 * D^4)  [per propeller]
        CT_bem = (
            -0.0051 * J**4
            + 0.0959 * J**3
            - 0.5888 * J**2
            + 1.0065 * J
            - 0.1353
        )
 
        # Back-calculate n [rps] from J = V / (n * D)
        # Guard against J=0 or NaN to avoid division by zero
        n_rps = V / (J.replace(0, np.nan) * D)
 
        # Dimensional thrust per propeller [N]
        T_one = CT_bem * rho * n_rps**2 * D**4
 
        # Total thrust [N] for all propellers
        T_total = n_props * T_one
 
        # Non-dimensionalise using balance convention: F / (q * S_wing)
        df[output_cft_col] = T_total / (q * S_wing)
 
        # Tc_star using BEM thrust: T_total / (q * S_prop)
        df[output_tcstar_col] = T_total / (q * S_prop)
 
        return df
 
    def compute_thrust_separation_BEM(
        self,
        ct_col_on: str = "CT",
        aoa_col_on: str = "AoA",
        aos_col_on: str = "AoS",
        S_wing: float = None,
        S_prop: float = None,
        D: float = None,
        n_props: int = None,
        recompute_cd: bool = True,
        recompute_cl: bool = True,
        recompute_cyaw: bool = True,
    ):
        """
        Separate propeller thrust from the measured axial body-frame force
        coefficient (CT) using the BEM polynomial from Appendix B of the
        AE4115 lab manual, and optionally recompute wind-axis coefficients
        with thrust removed.
 
        Workflow
        --------
        1. Evaluate the BEM polynomial CT_bem(J) per row to get the isolated
           propeller thrust coefficient (per propeller, T/(rho*n^2*D^4)).
        2. Convert to a body-frame force coefficient using the balance
           non-dimensionalisation (q * S_wing), summed over both propellers:
               CFt_thrust_BEM = T_total / (q * S_wing)
        3. Compute Tc_star_BEM = T_total / (q * S_prop).
        4. Subtract BEM thrust from the measured CT to isolate the aerodynamic
           axial force (airframe drag + interference drag, no thrust):
               CFt_aero_BEM = CT_measured - CFt_thrust_BEM
        5. Reproject CFt_aero_BEM to wind axes (BAL_forces formulae) for any
           coefficient whose recompute flag is True:
               CL_aero_BEM   = CFn*cos(alpha) - CFt_aero*sin(alpha)
               CD_aero_BEM   = (CFn*sin(alpha) + CFt_aero*cos(alpha))*cos(beta)
                               + CFs*sin(beta)
               CYaw_aero_BEM = -(CFn*sin(alpha) + CFt_aero*cos(alpha))*sin(beta)
                               + CFs*cos(beta)
           CFn (CN) and CFs (CY) are unchanged by propeller thrust in all cases.
 
        Always-added columns
        --------------------
        CFt_thrust_BEM  : BEM propeller thrust as body-frame force coefficient
        Tc_star_BEM     : thrust loading coefficient  T_total / (q * S_prop)
        CFt_aero_BEM    : body-frame axial aero force with thrust removed
 
        Conditionally-added columns (controlled by flags)
        --------------------------------------------------
        CD_aero_BEM   : wind-axis drag with thrust removed   (recompute_cd=True)
        CL_aero_BEM   : wind-axis lift with thrust removed   (recompute_cl=True)
        CYaw_aero_BEM : wind-axis yaw force with thrust removed (recompute_cyaw=True)
 
        Parameters
        ----------
        ct_col_on      : column name for measured axial body-frame force coefficient
        aoa_col_on     : column name for angle of attack [deg]
        aos_col_on     : column name for angle of sideslip [deg]
        S_wing         : wing reference area [m^2]
        S_prop         : propeller disk area for one propeller [m^2]
        recompute_cd   : if True, compute CD_aero_BEM  (default True)
        recompute_cl   : if True, compute CL_aero_BEM  (default True)
        recompute_cyaw : if True, compute CYaw_aero_BEM (default False)
        """
 
        df = self.df.copy()

        if D is None:
            D = self.PROP_DIAMETER
        if S_wing is None:
            S_wing = self.WING_AREA
        if S_prop is None:
            S_prop = self.PROP_AREA
        if n_props is None:
            n_props = self.N_PROPS
 
        self.require_columns(
            df,
            [ct_col_on, aoa_col_on, aos_col_on, "J", "V", "rho", "q", "CN", "CY", "CT"],
            context="compute_thrust_separation_BEM",
        )
 
        # -------------------------------------------------------------
        # Step 1–3: BEM thrust coefficient and Tc_star
        # -------------------------------------------------------------
        df = self.compute_ct_thrust_from_bem(
            df=df,
            j_col="J",
            v_col="V",
            rho_col="rho",
            q_col="q",
            D=D,
            S_wing=S_wing,
            S_prop=S_prop,
            n_props=n_props,
            output_cft_col="CFt_thrust_BEM",
            output_tcstar_col="Tc_star_BEM",
        )
 
        alpha = np.deg2rad(pd.to_numeric(df[aoa_col_on], errors="coerce"))
        beta  = np.deg2rad(pd.to_numeric(df[aos_col_on], errors="coerce"))
 
        # Step 4: Add BEM thrust back to recover aerodynamic-only axial force.
        #   CT_measured = aero_drag - thrust   (thrust acts forward, drag acts aft)
        #   CFt_aero    = CT_measured + thrust = aero drag with thrust removed
        #   Expected: positive (drag-like) once thrust is stripped out

        ct_on   = pd.to_numeric(df[ct_col_on],        errors="coerce")
        cft_bem = pd.to_numeric(df["CFt_thrust_BEM"], errors="coerce")

        df["CFt_aero_BEM"] = ct_on + cft_bem
 
        # -------------------------------------------------------------
        # Step 5: Reproject to wind axes using BAL_forces formulae.
        # CFn (CN) and CFs (CY) are unchanged by propeller thrust.
        # -------------------------------------------------------------
        CFn      = pd.to_numeric(df["CN"], errors="coerce")
        CFs      = pd.to_numeric(df["CY"], errors="coerce")
        CFt_aero = df["CFt_aero_BEM"]
 
        if recompute_cd:
            df["CD_aero_BEM"] = (
                (CFn * np.sin(alpha) + CFt_aero * np.cos(alpha)) * np.cos(beta)
                + CFs * np.sin(beta)
            )
 
        if recompute_cl:
            df["CL_aero_BEM"] = (
                CFn * np.cos(alpha) - CFt_aero * np.sin(alpha)
            )
 
        if recompute_cyaw:
            df["CYaw_aero_BEM"] = (
                -(CFn * np.sin(alpha) + CFt_aero * np.cos(alpha)) * np.sin(beta)
                + CFs * np.cos(beta)
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
            (self.WING_AREA / (4.0 * self.TUNNEL_AREA)) * df[cd0_col]
            + (5.0 * self.WING_AREA / (4.0 * self.TUNNEL_AREA)) * df["CDsep_fit"]
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
        suffix: str = "blockage_corr",
        cft_thrust_col: str = "CFt_thrust_BEM",
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

        CFt_thrust_BEM receives solid + wake blockage only (never slipstream),
        because it is the thrust coefficient that *causes* slipstream blockage
        rather than a measurement contaminated by it.
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

        velocity_factor    = 1.0 / (1.0 + df["e_total_blockage"])
        coefficient_factor = 1.0 / (1.0 + df["e_total_blockage"]) ** 2

        for col in velocity_cols:
            if col in df.columns:
                df[f"{col}_{suffix}"] = df[col] * velocity_factor

        for col in coefficient_cols:
            if col in df.columns:
                df[f"{col}_{suffix}"] = df[col] * coefficient_factor

        # ----------------------------------------------------------------
        # CFt_thrust_BEM: apply solid + wake blockage only, never slipstream.
        # e_esb_ewb is recomputed from just those two components so that
        # this correction is independent of the apply_ess flag.
        # ----------------------------------------------------------------
        if cft_thrust_col in df.columns:
            e_esb_ewb = pd.Series(0.0, index=df.index, dtype=float)
            if apply_esb:
                e_esb_ewb = e_esb_ewb + df[esb_col].fillna(0.0)
            if apply_ewb:
                e_esb_ewb = e_esb_ewb + df[ewb_col].fillna(0.0)
            cft_factor = 1.0 / (1.0 + e_esb_ewb) ** 2
            df[f"{cft_thrust_col}_{suffix}"] = df[cft_thrust_col] * cft_factor

        self.df = df

        if save_csv:
            save_name = filename or default_filename
            self.save_df(self.df, save_name)

        return self.df

    def rename_detected_final_force_moment_columns(
        self,
        base_cols = ("CD", "CL", "CYaw", "CMroll", "CMpitch", "CMyaw", "AoA", "V", "CFt_thrust_BEM"),
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
            "CD", "CYaw", "CL", "CMroll", "CMpitch", "CMyaw"
        ]
        self.require_columns(corr_df, required_corr_cols, context="ModelOff correction grid")

        corr_df = corr_df[required_corr_cols].copy()
        corr_df = corr_df.rename(columns={
            "CD": "CD_modeloff",
            "CYaw": "CYaw_modeloff",
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
        source_columns: Optional[dict[str, str]] = None,
        cmpitch25c_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply model-off correction to the provided dataframe.

        Uses AoA_round / AoS_round if present.
        Otherwise uses AoA / AoS rounded internally to nearest 0.5.
        Temporary merge columns are not kept in the returned dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe to correct.
        save_csv : bool, optional
            If True, save the corrected dataframe.
        filename : str, optional
            Output filename if save_csv=True.
        source_columns : dict[str, str], optional
            Mapping from canonical coefficient names to the actual dataframe
            columns that should be corrected.

            Canonical names allowed:
                "CD", "CYaw", "CL", "CMroll", "CMpitch", "CMyaw"

            Example
            -------
            source_columns = {
                "CD": "CD_SC",
                "CL": "CL_SC",
                "CMpitch": "CMpitch_SC"
            }

            Then the correction becomes:
                CD_SC      = CD_SC      - CD_modeloff
                CL_SC      = CL_SC      - CL_modeloff
                CMpitch_SC = CMpitch_SC - CMpitch_modeloff

            If None, the default behaviour is used:
                CD      = CD      - CD_modeloff
                CYaw    = CYaw    - CYaw_modeloff
                CL      = CL      - CL_modeloff
                CMroll  = CMroll  - CMroll_modeloff
                CMpitch = CMpitch - CMpitch_modeloff
                CMyaw   = CMyaw   - CMyaw_modeloff

        cmpitch25c_column : str, optional
            Column name to use for the 25%-chord pitching moment correction.
            If None, defaults to "CMpitch25c" when present.
        """
        correction_map = {
            "CD": "CD_modeloff",
            "CYaw": "CYaw_modeloff",
            "CL": "CL_modeloff",
            "CMroll": "CMroll_modeloff",
            "CMpitch": "CMpitch_modeloff",
            "CMyaw": "CMyaw_modeloff",
        }

        # Default: correct the standard columns themselves
        default_source_columns = {
            "CD": "CD",
            "CYaw": "CYaw",
            "CL": "CL",
            "CMroll": "CMroll",
            "CMpitch": "CMpitch",
            "CMyaw": "CMyaw",
        }

        if source_columns is None:
            source_columns = default_source_columns.copy()
        else:
            # Start from defaults and overwrite only what the user specified
            source_columns = {**default_source_columns, **source_columns}

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
        for canonical_name, corr_col in correction_map.items():
            data_col = source_columns.get(canonical_name)

            if data_col in merged.columns:
                merged[f"{data_col}_uncorrected"] = merged[data_col]
                merged[data_col] = merged[data_col] - merged[corr_col]

        # --------------------------------------------------------
        # Optional CMpitch25c correction
        # --------------------------------------------------------
        if self.apply_cmpitch_to_25c and "CMpitch_modeloff" in merged.columns:
            target_cmpitch25c_col = cmpitch25c_column or "CMpitch25c"

            if target_cmpitch25c_col in merged.columns:
                merged[f"{target_cmpitch25c_col}_uncorrected"] = merged[target_cmpitch25c_col]
                merged[target_cmpitch25c_col] = (
                    merged[target_cmpitch25c_col] - merged["CMpitch_modeloff"]
                )

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
        clip_negative_cdsep: bool = True,
        save_dir: str | Path | None = None,
    ) -> None:
        super().__init__(save_dir=save_dir)
        self.df = df.copy()
        self.clip_negative_cdsep = clip_negative_cdsep
        self.fit_df: Optional[pd.DataFrame] = None
        self.e_constant = self.E_SOLID_BLOCKAGE

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
        tau: float = None,
        delta: float = None,
        geom_factor: float = None,
        cl_source_col: str = "CL_blockage_corr",
        cm_source_col: str = "CMpitch_blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        
        #Get parameters, using defaults if not provided
        tau            = tau            if tau        is not None else self.TAU2_WING
        delta          = delta          if delta          is not None else self.DELTA_WING
        geom_factor    = geom_factor    if geom_factor    is not None else self.GEOM_FACTOR

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
        delta: float  = None,
        geom_factor: float = None,
        tau2_lt: float = None,
        dcmpitch_dalpha: float = None,
        dcmpitch_dalpha_unit: str = "per_rad",
        aoa_source_col: Optional[str] = None,
        cmpitch_source_col: str = "CMpitch_blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply tail correction using tail-off CLw data.
        """
        #Get parameters, using defaults if not provided
        delta = delta if delta is not None else self.DELTA_TAIL
        geom_factor = geom_factor if geom_factor is not None else self.GEOM_FACTOR
        tau2_lt = tau2_lt if tau2_lt is not None else self.TAU2_TAIL
        dcmpitch_dalpha = dcmpitch_dalpha if dcmpitch_dalpha is not None else self.DCMPITCH_DALPHA

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
        delta: float = None,
        geom_factor: float = None,
        aoa_source_col: Optional[str] = "AoA_streamline_curvature_corr",
        cd_source_col: str = "CD_blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        
        #Get parameters, using defaults if not provided
        delta          = delta          if delta          is not None else self.DELTA_WING
        geom_factor    = geom_factor    if geom_factor    is not None else self.GEOM_FACTOR

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
        clip_negative_cdsep: bool = True,
        velocity_tolerance: float = 1.0,
        save_dir: str | Path | None = None,
    ) -> None:
        super().__init__(save_dir=save_dir)
        self.df = df.copy()
        self.clip_negative_cdsep = clip_negative_cdsep
        self.velocity_tolerance = velocity_tolerance
        self.fit_df: Optional[pd.DataFrame] = None
        self.e_constant = self.E_SOLID_BLOCKAGE

    def apply_streamline_curvature_correction(
        self,
        tailoff,
        tau: float = None,
        delta: float =None,
        geom_factor: float = None,
        cl_source_col: str = "CL_aero_BEM_blockage_corr",
        cm_source_col: str = "CMpitch_blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        
        #Get parameters, using defaults if not provided
        tau            = tau            if tau        is not None else self.TAU2_WING
        delta          = delta          if delta          is not None else self.DELTA_WING
        geom_factor    = geom_factor    if geom_factor    is not None else self.GEOM_FACTOR

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
        delta: float = None,
        geom_factor: float = None,
        aoa_source_col: Optional[str] = "AoA_streamline_curvature_corr",
        cd_source_col: str = "CD_blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:

        #Get parameters, using defaults if not provided
        delta          = delta          if delta          is not None else self.DELTA_WING
        geom_factor    = geom_factor    if geom_factor    is not None else self.GEOM_FACTOR

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
        delta: float  = None,  #delta of tail
        geom_factor: float = None,
        tau2_lt: float = None,    #tau_2 of tail
        dcmpitch_dalpha: float = None,
        dcmpitch_dalpha_unit: str = "per_rad",
        aoa_source_col: Optional[str] = None,
        cmpitch_source_col: str = "CMpitch_blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply tail correction using tail-off CLw data.
        """
        #Get parameters, using defaults if not provided
        delta = delta if delta is not None else self.DELTA_TAIL
        geom_factor = geom_factor if geom_factor is not None else self.GEOM_FACTOR
        tau2_lt = tau2_lt if tau2_lt is not None else self.TAU2_TAIL
        dcmpitch_dalpha = dcmpitch_dalpha if dcmpitch_dalpha is not None else self.DCMPITCH_DALPHA

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
        tc_col: str = "Tc_star_BEM",
        D_prop: Optional[float] = None,
        S_prop: Optional[float] = None,
        tunnel_area: Optional[float] = None,
        output_col: str = "ess",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        df = self.df.copy()

        tunnel_area = tunnel_area if tunnel_area is not None else self.TUNNEL_AREA
        D_prop = D_prop if D_prop is not None else self.PROP_DIAMETER
        S_prop = S_prop if S_prop is not None else (0.25 * np.pi * D_prop ** 2)

        self.require_columns(df, [tc_col], context="compute slipstream blockage factor")

        tc = df[tc_col].astype(float)

        if (1.0 + 2.0 * tc < 0).any():
            raise ValueError("Encountered Tc* values for which (1 + 2 Tc*) < 0, making ess invalid.")

        df[output_col] = -(tc / (2.0 * np.sqrt(1.0 + 2.0 * tc))) * (S_prop / tunnel_area)

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
        cft_thrust_col: str = "CFt_thrust_BEM",
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
            cft_thrust_col=cft_thrust_col,
            suffix=suffix,
            save_csv=save_csv,
            filename=filename,
            default_filename="propOn_blockage_corrected.csv",
        )
    
    def apply_modeloff_correction(
        self,
        modeloff_corrector,
        save_csv: bool = False,
        filename: Optional[str] = None,
        source_columns: Optional[dict[str, str]] = None,
        cmpitch25c_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply model-off correction to the current prop-on dataframe.

        Parameters
        ----------
        modeloff_corrector : ModelOffCorrector
            An initialized ModelOffCorrector instance containing the correction grid.
        save_csv : bool, optional
            If True, save the corrected dataframe.
        filename : str, optional
            Output filename if save_csv=True.
        source_columns : dict[str, str], optional
            Mapping from canonical coefficient names to the actual dataframe
            columns that should be corrected.

            Allowed canonical keys:
                "CD", "CYaw", "CL", "CMroll", "CMpitch", "CMyaw"

            Example
            -------
            source_columns = {
                "CD": "CD_SC",
                "CL": "CL_SC",
                "CMpitch": "CMpitch_SC",
            }

            Then:
                CD_SC      = CD_SC      - CD_modeloff
                CL_SC      = CL_SC      - CL_modeloff
                CMpitch_SC = CMpitch_SC - CMpitch_modeloff

            If None, the default columns are corrected:
                CD, CYaw, CL, CMroll, CMpitch, CMyaw
        cmpitch25c_column : str, optional
            Column name to use for the 25%-chord pitching moment correction.
            If None, defaults to "CMpitch25c" when present.

        Returns
        -------
        pd.DataFrame
            The corrected dataframe, also stored in self.df.
        """
        self.df = modeloff_corrector.apply(
            df=self.df,
            save_csv=False,
            filename=None,
            source_columns=source_columns,
            cmpitch25c_column=cmpitch25c_column,
        )

        if save_csv:
            save_name = filename or "propOn_modeloff_corrected.csv"
            self.save_df(self.df, save_name)

        return self.df
    
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
        clip_negative_cdsep: bool = True,
        save_dir: str | Path | None = None,
    ) -> None:
        super().__init__(save_dir=save_dir)
        self.df = df.copy()
        self.e_constant = self.E_SOLID_BLOCKAGE_TAILOFF 
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