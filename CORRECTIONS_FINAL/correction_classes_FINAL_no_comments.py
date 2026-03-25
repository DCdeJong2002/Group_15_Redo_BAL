from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

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
    - shared downwash logic
    - shared tail correction logic
    - shared blockage factor computation
    - BEM thrust separation
    """

    # ============================================================
    # Shared tunnel / model constants
    # All subclasses (PropOffData, PropOnData, TailOffData) inherit
    # these automatically. Override at subclass level if needed.
    # ============================================================
    TUNNEL_AREA:              float = 2.07
    WING_AREA:                float = 0.2172
    GEOM_FACTOR:              float = WING_AREA / TUNNEL_AREA

    PROP_DIAMETER:            float = 0.2032
    PROP_AREA:                float = np.pi * 0.25 * (PROP_DIAMETER ** 2)
    N_PROPS:                  int   = 2

    DELTA_WING:               float = 0.1065
    DELTA_TAIL:               float = 0.1085
    TAU2_WING:                float = 0.045
    TAU2_TAIL:                float = 0.8 

    DCMPITCH_DALPHA:          float = -0.15676   # per rad

    E_SOLID_BLOCKAGE:         float = 0.007229438
    E_SOLID_BLOCKAGE_TAILOFF: float = 0.006406642

    # ============================================================
    def __init__(self, save_dir: str | Path | None = None) -> None:
        if save_dir is None:
            self.save_dir = Path(__file__).resolve().parent
        else:
            self.save_dir = Path(save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
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
        """Change the output directory for saved files."""
        self.save_dir = Path(directory)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_df(self, df: pd.DataFrame, filename: str) -> Path:
        out_path = self.save_dir / filename
        df.to_csv(out_path, index=False)
        print(f"Saved file: {out_path}")
        return out_path

    # ============================================================
    # Shared helper: streamline curvature correction
    # ============================================================
    def _apply_streamline_curvature_common(
        self,
        tailoff,
        tau: float = None,
        delta: float = None,
        geom_factor: float = None,
        cl_source_col: str = "CL_blockage_corr",
        cm_source_col: str = "CMpitch_blockage_corr",
        dataset_label: str = "dataset",
        save_csv: bool = False,
        filename: Optional[str] = None,
        default_filename: str = "streamline_curvature_corrected.csv",
    ) -> pd.DataFrame:
        """Shared streamline-curvature correction using tail-off data."""

        # Resolve defaults from class constants if not explicitly passed
        tau         = tau         if tau         is not None else self.TAU2_WING
        delta       = delta       if delta       is not None else self.DELTA_WING
        geom_factor = geom_factor if geom_factor is not None else self.GEOM_FACTOR

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
        df = self._ensure_round_keys(df, dataset_label)

        self.require_columns(
            df,
            ["V_round", "AoA_round", "AoS_round", cl_source_col, cm_source_col],
            context=f"{dataset_label} streamline curvature correction",
        )

        # --------------------------------------------------------
        # Tail-off CLw lookup
        # --------------------------------------------------------
        df = self._merge_clw_tailoff(df, tailoff, context="streamline curvature correction")

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

        for col in ["V_round", "AoS_round"]:
            if col in cl_a_lookup.columns:
                cl_a_lookup[col] = cl_a_lookup[col].astype(float)

        df = df.merge(cl_a_lookup, on=["V_round", "AoS_round"], how="left")

 
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

        df[f"{cl_source_col}_sc_corr"] = df[cl_source_col] + df["delta_CL_sc"]
        df[f"{cm_source_col}_sc_corr"] = df[cm_source_col] + df["delta_CMpitch_sc"]

        df["streamline_curvature_data_found"] = (
            df["CLw_tailoff"].notna() & df["CL_alpha_per_deg_tailoff"].notna()
        )

        self.df = df

        if save_csv:
            self.save_df(self.df, filename or default_filename)

        return self.df

    # ============================================================
    # Shared helper: downwash correction
    # ============================================================
    def _apply_downwash_correction_common(
        self,
        tailoff,
        delta: float = None,
        geom_factor: float = None,
        aoa_source_col: Optional[str] = None,
        cd_source_col: str = "CD_blockage_corr",
        dataset_label: str = "dataset",
        save_csv: bool = False,
        filename: Optional[str] = None,
        default_filename: str = "downwash_corrected.csv",
    ) -> pd.DataFrame:
        """
        Shared downwash correction using tail-off wing lift data.

        Formulae
        --------
        Delta_alpha_dw_deg = delta * geom_factor * CLw * 57.3
        Delta_CD_dw        = delta * geom_factor * CLw^2
        """

        # Resolve defaults from class constants if not explicitly passed
        delta       = delta       if delta       is not None else self.DELTA_WING
        geom_factor = geom_factor if geom_factor is not None else self.GEOM_FACTOR

        if tailoff.grid_df is None:
            raise ValueError(
                "tailoff.grid_df is not available. "
                "Run tailoff.build_alpha_slice_grid_by_velocity() first."
            )

        df = self.df.copy()

        # --------------------------------------------------------
        # Remove existing columns that may collide with merge output
        # --------------------------------------------------------
        cols_to_drop = [
            "CLw_tailoff", "delta_alpha_dw_deg", "delta_alpha_dw_rad",
            "delta_CD_dw", "AoA_downwash_corr", "downwash_data_found",
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        df = df.drop(columns=[c for c in df.columns if c.startswith("CLw_tailoff_")], errors="ignore")

        # --------------------------------------------------------
        # Ensure rounded matching columns exist
        # --------------------------------------------------------
        df = self._ensure_round_keys(df, dataset_label)

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
        df = self._merge_clw_tailoff(df, tailoff, context="downwash correction")

        # --------------------------------------------------------
        # Compute and apply downwash corrections
        # --------------------------------------------------------
        df["delta_alpha_dw_deg"] = delta * geom_factor * df["CLw_tailoff"] * 57.3
        df["delta_alpha_dw_rad"] = np.radians(df["delta_alpha_dw_deg"])
        df["delta_CD_dw"]        = delta * geom_factor * (df["CLw_tailoff"] ** 2)

        df["AoA_downwash_corr"]              = df[aoa_source_col] + df["delta_alpha_dw_deg"]
        df[f"{cd_source_col}_dw_corr"]       = df[cd_source_col]  + df["delta_CD_dw"]
        df["downwash_data_found"]            = df["CLw_tailoff"].notna()

        self.df = df

        if save_csv:
            self.save_df(self.df, filename or default_filename)

        return self.df

    # ============================================================
    # Shared helper: tail correction
    # ============================================================
    def _apply_tail_correction_common(
        self,
        tailoff,
        delta: float = None,
        geom_factor: float = None,
        tau2_lt: float = None,
        dcmpitch_dalpha: float = None,
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

        Formulae
        --------
        Delta_alpha_tail   = delta * geom_factor * CLw * tau2_lt
        Delta_CMpitch_tail = dcmpitch_dalpha * Delta_alpha_tail
        """

        # Resolve defaults from class constants if not explicitly passed
        delta           = delta           if delta           is not None else self.DELTA_TAIL
        geom_factor     = geom_factor     if geom_factor     is not None else self.GEOM_FACTOR
        tau2_lt         = tau2_lt         if tau2_lt         is not None else self.TAU2_TAIL
        dcmpitch_dalpha = dcmpitch_dalpha if dcmpitch_dalpha is not None else self.DCMPITCH_DALPHA

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
        cols_to_drop = [
            "CLw_tailoff", "delta_alpha_tail_rad", "delta_alpha_tail_deg",
            "delta_CMpitch_tail", "AoA_tail_corr", "tail_correction_data_found",
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        df = df.drop(columns=[c for c in df.columns if c.startswith("CLw_tailoff_")], errors="ignore")

        # --------------------------------------------------------
        # Ensure rounded matching columns exist
        # --------------------------------------------------------
        df = self._ensure_round_keys(df, dataset_label)

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
        df = self._merge_clw_tailoff(df, tailoff, context="tail correction")

        # --------------------------------------------------------
        # Compute tail corrections
        # --------------------------------------------------------
        df["delta_alpha_tail_rad"] = delta * geom_factor * df["CLw_tailoff"] * tau2_lt
        df["delta_alpha_tail_deg"] = np.degrees(df["delta_alpha_tail_rad"])

        if dcmpitch_dalpha_unit == "per_deg":
            df["delta_CMpitch_tail"] = dcmpitch_dalpha * df["delta_alpha_tail_deg"]
        else:
            df["delta_CMpitch_tail"] = dcmpitch_dalpha * df["delta_alpha_tail_rad"]

        df["AoA_tail_corr"] = df[aoa_source_col] + df["delta_alpha_tail_deg"]
        # minus sign: tail downwash reduces effective AoA → reduces CMpitch (more negative)
        df[f"{cmpitch_source_col}_tail_corr"] = df[cmpitch_source_col] - df["delta_CMpitch_tail"]
        df["tail_correction_data_found"]      = df["CLw_tailoff"].notna()

        self.df = df

        if save_csv:
            self.save_df(self.df, filename or default_filename)

        return self.df

    # ============================================================
    # Shared helper: ensure rounded key columns exist
    # ============================================================
    def _ensure_round_keys(self, df: pd.DataFrame, dataset_label: str = "dataset") -> pd.DataFrame:
        """
        Ensure AoA_round, AoS_round, and V_round columns exist on df.
        Creates them from raw columns if not already present.
        """
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

        return df

    # ============================================================
    # Shared helper: merge CLw from tail-off grid
    # ============================================================
    def _merge_clw_tailoff(
        self,
        df: pd.DataFrame,
        tailoff,
        context: str = "",
    ) -> pd.DataFrame:
        """
        Left-merge CLw_tailoff onto df from tailoff.grid_df,
        matching on (V_round, AoA_round, AoS_round).
        """
        tail_grid = tailoff.grid_df.copy()
        self.require_columns(
            tail_grid,
            ["V_round", "AoA_round", "AoS_round", "CL_solid_blockage_corr"],
            context=f"TailOff grid for {context}",
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

        df = df.merge(tail_grid_lookup, on=key_cols, how="left")

        if "CLw_tailoff" not in df.columns:
            raise ValueError(
                f"{context} merge did not produce 'CLw_tailoff'. "
                "This usually means a column name collision occurred before merge."
            )

        return df

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
        Compute and store the solid blockage factor in output_col.
        Does not yet apply the correction to coefficients.
        """
        if use_constant_e and use_e_column:
            raise ValueError("Choose either use_constant_e=True or use_e_column=True, not both.")
        if not use_constant_e and not use_e_column:
            raise ValueError("Choose one source for e.")

        df = self.df.copy()

        if use_constant_e:
            df[output_col] = self.E_SOLID_BLOCKAGE
        else:
            if e_column not in df.columns:
                raise ValueError(f"Column '{e_column}' not found in dataframe.")
            df[output_col] = df[e_column]

        self.df = df

        if save_csv:
            self.save_df(self.df, filename or default_filename)

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
        Compute and store the wake blockage factor in output_col.

        Formula:
            CDi   = k * CL^2
            CDsep = CD - CD0 - CDi
            e_wb  = (S/(4C)) * CD0 + (5S/(4C)) * CDsep
        """
        df = self.df.copy()

        self.require_columns(df, [cd0_col, cd_col, cl_col, k_col], context="compute wake blockage factor")

        df["CL2_fit"]   = df[cl_col] ** 2
        df["CDi_fit"]   = df[k_col] * df["CL2_fit"]
        df["CDsep_fit"] = df[cd_col] - df[cd0_col] - df["CDi_fit"]

        if clip_negative_cdsep:
            df["CDsep_fit"] = df["CDsep_fit"].clip(lower=0)

        df[output_col] = (
            (self.WING_AREA / (4.0 * self.TUNNEL_AREA)) * df[cd0_col]
            + (5.0 * self.WING_AREA / (4.0 * self.TUNNEL_AREA)) * df["CDsep_fit"]
        )

        self.df = df

        if save_csv:
            self.save_df(self.df, filename or default_filename)

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
        cft_thrust_col: Optional[str] = "CFt_thrust_BEM",
        save_csv: bool = False,
        filename: Optional[str] = None,
        default_filename: str = "combined_blockage_corrected.csv",
    ) -> pd.DataFrame:
        """
        Apply one-step blockage correction using selected e-columns.

        Velocity:     V_corr = V / (1 + e_total)
        Coefficients: C_corr = C / (1 + e_total)^2

        CFt_thrust_BEM receives solid + wake blockage only (never slipstream),
        because it is the thrust coefficient that *causes* slipstream blockage.
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
        # CFt_thrust_BEM: solid + wake only, never slipstream
        # ----------------------------------------------------------------
        if cft_thrust_col is not None and cft_thrust_col in df.columns:
            e_esb_ewb = pd.Series(0.0, index=df.index, dtype=float)
            if apply_esb:
                e_esb_ewb = e_esb_ewb + df[esb_col].fillna(0.0)
            if apply_ewb:
                e_esb_ewb = e_esb_ewb + df[ewb_col].fillna(0.0)
            cft_factor = 1.0 / (1.0 + e_esb_ewb) ** 2
            df[f"{cft_thrust_col}_{suffix}"] = df[cft_thrust_col] * cft_factor

        self.df = df

        if save_csv:
            self.save_df(self.df, filename or default_filename)

        return self.df

    # ============================================================
    # Shared: BEM thrust computation
    # ============================================================
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
        (Appendix B, Lab Manual AE4115).

        BEM polynomial (per propeller):
            CT_bem(J) = -0.0051*J^4 + 0.0959*J^3 - 0.5888*J^2 + 1.0065*J - 0.1353
        """
        df = df.copy()

        D       = D       if D       is not None else self.PROP_DIAMETER
        S_wing  = S_wing  if S_wing  is not None else self.WING_AREA
        S_prop  = S_prop  if S_prop  is not None else self.PROP_AREA
        n_props = n_props if n_props is not None else self.N_PROPS

        self.require_columns(df, [j_col, v_col, rho_col, q_col], context="compute_ct_thrust_from_bem")

        J   = pd.to_numeric(df[j_col],   errors="coerce")
        V   = pd.to_numeric(df[v_col],   errors="coerce")
        rho = pd.to_numeric(df[rho_col], errors="coerce")
        q   = pd.to_numeric(df[q_col],   errors="coerce")

        CT_bem = (
            -0.0051 * J**4
            + 0.0959 * J**3
            - 0.5888 * J**2
            + 1.0065 * J
            - 0.1353
        )

        n_rps   = V / (J.replace(0, np.nan) * D)
        T_one   = CT_bem * rho * n_rps**2 * D**4
        T_total = n_props * T_one

        df[output_cft_col]   = T_total / (q * S_wing)
        df[output_tcstar_col] = T_total / (q * S_prop)

        return df

    # ============================================================
    # Shared: BEM thrust separation
    # ============================================================
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
    ) -> pd.DataFrame:
        """
        Separate propeller thrust from the measured axial body-frame force
        coefficient (CT) using the BEM polynomial.

        Always-added columns:  CFt_thrust_BEM, Tc_star_BEM, CFt_aero_BEM
        Conditional columns:   CD_aero_BEM, CL_aero_BEM, CYaw_aero_BEM
        """
        df = self.df.copy()

        D       = D       if D       is not None else self.PROP_DIAMETER
        S_wing  = S_wing  if S_wing  is not None else self.WING_AREA
        S_prop  = S_prop  if S_prop  is not None else self.PROP_AREA
        n_props = n_props if n_props is not None else self.N_PROPS

        self.require_columns(
            df,
            [ct_col_on, aoa_col_on, aos_col_on, "J", "V", "rho", "q", "CN", "CY", "CT"],
            context="compute_thrust_separation_BEM",
        )

        df = self.compute_ct_thrust_from_bem(
            df=df, j_col="J", v_col="V", rho_col="rho", q_col="q",
            D=D, S_wing=S_wing, S_prop=S_prop, n_props=n_props,
            output_cft_col="CFt_thrust_BEM", output_tcstar_col="Tc_star_BEM",
        )

        alpha = np.deg2rad(pd.to_numeric(df[aoa_col_on], errors="coerce"))
        beta  = np.deg2rad(pd.to_numeric(df[aos_col_on], errors="coerce"))

        ct_on   = pd.to_numeric(df[ct_col_on],        errors="coerce")
        cft_bem = pd.to_numeric(df["CFt_thrust_BEM"], errors="coerce")

        # CT_measured = aero_drag - thrust  →  CFt_aero = CT_measured + thrust
        df["CFt_aero_BEM"] = ct_on + cft_bem

        CFn      = pd.to_numeric(df["CN"], errors="coerce")
        CFs      = pd.to_numeric(df["CY"], errors="coerce")
        CFt_aero = df["CFt_aero_BEM"]

        if recompute_cd:
            df["CD_aero_BEM"] = (
                (CFn * np.sin(alpha) + CFt_aero * np.cos(alpha)) * np.cos(beta)
                + CFs * np.sin(beta)
            )

        if recompute_cl:
            df["CL_aero_BEM"] = CFn * np.cos(alpha) - CFt_aero * np.sin(alpha)

        if recompute_cyaw:
            df["CYaw_aero_BEM"] = (
                -(CFn * np.sin(alpha) + CFt_aero * np.cos(alpha)) * np.sin(beta)
                + CFs * np.cos(beta)
            )

        self.df = df
        return self.df

    # ============================================================
    # Shared: rename final force/moment columns
    # ============================================================
    def rename_detected_final_force_moment_columns(
        self,
        base_cols=("CD", "CL", "CYaw", "CMroll", "CMpitch", "CMyaw", "AoA", "V", "CFt_thrust_BEM"),
        final_suffix: str = "_FINAL",
        save_csv: bool = False,
        filename: str = "final_results.csv",
        save_directory=None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Detect the most fully corrected aerodynamic force/moment columns,
        rename them to *_FINAL, and optionally save the dataframe.

        Detection logic: for each base name, find all columns starting with
        that name that contain 'corr', then pick the one with the most
        underscore-separated parts (most corrections applied).
        """
        df = self.df.copy()
        rename_dict = {}

        for base in base_cols:
            candidates = [col for col in df.columns if col.startswith(base)]
            if not candidates:
                continue

            corr_candidates = [c for c in candidates if "corr" in c]
            if corr_candidates:
                final_col = max(corr_candidates, key=lambda c: len(c.split("_")))
            else:
                final_col = max(candidates, key=lambda c: len(c.split("_")))

            rename_dict[final_col] = f"{base}{final_suffix}"

        if rename_dict:
            df = df.rename(columns=rename_dict)

        if verbose and rename_dict:
            print("\nDetected final force/moment columns:")
            for old, new in rename_dict.items():
                print(f"  {old} -> {new}")

        self.df = df

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


# ============================================================
# Model-off corrector
# ============================================================
class ModelOffCorrector(BaseCorrector):
    """Applies model-off correction to a dataframe."""

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
            "CD", "CYaw", "CL", "CMroll", "CMpitch", "CMyaw",
        ]
        self.require_columns(corr_df, required_corr_cols, context="ModelOff correction grid")

        corr_df = corr_df[required_corr_cols].copy()
        corr_df = corr_df.rename(columns={
            "CD":      "CD_modeloff",
            "CYaw":    "CYaw_modeloff",
            "CL":      "CL_modeloff",
            "CMroll":  "CMroll_modeloff",
            "CMpitch": "CMpitch_modeloff",
            "CMyaw":   "CMyaw_modeloff",
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

        Uses AoA_round / AoS_round if present, otherwise rounds AoA / AoS internally.

        Parameters
        ----------
        source_columns : dict[str, str], optional
            Mapping from canonical coefficient names to actual dataframe columns.
            Canonical names: "CD", "CYaw", "CL", "CMroll", "CMpitch", "CMyaw".
            If None, corrects the standard columns in place.
        """
        correction_map = {
            "CD":      "CD_modeloff",
            "CYaw":    "CYaw_modeloff",
            "CL":      "CL_modeloff",
            "CMroll":  "CMroll_modeloff",
            "CMpitch": "CMpitch_modeloff",
            "CMyaw":   "CMyaw_modeloff",
        }

        default_source_columns = {k: k for k in correction_map}

        if source_columns is None:
            source_columns = default_source_columns.copy()
        else:
            source_columns = {**default_source_columns, **source_columns}

        df_work = df.copy()

        if "AoA_round" in df_work.columns and "AoS_round" in df_work.columns:
            df_work["_AoA_merge"] = df_work["AoA_round"]
            df_work["_AoS_merge"] = df_work["AoS_round"]
        else:
            self.require_columns(df_work, ["AoA", "AoS"], context="ModelOff apply")
            df_work["_AoA_merge"] = self.round_to_half(df_work["AoA"])
            df_work["_AoS_merge"] = self.round_to_half(df_work["AoS"])

        corr_work = self.corr_df.copy().rename(columns={
            "AoA_round": "_AoA_merge",
            "AoS_round": "_AoS_merge",
        })

        merged = df_work.merge(corr_work, on=["_AoA_merge", "_AoS_merge"], how="left")
        merged["modeloff_correction_found"] = (
            merged[list(correction_map.values())].notna().all(axis=1)
        )

        for canonical_name, corr_col in correction_map.items():
            data_col = source_columns.get(canonical_name)
            if data_col in merged.columns:
                merged[f"{data_col}_uncorrected"] = merged[data_col]
                merged[data_col] = merged[data_col] - merged[corr_col]

        if self.apply_cmpitch_to_25c and "CMpitch_modeloff" in merged.columns:
            target_col = cmpitch25c_column or "CMpitch25c"
            if target_col in merged.columns:
                merged[f"{target_col}_uncorrected"] = merged[target_col]
                merged[target_col] = merged[target_col] - merged["CMpitch_modeloff"]

        merged = merged.drop(columns=["_AoA_merge", "_AoS_merge"], errors="ignore")

        if save_csv:
            self.save_df(merged, filename or "modeloff_corrected.csv")

        return merged


# ============================================================
# Prop-off data
# ============================================================
class PropOffData(BaseCorrector):
    """Holds and corrects a prop-off dataframe."""

    def __init__(
        self,
        df: pd.DataFrame,
        clip_negative_cdsep: bool = True,
        save_dir: str | Path | None = None,
    ) -> None:
        super().__init__(save_dir=save_dir)
        self.df                   = df.copy()
        self.clip_negative_cdsep  = clip_negative_cdsep
        self.fit_df: Optional[pd.DataFrame] = None

    @staticmethod
    def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
        """Fit y = intercept + slope * x. Returns intercept, slope, predicted y, R²."""
        slope, intercept = np.polyfit(x, y, 1)
        y_hat  = intercept + slope * x
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
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
        tau         = tau         if tau         is not None else self.TAU2_WING
        delta       = delta       if delta       is not None else self.DELTA_WING
        geom_factor = geom_factor if geom_factor is not None else self.GEOM_FACTOR
        return self._apply_streamline_curvature_common(
            tailoff=tailoff, tau=tau, delta=delta, geom_factor=geom_factor,
            cl_source_col=cl_source_col, cm_source_col=cm_source_col,
            dataset_label="PropOff", save_csv=save_csv, filename=filename,
            default_filename="propOff_streamline_curvature_corrected.csv",
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
        delta       = delta       if delta       is not None else self.DELTA_WING
        geom_factor = geom_factor if geom_factor is not None else self.GEOM_FACTOR
        return self._apply_downwash_correction_common(
            tailoff=tailoff, delta=delta, geom_factor=geom_factor,
            aoa_source_col=aoa_source_col, cd_source_col=cd_source_col,
            dataset_label="PropOff", save_csv=save_csv, filename=filename,
            default_filename="propOff_downwash_corrected.csv",
        )

    def apply_tail_correction(
        self,
        tailoff,
        delta: float = None,
        geom_factor: float = None,
        tau2_lt: float = None,
        dcmpitch_dalpha: float = None,
        dcmpitch_dalpha_unit: str = "per_rad",
        aoa_source_col: Optional[str] = None,
        cmpitch_source_col: str = "CMpitch_blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        delta           = delta           if delta           is not None else self.DELTA_TAIL
        geom_factor     = geom_factor     if geom_factor     is not None else self.GEOM_FACTOR
        tau2_lt         = tau2_lt         if tau2_lt         is not None else self.TAU2_TAIL
        dcmpitch_dalpha = dcmpitch_dalpha if dcmpitch_dalpha is not None else self.DCMPITCH_DALPHA
        return self._apply_tail_correction_common(
            tailoff=tailoff, delta=delta, geom_factor=geom_factor,
            tau2_lt=tau2_lt, dcmpitch_dalpha=dcmpitch_dalpha,
            dcmpitch_dalpha_unit=dcmpitch_dalpha_unit,
            aoa_source_col=aoa_source_col, cmpitch_source_col=cmpitch_source_col,
            dataset_label="PropOff", save_csv=save_csv, filename=filename,
            default_filename="propOff_tail_corrected.csv",
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
        Fit CD = CD0 + k * CL^2 grouped by (V_round, AoS_round, dE, dR).
        Stores summary fit table in self.fit_df.
        """
        df = self.df.copy()
        self.require_columns(df, ["AoA", "AoS", "dE", "dR", v_col, cl_col, cd_col],
                             context="PropOff fit_cd_polar")

        df["V_round"]   = self.round_to_half(df[v_col])
        df["AoA_round"] = self.round_to_half(df["AoA"])
        df["AoS_round"] = self.round_to_half(df["AoS"])
        df["CL2"]       = df[cl_col] ** 2

        summary_records = []
        groups_out      = []

        for keys, g in df.groupby(["V_round", "AoS_round", "dE", "dR"]):
            g = g.copy().sort_values("AoA_round")

            if g["AoA_round"].nunique() < min_aoa_points:
                g["fit_used"] = False
                groups_out.append(g)
                summary_records.append({
                    "V_round": keys[0], "AoS_round": keys[1],
                    "dE": keys[2], "dR": keys[3], "fit_used": False,
                })
                continue

            cd0, k, y_hat, r2 = self._linear_fit(g["CL2"].values, g[cd_col].values)

            g["fit_used"]    = True
            g["CD0_fit"]     = cd0
            g["k_fit"]       = k
            g["R2_fit"]      = r2
            g["CD_fit_pred"] = y_hat
            g["CDi_fit"]     = k * g["CL2"]
            groups_out.append(g)

            summary_records.append({
                "V_round": keys[0], "AoS_round": keys[1],
                "dE": keys[2], "dR": keys[3],
                "fit_used": True, "CD0_fit": cd0, "k_fit": k, "R2_fit": r2,
            })

        self.fit_df = pd.DataFrame(summary_records)
        self.df     = pd.concat(groups_out, ignore_index=True)

        if save_csv:
            self.save_df(self.df, filename or "propOff_with_CD_fit_values.csv")
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
            use_constant_e=use_constant_e, use_e_column=use_e_column,
            e_column=e_column, output_col=output_col,
            save_csv=save_csv, filename=filename,
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
            cd0_col=cd0_col, cd_col=cd_col, cl_col=cl_col, k_col=k_col,
            output_col=output_col, clip_negative_cdsep=self.clip_negative_cdsep,
            save_csv=save_csv, filename=filename,
            default_filename="propOff_with_ewb.csv",
        )

    def apply_blockage_correction(
        self,
        apply_esb: bool = True,
        apply_ewb: bool = True,
        esb_col: str = "esb",
        ewb_col: str = "ewb",
        velocity_cols: Sequence[str] = ("V",),
        coefficient_cols: Sequence[str] = ("CL", "CD", "CYaw", "CMroll", "CMpitch", "CMyaw"),
        suffix: str = "blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._apply_combined_blockage_from_e_columns(
            apply_esb=apply_esb, apply_ewb=apply_ewb, apply_ess=False,
            esb_col=esb_col, ewb_col=ewb_col,
            velocity_cols=velocity_cols, coefficient_cols=coefficient_cols,
            suffix=suffix, cft_thrust_col=None,
            save_csv=save_csv, filename=filename,
            default_filename="propOff_blockage_corrected.csv",
        )


# ============================================================
# Prop-on data
# ============================================================
class PropOnData(BaseCorrector):
    """Holds and corrects a prop-on dataframe."""

    def __init__(
        self,
        df: pd.DataFrame,
        clip_negative_cdsep: bool = True,
        velocity_tolerance: float = 1.0,
        save_dir: str | Path | None = None,
    ) -> None:
        super().__init__(save_dir=save_dir)
        self.df                   = df.copy()
        self.clip_negative_cdsep  = clip_negative_cdsep
        self.velocity_tolerance   = velocity_tolerance
        self.fit_df: Optional[pd.DataFrame] = None

    def apply_streamline_curvature_correction(
        self,
        tailoff,
        tau: float = None,
        delta: float = None,
        geom_factor: float = None,
        cl_source_col: str = "CL_aero_BEM_blockage_corr",
        cm_source_col: str = "CMpitch_blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        tau         = tau         if tau         is not None else self.TAU2_WING
        delta       = delta       if delta       is not None else self.DELTA_WING
        geom_factor = geom_factor if geom_factor is not None else self.GEOM_FACTOR
        return self._apply_streamline_curvature_common(
            tailoff=tailoff, tau=tau, delta=delta, geom_factor=geom_factor,
            cl_source_col=cl_source_col, cm_source_col=cm_source_col,
            dataset_label="PropOn", save_csv=save_csv, filename=filename,
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
        delta       = delta       if delta       is not None else self.DELTA_WING
        geom_factor = geom_factor if geom_factor is not None else self.GEOM_FACTOR
        return self._apply_downwash_correction_common(
            tailoff=tailoff, delta=delta, geom_factor=geom_factor,
            aoa_source_col=aoa_source_col, cd_source_col=cd_source_col,
            dataset_label="PropOn", save_csv=save_csv, filename=filename,
            default_filename="propOn_downwash_corrected.csv",
        )

    def apply_tail_correction(
        self,
        tailoff,
        delta: float = None,
        geom_factor: float = None,
        tau2_lt: float = None,
        dcmpitch_dalpha: float = None,
        dcmpitch_dalpha_unit: str = "per_rad",
        aoa_source_col: Optional[str] = None,
        cmpitch_source_col: str = "CMpitch_blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        delta           = delta           if delta           is not None else self.DELTA_TAIL
        geom_factor     = geom_factor     if geom_factor     is not None else self.GEOM_FACTOR
        tau2_lt         = tau2_lt         if tau2_lt         is not None else self.TAU2_TAIL
        dcmpitch_dalpha = dcmpitch_dalpha if dcmpitch_dalpha is not None else self.DCMPITCH_DALPHA
        return self._apply_tail_correction_common(
            tailoff=tailoff, delta=delta, geom_factor=geom_factor,
            tau2_lt=tau2_lt, dcmpitch_dalpha=dcmpitch_dalpha,
            dcmpitch_dalpha_unit=dcmpitch_dalpha_unit,
            aoa_source_col=aoa_source_col, cmpitch_source_col=cmpitch_source_col,
            dataset_label="PropOn", save_csv=save_csv, filename=filename,
            default_filename="propOn_tail_corrected.csv",
        )

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
            self.fit_df = pd.read_csv(fit_csv)
            self.require_columns(
                self.fit_df,
                ["V_round", "AoS_round", "dE", "dR", "CD0_fit", "k_fit"],
                context="PropOn attach_fits load fit_csv",
            )

        if self.fit_df is None:
            raise ValueError(
                "No fit table loaded. Provide input_fit_df or fit_csv, "
                "or set self.fit_df before calling attach_fits()."
            )

        df     = self.df.copy()
        fit_df = self.fit_df.copy()

        if "V_round" not in df.columns:
            if "V" not in df.columns:
                raise ValueError("Need either 'V_round' or 'V' in dataframe.")
            df["V_round"] = self.round_to_half(df["V"])

        if "AoA_round" not in df.columns and "AoA" in df.columns:
            df["AoA_round"] = self.round_to_half(df["AoA"])

        if "AoS_round" not in df.columns and "AoS" in df.columns:
            df["AoS_round"] = self.round_to_half(df["AoS"])

        self.require_columns(df, [vel_col_data, aos_col, de_col, dr_col], context="PropOn attach_fits")

        df["matched_fit_velocity"]   = np.nan
        df["velocity_match_error"]   = np.nan
        df["fit_found"]              = False
        df["fit_match_type"]         = pd.Series([None] * len(df), dtype="object")
        df["matched_fit_AoS"]        = np.nan
        df["matched_fit_dR"]         = np.nan
        df["matched_fit_source_row"] = np.nan

        for col in fit_value_cols:
            df[col] = np.nan

        for frame in [df, fit_df]:
            frame[aos_col] = frame[aos_col].astype(float).round(3).replace(-0.0, 0.0)
            frame[de_col]  = frame[de_col].astype(float).replace(-0.0, 0.0)
            frame[dr_col]  = frame[dr_col].astype(float).replace(-0.0, 0.0)

        for idx, row in df.iterrows():
            aos_val = row[aos_col]
            de_val  = row[de_col]
            dr_val  = row[dr_col]
            v_val   = row[vel_col_data]

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
                    np.isclose(fit_df[de_col],  de_key)  &
                    np.isclose(fit_df[dr_col],  dr_key)
                ].copy()

                if candidate.empty:
                    continue

                candidate["vel_diff"] = np.abs(candidate[vel_col_fit] - v_val)
                best_idx  = candidate["vel_diff"].idxmin()
                best_diff = candidate.loc[best_idx, "vel_diff"]

                if best_diff <= self.velocity_tolerance:
                    df.at[idx, "matched_fit_velocity"]   = candidate.at[best_idx, vel_col_fit]
                    df.at[idx, "velocity_match_error"]   = best_diff
                    df.at[idx, "fit_found"]              = True
                    df.at[idx, "fit_match_type"]         = label
                    df.at[idx, "matched_fit_AoS"]        = candidate.at[best_idx, aos_col]
                    df.at[idx, "matched_fit_dR"]         = candidate.at[best_idx, dr_col]
                    df.at[idx, "matched_fit_source_row"] = best_idx

                    for col in fit_value_cols:
                        df.at[idx, col] = candidate.at[best_idx, col]
                    break

        self.df = df

        if save_csv:
            self.save_df(self.df, filename or "propOn_with_attached_fits.csv")

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
        D_prop      = D_prop      if D_prop      is not None else self.PROP_DIAMETER
        S_prop      = S_prop      if S_prop      is not None else (0.25 * np.pi * (D_prop ** 2))

        self.require_columns(df, [tc_col], context="compute slipstream blockage factor")

        tc = df[tc_col].astype(float)

        if (1.0 + 2.0 * tc < 0).any():
            raise ValueError("Encountered Tc* values for which (1 + 2 Tc*) < 0, making ess invalid.")

        df[output_col] = -(tc / (2.0 * np.sqrt(1.0 + 2.0 * tc))) * (S_prop / tunnel_area)

        self.df = df

        if save_csv:
            self.save_df(self.df, filename or "propOn_with_ess.csv")

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
            use_constant_e=use_constant_e, use_e_column=use_e_column,
            e_column=e_column, output_col=output_col,
            save_csv=save_csv, filename=filename,
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
            cd0_col=cd0_col, cd_col=cd_col, cl_col=cl_col, k_col=k_col,
            output_col=output_col, clip_negative_cdsep=self.clip_negative_cdsep,
            save_csv=save_csv, filename=filename,
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
        coefficient_cols: Sequence[str] = ("CL", "CD", "CYaw", "CMroll", "CMpitch", "CMyaw"),
        cft_thrust_col: str = "CFt_thrust_BEM",
        suffix: str = "blockage_corr",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._apply_combined_blockage_from_e_columns(
            apply_esb=apply_esb, apply_ewb=apply_ewb, apply_ess=apply_ess,
            esb_col=esb_col, ewb_col=ewb_col, ess_col=ess_col,
            velocity_cols=velocity_cols, coefficient_cols=coefficient_cols,
            cft_thrust_col=cft_thrust_col, suffix=suffix,
            save_csv=save_csv, filename=filename,
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
        """Apply model-off correction to the current prop-on dataframe."""
        self.df = modeloff_corrector.apply(
            df=self.df, save_csv=False, filename=None,
            source_columns=source_columns, cmpitch25c_column=cmpitch25c_column,
        )

        if save_csv:
            self.save_df(self.df, filename or "propOn_modeloff_corrected.csv")

        return self.df


# ============================================================
# Tail-off data
# ============================================================
class TailOffData(BaseCorrector):
    """Holds and corrects a tail-off dataframe."""

    # Override the solid blockage constant for the tail-off configuration
    E_SOLID_BLOCKAGE: float = BaseCorrector.E_SOLID_BLOCKAGE_TAILOFF

    def __init__(
        self,
        df: pd.DataFrame,
        clip_negative_cdsep: bool = True,
        save_dir: str | Path | None = None,
    ) -> None:
        super().__init__(save_dir=save_dir)
        self.df                   = df.copy()
        self.clip_negative_cdsep  = clip_negative_cdsep
        self.grid_df: Optional[pd.DataFrame] = None
        self.cl_a_df: Optional[pd.DataFrame] = None

    def apply_solid_blockage(
        self,
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply solid blockage correction using the shared infrastructure.
        Produces *_solid_blockage_corr columns as required by downstream lookups.
        """
        # Step 1: store e as "esb" column using E_SOLID_BLOCKAGE
        # (overridden above to E_SOLID_BLOCKAGE_TAILOFF for this class)
        self._compute_solid_blockage_e_common(output_col="esb")

        # Step 2: apply with the suffix downstream code expects
        return self._apply_combined_blockage_from_e_columns(
            apply_esb=True,
            apply_ewb=False,
            apply_ess=False,
            esb_col="esb",
            velocity_cols=("V",),
            coefficient_cols=("CL", "CD", "CY", "CMroll", "CMpitch", "CMyaw"),
            suffix="solid_blockage_corr",   # produces CL_solid_blockage_corr etc.
            cft_thrust_col=None,
            save_csv=save_csv,
            filename=filename,
            default_filename="tailOff_solid_blockage_corrected.csv",
        )

    def build_alpha_slice_grid_by_velocity(
        self,
        coeff_cols=None,
        anchor_aoa_vals=(0.0, 5.0, 10.0),
        extra_aoa_vals=None,
        extra_aos_vals=None,
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build a full AoA_round / AoS_round grid for each V_round using the
        available AoS sweeps at fixed AoA as anchor slices, while also forcing
        additional AoA/AoS values into the grid.

        For each target (AoA_round, AoS_round):
        1. Use measured values where available.
        2. Interpolate in AoA between anchor slices at the same AoS_round.
        3. Linearly extrapolate if target AoA is outside the anchor range.
        4. Leave unresolved (NaN) if fewer than two anchor values are available.
        """
        df = self.df.copy()

        df = self._ensure_round_keys(df, "TailOff")

        if coeff_cols is None:
            coeff_cols = [
                c for c in [
                    "CL_solid_blockage_corr", "CD_solid_blockage_corr",
                    "CY_solid_blockage_corr", "CMroll_solid_blockage_corr",
                    "CMpitch_solid_blockage_corr", "CMyaw_solid_blockage_corr",
                    "CL", "CD", "CY", "CMroll", "CMpitch", "CMyaw",
                ]
                if c in df.columns
            ]

        if not coeff_cols:
            raise ValueError("No coefficient columns found for grid construction.")

        extra_aoa_vals = np.asarray(list(extra_aoa_vals or []), dtype=float)
        extra_aos_vals = np.asarray(list(extra_aos_vals or []), dtype=float)

        df_avg = (
            df.groupby(["V_round", "AoA_round", "AoS_round"], as_index=False)[coeff_cols].mean()
        )

        all_velocity_grids = []

        def interp_or_extrap_alpha(x_target, x_known, y_known):
            valid = ~(np.isnan(x_known) | np.isnan(y_known))
            x, y  = x_known[valid], y_known[valid]
            if len(x) < 2:
                return np.nan, "unresolved"
            order = np.argsort(x)
            x, y  = x[order], y[order]
            exact = np.isclose(x, x_target)
            if exact.any():
                return y[exact][0], "interp_alpha"
            if x.min() <= x_target <= x.max():
                return np.interp(x_target, x, y), "interp_alpha"
            x0, x1 = (x[0], x[1]) if x_target < x.min() else (x[-2], x[-1])
            y0, y1 = (y[0], y[1]) if x_target < x.min() else (y[-2], y[-1])
            if np.isclose(x1, x0):
                return np.nan, "unresolved"
            slope   = (y1 - y0) / (x1 - x0)
            return y0 + slope * (x_target - x0), "extrap_alpha"

        for v_val, g_v in df_avg.groupby("V_round"):
            aoa_vals = np.sort(np.unique(np.concatenate([
                g_v["AoA_round"].unique().astype(float), extra_aoa_vals
            ])))
            aos_vals = np.sort(np.unique(np.concatenate([
                g_v["AoS_round"].unique().astype(float), extra_aos_vals
            ])))

            grid = pd.MultiIndex.from_product(
                [aoa_vals, aos_vals], names=["AoA_round", "AoS_round"]
            ).to_frame(index=False)
            grid["V_round"] = v_val

            measured_keys = g_v[["AoA_round", "AoS_round"]].copy()
            measured_keys["case_available"] = True
            grid = grid.merge(measured_keys, on=["AoA_round", "AoS_round"], how="left")
            grid["case_available"] = grid["case_available"].fillna(False).infer_objects(copy=False)

            source_type_series = pd.Series(index=grid.index, dtype="object")
            measured_lookup    = g_v.set_index(["AoA_round", "AoS_round"])

            available_anchor_aoa = sorted(
                a for a in anchor_aoa_vals
                if np.isclose(g_v["AoA_round"].unique(), a).any()
            )
            if not available_anchor_aoa:
                raise ValueError(
                    f"At V_round={v_val}, none of the requested anchor AoA slices "
                    f"{anchor_aoa_vals} are available."
                )

            for coeff in coeff_cols:
                values       = []
                local_source = []

                anchor_maps = {}
                for a_anchor in available_anchor_aoa:
                    g_anchor = g_v[np.isclose(g_v["AoA_round"], a_anchor)][["AoS_round", coeff]].copy()
                    g_anchor = g_anchor.dropna(subset=[coeff])
                    anchor_maps[a_anchor] = dict(zip(g_anchor["AoS_round"], g_anchor[coeff]))

                for _, row in grid.iterrows():
                    aoa_t = row["AoA_round"]
                    aos_t = row["AoS_round"]

                    if (aoa_t, aos_t) in measured_lookup.index:
                        measured_val = measured_lookup.loc[(aoa_t, aos_t), coeff]
                        if not pd.isna(measured_val):
                            values.append(measured_val)
                            local_source.append("measured")
                            continue

                    x_known = np.asarray(
                        [a for a in available_anchor_aoa if not pd.isna(anchor_maps[a].get(aos_t, np.nan))],
                        dtype=float
                    )
                    y_known = np.asarray(
                        [anchor_maps[a][aos_t] for a in available_anchor_aoa
                         if not pd.isna(anchor_maps[a].get(aos_t, np.nan))],
                        dtype=float
                    )

                    val, src = interp_or_extrap_alpha(float(aoa_t), x_known, y_known)
                    values.append(val)
                    local_source.append(src)

                grid[coeff] = values

                priority = {"unresolved": 4, "extrap_alpha": 3, "interp_alpha": 2, "measured": 1, None: 0}
                if source_type_series.isna().all():
                    source_type_series[:] = local_source
                else:
                    source_type_series[:] = [
                        old if priority.get(old, 0) >= priority.get(new, 0) else new
                        for old, new in zip(source_type_series.tolist(), local_source)
                    ]

            grid["source_type"] = source_type_series.fillna("unresolved")

            ordered = ["V_round", "AoA_round", "AoS_round", "case_available", "source_type"]
            ordered += [c for c in coeff_cols if c in grid.columns]
            other   = [c for c in grid.columns if c not in ordered]
            grid    = grid[ordered + other]

            all_velocity_grids.append(grid)

        grid_df      = pd.concat(all_velocity_grids, ignore_index=True)
        self.grid_df = grid_df

        if save_csv:
            self.save_df(grid_df, filename or "TAILOFF_grid_by_velocity_alpha_slice.csv")

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
        Compute CL-alpha slope for each (V_round, AoS_round) case using a
        linear fit over a specified AoA range.
        """
        if self.grid_df is None:
            raise ValueError("grid_df is not available. Run build_alpha_slice_grid_by_velocity() first.")

        df = self.grid_df.copy()
        df = self._ensure_round_keys(df, "TailOff cl_alpha")

        self.require_columns(df, [v_col, aos_col, aoa_col, cl_col],
                             context="compute_cl_alpha_slope_by_case")

        df_fit  = df[(df[aoa_col] >= aoa_min) & (df[aoa_col] <= aoa_max)].copy()
        results = []

        for (v_val, aos_val), g in df_fit.groupby([v_col, aos_col]):
            g = g[[aoa_col, cl_col]].dropna().copy()
            g = g.groupby(aoa_col, as_index=False)[cl_col].mean()

            n_points    = len(g)
            n_unique    = g[aoa_col].nunique()

            if n_unique < min_points:
                results.append({
                    v_col: v_val, aos_col: aos_val,
                    "aoa_min_fit": aoa_min, "aoa_max_fit": aoa_max,
                    "n_points": n_points, "n_unique_aoa": n_unique,
                    "cl_alpha_slope_per_deg": np.nan, "cl_at_aoa0": np.nan,
                    "r2": np.nan, "fit_success": False,
                })
                continue

            x     = g[aoa_col].to_numpy(dtype=float)
            y     = g[cl_col].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            y_hat = intercept + slope * x
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2     = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

            results.append({
                v_col: v_val, aos_col: aos_val,
                "aoa_min_fit": aoa_min, "aoa_max_fit": aoa_max,
                "n_points": n_points, "n_unique_aoa": n_unique,
                "cl_alpha_slope_per_deg": slope, "cl_at_aoa0": intercept,
                "r2": r2, "fit_success": True,
            })

        result_df    = pd.DataFrame(results).sort_values([v_col, aos_col]).reset_index(drop=True)
        self.cl_a_df = result_df

        if save_csv:
            self.save_df(result_df, filename or "tailoff_cl_alpha_slope_by_case.csv")

        return result_df