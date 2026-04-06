"""
Wind Tunnel Wall-Correction Pipeline
=====================================
Implements solid-blockage, wake-blockage, slipstream-blockage, streamline-
curvature, downwash, and tail corrections for three wind-tunnel test
configurations:

    - PropOffData  : propeller-off (clean-wing / power-off) measurements
    - PropOnData   : propeller-on (powered) measurements
    - TailOffData  : tail-off (no horizontal tail) measurements used as
                     the reference lift distribution for interference
                     corrections

All correction formulae follow the methodology described in the
AE4115 Lab Manual (TU Delft), in particular Appendix B (BEM polynomial)
and the standard AGARD / ESDU wall-correction framework.

Correction sequence prop-on (recommended order)
----------------------------------------
1.  BEM thrust separation              (compute_thrust_separation_BEM)
2.  Model-off tare subtraction         (ModelOffCorrector)
3.  Attach CL^2-CDO fit from prop-off  (attach_fits)
4.  Solid-blockage factor              (compute_solid_blockage_e)
5.  Wake-blockage factor               (compute_wake_blockage_e)
6.  Slipstream-blockage factor         (compute_slipstream_blockage_e) 
7.  Combined blockage correction       (apply_blockage_correction)
8.  Streamline-curvature correction    (apply_streamline_curvature_correction)
9.  Downwash correction                (apply_downwash_correction)
10. Tail correction                    (apply_tail_correction)
11. Rename final columns               (rename_detected_final_force_moment_columns)

Correction sequence prop-off (recommended order)
----------------------------------------
1.  Model-off tare subtraction         (ModelOffCorrector)
2.  Compute CL^2-CDO fit               (fit_cd_polar)
3.  Solid-blockage factor              (compute_solid_blockage_e)
4.  Wake-blockage factor               (compute_wake_blockage_e)
5.  Combined blockage correction       (apply_blockage_correction)
6.  Streamline-curvature correction    (apply_streamline_curvature_correction)
7.  Downwash correction                (apply_downwash_correction)
8.  Tail correction                    (apply_tail_correction)
9.  Rename final columns               (rename_detected_final_force_moment_columns)

Correction sequence tail-off (recommended order)
----------------------------------------
1.  Model-off tare subtraction         (ModelOffCorrector)
2.  Solid-blockage factor              (compute_solid_blockage_e)
3.  build AoA-AoS-V grid to find CLw   (build_alpha_slice_grid_by_velocity)
4.  find CLW-alpha slopes              (compute_cl_alpha_slope_by_case)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Set

import numpy as np
import pandas as pd
import re


pd.set_option('future.no_silent_downcasting', True)


class BaseCorrector:
    """
    Abstract base class shared by PropOffData, PropOnData, and TailOffData.

    Provides
    --------
    - Configurable output directory and CSV save helper.
    - Required-column validation.
    - round_to_half utility (rounds to nearest 0.5).
    - Solid-blockage factor computation.
    - Wake-blockage factor computation.
    - Combined blockage correction (velocity + coefficients).
    - Streamline-curvature correction (shared implementation).
    - Downwash correction (shared implementation).
    - Tail interference correction (shared implementation).
    - Tail-off CLw lookup / merge helper.
    - BEM-based propeller thrust computation.
    - BEM-based thrust separation from body-frame axial force.
    - Final-column auto-detection and renaming.

    Tunnel / Model Constants
    -------------------------
    All subclasses inherit the following geometric constants.
    Override at the subclass level if needed.

        TUNNEL_AREA              = 2.07       m^2   wind-tunnel cross-sectional area C
        WING_AREA                = 0.2172     m^2   wing reference area S
        GEOM_FACTOR              = S / C            geometric blockage ratio

        PROP_DIAMETER            = 0.2032     m
        PROP_AREA                = pi/4 * D^2       single propeller disc area
        N_PROPS                  = 2                number of propellers

        DELTA_WING               = 0.1065           wing solid-angle factor delta (ESDU)
        DELTA_TAIL               = 0.1085           tail solid-angle factor delta_t
        TAU2_WING                = 0.045            streamline-curvature factor tau2 (wing)
        TAU2_TAIL                = 0.8              streamline-curvature factor tau2 (tail)

        DCMPITCH_DALPHA          = -0.15676  rad^-1 pitching-moment slope dCm/dalpha of tail

        E_SOLID_BLOCKAGE         = 0.007229438      e_sb for full configuration
        E_SOLID_BLOCKAGE_TAILOFF = 0.006406642      e_sb for tail-off configuration
    """

    # ============================================================
    # Shared tunnel / model constants
    # ============================================================
    TUNNEL_AREA:              float = 2.07
    WING_AREA:                float = 0.2172
    GEOM_FACTOR:              float = WING_AREA / TUNNEL_AREA

    PROP_DIAMETER:            float = 0.2032
    PROP_AREA:                float = np.pi * 0.25 * (PROP_DIAMETER ** 2)
    N_PROPS:                  int   = 2

    #whether or not to use the new tau and delta values
    DELTA_WING:           float = 0.103691
    DELTA_TAIL:           float = 0.103691
    TAU2_WING:            float = 0.069616
    TAU2_TAIL:            float = 0.749271

    DCMPITCH_DALPHA:          float = -0.15676   # per deg

    E_SOLID_BLOCKAGE:         float = 0.007219591
    E_SOLID_BLOCKAGE_TAILOFF: float = 0.006399352

    # ============================================================
    def __init__(self, save_dir: str | Path | None = None) -> None:
        """
        Parameters
        ----------
        save_dir : str or Path, optional
            Directory used for all save_df calls. Defaults to the directory
            containing this source file. Created automatically if it does
            not exist.
        """
        if save_dir is None:
            self.save_dir = Path(__file__).resolve().parent
        else:
            self.save_dir = Path(save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    @staticmethod
    def round_to_half(series: pd.Series) -> pd.Series:
        """
        Round each element of series to the nearest 0.5.

        Formula
        -------
            x_rounded = round(2 * x) / 2

        Parameters
        ----------
        series : pd.Series
            Numeric series (typically an AoA, AoS, or velocity column).

        Returns
        -------
        pd.Series
            Rounded values with the same index.
        """
        return np.round(series * 2.0) / 2.0

    @staticmethod
    def require_columns(df: pd.DataFrame, required_cols: Sequence[str], context: str = "") -> None:
        """
        Raise ValueError if any of required_cols are absent from df.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to inspect.
        required_cols : sequence of str
            Column names that must be present.
        context : str, optional
            Descriptive label prepended to the error message.

        Raises
        ------
        ValueError
            Lists all missing columns in the error message.
        """
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            prefix = f"{context}: " if context else ""
            raise ValueError(f"{prefix}missing required columns: {missing}")
        
    @staticmethod
    def _ct_interp_with_extrap(
        J_query: float,
        J_arr: np.ndarray,
        CT_arr: np.ndarray,
    ) -> float:
        """
        Linear interpolation inside [J_arr[0], J_arr[-1]]; linear extrapolation
        outside, using the slope of the two nearest endpoint points.
    
        Parameters
        ----------
        J_query : float
            Advance ratio value to evaluate.
        J_arr : np.ndarray
            Sorted (ascending) digitised J values for one velocity curve.
        CT_arr : np.ndarray
            CT values corresponding to J_arr.
    
        Returns
        -------
        float
            Interpolated or linearly extrapolated CT value.
        """
        if J_query <= J_arr[0]:
            slope = (CT_arr[1] - CT_arr[0]) / (J_arr[1] - J_arr[0])
            return float(CT_arr[0] + slope * (J_query - J_arr[0]))
        elif J_query >= J_arr[-1]:
            slope = (CT_arr[-1] - CT_arr[-2]) / (J_arr[-1] - J_arr[-2])
            return float(CT_arr[-1] + slope * (J_query - J_arr[-1]))
        else:
            return float(np.interp(J_query, J_arr, CT_arr))

    def set_save_directory(self, directory: str | Path) -> None:
        """
        Change the output directory used by save_df.

        Parameters
        ----------
        directory : str or Path
            New directory path. Created automatically if it does not exist.
        """
        self.save_dir = Path(directory)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_df(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Write df to <save_dir>/<filename> as a CSV (no row index).

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to persist.
        filename : str
            File name (not a full path) relative to save_dir.

        Returns
        -------
        Path
            Absolute path of the written file.
        """
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
        aoa_source_col: Optional[str] = "AoA",
        dataset_label: str = "dataset",
        save_csv: bool = False,
        filename: Optional[str] = None,
        default_filename: str = "streamline_curvature_corrected.csv",
    ) -> pd.DataFrame:
        """
        Apply the streamline-curvature (tunnel-induced camber) correction.

        The curved streamlines induced by the tunnel walls impose an effective
        camber on the model, shifting the apparent angle of attack and the
        pitching moment. The correction references the tail-off lift
        coefficient CLw (from tailoff.grid_df) and the tail-off CL-alpha
        slope (from tailoff.cl_a_df).

        Formulae
        --------
        Induced incidence correction [radians]:

            delta_alpha_sc = tau2 * delta * (S/C) * CL_w

        Converted to degrees:

            delta_alpha_sc_deg = delta_alpha_sc * (180 / pi)

        Corrected angle of attack:

            AoA_sc = AoA + delta_alpha_sc_deg

        Lift-coefficient correction:

            delta_CL_sc = -delta_alpha_sc_deg * (dCL/dalpha)_tailoff

        Pitching-moment correction (factor 0.25 assumes aerodynamic
        centre at quarter-chord):

            delta_CMpitch_sc = -0.25 * delta_CL_sc

        Corrected coefficients:

            CL_corr = CL + delta_CL_sc
            CM_corr = CM + delta_CMpitch_sc

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset. Must have grid_df and cl_a_df
            populated before calling this method.
        tau : float, optional
            Streamline-curvature parameter tau2. Defaults to TAU2_WING.
        delta : float, optional
            Solid-angle factor delta. Defaults to DELTA_WING.
        geom_factor : float, optional
            Geometric blockage ratio S/C. Defaults to GEOM_FACTOR.
        cl_source_col : str
            Column name of the blockage-corrected lift coefficient in
            self.df (input).
        cm_source_col : str
            Column name of the blockage-corrected pitching-moment
            coefficient in self.df (input).
        dataset_label : str
            Label used in error messages.
        save_csv : bool
            If True, write the corrected dataframe to disk.
        filename : str, optional
            Override output file name.
        default_filename : str
            Fallback file name when filename is None.

        Returns
        -------
        pd.DataFrame
            self.df with the following new columns added:

            - delta_alpha_sc_rad            delta_alpha_sc in radians
            - delta_alpha_sc_deg            delta_alpha_sc in degrees
            - AoA_streamline_curvature_corr corrected angle of attack
            - delta_CL_sc                   CL correction
            - delta_CMpitch_sc              CM correction
            - {cl_source_col}_sc_corr       corrected CL
            - {cm_source_col}_sc_corr       corrected CM
            - streamline_curvature_data_found  boolean lookup flag
        """

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

        if aoa_source_col in df.columns:
            df[f"{aoa_source_col}_sc_corr"] = df[aoa_source_col] + df["delta_alpha_sc_deg"]
        else:
            df[f"{aoa_source_col}_sc_corr"] = df["AoA_round"] + df["delta_alpha_sc_deg"]

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
        Apply the tunnel-induced downwash (angle-of-attack) correction.

        The finite tunnel cross-section introduces a downwash at the model
        that increases the apparent angle of attack and adds induced drag.
        The correction uses the tail-off wing lift coefficient CLw as the
        reference lifting load.

        Formulae
        --------
        Downwash-induced incidence increment [degrees]:

            delta_alpha_dw = delta * (S/C) * CL_w * 57.3

        Converted to radians:

            delta_alpha_dw_rad = delta_alpha_dw * (pi / 180)

        Corrected angle of attack:

            AoA_dw = AoA + delta_alpha_dw

        Induced-drag correction:

            delta_CD_dw = delta * (S/C) * CL_w^2

        Corrected drag coefficient:

            CD_corr = CD + delta_CD_dw

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with grid_df populated.
        delta : float, optional
            Solid-angle factor delta. Defaults to DELTA_WING.
        geom_factor : float, optional
            Geometric blockage ratio S/C. Defaults to GEOM_FACTOR.
        aoa_source_col : str, optional
            Column in self.df used as the angle-of-attack input. If None,
            falls back to 'AoA' then 'AoA_round'.
        cd_source_col : str
            Column name of the blockage-corrected drag coefficient.
        dataset_label : str
            Label used in error messages.
        save_csv : bool
            If True, write the corrected dataframe to disk.
        filename : str, optional
            Override output file name.
        default_filename : str
            Fallback file name when filename is None.

        Returns
        -------
        pd.DataFrame
            self.df with the following new columns added:

            - delta_alpha_dw_deg        delta_alpha_dw in degrees
            - delta_alpha_dw_rad        delta_alpha_dw in radians
            - delta_CD_dw               induced-drag correction
            - AoA_downwash_corr         corrected angle of attack
            - {cd_source_col}_dw_corr   corrected CD
            - downwash_data_found        boolean lookup flag
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

        df[f"{aoa_source_col}_dw_corr"]      = df[aoa_source_col] + df["delta_alpha_dw_deg"]
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
        dcmpitch_dalpha_unit: str = "per_deg",
        aoa_source_col: Optional[str] = None,
        cmpitch_source_col: str = "CMpitch_blockage_corr",
        dataset_label: str = "dataset",
        save_csv: bool = False,
        filename: Optional[str] = None,
        default_filename: str = "tail_corrected.csv",
    ) -> pd.DataFrame:
        """
        Apply the tail-plane interference (tail-correction) correction.

        The tunnel-wall constraint distorts the downwash at the horizontal
        tail, creating an error in the measured pitching moment. The
        correction estimates the additional incidence felt by the tail using
        the tail streamline-curvature parameter and then infers the pitching-
        moment change via the tail pitching-moment slope.

        Formulae
        --------
        Tail-induced angle-of-attack increment [radians]:

            delta_alpha_tail = delta_t * (S/C) * CL_w * tau2_lt

        Converted to degrees:

            delta_alpha_tail_deg = delta_alpha_tail * (180 / pi)

        Pitching-moment correction:

            If dcmpitch_dalpha_unit == 'per_deg':
                delta_CMpitch_tail = (dCm/dalpha) * delta_alpha_tail_deg

            If dcmpitch_dalpha_unit == 'per_rad':
                delta_CMpitch_tail = (dCm/dalpha) * delta_alpha_tail

        where dCm/dalpha defaults to DCMPITCH_DALPHA = -0.15676 rad^-1.

        Sign convention: tail downwash reduces effective AoA, which
        reduces CMpitch (makes it more negative), hence the minus sign:

            CMpitch_corr = CMpitch - delta_CMpitch_tail

        Corrected angle of attack:

            AoA_tail_corr = AoA + delta_alpha_tail_deg

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with grid_df populated.
        delta : float, optional
            Tail solid-angle factor delta_t. Defaults to DELTA_TAIL.
        geom_factor : float, optional
            Geometric blockage ratio S/C. Defaults to GEOM_FACTOR.
        tau2_lt : float, optional
            Tail streamline-curvature parameter tau2 at the tail location.
            Defaults to TAU2_TAIL.
        dcmpitch_dalpha : float, optional
            Pitching-moment slope dCm/dalpha of the tail.
            Defaults to DCMPITCH_DALPHA.
        dcmpitch_dalpha_unit : str, 'per_rad' or 'per_deg'
            Unit of dcmpitch_dalpha. Determines whether delta_alpha_tail
            is expressed in radians or degrees when computing delta_CMpitch.
        aoa_source_col : str, optional
            Column in self.df used as the AoA input. Falls back to 'AoA'
            then 'AoA_round' when None.
        cmpitch_source_col : str
            Column name of the pitching-moment coefficient to correct.
        dataset_label : str
            Label used in error messages.
        save_csv : bool
            If True, write the corrected dataframe to disk.
        filename : str, optional
            Override output file name.
        default_filename : str
            Fallback file name when filename is None.

        Returns
        -------
        pd.DataFrame
            self.df with the following new columns added:

            - delta_alpha_tail_rad               tail AoA increment in radians
            - delta_alpha_tail_deg               tail AoA increment in degrees
            - delta_CMpitch_tail                 CMpitch correction
            - AoA_tail_corr                      corrected AoA
            - {cmpitch_source_col}_tail_corr     corrected CMpitch
            - tail_correction_data_found         boolean lookup flag

        Raises
        ------
        ValueError
            If dcmpitch_dalpha_unit is not 'per_deg' or 'per_rad'.
        """

        # Resolve defaults from class constants if not explicitly passed
        delta           = delta           if delta           is not None else self.DELTA_WING
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
        # Compute tail corrections second method was chosen
        # --------------------------------------------------------
        if False:
            df["delta_alpha_tail_rad"] = delta * geom_factor * df["CLw_tailoff"] * tau2_lt
        else:
            df["delta_alpha_tail_rad"] = delta * geom_factor * df["CLw_tailoff"] * (1+tau2_lt)

        df["delta_alpha_tail_deg"] = df["delta_alpha_tail_rad"] * 57.3

        if dcmpitch_dalpha_unit == "per_deg":
            df["delta_CMpitch_tail"] = dcmpitch_dalpha * df["delta_alpha_tail_deg"]
        else:
            df["delta_CMpitch_tail"] = dcmpitch_dalpha * df["delta_alpha_tail_rad"]

        df[f"TAILONLY_{aoa_source_col}_tail_corr"] = df[aoa_source_col] + df["delta_alpha_tail_deg"]
        # minus sign: tail downwash reduces effective AoA -> reduces CMpitch (more negative)
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
        Guarantee that AoA_round, AoS_round, and V_round columns are present
        in df, creating them from raw columns when absent.

        Priority for velocity source:
            1. V_round               (already exists, no-op)
            2. V_solid_blockage_corr (preferred after solid-blockage step)
            3. V                     (raw freestream velocity)

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to augment (mutated in place, not copied).
        dataset_label : str
            Label used in error messages.

        Returns
        -------
        pd.DataFrame
            The (possibly augmented) dataframe.

        Raises
        ------
        ValueError
            If neither the rounded column nor any valid source column is found.
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
        Left-merge the tail-off wing lift coefficient CLw_tailoff onto df
        from tailoff.grid_df, matching on (V_round, AoA_round, AoS_round).

        The source column in tailoff.grid_df is CL_solid_blockage_corr,
        which is the blockage-corrected lift coefficient of the wing-only
        (tail-off) configuration. Rows in df that have no matching entry
        in the grid receive NaN.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to enrich. Must already contain V_round, AoA_round,
            AoS_round.
        tailoff : TailOffData
            Reference tail-off dataset.
        context : str, optional
            Descriptive label for error messages.

        Returns
        -------
        pd.DataFrame
            df with CLw_tailoff column appended.

        Raises
        ------
        ValueError
            If the post-merge dataframe does not contain CLw_tailoff,
            indicating a column-name collision before the merge.
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
        Compute and store the solid-blockage factor e_sb.

        The solid-blockage factor accounts for the reduction in effective
        tunnel cross-section caused by the physical volume of the model.
        Two modes are supported:

        Constant mode (use_constant_e=True):
            Uses the class-level constant E_SOLID_BLOCKAGE (or the
            overridden value in TailOffData).

                e_sb = E_SOLID_BLOCKAGE   (scalar, same for all rows)

        Per-row mode (use_e_column=True):
            Reads e_sb from a column already present in self.df.

                e_sb = df[e_column]

        The factor is stored in output_col (default 'esb') and is
        subsequently consumed by _apply_combined_blockage_from_e_columns.

        Parameters
        ----------
        use_constant_e : bool
            Use the class constant E_SOLID_BLOCKAGE.
        use_e_column : bool
            Read e_sb from e_column in self.df.
        e_column : str
            Source column name when use_e_column is True.
        output_col : str
            Destination column name for e_sb.
        save_csv : bool
            If True, write the updated dataframe to disk.
        filename : str, optional
            Override output file name.
        default_filename : str
            Fallback file name.

        Returns
        -------
        pd.DataFrame
            self.df with output_col added.

        Raises
        ------
        ValueError
            If both or neither of use_constant_e / use_e_column are True,
            or if e_column is missing from the dataframe.
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
        Compute and store the wake-blockage factor e_wb.

        The wake-blockage factor accounts for the reduction in effective
        freestream velocity due to the momentum deficit in the model wake.
        It is split into a zero-lift (pressure drag) term and a separated
        flow (form drag) term.

        Formulae
        --------
        Induced drag from polar fit:

            CDi = k * CL^2

        Separated / form drag:

            CDsep = CD - CD0 - CDi

        Note: negative CDsep values are physically unrealistic (scatter)
        and are clipped to zero by default.

        Wake-blockage factor:

            e_wb = (S / (4*C)) * CD0  +  (5*S / (4*C)) * CDsep

        where S = WING_AREA and C = TUNNEL_AREA.

        Parameters
        ----------
        cd0_col : str
            Column name of the zero-lift drag coefficient CD0 (from polar fit).
        cd_col : str
            Column name of the measured drag coefficient CD.
        cl_col : str
            Column name of the measured lift coefficient CL.
        k_col : str
            Column name of the induced-drag factor k (from polar fit).
        output_col : str
            Destination column name for e_wb.
        clip_negative_cdsep : bool
            If True, clip CDsep to zero from below.
        save_csv : bool
            If True, write the updated dataframe to disk.
        filename : str, optional
            Override output file name.
        default_filename : str
            Fallback file name.

        Returns
        -------
        pd.DataFrame
            self.df with intermediate columns CL2_fit, CDi_fit, CDsep_fit,
            and output_col added.
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
        Apply the combined blockage correction to velocities and aerodynamic
        force/moment coefficients using pre-computed blockage factor columns.

        Formulae
        --------
        Total blockage factor (sum of selected terms):

            e_total = e_sb + e_wb + e_ss

        Any of the three terms can be excluded via the apply_* flags.

        Corrected freestream velocity:

            V_corr = V / (1 + e_total)

        Corrected aerodynamic coefficients:

            C_corr = C / (1 + e_total)^2

        Special handling for thrust coefficient
        ----------------------------------------
        CFt_thrust_BEM is the propeller thrust coefficient that causes
        slipstream blockage (e_ss). Correcting it with e_ss would be
        circular, so it always receives solid + wake blockage only:

            CFt_corr = CFt / (1 + e_sb + e_wb)^2

        Parameters
        ----------
        apply_esb : bool
            Include solid-blockage factor e_sb in e_total.
        apply_ewb : bool
            Include wake-blockage factor e_wb in e_total.
        apply_ess : bool
            Include slipstream-blockage factor e_ss in e_total
            (prop-on only).
        esb_col : str
            Column name for e_sb.
        ewb_col : str
            Column name for e_wb.
        ess_col : str
            Column name for e_ss.
        velocity_cols : sequence of str
            Velocity columns to correct. Each produces {col}_{suffix}.
        coefficient_cols : sequence of str
            Coefficient columns to correct. Each produces {col}_{suffix}.
        suffix : str
            Suffix appended to all output column names.
        cft_thrust_col : str or None
            Column name of CFt_thrust_BEM. Set to None to skip the special
            thrust correction.
        save_csv : bool
            If True, write the updated dataframe to disk.
        filename : str, optional
            Override output file name.
        default_filename : str
            Fallback file name.

        Returns
        -------
        pd.DataFrame
            self.df with corrected velocity/coefficient columns and
            e_total_blockage added.
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

        velocity_factor    = (1.0 + df["e_total_blockage"])
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
        Compute propeller thrust from the BEM polynomial and separate it from
        the measured axial body-frame force coefficient CT, yielding
        aerodynamic-only lift, drag, and side-force coefficients.
    
        All output columns carry the suffix _BEM. Call compute_thrust_separation_exp
        afterwards to produce the equivalent _EXP columns for comparison.
    
        Formulae
        --------
        BEM polynomial (per propeller):
    
            CT_bem(J) = -0.0051*J^4 + 0.0959*J^3 - 0.5888*J^2 + 1.0065*J - 0.1353
    
        Propeller rotational speed:
    
            n = V / (J * D)     [rev/s]
    
        Thrust of a single propeller:
    
            T1 = CT_bem * rho * n^2 * D^4
    
        Total thrust:
    
            T_total = n_props * T1
    
        Thrust force coefficient referenced to wing area:
    
            CFt_thrust_BEM = T_total / (q * S_wing)
    
        Thrust loading coefficient:
    
            Tc_star_BEM = T_total / (q * S_prop)
    
        Aerodynamic axial force:
    
            CFt_aero_BEM = CT_measured + CFt_thrust_BEM
    
        Wind-axis transformation (alpha = AoA [rad], beta = AoS [rad]):
    
            CD_aero = (CN*sin(alpha) + CFt_aero*cos(alpha)) * cos(beta)
                    + CS*sin(beta)
    
            CL_aero = CN*cos(alpha) - CFt_aero*sin(alpha)
    
            CYaw    = -(CN*sin(alpha) + CFt_aero*cos(alpha)) * sin(beta)
                    + CS*cos(beta)
    
        Parameters
        ----------
        ct_col_on : str
            Column name of the measured axial body-frame force coefficient.
        aoa_col_on : str
            Column name of the angle of attack [degrees].
        aos_col_on : str
            Column name of the sideslip angle [degrees].
        S_wing : float, optional
            Wing reference area [m^2]. Defaults to WING_AREA.
        S_prop : float, optional
            Single propeller disc area [m^2]. Defaults to PROP_AREA.
        D : float, optional
            Propeller diameter [m]. Defaults to PROP_DIAMETER.
        n_props : int, optional
            Number of propellers. Defaults to N_PROPS.
        recompute_cd : bool
            If True, compute and store CD_aero_BEM.
        recompute_cl : bool
            If True, compute and store CL_aero_BEM.
        recompute_cyaw : bool
            If True, compute and store CYaw_aero_BEM.
    
        Returns
        -------
        pd.DataFrame
            self.df with the following new columns:
    
            Always added:
            - CT_props_total_BEM    n_props * CT_bem per row
            - CFt_thrust_BEM        propeller thrust force coefficient
            - Tc_star_BEM           thrust loading coefficient T/(q * S_prop)
            - CFt_aero_BEM          aerodynamic axial force coefficient
    
            Conditional (controlled by recompute_* flags):
            - CD_aero_BEM
            - CL_aero_BEM
            - CYaw_aero_BEM
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
    
        # ------------------------------------------------------------------
        # BEM polynomial CT and thrust conversion
        # ------------------------------------------------------------------
        J   = pd.to_numeric(df["J"],   errors="coerce")
        V   = pd.to_numeric(df["V"],   errors="coerce")
        rho = pd.to_numeric(df["rho"], errors="coerce")
        q   = pd.to_numeric(df["q"],   errors="coerce")
    
        CT_bem = (
            -0.0051 * J**4
            + 0.0959 * J**3
            - 0.5888 * J**2
            + 1.0065 * J
            - 0.1353
        )
    
        n_rps   = V / (J.replace(0, np.nan) * D)
        T_one   = CT_bem * rho * n_rps**2 * D**4
        T_total = T_one * n_props
    
        df["CT_one_prop_BEM"]    = CT_bem 
        df["CFt_thrust_BEM"]     = T_total / (q * S_wing)
        df["Tc_star_BEM"]        = T_total / (q * S_prop * n_props)
    
        # ------------------------------------------------------------------
        # Thrust separation and wind-axis transformation
        # ------------------------------------------------------------------
        alpha    = np.deg2rad(pd.to_numeric(df[aoa_col_on], errors="coerce"))
        beta     = np.deg2rad(pd.to_numeric(df[aos_col_on], errors="coerce"))
        ct_on    = pd.to_numeric(df[ct_col_on],        errors="coerce")
        cft_bem  = pd.to_numeric(df["CFt_thrust_BEM"], errors="coerce")
    
        mode = 1
        if mode == 1:
            df["CFt_aero_BEM"] = ct_on + cft_bem
        elif mode == 2:
            df["CFt_aero_BEM"] = ct_on - cft_bem
        elif mode == 3:
            df["CFt_aero_BEM"] = -ct_on + cft_bem
    
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


    def compute_thrust_separation_EXP(
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
        exp_ct_path: str | Path = None,
        extrap_velocities: frozenset[float] = frozenset({20., 40.}),
        extrap_j_min: float = 1.55,
        extrap_j_max: float = 2.85,
    ) -> pd.DataFrame:
        """
        Compute propeller thrust by interpolating experimental CT-J curves from
        WebPlotDigitizer and separate it from the measured axial body-frame force
        coefficient CT, yielding aerodynamic-only lift, drag, and side-force
        coefficients.
    
        All output columns carry the suffix _EXP. Call compute_thrust_separation_BEM
        first to produce the equivalent _BEM columns; both sets then sit
        side-by-side in self.df for direct comparison.
    
        CT lookup procedure (per row)
        ------------------------------
        1. Parse exp_ct_path CSV into one (J, CT) curve per freestream velocity.
        The velocity is read from the series header, e.g. "Ct_V20" -> 20 m/s.
        2. Find the two velocity curves bracketing the row's V (or the nearest
        curve if V is outside the available range).
        3. Evaluate CT on each bracketing curve at the row's J:
            - linear interpolation if J is within the digitised range.
            - linear extrapolation from the two nearest endpoint points if J
                is outside the range AND the curve velocity is in
                extrap_velocities AND J is within [extrap_j_min, extrap_j_max].
            - flat clamp (np.interp default) in all other out-of-range cases.
        4. Linearly interpolate the two CT values in V to the row's exact V.
    
        Thrust conversion (same as BEM path):
    
            n     = V / (J * D)
            T1    = CT_exp * rho * n^2 * D^4
            T     = n_props * T1
            CFt   = T / (q * S_wing)
            Tc*   = T / (q * S_prop)
    
        Thrust separation and wind-axis transformation:
    
            CFt_aero = CT_measured + CFt_thrust_EXP
    
            CD_aero = (CN*sin(alpha) + CFt_aero*cos(alpha)) * cos(beta)
                    + CS*sin(beta)
    
            CL_aero = CN*cos(alpha) - CFt_aero*sin(alpha)
    
            CYaw    = -(CN*sin(alpha) + CFt_aero*cos(alpha)) * sin(beta)
                    + CS*cos(beta)
    
        Parameters
        ----------
        ct_col_on : str
            Column name of the measured axial body-frame force coefficient.
        aoa_col_on : str
            Column name of the angle of attack [degrees].
        aos_col_on : str
            Column name of the sideslip angle [degrees].
        S_wing : float, optional
            Wing reference area [m^2]. Defaults to WING_AREA.
        S_prop : float, optional
            Single propeller disc area [m^2]. Defaults to PROP_AREA.
        D : float, optional
            Propeller diameter [m]. Defaults to PROP_DIAMETER.
        n_props : int, optional
            Number of propellers. Defaults to N_PROPS.
        recompute_cd : bool
            If True, compute and store CD_aero_EXP.
        recompute_cl : bool
            If True, compute and store CL_aero_EXP.
        recompute_cyaw : bool
            If True, compute and store CYaw_aero_EXP.
        exp_ct_path : str or Path
            Path to the WebPlotDigitizer CSV file.
            Default: "Ct_V_exp_data.csv"
        extrap_velocities : frozenset of float
            Freestream velocities [m/s] for which linear J-extrapolation beyond
            the digitised range is permitted. Rows at other velocities are clamped
            flat at the curve endpoints.
            Default: frozenset({20., 40.})
        extrap_j_min : float
            Lower J bound of the extrapolation window. Default: 1.6.
        extrap_j_max : float
            Upper J bound of the extrapolation window. Default: 2.8.
    
        Returns
        -------
        pd.DataFrame
            self.df with the following new columns added alongside any existing
            _BEM columns:
    
            Always added:
            - CT_props_total_EXP    total propeller thrust coefficient
            - CFt_thrust_EXP        propeller thrust force coefficient
            - Tc_star_EXP           thrust loading coefficient T/(q * S_prop)
            - CFt_aero_EXP          aerodynamic axial force coefficient
    
            Conditional (controlled by recompute_* flags):
            - CD_aero_EXP
            - CL_aero_EXP
            - CYaw_aero_EXP
    
        Raises
        ------
        ValueError
            If no valid velocity curves are found in the CSV, or if required
            columns are missing from self.df.
        """
        df = self.df.copy()
    
        D       = D       if D       is not None else self.PROP_DIAMETER
        S_wing  = S_wing  if S_wing  is not None else self.WING_AREA
        S_prop  = S_prop  if S_prop  is not None else self.PROP_AREA
        n_props = n_props if n_props is not None else self.N_PROPS
    
        self.require_columns(
            df,
            [ct_col_on, aoa_col_on, aos_col_on, "J", "V", "rho", "q", "CN", "CY", "CT"],
            context="compute_thrust_separation_exp",
        )

        if exp_ct_path is None:
            raise ValueError("exp_ct_path is not provided, provide the path to the C_T digitized data")
    
        # ------------------------------------------------------------------
        # Parse the WebPlotDigitizer CSV into {V_value: (J_arr, CT_arr)}
        # ------------------------------------------------------------------
        raw          = pd.read_csv(exp_ct_path, header=None)
        series_names = raw.iloc[0].tolist()
        curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    
        col_idx = 0
        while col_idx < len(series_names):
            name = str(series_names[col_idx]).strip()
            if name and name.lower() != "nan":
                match = re.search(r"(\d+(?:\.\d+)?)$", name)
                if match:
                    v_key   = float(match.group(1))
                    j_vals  = pd.to_numeric(
                        raw.iloc[2:, col_idx], errors="coerce"
                    ).dropna().to_numpy()
                    ct_vals = pd.to_numeric(
                        raw.iloc[2:, col_idx + 1], errors="coerce"
                    ).iloc[: len(j_vals)].to_numpy()
                    order         = np.argsort(j_vals)
                    curves[v_key] = (j_vals[order], ct_vals[order])
            col_idx += 2
    
        if not curves:
            raise ValueError(
                f"No valid velocity curves found in {exp_ct_path}. "
                "Check that series headers follow the pattern 'Ct_V<number>'."
            )
    
        v_keys = np.array(sorted(curves.keys()))
        # ------------------------------------------------------------------
        # Helper: evaluate CT on one curve, with optional extrapolation
        # ------------------------------------------------------------------
        def ct_at_v(vk: float, J_i: float) -> float:
            j_arr, ct_arr = curves[vk]
            allow_extrap  = (
                vk in extrap_velocities
                and extrap_j_min <= J_i <= extrap_j_max
            )
            if allow_extrap:
                return self._ct_interp_with_extrap(J_i, j_arr, ct_arr)
            else:
                return float(np.interp(J_i, j_arr, ct_arr))
    
        # ------------------------------------------------------------------
        # Row-wise CT lookup with bracketing interpolation in V
        # ------------------------------------------------------------------
        J_series   = pd.to_numeric(df["J"],   errors="coerce").to_numpy()
        V_series   = pd.to_numeric(df["V_round"],   errors="coerce").to_numpy()
        rho_series = pd.to_numeric(df["rho"], errors="coerce").to_numpy()
        q_series   = pd.to_numeric(df["q"],   errors="coerce").to_numpy()
    
        CT_exp = np.full(len(df), np.nan)
    
        for i, (J_i, V_i) in enumerate(zip(J_series, V_series)):
            if np.isnan(J_i) or np.isnan(V_i):
                continue
    
            idx_hi = int(np.searchsorted(v_keys, V_i))
            idx_lo = max(idx_hi - 1, 0)
            idx_hi = min(idx_hi, len(v_keys) - 1)
    
            if idx_lo == idx_hi:
                CT_exp[i] = ct_at_v(v_keys[idx_lo], J_i)
            else:
                v_lo, v_hi = v_keys[idx_lo], v_keys[idx_hi]
                ct_lo      = ct_at_v(v_lo, J_i)
                ct_hi      = ct_at_v(v_hi, J_i)
                frac       = (V_i - v_lo) / (v_hi - v_lo)
                CT_exp[i]  = ct_lo + frac * (ct_hi - ct_lo)
    
        # ------------------------------------------------------------------
        # Convert CT -> thrust -> force coefficients
        # ------------------------------------------------------------------
        J_safe  = np.where(J_series == 0, np.nan, J_series)
        n_rps   = V_series / (J_safe * D)
        T_one   = CT_exp * rho_series * n_rps**2 * D**4
        T_total = T_one * n_props
    
        df["CT_one_prop_EXP"] = CT_exp  
        df["CFt_thrust_EXP"]     = T_total / (q_series * S_wing)
        df["Tc_star_EXP"]        = T_total / (q_series * S_prop * n_props)
    
        # ------------------------------------------------------------------
        # Thrust separation and wind-axis transformation
        # ------------------------------------------------------------------
        alpha    = np.deg2rad(pd.to_numeric(df[aoa_col_on], errors="coerce"))
        beta     = np.deg2rad(pd.to_numeric(df[aos_col_on], errors="coerce"))
        ct_on    = pd.to_numeric(df[ct_col_on],        errors="coerce")
        cft_exp  = pd.to_numeric(df["CFt_thrust_EXP"], errors="coerce")

        mode = 1
        if mode == 1:
            df["CFt_aero_EXP"] = ct_on + cft_exp
        elif mode == 2:
            df["CFt_aero_EXP"] = ct_on - cft_exp    
        elif mode == 3:
            df["CFt_aero_EXP"] = -ct_on + cft_exp

        CFn      = pd.to_numeric(df["CN"], errors="coerce")
        CFs      = pd.to_numeric(df["CY"], errors="coerce")
        CFt_aero = df["CFt_aero_EXP"]
    
        if recompute_cd:
            df["CD_aero_EXP"] = (
                (CFn * np.sin(alpha) + CFt_aero * np.cos(alpha)) * np.cos(beta)
                + CFs * np.sin(beta)
            )
        if recompute_cl:
            df["CL_aero_EXP"] = CFn * np.cos(alpha) - CFt_aero * np.sin(alpha)
        if recompute_cyaw:
            df["CYaw_aero_EXP"] = (
                -(CFn * np.sin(alpha) + CFt_aero * np.cos(alpha)) * np.sin(beta)
                + CFs * np.cos(beta)
            )
    
        self.df = df
        return self.df


    # ============================================================
    # Shared: rename final force/moment columns
    # ============================================================
    def create_final_output_df(
        self,
        base_cols=("CD", "CL", "CYaw", "CMroll", "CMpitch", "CMyaw", "AoA", "V", "CFt"),
        final_suffix: str = "_FINAL",
        active_columns: Optional[dict[str, str]] = None,
        save_csv: bool = False,
        filename: str = "final_results.csv",
        save_directory=None,
        verbose: bool = True,
        print_corrections: bool = True,
        save_slim: bool = False,
        slim_filename: str = "final_results_slim.csv",
        extra_slim_cols: Optional[Sequence[str]] = None,
        slim_col_order: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Detect the most fully corrected column for each base aerodynamic
        quantity and rename it with final_suffix (default '_FINAL').

        Detection logic
        ---------------
        For each base name (e.g. 'CD'):

        1. If active_columns is provided and the base name is a key in it,
           use the specified column directly, bypassing auto-detection.
        2. Otherwise, collect all columns whose name starts with that base.
        3. Among those, prefer columns whose name contains 'corr'.
        4. Among the filtered candidates, select the one with the most
           underscore-separated parts (most corrections applied).
        5. Rename it to {base}_FINAL.

        Correction suffix legend (used by print_corrections):
            blockage_corr   Blockage correction (solid + wake [+ slipstream]).
                            For CFt specifically: solid + wake only, slipstream
                            is never applied because CFt causes slipstream blockage.
            sc_corr         Streamline-curvature correction
            dw_corr         Downwash correction
            tail_corr       Tail-plane interference correction

        Slim output
        -----------
        When save_slim=True a reduced CSV is saved containing only the most
        relevant columns. The standard slim columns are always included if
        present in the dataframe:

            AoA_FINAL, AoS, AoS_round, V_FINAL, AoA_round, V_round,
            dE, dR,
            CL_FINAL, CD_FINAL, CYaw_FINAL, CMroll_FINAL,
            CMpitch_FINAL, CMyaw_FINAL, CFt_FINAL

        Additional columns can be appended via extra_slim_cols.

        The column order in the slim CSV follows slim_col_order if provided,
        otherwise the standard order above is used. slim_col_order only needs
        to list the columns you want to reorder — any remaining columns are
        appended at the end in their natural order. Columns listed in
        slim_col_order that are not present in the dataframe are silently
        skipped.

        Parameters
        ----------
        base_cols : tuple of str
            Base column names to search for.
        final_suffix : str
            Suffix appended to each selected column name.
        active_columns : dict[str, str], optional
            Explicit mapping of base name to the exact column to use as
            final. For example:
                {'AoA': 'AoA_downwash_corr', 'CD': 'CD_blockage_corr_dw_corr'}
            Bases present in this dict bypass auto-detection entirely.
            Bases not in this dict still use auto-detection as normal.
        save_csv : bool
            If True, write the full renamed dataframe to disk.
        filename : str
            Output file name for the full dataframe.
        save_directory : str or Path, optional
            Override the output directory. Falls back to save_dir.
        verbose : bool
            If True, print the raw column rename mapping (old -> new).
        print_corrections : bool
            If True, print a human-readable summary of which corrections
            were applied to each final column, inferred from the suffixes
            present in the source column name. CFt always shows
            'solid + wake only' for blockage, never slipstream.
        save_slim : bool
            If True, save a reduced CSV with only the most relevant columns.
        slim_filename : str
            Output file name for the slim dataframe.
        extra_slim_cols : sequence of str, optional
            Additional column names to include in the slim output beyond
            the standard set. For example:
                ['J', 'rho', 'Tc_star_BEM']
        slim_col_order : sequence of str, optional
            Explicit column order for the slim dataframe. Only the columns
            listed here are reordered — any remaining slim columns are
            appended afterwards in their natural order. Columns not present
            in the dataframe are silently skipped. For example:
                ['AoA_FINAL', 'V_FINAL', 'CL_FINAL', 'CD_FINAL', 'J']

        Returns
        -------
        pd.DataFrame
            self.df with selected columns renamed to {base}_FINAL.

        Raises
        ------
        ValueError
            If a column specified in active_columns does not exist in self.df.
        """

        # Correction suffix legend for all standard coefficients.
        SUFFIX_LEGEND = {
            "blockage_corr": "Blockage (solid + wake [+ slipstream])",
            "sc_corr":       "Streamline-curvature",
            "dw_corr":       "Downwash",
            "tail_corr":     "Tail-plane interference",
        }

        # CFt-specific legend: blockage is always solid + wake only,
        # slipstream is never applied to CFt by design.
        CFT_SUFFIX_LEGEND = {
            "blockage_corr": "Blockage (solid + wake only)",
            "sc_corr":       "Streamline-curvature",
            "dw_corr":       "Downwash",
            "tail_corr":     "Tail-plane interference",
        }

        # Standard columns always included in the slim output if present.
        SLIM_STANDARD_COLS = [
            "AoA_FINAL", "AoS", "AoS_round",
            "V_FINAL", "AoA_round", "V_round",
            "dE", "dR",
            "CL_FINAL", "CD_FINAL", "CYaw_FINAL",
            "CMroll_FINAL", "CMpitch_FINAL", "CMyaw_FINAL",
            "CFt_FINAL", "CT_props_total",
        ]

        df = self.df.copy()
        rename_dict = {}

        for base in base_cols:
            # --------------------------------------------------------
            # If the user explicitly specified a column for this base,
            # use it directly and skip auto-detection
            # --------------------------------------------------------
            if active_columns is not None and base in active_columns:
                explicit_col = active_columns[base]
                if explicit_col not in df.columns:
                    raise ValueError(
                        f"active_columns['{base}'] = '{explicit_col}' "
                        f"does not exist in the dataframe."
                    )
                rename_dict[explicit_col] = f"{base}{final_suffix}"
                continue

            # --------------------------------------------------------
            # Auto-detection fallback
            # --------------------------------------------------------
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

        # --------------------------------------------------------
        # Verbose: raw rename mapping
        # --------------------------------------------------------
        if verbose and rename_dict:
            print("\nDetected final force/moment columns:")
            for old, new in rename_dict.items():
                print(f"  {old} -> {new}")

        # --------------------------------------------------------
        # Print corrections: human-readable correction summary
        # --------------------------------------------------------
        if print_corrections and rename_dict:
            col_width = max(len(base) for base in rename_dict.values()) + 2
            print("\n" + "=" * 60)
            print("  CORRECTIONS APPLIED PER FINAL COLUMN")
            print("=" * 60)
            for source_col, final_col in rename_dict.items():
                base_name = final_col.replace(final_suffix, "")
                legend = CFT_SUFFIX_LEGEND if base_name == "CFt" else SUFFIX_LEGEND
                applied = [
                    label
                    for suffix, label in legend.items()
                    if suffix in source_col
                ]
                corrections_str = ", ".join(applied) if applied else "None (raw column)"
                print(f"  {base_name:<{col_width}} {corrections_str}")
            print("=" * 60)

        self.df = df

        # --------------------------------------------------------
        # Save full dataframe
        # --------------------------------------------------------
        if save_csv:
            if save_directory is None:
                save_directory = getattr(self, "save_dir", None)
            if save_directory is None:
                raise ValueError("No save directory specified.")

            save_path = Path(save_directory) / filename
            df.to_csv(save_path, index=False)
            if verbose:
                print(f"\nSaved file: {save_path}")

        # --------------------------------------------------------
        # Save slim dataframe
        # --------------------------------------------------------
        if save_slim:
            if save_directory is None:
                save_directory = getattr(self, "save_dir", None)
            if save_directory is None:
                raise ValueError("No save directory specified for slim output.")

            # Build the full list of desired slim columns
            slim_cols_wanted = list(SLIM_STANDARD_COLS)
            if extra_slim_cols:
                for col in extra_slim_cols:
                    if col not in slim_cols_wanted:
                        slim_cols_wanted.append(col)

            # Keep only those that actually exist in the dataframe
            slim_cols_available = [c for c in slim_cols_wanted if c in df.columns]

            # Apply custom column order if provided
            if slim_col_order:
                # Start with the ordered columns that are available
                ordered = [c for c in slim_col_order if c in slim_cols_available]
                # Append any remaining available columns not in slim_col_order
                remaining = [c for c in slim_cols_available if c not in ordered]
                slim_cols_available = ordered + remaining

            df_slim = df[slim_cols_available].copy()

            slim_path = Path(save_directory) / slim_filename
            df_slim.to_csv(slim_path, index=False)
            if verbose:
                print(f"Saved slim file: {slim_path}")
                print(f"  Slim columns ({len(slim_cols_available)}): {slim_cols_available}")

        return self.df


# ============================================================
# Model-off corrector
# ============================================================
class ModelOffCorrector(BaseCorrector):
    """
    Apply the model-off tare correction to a measured dataframe.

    The model-off correction removes the aerodynamic contribution of the
    model support structure (sting, struts, etc.) that was measured in a
    separate run with the model absent. The correction subtracts the tare
    coefficients, looked up by (AoA_round, AoS_round), from the
    corresponding coefficients in the measured dataframe:

        C_corrected = C_measured - C_tare

    Parameters
    ----------
    correction_csv : str or Path
        Path to the CSV file containing the model-off tare grid. Must
        include columns: AoA_round, AoS_round, CD, CYaw, CL, CMroll,
        CMpitch, CMyaw.
    apply_cmpitch_to_25c : bool
        If True (default), also subtract the pitching-moment tare from
        the CMpitch25c column (moment referenced to 25% chord).
    save_dir : str or Path, optional
        Output directory for saved files.
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
        """
        Load and validate the model-off correction grid from correction_csv.

        The grid is deduplicated on (AoA_round, AoS_round) and coefficient
        columns are renamed with the _modeloff suffix to avoid collisions
        during the subsequent merge.

        Returns
        -------
        pd.DataFrame
            Cleaned tare grid with columns:
            AoA_round, AoS_round, CD_modeloff, CYaw_modeloff, CL_modeloff,
            CMroll_modeloff, CMpitch_modeloff, CMyaw_modeloff.

        Raises
        ------
        ValueError
            If required columns are absent from the CSV.
        """
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
        Apply the model-off tare correction to df.

        For each canonical coefficient C, the corrected value is:

            C_corrected = C_measured - C_tare

        where C_tare is the value from the model-off grid matched by
        (AoA_round, AoS_round). If apply_cmpitch_to_25c is True, the
        same subtraction is applied to the CMpitch25c column.

        Original uncorrected values are preserved in {col}_uncorrected
        columns.

        Parameters
        ----------
        df : pd.DataFrame
            Measured dataframe to correct. Should contain the coefficient
            columns to be corrected, plus either (AoA_round, AoS_round)
            or (AoA, AoS) which are rounded internally.
        save_csv : bool
            If True, write the corrected dataframe to disk.
        filename : str, optional
            Override output file name.
        source_columns : dict[str, str], optional
            Mapping {canonical_name: actual_column_name} for cases where
            coefficient columns have non-standard names. Canonical names
            are 'CD', 'CYaw', 'CL', 'CMroll', 'CMpitch', 'CMyaw'. If
            None, canonical names are used directly.
        cmpitch25c_column : str, optional
            Column name of the pitching moment at 25% chord. Defaults
            to 'CMpitch25c'.

        Returns
        -------
        pd.DataFrame
            Corrected dataframe with {col}_uncorrected backup columns and
            a modeloff_correction_found boolean column indicating whether
            a tare entry was found for each row.
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
    """
    Container and correction pipeline for propeller-off (clean-wing,
    power-off) wind-tunnel measurements.

    This class stores the raw/partially-processed dataframe in self.df
    and provides methods to apply the full sequence of wall corrections
    in the recommended order:

    1.  fit_cd_polar                        fit CD = CD0 + k*CL^2
    2.  compute_solid_blockage_e            compute e_sb
    3.  compute_wake_blockage_e             compute e_wb
    4.  apply_blockage_correction           apply combined blockage
    5.  apply_streamline_curvature_correction
    6.  apply_downwash_correction
    7.  apply_tail_correction

    Parameters
    ----------
    df : pd.DataFrame
        Raw measurement dataframe.
    clip_negative_cdsep : bool
        If True (default), clip CDsep to zero when computing the
        wake-blockage factor. Negative values are physically unrealistic
        and arise from measurement scatter.
    save_dir : str or Path, optional
        Output directory for all CSV exports.
    """

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
        self.raw_df               = df.copy()

    @staticmethod
    def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
        """
        Fit the linear model y = intercept + slope * x using ordinary
        least squares.

        Parameters
        ----------
        x : np.ndarray
            Independent variable (e.g. CL^2).
        y : np.ndarray
            Dependent variable (e.g. CD).

        Returns
        -------
        intercept : float
            Fitted intercept (CD0 in a drag-polar context).
        slope : float
            Fitted slope (k in a drag-polar context).
        y_hat : np.ndarray
            Predicted values at each x.
        r2 : float
            Coefficient of determination R^2. Returns nan if total sum
            of squares is zero.
        """
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
        """
        Apply the streamline-curvature correction to the prop-off dataset.

        Delegates to BaseCorrector._apply_streamline_curvature_common.
        See that method for the full formula description.

        Formulae (summary)
        ------------------
            delta_alpha_sc = tau2 * delta * (S/C) * CL_w        [rad]
            AoA_sc         = AoA + delta_alpha_sc_deg
            delta_CL_sc    = -delta_alpha_sc_deg * (dCL/dalpha)_tailoff
            delta_CM_sc    = -0.25 * delta_CL_sc

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with grid_df and cl_a_df populated.
        tau : float, optional
            Streamline-curvature parameter tau2. Defaults to TAU2_WING.
        delta : float, optional
            Solid-angle factor delta. Defaults to DELTA_WING.
        geom_factor : float, optional
            Geometric blockage ratio S/C. Defaults to GEOM_FACTOR.
        cl_source_col : str
            Input lift-coefficient column (post blockage correction).
        cm_source_col : str
            Input pitching-moment column (post blockage correction).
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            Updated self.df.
        """
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
        """
        Apply the downwash correction to the prop-off dataset.

        Delegates to BaseCorrector._apply_downwash_correction_common.
        See that method for the full formula description.

        Formulae (summary)
        ------------------
            delta_alpha_dw = delta * (S/C) * CL_w * 57.3        [deg]
            AoA_dw         = AoA + delta_alpha_dw
            delta_CD_dw    = delta * (S/C) * CL_w^2

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with grid_df populated.
        delta : float, optional
            Solid-angle factor delta. Defaults to DELTA_WING.
        geom_factor : float, optional
            Geometric blockage ratio S/C. Defaults to GEOM_FACTOR.
        aoa_source_col : str, optional
            AoA input column. Defaults to 'AoA_streamline_curvature_corr'
            so the streamline-curvature-corrected AoA is used as input.
        cd_source_col : str
            Drag-coefficient input column.
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            Updated self.df.
        """
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
        """
        Apply the tail-plane interference correction to the prop-off dataset.

        Delegates to BaseCorrector._apply_tail_correction_common.
        See that method for the full formula description.

        Formulae (summary)
        ------------------
            delta_alpha_tail = delta_t * (S/C) * CL_w * tau2_lt  [rad]
            delta_CMpitch    = (dCm/dalpha) * delta_alpha_tail
            CMpitch_corr     = CMpitch - delta_CMpitch

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with grid_df populated.
        delta : float, optional
            Tail solid-angle factor delta_t. Defaults to DELTA_TAIL.
        geom_factor : float, optional
            Geometric blockage ratio S/C. Defaults to GEOM_FACTOR.
        tau2_lt : float, optional
            Tail streamline-curvature parameter tau2. Defaults to TAU2_TAIL.
        dcmpitch_dalpha : float, optional
            Tail pitching-moment slope dCm/dalpha. Defaults to
            DCMPITCH_DALPHA = -0.15676 rad^-1.
        dcmpitch_dalpha_unit : str, 'per_rad' or 'per_deg'
            Unit of dcmpitch_dalpha.
        aoa_source_col : str, optional
            AoA input column. Falls back to 'AoA' then 'AoA_round'.
        cmpitch_source_col : str
            Pitching-moment input column.
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            Updated self.df.
        """
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
        Fit the parabolic drag polar for each test condition.

        For each unique combination of (V_round, AoS_round, dE, dR),
        fits the standard drag polar:

            CD = CD0 + k * CL^2

        using ordinary least squares on CL^2 vs CD. Groups with fewer
        than min_aoa_points unique AoA values are skipped and flagged
        with fit_used = False.

        The fitted coefficients are stored back in self.df (per row) and
        in a compact summary table self.fit_df (one row per condition).

        Parameters
        ----------
        save_csv : bool
            If True, write the per-row result to filename and, optionally,
            the summary fit table to fit_params_filename.
        filename : str, optional
            Override output file name for the per-row dataframe.
        fit_params_filename : str, optional
            File name for the summary fit table. If None, not written.
        v_col : str
            Velocity column name.
        cl_col : str
            Lift-coefficient column name.
        cd_col : str
            Drag-coefficient column name.
        min_aoa_points : int
            Minimum number of distinct AoA values required to attempt a fit.

        Returns
        -------
        pd.DataFrame
            self.df with new columns:

            - fit_used      boolean flag indicating fit was performed
            - CD0_fit       fitted zero-lift drag coefficient
            - k_fit         fitted induced-drag factor
            - R2_fit        coefficient of determination R^2
            - CD_fit_pred   predicted CD from fit
            - CDi_fit       induced drag component k * CL^2
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
        """
        Compute and store the solid-blockage factor e_sb for the prop-off
        configuration.

        Delegates to BaseCorrector._compute_solid_blockage_e_common using
        the class constant E_SOLID_BLOCKAGE = 0.007229438.

        Formula (constant mode):

            e_sb = 0.007229438   (same for all rows)

        Parameters
        ----------
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.
        use_constant_e : bool
            Use the class constant (default True).
        use_e_column : bool
            Use a per-row value from e_column instead.
        e_column : str
            Source column when use_e_column is True.
        output_col : str
            Destination column name for e_sb.

        Returns
        -------
        pd.DataFrame
            self.df with output_col added.
        """
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
        """
        Compute and store the wake-blockage factor e_wb for the prop-off
        configuration.

        Delegates to BaseCorrector._compute_wake_blockage_e_common.

        Formulae
        --------
            CDi    = k * CL^2
            CDsep  = CD - CD0 - CDi          (clipped to >= 0)
            e_wb   = (S/(4*C))*CD0 + (5*S/(4*C))*CDsep

        Parameters
        ----------
        cd0_col : str
            Zero-lift drag coefficient column (from fit_cd_polar).
        cd_col : str
            Measured drag coefficient column.
        cl_col : str
            Measured lift coefficient column.
        k_col : str
            Induced-drag factor column (from fit_cd_polar).
        output_col : str
            Destination column name for e_wb.
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            self.df with output_col added.
        """
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
        """
        Apply the combined solid + wake blockage correction to the prop-off
        dataset. Slipstream blockage is not applicable and is excluded.

        Delegates to BaseCorrector._apply_combined_blockage_from_e_columns.

        Formulae
        --------
            e_total = e_sb + e_wb

            V_corr = V / (1 + e_total)
            C_corr = C / (1 + e_total)^2

        Parameters
        ----------
        apply_esb : bool
            Include solid-blockage correction.
        apply_ewb : bool
            Include wake-blockage correction.
        esb_col : str
            Column name for e_sb.
        ewb_col : str
            Column name for e_wb.
        velocity_cols : sequence of str
            Velocity columns to correct.
        coefficient_cols : sequence of str
            Aerodynamic coefficient columns to correct.
        suffix : str
            Suffix appended to all output column names.
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            self.df with corrected columns and e_total_blockage added.
        """
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
    """
    Container and correction pipeline for propeller-on (powered) wind-tunnel
    measurements.

    In addition to the wall corrections applied to the prop-off configuration,
    this class handles:

    - BEM-based thrust separation from the measured body-frame axial force
      (compute_thrust_separation_BEM).
    - Slipstream blockage factor computation
      (compute_slipstream_blockage_e).
    - Attachment of prop-off drag-polar fit parameters to prop-on rows
      (attach_fits), needed for the wake-blockage computation.
    - Model-off tare correction (apply_modeloff_correction).

    Parameters
    ----------
    df : pd.DataFrame
        Raw measurement dataframe.
    clip_negative_cdsep : bool
        Clip CDsep to zero when computing wake blockage (default True).
    velocity_tolerance : float
        Maximum allowable velocity difference [m/s] when matching prop-off
        fit entries to prop-on conditions in attach_fits (default 1.0 m/s).
    save_dir : str or Path, optional
        Output directory for CSV exports.
    """

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
        """
        Apply the streamline-curvature correction to the prop-on dataset.

        Delegates to BaseCorrector._apply_streamline_curvature_common.
        The default input lift column is 'CL_aero_BEM_blockage_corr'
        (aerodynamic lift after BEM thrust separation and blockage
        correction) rather than the raw 'CL_blockage_corr' used for
        prop-off data.

        Formulae (summary)
        ------------------
            delta_alpha_sc = tau2 * delta * (S/C) * CL_w        [rad]
            AoA_sc         = AoA + delta_alpha_sc_deg
            delta_CL_sc    = -delta_alpha_sc_deg * (dCL/dalpha)_tailoff
            delta_CM_sc    = -0.25 * delta_CL_sc

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with grid_df and cl_a_df populated.
        tau : float, optional
            Streamline-curvature parameter tau2. Defaults to TAU2_WING.
        delta : float, optional
            Solid-angle factor delta. Defaults to DELTA_WING.
        geom_factor : float, optional
            Geometric blockage ratio S/C. Defaults to GEOM_FACTOR.
        cl_source_col : str
            Input lift-coefficient column.
        cm_source_col : str
            Input pitching-moment column.
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            Updated self.df.
        """
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
        """
        Apply the downwash correction to the prop-on dataset.

        Delegates to BaseCorrector._apply_downwash_correction_common.

        Formulae (summary)
        ------------------
            delta_alpha_dw = delta * (S/C) * CL_w * 57.3        [deg]
            AoA_dw         = AoA + delta_alpha_dw
            delta_CD_dw    = delta * (S/C) * CL_w^2

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with grid_df populated.
        delta : float, optional
            Solid-angle factor delta. Defaults to DELTA_WING.
        geom_factor : float, optional
            Geometric blockage ratio S/C. Defaults to GEOM_FACTOR.
        aoa_source_col : str, optional
            AoA input column. Defaults to 'AoA_streamline_curvature_corr'.
        cd_source_col : str
            Drag-coefficient input column.
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            Updated self.df.
        """
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
        """
        Apply the tail-plane interference correction to the prop-on dataset.

        Delegates to BaseCorrector._apply_tail_correction_common.

        Formulae (summary)
        ------------------
            delta_alpha_tail = delta_t * (S/C) * CL_w * tau2_lt  [rad]
            delta_CMpitch    = (dCm/dalpha) * delta_alpha_tail
            CMpitch_corr     = CMpitch - delta_CMpitch

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with grid_df populated.
        delta : float, optional
            Tail solid-angle factor delta_t. Defaults to DELTA_TAIL.
        geom_factor : float, optional
            Geometric blockage ratio S/C. Defaults to GEOM_FACTOR.
        tau2_lt : float, optional
            Tail streamline-curvature parameter tau2. Defaults to TAU2_TAIL.
        dcmpitch_dalpha : float, optional
            Tail pitching-moment slope dCm/dalpha. Defaults to
            DCMPITCH_DALPHA = -0.15676 rad^-1.
        dcmpitch_dalpha_unit : str, 'per_rad' or 'per_deg'
            Unit of dcmpitch_dalpha.
        aoa_source_col : str, optional
            AoA input column.
        cmpitch_source_col : str
            Pitching-moment input column.
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            Updated self.df.
        """
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
        Attach prop-off drag-polar fit values (CD0, k) to each row of the
        prop-on dataframe.

        Because the prop-on runs may not sweep the full AoA range needed to
        fit a drag polar independently, the fit coefficients from the matching
        prop-off condition are used instead. Matching is performed on
        (AoS_round, dE, dR) with closest-velocity selection, within the
        tolerance set by self.velocity_tolerance.

        Fallback order
        --------------
        For each prop-on row, the following match strategies are attempted
        in order and the first successful match is used:

            1. exact AoS, exact dR
            2. exact AoS, flipped dR   (sign reversal)
            3. exact AoS, dR = 0
            4. flipped AoS, exact dR
            5. flipped AoS, flipped dR
            6. flipped AoS, dR = 0

        Diagnostic columns added to self.df:

            - matched_fit_velocity      velocity of the matched fit entry
            - velocity_match_error      |V_data - V_fit|
            - fit_found                 boolean success flag
            - fit_match_type            label of the match strategy used
            - matched_fit_AoS           AoS of the matched fit entry
            - matched_fit_dR            dR of the matched fit entry
            - matched_fit_source_row    index of matched row in fit_df

        Parameters
        ----------
        input_fit_df : pd.DataFrame, optional
            Prop-off fit-summary table (output of PropOffData.fit_cd_polar).
            If None, uses self.fit_df or loads from fit_csv.
        fit_csv : str or Path, optional
            Path to a CSV file containing the fit summary table.
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.
        vel_col_data : str
            Velocity column in self.df for matching.
        vel_col_fit : str
            Velocity column in the fit table for matching.
        aos_col : str
            AoS column name (must exist in both dataframes).
        de_col : str
            Elevator deflection column name.
        dr_col : str
            Rudder deflection column name.
        fit_value_cols : tuple of str
            Fit-coefficient columns to attach from the fit table.

        Returns
        -------
        pd.DataFrame
            self.df with fit_value_cols and diagnostic columns added.

        Raises
        ------
        ValueError
            If no fit table is available from any source.
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
        n_props: Optional[int] = None,
        tunnel_area: Optional[float] = None,
        output_col: str = "ess",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute the slipstream-blockage factor e_ss for powered conditions.

        The propeller slipstream accelerates the flow behind the disc,
        reducing the effective freestream cross-section available to the
        rest of the model.

        Formula
        -------
            e_ss = -(Tc_star / (2 * sqrt(1 + 2*Tc_star))) * (S_prop / C)

        where:
            Tc_star  = thrust loading coefficient T/(q * S_prop)
            S_prop   = single propeller disc area  [m^2]
            C        = tunnel cross-sectional area [m^2]

        Note: e_ss is negative by construction. The slipstream increases
        the effective dynamic pressure, so it reduces the blockage
        correction relative to the solid and wake terms.

        Parameters
        ----------
        tc_col : str
            Column name of the thrust loading coefficient Tc_star_BEM.
        D_prop : float, optional
            Propeller diameter [m]. Defaults to PROP_DIAMETER.
        S_prop : float, optional
            Single propeller disc area [m^2]. Computed from D_prop if
            not supplied.
        tunnel_area : float, optional
            Tunnel cross-sectional area [m^2]. Defaults to TUNNEL_AREA.
        output_col : str
            Destination column name for e_ss.
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            self.df with output_col added.

        Raises
        ------
        ValueError
            If any row has (1 + 2*Tc_star) < 0, making the formula
            undefined (square root of a negative number).
        """
        df = self.df.copy()

        tunnel_area = tunnel_area if tunnel_area is not None else self.TUNNEL_AREA
        D_prop      = D_prop      if D_prop      is not None else self.PROP_DIAMETER
        S_prop      = S_prop      if S_prop      is not None else (0.25 * np.pi * (D_prop ** 2))
        n_props     = n_props     if n_props     is not None else self.N_PROPS

        self.require_columns(df, [tc_col], context="compute slipstream blockage factor")

        tc = df[tc_col].astype(float)

        if (1.0 + 2.0 * tc < 0).any():
            raise ValueError("Encountered Tc* values for which (1 + 2 Tc*) < 0, making ess invalid.")

        df[output_col] = -n_props * ((tc / (2.0 * np.sqrt(1.0 + 2.0 * tc))) * (S_prop / tunnel_area))

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
        """
        Compute and store the solid-blockage factor e_sb for the prop-on
        configuration.

        Delegates to BaseCorrector._compute_solid_blockage_e_common using
        E_SOLID_BLOCKAGE = 0.007229438.

        Formula (constant mode):

            e_sb = 0.007229438   (same for all rows)

        Parameters
        ----------
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.
        use_constant_e : bool
            Use the class constant (default True).
        use_e_column : bool
            Use a per-row value from e_column instead.
        e_column : str
            Source column when use_e_column is True.
        output_col : str
            Destination column name for e_sb.

        Returns
        -------
        pd.DataFrame
            self.df with output_col added.
        """
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
        """
        Compute and store the wake-blockage factor e_wb for the prop-on
        configuration.

        Delegates to BaseCorrector._compute_wake_blockage_e_common.
        The drag-polar fit values are typically attached first via
        attach_fits.

        Formulae
        --------
            CDi    = k * CL^2
            CDsep  = CD - CD0 - CDi          (clipped to >= 0)
            e_wb   = (S/(4*C))*CD0 + (5*S/(4*C))*CDsep

        Parameters
        ----------
        cd0_col : str
            Zero-lift drag coefficient column.
        cd_col : str
            Measured drag coefficient column.
        cl_col : str
            Measured lift coefficient column.
        k_col : str
            Induced-drag factor column.
        output_col : str
            Destination column name for e_wb.
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            self.df with output_col added.
        """
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
        """
        Apply the combined solid + wake + slipstream blockage correction
        to the prop-on dataset.

        Delegates to BaseCorrector._apply_combined_blockage_from_e_columns.

        Formulae
        --------
            e_total = e_sb + e_wb + e_ss

            V_corr  = V / (1 + e_total)
            C_corr  = C / (1 + e_total)^2

        Special case for thrust coefficient (avoids circular correction):

            CFt_corr = CFt / (1 + e_sb + e_wb)^2

        Parameters
        ----------
        apply_esb : bool
            Include solid-blockage correction.
        apply_ewb : bool
            Include wake-blockage correction.
        apply_ess : bool
            Include slipstream-blockage correction (default True).
        esb_col : str
            Column name for e_sb.
        ewb_col : str
            Column name for e_wb.
        ess_col : str
            Column name for e_ss.
        velocity_cols : sequence of str
            Velocity columns to correct.
        coefficient_cols : sequence of str
            Aerodynamic coefficient columns to correct.
        cft_thrust_col : str
            Thrust force coefficient column (receives solid + wake only).
        suffix : str
            Suffix appended to all output column names.
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            self.df with corrected columns and e_total_blockage added.
        """
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
        """
        Apply the model-off tare correction to the current prop-on dataframe.

        Delegates to ModelOffCorrector.apply. The correction subtracts the
        support-structure aerodynamic tare from each measured coefficient,
        matched by (AoA_round, AoS_round):

            C_corrected = C_measured - C_tare

        Parameters
        ----------
        modeloff_corrector : ModelOffCorrector
            Pre-configured model-off corrector instance.
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.
        source_columns : dict[str, str], optional
            Mapping from canonical coefficient names to actual column names
            in self.df.
        cmpitch25c_column : str, optional
            Column name of the pitching moment at 25% chord.

        Returns
        -------
        pd.DataFrame
            Updated self.df.
        """
        self.df = modeloff_corrector.apply(
            df=self.df, save_csv=False, filename=None,
            source_columns=source_columns, cmpitch25c_column=cmpitch25c_column,
        )

        if save_csv:
            self.save_df(self.df, filename or "propOn_modeloff_corrected.csv")

        return self.df

    def compute_delta_CT_from_propoff(
        self,
        df_propoff: pd.DataFrame,
        ct_col_on: str = "CT",
        cl_col_off: str = "CL",
        cd_col_off: str = "CD",
        cyaw_col_off: str = "CYaw",
        aoa_col: str = "AoA_round",
        aos_col: str = "AoS_round",
        v_col: str = "V_round",
        dr_col: str = "dR",
        de_col: str = "dE",
        S_wing: float = None,
        S_prop: float = None,
        D: float = None,
        n_props: int = None,
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute the net propeller axial force as the difference in body-frame
        axial force coefficient CT between the prop-on and prop-off conditions.
        Convert this delta to a standard propeller thrust coefficient CT_props.

        Only rows where dR == 0 and dE == 0 are processed; rows without a
        matching prop-off condition receive NaN in the output columns.

        The prop-off CT is recovered by inverting the wind-axis transformation:

            CT_propoff = -CL * sin(alpha) + (CD * cos(beta) - CYaw * sin(beta)) * cos(alpha)

        where alpha = AoA [deg -> rad], beta = AoS [deg -> rad].
        This formula is algebraically exact for all alpha and beta
        (CY cancels out entirely via cos^2 + sin^2 = 1).

        The net propeller axial force coefficient (referenced to wing area):

            delta_CT = CT_propon - CT_propoff

        Standard propeller thrust coefficient (per propeller, referenced to
        propeller disc area and rotational speed):

            CT_props = -delta_CT * q * S_wing / (rho * n_rps^2 * D^4 * n_props)

        The negative sign reflects the convention that thrust opposes drag
        (delta_CT is negative when thrust dominates, CT_props is positive).

        Parameters
        ----------
        df_propoff : pd.DataFrame
            Prop-off dataset containing at minimum: AoA (raw), AoS (raw),
            V (raw), dR, dE, and the wind-axis force columns CL, CD, CYaw.
            Does NOT need to have AoA_round / V_round pre-computed.
        ct_col_on : str
            Column in self.df with the measured body-frame axial force
            coefficient. Default: "CT".
        cl_col_off, cd_col_off, cyaw_col_off : str
            Column names in df_propoff for lift, drag, and yaw force
            coefficients. Defaults: "CL", "CD", "CYaw".
        aoa_col : str
            Rounded AoA column used for matching. Default: "AoA_round".
        aos_col : str
            Rounded AoS column used for matching. Default: "AoS_round".
        v_col : str
            Rounded velocity column used for matching. Default: "V_round".
        dr_col : str
            Rudder deflection column. Default: "dR".
        de_col : str
            Elevator deflection column. Default: "dE".
        S_wing : float, optional
            Wing reference area [m^2]. Defaults to WING_AREA.
        S_prop : float, optional
            Single propeller disc area [m^2]. Defaults to PROP_AREA.
        D : float, optional
            Propeller diameter [m]. Defaults to PROP_DIAMETER.
        n_props : int, optional
            Number of propellers. Defaults to N_PROPS.
        save_csv : bool
            If True, write the updated self.df to disk.
        filename : str, optional
            Override output filename.

        Returns
        -------
        pd.DataFrame
            self.df with the following new columns:

            - CT_propoff_inv       body-frame CT recovered from prop-off CL/CD/CYaw
            - delta_CT             CT_propon - CT_propoff_inv  (prop net axial force)
            - delta_T              propeller thrust per unit dynamic pressure and wing area
            - CT_one_prop_delta    propeller thrust coefficient per propeller
            - CT_both_prop_delta   combined thrust coefficient for both propellers
            - propoff_match_found  bool flag: True where a prop-off match existed
        """
        S_wing  = S_wing  if S_wing  is not None else self.WING_AREA
        S_prop  = S_prop  if S_prop  is not None else self.PROP_AREA
        D       = D       if D       is not None else self.PROP_DIAMETER
        n_props = n_props if n_props is not None else self.N_PROPS

        # to:
        df_on = self.df.copy().reset_index(drop=True)

        # ------------------------------------------------------------------
        # Drop output columns if they already exist from a previous call
        # to prevent pandas _x/_y suffix collision on merge
        # ------------------------------------------------------------------
        _out_cols = [
            "CT_propoff_inv",
            "delta_CT",
            "delta_T",
            "CT_one_prop_delta",
            "CT_both_prop_delta",
            "CT_props_delta",
            "propoff_match_found",
        ]
        df_on = df_on.drop(columns=[c for c in _out_cols if c in df_on.columns])

        # ------------------------------------------------------------------
        # Validate required columns in prop-on df
        # ------------------------------------------------------------------
        self.require_columns(
            df_on,
            [ct_col_on, aoa_col, aos_col, v_col, dr_col, de_col, "q", "rho", "J"],
            context="compute_delta_CT_from_propoff (prop-on)",
        )

        # ------------------------------------------------------------------
        # Prepare prop-off df: add rounded keys if not present
        # ------------------------------------------------------------------
        df_off = df_propoff.copy()

        if "AoA_round" not in df_off.columns:
            df_off["AoA_round"] = self.round_to_half(df_off["AoA"])
        if "AoS_round" not in df_off.columns:
            df_off["AoS_round"] = self.round_to_half(df_off["AoS"])
        if "V_round" not in df_off.columns:
            df_off["V_round"] = df_off["V"].round().astype(int)

        self.require_columns(
            df_off,
            [cl_col_off, cd_col_off, cyaw_col_off, "AoA_round", "AoS_round",
             "V_round", dr_col, de_col],
            context="compute_delta_CT_from_propoff (prop-off)",
        )

        # ------------------------------------------------------------------
        # Invert wind-axis transform on prop-off to recover CT_propoff
        #
        #   CT_propoff = -CL*sin(alpha) + (CD*cos(beta) - CYaw*sin(beta))*cos(alpha)
        #
        # This formula is algebraically exact for all alpha and beta.
        # CY cancels out entirely via cos^2 + sin^2 = 1.
        # ------------------------------------------------------------------
        alpha_off = np.deg2rad(pd.to_numeric(df_off["AoA"], errors="coerce"))
        beta_off  = np.deg2rad(pd.to_numeric(df_off["AoS"], errors="coerce"))

        CL_off   = pd.to_numeric(df_off[cl_col_off],   errors="coerce")
        CD_off   = pd.to_numeric(df_off[cd_col_off],   errors="coerce")
        CYaw_off = pd.to_numeric(df_off[cyaw_col_off], errors="coerce")

        df_off["CT_inv"] = (
            -CL_off * np.sin(alpha_off)
            + (CD_off * np.cos(beta_off) - CYaw_off * np.sin(beta_off)) * np.cos(alpha_off)
        )

        # ------------------------------------------------------------------
        # Build prop-off lookup: average CT_inv per (AoA_round, AoS_round,
        # V_round) — only rows with dR==0, dE==0
        # ------------------------------------------------------------------
        df_off_clean = df_off[
            (df_off[dr_col] == 0) & (df_off[de_col] == 0)
        ].copy()

        lookup_keys = ["AoA_round", "AoS_round", "V_round"]

        for key in lookup_keys:
            df_off_clean[key] = pd.to_numeric(df_off_clean[key], errors="coerce").astype(float)

        ct_off_lookup = (
            df_off_clean
            .groupby(lookup_keys, as_index=False)["CT_inv"]
            .mean()
            .rename(columns={"CT_inv": "CT_propoff_inv"})
        )

        # ------------------------------------------------------------------
        # Filter prop-on rows: only dR==0 and dE==0
        # NOTE: output columns are NOT added to df_on before this point —
        # doing so would cause df_valid to inherit CT_propoff_inv, which
        # then collides with the merge and produces _x/_y suffixed columns.
        # ------------------------------------------------------------------
        valid_mask = (
            (pd.to_numeric(df_on[dr_col], errors="coerce") == 0) &
            (pd.to_numeric(df_on[de_col], errors="coerce") == 0)
        )

        df_valid = df_on[valid_mask].copy()

        if df_valid.empty:
            print("compute_delta_CT_from_propoff: no rows with dR==0, dE==0 found.")
            df_on["CT_propoff_inv"]      = np.nan
            df_on["delta_CT"]            = np.nan
            df_on["CT_props_delta"]      = np.nan
            df_on["propoff_match_found"] = False
            self.df = df_on
            return self.df

        # ------------------------------------------------------------------
        # Align merge key dtypes — both sides forced to float64
        # ------------------------------------------------------------------
        for key in lookup_keys:
            df_valid[key]      = pd.to_numeric(df_valid[key],      errors="coerce").astype(float)
            ct_off_lookup[key] = pd.to_numeric(ct_off_lookup[key], errors="coerce").astype(float)

        # ------------------------------------------------------------------
        # Merge prop-off CT onto valid prop-on rows
        # ------------------------------------------------------------------
        df_valid = df_valid.merge(
            ct_off_lookup,
            on=lookup_keys,
            how="left",
        )

        if "CT_propoff_inv" not in df_valid.columns:
            raise ValueError(
                "compute_delta_CT_from_propoff: merge produced no CT_propoff_inv column.\n"
                f"df_valid key dtypes:       { {k: df_valid[k].dtype for k in lookup_keys} }\n"
                f"ct_off_lookup key dtypes:  { {k: ct_off_lookup[k].dtype for k in lookup_keys} }\n"
                f"df_valid shape: {df_valid.shape}, "
                f"ct_off_lookup shape: {ct_off_lookup.shape}"
            )

        df_valid["propoff_match_found"] = df_valid["CT_propoff_inv"].notna()

        # ------------------------------------------------------------------
        # delta_CT = CT_propon - CT_propoff
        # ------------------------------------------------------------------
        CT_on  = pd.to_numeric(df_valid[ct_col_on],       errors="coerce")
        CT_off = pd.to_numeric(df_valid["CT_propoff_inv"], errors="coerce")

        df_valid["delta_CT"] = CT_on - CT_off

        # ------------------------------------------------------------------
        # CT_props = -delta_CT * q * S_wing / (rho * n_rps^2 * D^4 * n_props)
        #
        # n_rps = V / (J * D)  [rev/s]
        # ------------------------------------------------------------------
        q   = pd.to_numeric(df_valid["q"],   errors="coerce")
        rho = pd.to_numeric(df_valid["rho"], errors="coerce")
        J   = pd.to_numeric(df_valid["J"],   errors="coerce")
        V   = pd.to_numeric(df_valid["V"],   errors="coerce")

        J_safe = J.replace(0, np.nan)
        n_rps  = V / (J_safe * D)

        df_valid["delta_T"] = df_valid["delta_CT"] * q * S_wing 

        df_valid["CT_one_prop_delta"] = (
            -df_valid["delta_CT"] * q * S_wing
            / (rho * n_rps**2 * D**4 * n_props)
        )

        df_valid["CT_both_prop_delta"] = df_valid["CT_one_prop_delta"] * n_props

        # ------------------------------------------------------------------
        # Initialise output columns on df_on (AFTER merge, no collision risk)
        # then write computed values back at the correct indices
        # ------------------------------------------------------------------
        df_on["CT_propoff_inv"]      = np.nan
        df_on["delta_CT"]            = np.nan
        df_on["delta_T"]             = np.nan
        df_on["CT_one_prop_delta"]   = np.nan
        df_on["CT_both_prop_delta"]  = np.nan
        df_on["propoff_match_found"] = False

        for col in ["CT_propoff_inv", "delta_CT", "delta_T", "CT_one_prop_delta", "CT_both_prop_delta", "propoff_match_found"]:
            df_on.loc[valid_mask, col] = df_valid[col].values

        self.df = df_on

        if save_csv:
            self.save_df(self.df, filename or "propOn_delta_CT.csv")

        return self.df
    
# ============================================================
# Tail-off data
# ============================================================
class TailOffData(BaseCorrector):
    """
    Container and correction pipeline for tail-off (no horizontal tail)
    wind-tunnel measurements.

    The tail-off dataset serves a dual role:

    1. Self-correction: applies a solid-blockage correction to the tail-off
       coefficients (no wake or slipstream blockage is applied).

    2. Reference for other datasets: PropOffData and PropOnData look up
       CLw_tailoff (the tail-off wing lift coefficient at matching
       conditions) from self.grid_df to drive the streamline-curvature,
       downwash, and tail corrections.

    Recommended usage order:
        1. apply_solid_blockage()
        2. build_alpha_slice_grid_by_velocity()
        3. compute_cl_alpha_slope_by_case()    (needed for streamline curvature)

    Class-level override
    --------------------
    E_SOLID_BLOCKAGE is overridden to E_SOLID_BLOCKAGE_TAILOFF = 0.006406642,
    reflecting the smaller model volume with the tail removed.

    Parameters
    ----------
    df : pd.DataFrame
        Raw tail-off measurement dataframe.
    clip_negative_cdsep : bool
        Retained for API consistency (unused in tail-off pipeline).
    save_dir : str or Path, optional
        Output directory for CSV exports.
    """

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
        Apply the solid-blockage correction to the tail-off dataset.

        Uses the tail-off solid-blockage constant:

            e_sb = E_SOLID_BLOCKAGE_TAILOFF = 0.006406642

        This is smaller than the full-configuration value (0.007229438)
        because the horizontal tail is absent, reducing the model volume.

        The correction uses the suffix 'solid_blockage_corr' (rather than
        the generic 'blockage_corr') so that downstream code can
        unambiguously identify the corrected lift column
        'CL_solid_blockage_corr' during the CLw lookup.

        Formulae
        --------
            V_corr = V / (1 + e_sb)
            C_corr = C / (1 + e_sb)^2

        Parameters
        ----------
        save_csv : bool
            Write output CSV if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            self.df with V_solid_blockage_corr, CL_solid_blockage_corr,
            CD_solid_blockage_corr, and other *_solid_blockage_corr
            columns added.
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
        Build a complete (AoA_round, AoS_round) grid of tail-off coefficients
        for each V_round.

        The tail-off test matrix typically sweeps AoS at a few fixed AoA
        anchor values. This method constructs a dense grid over all
        combinations of AoA and AoS using the following priority:

            1. Measured: use the value directly where available.
            2. Interpolation in AoA: linearly interpolate between anchor
               slices when the target AoA lies within the measured range
               at that AoS.
            3. Extrapolation in AoA: linearly extrapolate beyond the range
               using the two nearest anchor points when the target AoA is
               outside the measured range.
            4. Unresolved: leave as NaN if fewer than two anchor slices
               are available at the requested AoS.

        The source type of each grid cell is recorded in the column
        'source_type' with values:
            'measured', 'interp_alpha', 'extrap_alpha', 'unresolved'

        The resulting grid_df is stored in self.grid_df and used as the
        CLw lookup table by PropOffData and PropOnData correction methods.

        Parameters
        ----------
        coeff_cols : list of str, optional
            Coefficient columns to include in the grid. Defaults to
            CL_solid_blockage_corr, CD_solid_blockage_corr, and their
            raw equivalents if present in self.df.
        anchor_aoa_vals : tuple of float
            AoA values [degrees] at which AoS sweeps are available.
            Used as interpolation anchors.
        extra_aoa_vals : iterable of float, optional
            Additional AoA values to force into the grid even if not
            measured.
        extra_aos_vals : iterable of float, optional
            Additional AoS values to force into the grid even if not
            measured.
        save_csv : bool
            Write self.grid_df to disk if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            self.grid_df with columns: V_round, AoA_round, AoS_round,
            case_available, source_type, and all coeff_cols.

        Raises
        ------
        ValueError
            If none of the anchor_aoa_vals are present at a given velocity,
            or if no coefficient columns are found.
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
            """
            Interpolate or extrapolate y at x_target given paired arrays
            (x_known, y_known). Returns (value, source_label) where
            source_label is one of 'interp_alpha', 'extrap_alpha', or
            'unresolved'. Returns (nan, 'unresolved') if fewer than two
            valid points exist.

            Extrapolation uses the two nearest known points (linear):

                slope = (y1 - y0) / (x1 - x0)
                y_extrap = y0 + slope * (x_target - x0)
            """
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
        Compute the CL-alpha slope for each (V_round, AoS_round) case
        using a linear regression over a specified AoA range.

        For each test condition, fits the linear model:

            CL = CL0 + (dCL/dalpha) * alpha

        using ordinary least squares, where the slope is in units of
        per degree. The AoA range used for the fit is [aoa_min, aoa_max].

        The result is stored in self.cl_a_df and is subsequently used by
        the streamline-curvature correction to scale the AoA and CL
        corrections.

        Goodness of fit:

            SS_res = sum( (CL - CL_predicted)^2 )
            SS_tot = sum( (CL - mean(CL))^2 )
            R^2    = 1 - SS_res / SS_tot

        Parameters
        ----------
        aoa_min : float
            Minimum AoA [degrees] to include in the linear fit.
        aoa_max : float
            Maximum AoA [degrees] to include in the linear fit.
        cl_col : str
            Lift-coefficient column in self.grid_df.
        aoa_col : str
            AoA column name.
        aos_col : str
            AoS column name.
        v_col : str
            Velocity column name.
        min_points : int
            Minimum number of distinct AoA points required to attempt a
            fit. Cases with fewer points receive NaN and fit_success=False.
        save_csv : bool
            Write self.cl_a_df to disk if True.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            self.cl_a_df with one row per (V_round, AoS_round) case,
            containing:

            - cl_alpha_slope_per_deg    dCL/dalpha [deg^-1]
            - cl_at_aoa0               intercept CL0
            - r2                        coefficient of determination R^2
            - n_points                  number of data points used
            - n_unique_aoa              number of unique AoA values
            - aoa_min_fit               the aoa_min argument
            - aoa_max_fit               the aoa_max argument
            - fit_success               boolean flag

        Raises
        ------
        ValueError
            If self.grid_df has not been built yet.
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