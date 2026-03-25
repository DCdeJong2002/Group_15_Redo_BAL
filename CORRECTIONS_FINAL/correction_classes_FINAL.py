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

All correction formulae follow the methodology described in
AE4115 Lab Manual (TU Delft), in particular Appendix B (BEM polynomial)
and the standard AGARD / ESDU wall-correction framework.

Correction sequence (recommended order)
----------------------------------------
1.  Model-off tare subtraction         (ModelOffCorrector)
2.  BEM thrust separation              (compute_thrust_separation_BEM)
3.  Solid-blockage factor              (compute_solid_blockage_e)
4.  Wake-blockage factor               (compute_wake_blockage_e)
5.  Slipstream-blockage factor         (compute_slipstream_blockage_e) – prop-on only
6.  Combined blockage correction       (apply_blockage_correction)
7.  Streamline-curvature correction    (apply_streamline_curvature_correction)
8.  Downwash correction                (apply_downwash_correction)
9.  Tail correction                    (apply_tail_correction)
10. Rename final columns               (rename_detected_final_force_moment_columns)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)


class BaseCorrector:
    """
    Abstract base class shared by :class:`PropOffData`, :class:`PropOnData`,
    and :class:`TailOffData`.

    Provides
    --------
    * Configurable output directory and CSV helper.
    * Required-column validation.
    * ``round_to_half`` utility (rounds to nearest 0.5).
    * Solid-blockage factor computation.
    * Wake-blockage factor computation.
    * Combined blockage correction (velocity + coefficients).
    * Streamline-curvature correction (shared implementation).
    * Downwash correction (shared implementation).
    * Tail interference correction (shared implementation).
    * Tail-off CLw lookup / merge helper.
    * BEM-based propeller thrust computation.
    * BEM-based thrust separation from body-frame axial force.
    * Final-column auto-detection and renaming.

    Tunnel / Model Constants
    -------------------------
    All subclasses inherit the following geometric constants.  Override at the
    subclass level if needed.

    TUNNEL_AREA   : float = 2.07 m²   – wind-tunnel cross-sectional area *C*
    WING_AREA     : float = 0.2172 m² – wing reference area *S*
    GEOM_FACTOR   : float = S / C     – geometric blockage ratio

    PROP_DIAMETER : float = 0.2032 m
    PROP_AREA     : float = π/4 · D²  – single propeller disc area
    N_PROPS       : int   = 2         – number of propellers

    DELTA_WING    : float = 0.1065    – wing solid-angle factor δ (ESDU)
    DELTA_TAIL    : float = 0.1085    – tail solid-angle factor δ_t
    TAU2_WING     : float = 0.045     – streamline-curvature factor τ₂ (wing)
    TAU2_TAIL     : float = 0.8       – streamline-curvature factor τ₂ (tail location)

    DCMPITCH_DALPHA : float = −0.15676 rad⁻¹  – pitching-moment slope ∂Cm/∂α of the
                                                 tail (used in tail interference)

    E_SOLID_BLOCKAGE        : float = 0.007229438  – ε_sb for full configuration
    E_SOLID_BLOCKAGE_TAILOFF: float = 0.006406642  – ε_sb for tail-off configuration
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

    DELTA_WING:               float = 0.1065
    DELTA_TAIL:               float = 0.1085
    TAU2_WING:                float = 0.045
    TAU2_TAIL:                float = 0.8

    DCMPITCH_DALPHA:          float = -0.15676   # per rad

    E_SOLID_BLOCKAGE:         float = 0.007229438
    E_SOLID_BLOCKAGE_TAILOFF: float = 0.006406642

    # ============================================================
    def __init__(self, save_dir: str | Path | None = None) -> None:
        """
        Parameters
        ----------
        save_dir : str or Path, optional
            Directory used for all :meth:`save_df` calls.  Defaults to the
            directory containing this source file.  Created automatically if it
            does not exist.
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
        Round each element of *series* to the nearest 0.5.

        Formula
        -------
        .. math::
            x_{\\text{rounded}} = \\frac{\\text{round}(2x)}{2}

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
        Raise :class:`ValueError` if any of *required_cols* are absent from *df*.

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

    def set_save_directory(self, directory: str | Path) -> None:
        """
        Change the output directory used by :meth:`save_df`.

        Parameters
        ----------
        directory : str or Path
            New directory path.  Created automatically if it does not exist.
        """
        self.save_dir = Path(directory)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_df(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Write *df* to ``<save_dir>/<filename>`` as a headerless-index CSV.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to persist.
        filename : str
            File name (not a full path) relative to :attr:`save_dir`.

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
        dataset_label: str = "dataset",
        save_csv: bool = False,
        filename: Optional[str] = None,
        default_filename: str = "streamline_curvature_corrected.csv",
    ) -> pd.DataFrame:
        """
        Apply the streamline-curvature (tunnel-induced camber) correction.

        The curved streamlines induced by the tunnel walls impose an effective
        camber on the model, shifting the apparent angle of attack and the
        pitching moment.  The correction references the tail-off lift
        coefficient *CL_w* (from ``tailoff.grid_df``) and the tail-off
        CL-alpha slope (from ``tailoff.cl_a_df``).

        Formulae
        --------
        Induced incidence correction:

        .. math::
            \\Delta\\alpha_{sc} = \\tau_2 \\, \\delta \\, \\frac{S}{C} \\, C_{L_w}
            \\quad [\\text{radians}]

        Corrected angle of attack:

        .. math::
            \\alpha_{sc} = \\alpha + \\Delta\\alpha_{sc}

        Lift-coefficient correction:

        .. math::
            \\Delta C_L^{sc} = -\\Delta\\alpha_{sc,\\deg} \\cdot
            \\left(\\frac{\\partial C_L}{\\partial \\alpha}\\right)_{\\text{tail-off}}

        Pitching-moment correction:

        .. math::
            \\Delta C_m^{sc} = -0.25 \\, \\Delta C_L^{sc}

        where the factor 0.25 assumes the aerodynamic centre at the
        quarter-chord.

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset.  Must have ``grid_df`` and ``cl_a_df``
            populated before calling this method.
        tau : float, optional
            Streamline-curvature parameter τ₂.  Defaults to :attr:`TAU2_WING`.
        delta : float, optional
            Solid-angle factor δ.  Defaults to :attr:`DELTA_WING`.
        geom_factor : float, optional
            Geometric blockage ratio S/C.  Defaults to :attr:`GEOM_FACTOR`.
        cl_source_col : str
            Column name of the blockage-corrected lift coefficient in
            ``self.df`` (input).
        cm_source_col : str
            Column name of the blockage-corrected pitching-moment coefficient in
            ``self.df`` (input).
        dataset_label : str
            Label used in error messages.
        save_csv : bool
            If ``True``, write the corrected dataframe to disk.
        filename : str, optional
            Override output file name.
        default_filename : str
            Fallback file name when *filename* is ``None``.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with the following new columns added:

            * ``delta_alpha_sc_rad``          – Δα_sc in radians
            * ``delta_alpha_sc_deg``          – Δα_sc in degrees
            * ``AoA_streamline_curvature_corr``
            * ``delta_CL_sc``                 – ΔC_L correction
            * ``delta_CMpitch_sc``            – ΔC_m correction
            * ``{cl_source_col}_sc_corr``     – corrected C_L
            * ``{cm_source_col}_sc_corr``     – corrected C_m
            * ``streamline_curvature_data_found`` – boolean lookup flag
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
        Apply the tunnel-induced downwash (angle-of-attack) correction.

        The finite tunnel cross-section introduces a downwash at the model
        that increases the apparent angle of attack and adds induced drag.
        The correction uses the tail-off wing lift coefficient *CL_w* as the
        reference lifting load.

        Formulae
        --------
        Downwash-induced incidence increment:

        .. math::
            \\Delta\\alpha_{dw} = \\delta \\, \\frac{S}{C} \\, C_{L_w} \\times 57.3
            \\quad [\\text{degrees}]

        Corrected angle of attack:

        .. math::
            \\alpha_{dw} = \\alpha + \\Delta\\alpha_{dw}

        Induced-drag correction:

        .. math::
            \\Delta C_D^{dw} = \\delta \\, \\frac{S}{C} \\, C_{L_w}^{2}

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with ``grid_df`` populated.
        delta : float, optional
            Solid-angle factor δ.  Defaults to :attr:`DELTA_WING`.
        geom_factor : float, optional
            Geometric blockage ratio S/C.  Defaults to :attr:`GEOM_FACTOR`.
        aoa_source_col : str, optional
            Column in ``self.df`` used as the angle-of-attack input.  If
            ``None``, falls back to ``"AoA"`` then ``"AoA_round"``.
        cd_source_col : str
            Column name of the blockage-corrected drag coefficient.
        dataset_label : str
            Label used in error messages.
        save_csv : bool
            If ``True``, write the corrected dataframe to disk.
        filename : str, optional
            Override output file name.
        default_filename : str
            Fallback file name when *filename* is ``None``.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with the following new columns added:

            * ``delta_alpha_dw_deg``        – Δα_dw in degrees
            * ``delta_alpha_dw_rad``        – Δα_dw in radians
            * ``delta_CD_dw``               – ΔC_D induced-drag correction
            * ``AoA_downwash_corr``         – corrected angle of attack
            * ``{cd_source_col}_dw_corr``   – corrected C_D
            * ``downwash_data_found``       – boolean lookup flag
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
        Apply the tail-plane interference (tail-correction) correction.

        The tunnel-wall constraint distorts the downwash at the horizontal
        tail, creating an error in the measured pitching moment.  The
        correction estimates the additional incidence felt by the tail using
        the tail streamline-curvature parameter and then infers the pitching-
        moment change via the tail's pitching-moment slope.

        Formulae
        --------
        Tail-induced angle-of-attack increment (radians):

        .. math::
            \\Delta\\alpha_{tail} = \\delta_t \\, \\frac{S}{C} \\, C_{L_w} \\, \\tau_{2,lt}

        In degrees:

        .. math::
            \\Delta\\alpha_{tail,\\deg} = \\frac{180}{\\pi} \\, \\Delta\\alpha_{tail}

        Pitching-moment correction (sign convention: tail downwash reduces
        effective incidence, making C_m more negative):

        .. math::
            \\Delta C_m^{tail} = \\frac{\\partial C_m}{\\partial \\alpha} \\,
            \\Delta\\alpha_{tail}

        where *∂Cm/∂α* is supplied via *dcmpitch_dalpha* (default
        :attr:`DCMPITCH_DALPHA` = −0.15676 rad⁻¹).

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with ``grid_df`` populated.
        delta : float, optional
            Tail solid-angle factor δ_t.  Defaults to :attr:`DELTA_TAIL`.
        geom_factor : float, optional
            Geometric blockage ratio S/C.  Defaults to :attr:`GEOM_FACTOR`.
        tau2_lt : float, optional
            Tail streamline-curvature parameter τ₂ at the tail location.
            Defaults to :attr:`TAU2_TAIL`.
        dcmpitch_dalpha : float, optional
            Pitching-moment slope ∂Cm/∂α of the tail.
            Defaults to :attr:`DCMPITCH_DALPHA`.
        dcmpitch_dalpha_unit : {"per_rad", "per_deg"}
            Unit of *dcmpitch_dalpha*.  Determines whether Δα_tail is
            expressed in radians or degrees when computing ΔC_m.
        aoa_source_col : str, optional
            Column in ``self.df`` used as the angle-of-attack input.  Falls
            back to ``"AoA"`` then ``"AoA_round"`` when ``None``.
        cmpitch_source_col : str
            Column name of the pitching-moment coefficient to correct.
        dataset_label : str
            Label used in error messages.
        save_csv : bool
            If ``True``, write the corrected dataframe to disk.
        filename : str, optional
            Override output file name.
        default_filename : str
            Fallback file name when *filename* is ``None``.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with the following new columns added:

            * ``delta_alpha_tail_rad``                  – Δα_tail in radians
            * ``delta_alpha_tail_deg``                  – Δα_tail in degrees
            * ``delta_CMpitch_tail``                    – ΔC_m tail correction
            * ``AoA_tail_corr``                         – corrected AoA
            * ``{cmpitch_source_col}_tail_corr``        – corrected C_m
            * ``tail_correction_data_found``            – boolean lookup flag

        Raises
        ------
        ValueError
            If ``dcmpitch_dalpha_unit`` is not ``"per_deg"`` or ``"per_rad"``.
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
        Guarantee that ``AoA_round``, ``AoS_round``, and ``V_round`` columns
        are present in *df*, creating them from the raw columns when absent.

        The priority for velocity is:

        1. ``V_round``               (already exists – no-op)
        2. ``V_solid_blockage_corr`` (preferred velocity after solid-blockage)
        3. ``V``                     (raw freestream velocity)

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to augment (a copy is **not** made – mutated in place).
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
        Left-merge the tail-off wing lift coefficient *CLw_tailoff* onto *df*
        from ``tailoff.grid_df``, matching on the key triple
        ``(V_round, AoA_round, AoS_round)``.

        The source column in ``tailoff.grid_df`` is
        ``CL_solid_blockage_corr``, which is the blockage-corrected lift
        coefficient of the wing-only (tail-off) configuration.  Rows in *df*
        that have no matching entry in the grid receive ``NaN``.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to enrich.  Must already contain
            ``V_round``, ``AoA_round``, ``AoS_round``.
        tailoff : TailOffData
            Reference tail-off dataset.
        context : str, optional
            Descriptive label for error messages.

        Returns
        -------
        pd.DataFrame
            *df* with ``CLw_tailoff`` column appended.

        Raises
        ------
        ValueError
            If the post-merge dataframe does not contain ``CLw_tailoff``
            (indicating a column-name collision).
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
        Compute and store the solid-blockage factor ε_sb.

        The solid-blockage factor accounts for the reduction in effective
        tunnel cross-section caused by the physical volume of the model.
        Two modes are supported:

        * **Constant** (``use_constant_e=True``): uses the class-level
          constant :attr:`E_SOLID_BLOCKAGE` (or the overridden value in
          :class:`TailOffData`).
        * **Per-row** (``use_e_column=True``): reads ε_sb from a column
          already present in ``self.df``.

        The factor is stored in *output_col* (default ``"esb"``) and is
        subsequently consumed by :meth:`_apply_combined_blockage_from_e_columns`.

        Parameters
        ----------
        use_constant_e : bool
            Use the class constant :attr:`E_SOLID_BLOCKAGE`.
        use_e_column : bool
            Read ε_sb from *e_column* in ``self.df``.
        e_column : str
            Source column name when *use_e_column* is ``True``.
        output_col : str
            Destination column name for ε_sb.
        save_csv : bool
            If ``True``, write the updated dataframe to disk.
        filename : str, optional
            Override output file name.
        default_filename : str
            Fallback file name.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with *output_col* added.

        Raises
        ------
        ValueError
            If both or neither of *use_constant_e* / *use_e_column* are
            ``True``, or if *e_column* is missing.
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
        Compute and store the wake-blockage factor ε_wb.

        The wake-blockage factor accounts for the reduction in effective
        freestream velocity due to the momentum deficit in the model's wake.
        It is separated into a zero-lift (pressure drag) term and a
        separated-flow (form drag) term.

        Formulae
        --------
        Induced drag (from polar fit):

        .. math::
            C_{D_i} = k \\, C_L^{2}

        Separated / form drag:

        .. math::
            C_{D_{sep}} = C_D - C_{D_0} - C_{D_i}

        Wake-blockage factor:

        .. math::
            \\varepsilon_{wb} = \\frac{S}{4C} \\, C_{D_0}
            + \\frac{5S}{4C} \\, C_{D_{sep}}

        where *S* is the wing area (:attr:`WING_AREA`) and *C* is the
        tunnel cross-sectional area (:attr:`TUNNEL_AREA`).

        Note
        ----
        Negative values of *C_{D_{sep}}* are physically unrealistic and are
        clipped to zero by default (``clip_negative_cdsep=True``).

        Parameters
        ----------
        cd0_col : str
            Column name of the zero-lift drag coefficient C_D0 (from polar fit).
        cd_col : str
            Column name of the measured drag coefficient C_D.
        cl_col : str
            Column name of the measured lift coefficient C_L.
        k_col : str
            Column name of the induced-drag factor *k* (from polar fit).
        output_col : str
            Destination column name for ε_wb.
        clip_negative_cdsep : bool
            If ``True``, clip *C_{D_{sep}}* to zero from below.
        save_csv : bool
            If ``True``, write the updated dataframe to disk.
        filename : str, optional
            Override output file name.
        default_filename : str
            Fallback file name.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with intermediate columns ``CL2_fit``, ``CDi_fit``,
            ``CDsep_fit``, and *output_col* added.
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
        force/moment coefficients using the pre-computed blockage factor
        columns.

        The total blockage factor is:

        .. math::
            \\varepsilon_{total} = \\varepsilon_{sb} + \\varepsilon_{wb}
            + \\varepsilon_{ss}

        where any of the three terms can be excluded via the ``apply_*``
        flags.

        Corrected freestream velocity:

        .. math::
            V_{corr} = \\frac{V}{1 + \\varepsilon_{total}}

        Corrected aerodynamic coefficients:

        .. math::
            C_{corr} = \\frac{C}{(1 + \\varepsilon_{total})^{2}}

        Special handling for thrust coefficient
        ----------------------------------------
        ``CFt_thrust_BEM`` is the propeller thrust coefficient that *causes*
        slipstream blockage (ε_ss).  Correcting it with ε_ss would be
        circular, so it always receives solid + wake blockage only:

        .. math::
            C_{F_t,\\text{corr}} = \\frac{C_{F_t}}{(1 + \\varepsilon_{sb}
            + \\varepsilon_{wb})^{2}}

        Parameters
        ----------
        apply_esb : bool
            Include solid-blockage factor ε_sb.
        apply_ewb : bool
            Include wake-blockage factor ε_wb.
        apply_ess : bool
            Include slipstream-blockage factor ε_ss (prop-on only).
        esb_col : str
            Column name for ε_sb.
        ewb_col : str
            Column name for ε_wb.
        ess_col : str
            Column name for ε_ss.
        velocity_cols : sequence of str
            Velocity columns to correct.  Each produces ``{col}_{suffix}``.
        coefficient_cols : sequence of str
            Coefficient columns to correct.  Each produces ``{col}_{suffix}``.
        suffix : str
            Suffix appended to all output columns.
        cft_thrust_col : str or None
            Column name of ``CFt_thrust_BEM``.  Set to ``None`` to skip the
            special thrust correction.
        save_csv : bool
            If ``True``, write the updated dataframe to disk.
        filename : str, optional
            Override output file name.
        default_filename : str
            Fallback file name.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with corrected velocity/coefficient columns and
            ``e_total_blockage`` added.
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
        (Appendix B, AE4115 Lab Manual).

        BEM polynomial (per propeller), expressed as a function of advance
        ratio *J*:

        .. math::
            C_{T,\\text{BEM}}(J) = -0.0051\\,J^{4}
            + 0.0959\\,J^{3}
            - 0.5888\\,J^{2}
            + 1.0065\\,J
            - 0.1353

        Propeller rotational speed derived from the advance ratio:

        .. math::
            n = \\frac{V}{J \\, D}  \\quad [\\text{rev/s}]

        Thrust of a single propeller:

        .. math::
            T_1 = C_{T,\\text{BEM}} \\, \\rho \\, n^2 \\, D^4

        Total thrust (for *N* propellers):

        .. math::
            T_{total} = N_{\\text{props}} \\cdot T_1

        Thrust force coefficient (referenced to wing area):

        .. math::
            C_{F_t} = \\frac{T_{total}}{q \\, S}

        Thrust loading coefficient (referenced to propeller disc area):

        .. math::
            T_c^{*} = \\frac{T_{total}}{q \\, S_{prop}}

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe (not mutated; a copy is returned).
        j_col : str
            Column name of the advance ratio *J*.
        v_col : str
            Column name of the freestream velocity *V* [m/s].
        rho_col : str
            Column name of the air density *ρ* [kg/m³].
        q_col : str
            Column name of the dynamic pressure *q* [Pa].
        D : float, optional
            Propeller diameter [m].  Defaults to :attr:`PROP_DIAMETER`.
        S_wing : float, optional
            Wing reference area [m²].  Defaults to :attr:`WING_AREA`.
        S_prop : float, optional
            Single propeller disc area [m²].  Defaults to :attr:`PROP_AREA`.
        n_props : int, optional
            Number of propellers.  Defaults to :attr:`N_PROPS`.
        output_cft_col : str
            Column name for the output ``CFt_thrust_BEM``.
        output_tcstar_col : str
            Column name for the output ``Tc_star_BEM``.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with ``CFt_thrust_BEM`` and ``Tc_star_BEM`` columns
            added.
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
        Separate the propeller thrust contribution from the measured axial
        body-frame force coefficient *CT*, yielding aerodynamic-only lift,
        drag, and side-force coefficients.

        The measured axial coefficient contains both aerodynamic drag and
        the propeller thrust (acting in the opposite sense):

        .. math::
            C_{T,\\text{measured}} = C_{D,\\text{aero}} - C_{F_t,\\text{thrust}}

        Rearranging for the aerodynamic axial force:

        .. math::
            C_{F_t,\\text{aero}} = C_{T,\\text{measured}} + C_{F_t,\\text{thrust,BEM}}

        Wind-axis transformation from body-frame force coefficients
        (*C_N* = normal, *C_S* = side, *C_{Ft,aero}* = axial) to
        stability-axis aerodynamic coefficients:

        Aerodynamic drag:

        .. math::
            C_D = \\bigl(C_N \\sin\\alpha + C_{F_t,\\text{aero}} \\cos\\alpha\\bigr)
            \\cos\\beta + C_S \\sin\\beta

        Aerodynamic lift:

        .. math::
            C_L = C_N \\cos\\alpha - C_{F_t,\\text{aero}} \\sin\\alpha

        Aerodynamic yaw force:

        .. math::
            C_{Y,\\text{aero}} = -\\bigl(C_N \\sin\\alpha
            + C_{F_t,\\text{aero}} \\cos\\alpha\\bigr) \\sin\\beta
            + C_S \\cos\\beta

        where α is the angle of attack and β is the sideslip angle (both in
        radians).

        Parameters
        ----------
        ct_col_on : str
            Column name of the measured axial body-frame force coefficient.
        aoa_col_on : str
            Column name of the angle of attack [degrees].
        aos_col_on : str
            Column name of the sideslip angle [degrees].
        S_wing : float, optional
            Wing reference area [m²].  Defaults to :attr:`WING_AREA`.
        S_prop : float, optional
            Single propeller disc area [m²].  Defaults to :attr:`PROP_AREA`.
        D : float, optional
            Propeller diameter [m].  Defaults to :attr:`PROP_DIAMETER`.
        n_props : int, optional
            Number of propellers.  Defaults to :attr:`N_PROPS`.
        recompute_cd : bool
            If ``True``, compute and store ``CD_aero_BEM``.
        recompute_cl : bool
            If ``True``, compute and store ``CL_aero_BEM``.
        recompute_cyaw : bool
            If ``True``, compute and store ``CYaw_aero_BEM``.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with the following new columns:

            Always added:

            * ``CFt_thrust_BEM``  – propeller thrust force coefficient
            * ``Tc_star_BEM``     – thrust loading coefficient T/(q S_prop)
            * ``CFt_aero_BEM``    – aerodynamic axial force coefficient

            Conditional (controlled by ``recompute_*`` flags):

            * ``CD_aero_BEM``
            * ``CL_aero_BEM``
            * ``CYaw_aero_BEM``
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
        Detect the most fully corrected column for each base aerodynamic
        quantity and rename it with *final_suffix* (default ``"_FINAL"``).

        Detection logic
        ---------------
        For each base name (e.g. ``"CD"``):

        1. Collect all columns whose name starts with that base.
        2. Among those, prefer columns whose name contains ``"corr"``
           (indicating at least one correction has been applied).
        3. Among the filtered candidates, select the one with the **most
           underscore-separated parts** — i.e., the column that has gone
           through the most correction steps.
        4. Rename it to ``{base}_FINAL``.

        Parameters
        ----------
        base_cols : tuple of str
            Base column names to search for.
        final_suffix : str
            Suffix appended to each selected column.
        save_csv : bool
            If ``True``, write the renamed dataframe to disk.
        filename : str
            Output file name.
        save_directory : str or Path, optional
            Override the output directory.  Falls back to :attr:`save_dir`.
        verbose : bool
            If ``True``, print the detected rename mapping to stdout.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with selected columns renamed to ``{base}_FINAL``.
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
    """
    Apply the model-off tare correction to a measured dataframe.

    The model-off correction removes the aerodynamic contribution of the
    model support structure (sting, struts, etc.) that was measured in a
    separate run with the model absent.  The correction is applied by
    subtracting the tare coefficients, looked up by ``(AoA_round, AoS_round)``,
    from the corresponding coefficients in the measured dataframe.

    Parameters
    ----------
    correction_csv : str or Path
        Path to the CSV file containing the model-off tare grid.  Must
        include columns ``AoA_round``, ``AoS_round``, ``CD``, ``CYaw``,
        ``CL``, ``CMroll``, ``CMpitch``, ``CMyaw``.
    apply_cmpitch_to_25c : bool
        If ``True`` (default), also subtract the pitching-moment tare from the
        ``CMpitch25c`` column (moment transferred to the 25 % chord reference
        point).
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
        Load and validate the model-off correction grid from
        :attr:`correction_csv`.

        The grid is deduplicated on ``(AoA_round, AoS_round)`` and the
        coefficient columns are renamed with the ``_modeloff`` suffix for
        clarity during the subsequent merge.

        Returns
        -------
        pd.DataFrame
            Cleaned tare grid with columns:
            ``AoA_round``, ``AoS_round``,
            ``CD_modeloff``, ``CYaw_modeloff``, ``CL_modeloff``,
            ``CMroll_modeloff``, ``CMpitch_modeloff``, ``CMyaw_modeloff``.

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
        Apply the model-off tare correction to *df*.

        For each canonical coefficient *C*, the corrected value is:

        .. math::
            C_{\\text{corrected}} = C_{\\text{measured}} - C_{\\text{tare}}

        where *C_tare* is the value from the model-off grid matched by
        ``(AoA_round, AoS_round)``.  If ``apply_cmpitch_to_25c`` is ``True``,
        the same subtraction is applied to the ``CMpitch25c`` column.

        Original uncorrected values are preserved in ``{col}_uncorrected``
        columns.

        Parameters
        ----------
        df : pd.DataFrame
            Measured dataframe to correct.  Should contain the coefficient
            columns to be corrected, plus either ``(AoA_round, AoS_round)``
            or ``(AoA, AoS)`` (which are rounded internally).
        save_csv : bool
            If ``True``, write the corrected dataframe to disk.
        filename : str, optional
            Override output file name.
        source_columns : dict[str, str], optional
            Mapping ``{canonical_name: actual_column_name}`` for cases where
            the coefficient columns have non-standard names.  Canonical names
            are ``"CD"``, ``"CYaw"``, ``"CL"``, ``"CMroll"``, ``"CMpitch"``,
            ``"CMyaw"``.  If ``None``, the canonical names are used directly.
        cmpitch25c_column : str, optional
            Column name of the pitching moment at the 25 % chord.  Defaults
            to ``"CMpitch25c"``.

        Returns
        -------
        pd.DataFrame
            Corrected dataframe with ``{col}_uncorrected`` backup columns and
            a ``modeloff_correction_found`` boolean column indicating whether a
            tare entry was found for each row.
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

    This class stores the raw/partially-processed dataframe in ``self.df``
    and provides methods to apply the full sequence of wall corrections in
    the recommended order:

    1.  :meth:`fit_cd_polar`                       – fit C_D = C_D0 + k C_L²
    2.  :meth:`compute_solid_blockage_e`           – compute ε_sb
    3.  :meth:`compute_wake_blockage_e`            – compute ε_wb
    4.  :meth:`apply_blockage_correction`          – apply combined blockage
    5.  :meth:`apply_streamline_curvature_correction`
    6.  :meth:`apply_downwash_correction`
    7.  :meth:`apply_tail_correction`

    Parameters
    ----------
    df : pd.DataFrame
        Raw measurement dataframe.
    clip_negative_cdsep : bool
        If ``True`` (default), clip the separated drag *C_{D_{sep}}* to zero
        when computing the wake-blockage factor.  Negative values are
        physically unrealistic and arise from measurement scatter.
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

    @staticmethod
    def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
        """
        Fit the linear model ``y = intercept + slope * x`` using ordinary
        least squares.

        Parameters
        ----------
        x : np.ndarray
            Independent variable (e.g. C_L²).
        y : np.ndarray
            Dependent variable (e.g. C_D).

        Returns
        -------
        intercept : float
            Fitted intercept (C_D0 in a drag-polar context).
        slope : float
            Fitted slope (*k* in a drag-polar context).
        y_hat : np.ndarray
            Predicted values at each *x*.
        r2 : float
            Coefficient of determination R².  Returns ``nan`` if total sum of
            squares is zero.
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

        Delegates to :meth:`BaseCorrector._apply_streamline_curvature_common`.
        See that method for the full description of the correction formulae.

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with ``grid_df`` and ``cl_a_df``
            populated.
        tau : float, optional
            Streamline-curvature parameter τ₂.  Defaults to :attr:`TAU2_WING`.
        delta : float, optional
            Solid-angle factor δ.  Defaults to :attr:`DELTA_WING`.
        geom_factor : float, optional
            Geometric blockage ratio S/C.  Defaults to :attr:`GEOM_FACTOR`.
        cl_source_col : str
            Input lift-coefficient column (post blockage correction).
        cm_source_col : str
            Input pitching-moment column (post blockage correction).
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            Updated ``self.df``.
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

        Delegates to :meth:`BaseCorrector._apply_downwash_correction_common`.
        See that method for the full description of the correction formulae.

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with ``grid_df`` populated.
        delta : float, optional
            Solid-angle factor δ.  Defaults to :attr:`DELTA_WING`.
        geom_factor : float, optional
            Geometric blockage ratio S/C.  Defaults to :attr:`GEOM_FACTOR`.
        aoa_source_col : str, optional
            AoA input column.  Defaults to ``"AoA_streamline_curvature_corr"``
            so that the AoA corrected for streamline curvature is used as input.
        cd_source_col : str
            Drag-coefficient input column.
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            Updated ``self.df``.
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

        Delegates to :meth:`BaseCorrector._apply_tail_correction_common`.
        See that method for the full description of the correction formulae.

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with ``grid_df`` populated.
        delta : float, optional
            Tail solid-angle factor δ_t.  Defaults to :attr:`DELTA_TAIL`.
        geom_factor : float, optional
            Geometric blockage ratio S/C.  Defaults to :attr:`GEOM_FACTOR`.
        tau2_lt : float, optional
            Tail streamline-curvature parameter τ₂.  Defaults to
            :attr:`TAU2_TAIL`.
        dcmpitch_dalpha : float, optional
            Tail pitching-moment slope ∂Cm/∂α.  Defaults to
            :attr:`DCMPITCH_DALPHA`.
        dcmpitch_dalpha_unit : {"per_rad", "per_deg"}
            Unit of *dcmpitch_dalpha*.
        aoa_source_col : str, optional
            AoA input column.  Falls back to ``"AoA"`` then ``"AoA_round"``.
        cmpitch_source_col : str
            Pitching-moment input column.
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            Updated ``self.df``.
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

        For each unique combination of ``(V_round, AoS_round, dE, dR)``,
        fits:

        .. math::
            C_D = C_{D_0} + k \\, C_L^{2}

        using ordinary least squares on *C_L²* vs *C_D*.  Groups with fewer
        than *min_aoa_points* unique AoA values are skipped.

        The fitted coefficients are stored both back in ``self.df`` (per row)
        and in a compact summary table ``self.fit_df``
        (one row per condition).

        Parameters
        ----------
        save_csv : bool
            If ``True``, write the per-row result to *filename* and,
            optionally, the summary fit table to *fit_params_filename*.
        filename : str, optional
            Override output file name for the per-row dataframe.
        fit_params_filename : str, optional
            File name for the summary fit table.  If ``None``, the summary is
            not written.
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
            ``self.df`` with new columns:

            * ``fit_used``    – boolean flag
            * ``CD0_fit``     – fitted zero-lift drag coefficient C_D0
            * ``k_fit``       – fitted induced-drag factor *k*
            * ``R2_fit``      – coefficient of determination R²
            * ``CD_fit_pred`` – predicted C_D from fit
            * ``CDi_fit``     – induced drag component k·C_L²
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
        Compute and store the solid-blockage factor ε_sb for the prop-off
        configuration.

        Delegates to
        :meth:`BaseCorrector._compute_solid_blockage_e_common` using the
        class constant :attr:`E_SOLID_BLOCKAGE` = 0.007229438.

        Parameters
        ----------
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.
        use_constant_e : bool
            Use the class constant (default ``True``).
        use_e_column : bool
            Use a per-row value from *e_column*.
        e_column : str
            Source column when *use_e_column* is ``True``.
        output_col : str
            Destination column name for ε_sb.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with *output_col* added.
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
        Compute and store the wake-blockage factor ε_wb for the prop-off
        configuration.

        Delegates to
        :meth:`BaseCorrector._compute_wake_blockage_e_common`.
        See that method for the full formula description.

        Parameters
        ----------
        cd0_col : str
            Zero-lift drag coefficient column (from :meth:`fit_cd_polar`).
        cd_col : str
            Measured drag coefficient column.
        cl_col : str
            Measured lift coefficient column.
        k_col : str
            Induced-drag factor column (from :meth:`fit_cd_polar`).
        output_col : str
            Destination column name for ε_wb.
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with *output_col* added.
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
        dataset.

        Delegates to
        :meth:`BaseCorrector._apply_combined_blockage_from_e_columns`.
        Slipstream blockage (ε_ss) is not applicable for prop-off data and is
        therefore excluded.

        Correction formulae:

        .. math::
            V_{corr} = \\frac{V}{1 + \\varepsilon_{sb} + \\varepsilon_{wb}}

        .. math::
            C_{corr} = \\frac{C}{(1 + \\varepsilon_{sb} + \\varepsilon_{wb})^{2}}

        Parameters
        ----------
        apply_esb : bool
            Include solid-blockage correction.
        apply_ewb : bool
            Include wake-blockage correction.
        esb_col : str
            Column name for ε_sb.
        ewb_col : str
            Column name for ε_wb.
        velocity_cols : sequence of str
            Velocity columns to correct.
        coefficient_cols : sequence of str
            Aerodynamic coefficient columns to correct.
        suffix : str
            Suffix appended to all output column names.
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with corrected columns and ``e_total_blockage`` added.
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

    * BEM-based thrust separation from the measured body-frame axial force
      (:meth:`compute_thrust_separation_BEM`).
    * Slipstream blockage factor computation
      (:meth:`compute_slipstream_blockage_e`).
    * Attachment of prop-off drag-polar fit parameters to the prop-on rows
      (:meth:`attach_fits`), needed for the wake-blockage computation.
    * Model-off tare correction (:meth:`apply_modeloff_correction`).

    Parameters
    ----------
    df : pd.DataFrame
        Raw measurement dataframe.
    clip_negative_cdsep : bool
        Clip *C_{D_{sep}}* to zero when computing wake blockage (default
        ``True``).
    velocity_tolerance : float
        Maximum allowable velocity difference [m/s] when matching prop-off
        fit entries to prop-on conditions in :meth:`attach_fits`
        (default 1.0 m/s).
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

        Delegates to :meth:`BaseCorrector._apply_streamline_curvature_common`.
        The default input lift column is ``"CL_aero_BEM_blockage_corr"``
        (the aerodynamic lift after BEM thrust separation and blockage
        correction) rather than the raw ``"CL_blockage_corr"`` used for
        prop-off data.

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with ``grid_df`` and ``cl_a_df``
            populated.
        tau : float, optional
            Streamline-curvature parameter τ₂.  Defaults to :attr:`TAU2_WING`.
        delta : float, optional
            Solid-angle factor δ.  Defaults to :attr:`DELTA_WING`.
        geom_factor : float, optional
            Geometric blockage ratio S/C.  Defaults to :attr:`GEOM_FACTOR`.
        cl_source_col : str
            Input lift-coefficient column.
        cm_source_col : str
            Input pitching-moment column.
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            Updated ``self.df``.
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

        Delegates to :meth:`BaseCorrector._apply_downwash_correction_common`.

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with ``grid_df`` populated.
        delta : float, optional
            Solid-angle factor δ.  Defaults to :attr:`DELTA_WING`.
        geom_factor : float, optional
            Geometric blockage ratio S/C.  Defaults to :attr:`GEOM_FACTOR`.
        aoa_source_col : str, optional
            AoA input column.  Defaults to
            ``"AoA_streamline_curvature_corr"``.
        cd_source_col : str
            Drag-coefficient input column.
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            Updated ``self.df``.
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

        Delegates to :meth:`BaseCorrector._apply_tail_correction_common`.

        Parameters
        ----------
        tailoff : TailOffData
            Reference tail-off dataset with ``grid_df`` populated.
        delta : float, optional
            Tail solid-angle factor δ_t.  Defaults to :attr:`DELTA_TAIL`.
        geom_factor : float, optional
            Geometric blockage ratio S/C.  Defaults to :attr:`GEOM_FACTOR`.
        tau2_lt : float, optional
            Tail streamline-curvature parameter τ₂.  Defaults to
            :attr:`TAU2_TAIL`.
        dcmpitch_dalpha : float, optional
            Tail pitching-moment slope ∂Cm/∂α.  Defaults to
            :attr:`DCMPITCH_DALPHA`.
        dcmpitch_dalpha_unit : {"per_rad", "per_deg"}
            Unit of *dcmpitch_dalpha*.
        aoa_source_col : str, optional
            AoA input column.
        cmpitch_source_col : str
            Pitching-moment input column.
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            Updated ``self.df``.
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
        Attach prop-off drag-polar fit values (C_D0, *k*) to each row of the
        prop-on dataframe.

        Because the prop-on runs may not sweep the full AoA range needed to
        fit a drag polar independently, the fit coefficients from the matching
        prop-off condition are used instead.  Matching is performed on
        ``(AoS_round, dE, dR)`` with closest-velocity selection within the
        tolerance :attr:`velocity_tolerance`.

        Fallback order
        --------------
        For each prop-on row, the following match strategies are attempted in
        order, and the first successful match is used:

        1. exact AoS, exact dR
        2. exact AoS, flipped dR  (sign reversal)
        3. exact AoS, dR = 0
        4. flipped AoS, exact dR
        5. flipped AoS, flipped dR
        6. flipped AoS, dR = 0

        Diagnostic columns added to ``self.df``:

        * ``matched_fit_velocity``   – velocity of the matched fit entry
        * ``velocity_match_error``   – |V_data − V_fit|
        * ``fit_found``              – boolean success flag
        * ``fit_match_type``         – label describing the match strategy used
        * ``matched_fit_AoS``        – AoS of the matched fit entry
        * ``matched_fit_dR``         – dR of the matched fit entry
        * ``matched_fit_source_row`` – index of the matched row in fit_df

        Parameters
        ----------
        input_fit_df : pd.DataFrame, optional
            Prop-off fit-summary table (output of
            :meth:`PropOffData.fit_cd_polar`).  If ``None``, uses
            ``self.fit_df`` or loads from *fit_csv*.
        fit_csv : str or Path, optional
            Path to a CSV file containing the fit summary table.
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.
        vel_col_data : str
            Velocity column in ``self.df`` for matching.
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
            ``self.df`` with *fit_value_cols* and diagnostic columns added.

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
        tunnel_area: Optional[float] = None,
        output_col: str = "ess",
        save_csv: bool = False,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute the slipstream-blockage factor ε_ss for powered conditions.

        The propeller slipstream accelerates the flow behind the disc,
        effectively reducing the freestream cross-section available to the
        rest of the model.  The blockage factor is derived from the thrust
        loading coefficient *T_c** (Tc_star_BEM).

        Formula
        -------
        .. math::
            \\varepsilon_{ss} = -\\frac{T_c^{*}}{2\\sqrt{1 + 2\\,T_c^{*}}}
            \\cdot \\frac{S_{prop}}{C}

        where *S_prop* is the single propeller disc area and *C* is the
        tunnel cross-sectional area (:attr:`TUNNEL_AREA`).

        Note
        ----
        ε_ss is negative by construction (the slipstream increases the
        effective dynamic pressure, thus *reducing* the blockage
        correction relative to solid/wake blockage).

        Parameters
        ----------
        tc_col : str
            Column name of the thrust loading coefficient T_c*.
        D_prop : float, optional
            Propeller diameter [m].  Defaults to :attr:`PROP_DIAMETER`.
        S_prop : float, optional
            Single propeller disc area [m²].  Computed from *D_prop* if not
            supplied.
        tunnel_area : float, optional
            Tunnel cross-sectional area [m²].  Defaults to
            :attr:`TUNNEL_AREA`.
        output_col : str
            Destination column name for ε_ss.
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with *output_col* added.

        Raises
        ------
        ValueError
            If any row has ``(1 + 2 T_c*) < 0``, making the formula undefined.
        """
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
        """
        Compute and store the solid-blockage factor ε_sb for the prop-on
        configuration.

        Delegates to
        :meth:`BaseCorrector._compute_solid_blockage_e_common` using
        :attr:`E_SOLID_BLOCKAGE` = 0.007229438.

        Parameters
        ----------
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.
        use_constant_e : bool
            Use the class constant (default ``True``).
        use_e_column : bool
            Use a per-row value from *e_column*.
        e_column : str
            Source column when *use_e_column* is ``True``.
        output_col : str
            Destination column name for ε_sb.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with *output_col* added.
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
        Compute and store the wake-blockage factor ε_wb for the prop-on
        configuration.

        Delegates to
        :meth:`BaseCorrector._compute_wake_blockage_e_common`.
        The drag-polar fit values are typically attached first via
        :meth:`attach_fits`.

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
            Destination column name for ε_wb.
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with *output_col* added.
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
        Apply the combined solid + wake + slipstream blockage correction to
        the prop-on dataset.

        Delegates to
        :meth:`BaseCorrector._apply_combined_blockage_from_e_columns`.

        The total blockage factor includes the slipstream term:

        .. math::
            \\varepsilon_{total} = \\varepsilon_{sb} + \\varepsilon_{wb}
            + \\varepsilon_{ss}

        Velocity and coefficient corrections:

        .. math::
            V_{corr} = \\frac{V}{1 + \\varepsilon_{total}}, \\qquad
            C_{corr} = \\frac{C}{(1 + \\varepsilon_{total})^{2}}

        The thrust coefficient ``CFt_thrust_BEM`` receives only solid + wake
        blockage (never slipstream) to avoid circular correction.

        Parameters
        ----------
        apply_esb : bool
            Include solid-blockage correction.
        apply_ewb : bool
            Include wake-blockage correction.
        apply_ess : bool
            Include slipstream-blockage correction (default ``True``).
        esb_col : str
            Column name for ε_sb.
        ewb_col : str
            Column name for ε_wb.
        ess_col : str
            Column name for ε_ss.
        velocity_cols : sequence of str
            Velocity columns to correct.
        coefficient_cols : sequence of str
            Aerodynamic coefficient columns to correct.
        cft_thrust_col : str
            Thrust force coefficient column receiving solid + wake only.
        suffix : str
            Suffix appended to all output column names.
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with corrected columns and ``e_total_blockage`` added.
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

        Delegates to :meth:`ModelOffCorrector.apply`.  The correction
        subtracts the support-structure aerodynamic tare from each measured
        coefficient, matched by ``(AoA_round, AoS_round)``:

        .. math::
            C_{\\text{corrected}} = C_{\\text{measured}} - C_{\\text{tare}}

        Parameters
        ----------
        modeloff_corrector : ModelOffCorrector
            Pre-configured model-off corrector instance.
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.
        source_columns : dict[str, str], optional
            Mapping from canonical coefficient names to actual column names
            in ``self.df``.
        cmpitch25c_column : str, optional
            Column name of the pitching moment at the 25 % chord.

        Returns
        -------
        pd.DataFrame
            Updated ``self.df``.
        """
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
    """
    Container and correction pipeline for tail-off (no horizontal tail)
    wind-tunnel measurements.

    The tail-off dataset serves a dual role:

    1. **Self-correction**: applies a solid-blockage correction to the
       tail-off coefficients (no wake or slipstream blockage).
    2. **Reference for other datasets**: :class:`PropOffData` and
       :class:`PropOnData` look up *CLw_tailoff* (the tail-off wing lift
       coefficient at matching conditions) from ``self.grid_df`` to drive
       the streamline-curvature, downwash, and tail corrections.

    After calling :meth:`apply_solid_blockage`, call
    :meth:`build_alpha_slice_grid_by_velocity` to construct the reference
    grid ``self.grid_df``, then optionally
    :meth:`compute_cl_alpha_slope_by_case` to populate ``self.cl_a_df``
    for the streamline-curvature correction.

    Class-level override
    --------------------
    ``E_SOLID_BLOCKAGE`` is overridden to
    :attr:`BaseCorrector.E_SOLID_BLOCKAGE_TAILOFF` = 0.006406642,
    reflecting the smaller model volume with the tail removed.

    Parameters
    ----------
    df : pd.DataFrame
        Raw tail-off measurement dataframe.
    clip_negative_cdsep : bool
        (Unused in tail-off pipeline; retained for API consistency.)
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

        Uses the tail-off solid-blockage constant
        ε_sb = :attr:`E_SOLID_BLOCKAGE` = 0.006406642, which is smaller than
        the full-configuration value because the horizontal tail is absent.

        The correction is applied with the suffix ``"solid_blockage_corr"``
        (rather than the generic ``"blockage_corr"``) so that downstream code
        can unambiguously identify the corrected tail-off lift column
        ``CL_solid_blockage_corr`` during the CLw lookup.

        Correction formulae (solid blockage only):

        .. math::
            V_{corr} = \\frac{V}{1 + \\varepsilon_{sb}}

        .. math::
            C_{corr} = \\frac{C}{(1 + \\varepsilon_{sb})^{2}}

        Parameters
        ----------
        save_csv : bool
            Write output CSV if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            ``self.df`` with ``V_solid_blockage_corr``,
            ``CL_solid_blockage_corr``, ``CD_solid_blockage_corr``,
            and other ``*_solid_blockage_corr`` columns added.
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
        Build a complete ``(AoA_round, AoS_round)`` grid of tail-off
        coefficients for each ``V_round``.

        The tail-off test matrix typically sweeps AoS at a few fixed AoA
        anchor values.  This method constructs a dense grid over all
        combinations of AoA and AoS by:

        1. **Measured**: use the value directly where available.
        2. **Interpolation in AoA**: linearly interpolate between anchor
           slices when the target AoA lies within the measured range at that
           AoS.
        3. **Extrapolation in AoA**: linearly extrapolate beyond the range
           (using the two nearest anchor points) when the target AoA is
           outside the measured range.
        4. **Unresolved**: leave as NaN if fewer than two anchor slices are
           available at the requested AoS.

        The source type of each grid cell is recorded in the column
        ``source_type`` with values ``"measured"``, ``"interp_alpha"``,
        ``"extrap_alpha"``, or ``"unresolved"``.

        The resulting ``grid_df`` is stored in ``self.grid_df`` and is used
        as the CLw lookup table by the blockage, streamline-curvature,
        downwash, and tail correction methods of
        :class:`PropOffData` and :class:`PropOnData`.

        Parameters
        ----------
        coeff_cols : list of str, optional
            Coefficient columns to include in the grid.  Defaults to
            ``CL_solid_blockage_corr``, ``CD_solid_blockage_corr``, and
            their raw equivalents if present.
        anchor_aoa_vals : tuple of float
            AoA values (degrees) at which AoS sweeps are available.  Used as
            interpolation anchors.
        extra_aoa_vals : iterable of float, optional
            Additional AoA values to include in the grid even if not measured.
        extra_aos_vals : iterable of float, optional
            Additional AoS values to include in the grid even if not measured.
        save_csv : bool
            Write ``self.grid_df`` to disk if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            The grid stored in ``self.grid_df``, with columns:
            ``V_round``, ``AoA_round``, ``AoS_round``,
            ``case_available``, ``source_type``, and all *coeff_cols*.

        Raises
        ------
        ValueError
            If none of the *anchor_aoa_vals* are present at a given velocity,
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
            Interpolate or extrapolate *y* at *x_target* given paired arrays
            (*x_known*, *y_known*).  Returns the interpolated/extrapolated value
            and a source label ``"interp_alpha"`` or ``"extrap_alpha"``.
            Returns ``(nan, "unresolved")`` if fewer than two valid points exist.
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
        Compute the CL-alpha slope for each ``(V_round, AoS_round)`` case
        using a linear regression over the specified AoA range.

        For each test condition, fits:

        .. math::
            C_L = C_{L_0} + \\left(\\frac{\\partial C_L}{\\partial \\alpha}
            \\right) \\alpha

        where the slope is in units of per degree.

        The result is stored in ``self.cl_a_df`` and is subsequently used by
        the streamline-curvature correction method to scale the AoA and CL
        corrections.

        Parameters
        ----------
        aoa_min : float
            Minimum AoA [degrees] to include in the linear fit.
        aoa_max : float
            Maximum AoA [degrees] to include in the linear fit.
        cl_col : str
            Lift-coefficient column in ``self.grid_df``.
        aoa_col : str
            AoA column name.
        aos_col : str
            AoS column name.
        v_col : str
            Velocity column name.
        min_points : int
            Minimum number of distinct AoA points required to attempt a fit.
            Cases with fewer points receive ``NaN`` and ``fit_success=False``.
        save_csv : bool
            Write ``self.cl_a_df`` to disk if ``True``.
        filename : str, optional
            Override output file name.

        Returns
        -------
        pd.DataFrame
            ``self.cl_a_df`` with one row per ``(V_round, AoS_round)``
            case, containing:

            * ``cl_alpha_slope_per_deg`` – ∂CL/∂α [deg⁻¹]
            * ``cl_at_aoa0``            – intercept C_L0
            * ``r2``                    – coefficient of determination R²
            * ``n_points``              – number of data points used
            * ``n_unique_aoa``          – number of unique AoA values
            * ``aoa_min_fit``           – the *aoa_min* argument
            * ``aoa_max_fit``           – the *aoa_max* argument
            * ``fit_success``           – boolean flag

        Raises
        ------
        ValueError
            If ``self.grid_df`` has not been built yet.
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