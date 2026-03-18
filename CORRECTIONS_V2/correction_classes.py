from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import pandas as pd


class BaseCorrector:
    """
    Small utility base class.

    Provides:
    - configurable save directory
    - dataframe saving helper
    - required-column validation
    - round-to-half utility
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

    def _reorder_solid_blockage_columns(self) -> None:
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

    def apply_solid_blockage(
        self,
        save_csv: bool = False,
        filename: Optional[str] = None,
        use_constant_e: bool = True,
        use_e_column: bool = False,
        e_column: str = "e",
    ) -> pd.DataFrame:
        """
        Apply solid blockage correction:
            V_corr = V / (1 + e)
            coeff_corr = coeff / (1 + e)^2
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
            save_name = filename or "propOff_solid_blockage_corrected.csv"
            self.save_df(self.df, save_name)

        return self.df

    def fit_cd_polar(
        self,
        save_csv: bool = False,
        filename: Optional[str] = None,
        fit_params_filename: Optional[str] = None,
        v_col: str = "V_solid_blockage_corr",
        cl_col: str = "CL_solid_blockage_corr",
        cd_col: str = "CD_solid_blockage_corr",
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

    def apply_wake_blockage(
        self,
        save_csv: bool = False,
        filename: Optional[str] = None,
        v_col: str = "V_solid_blockage_corr",
        cl_col: str = "CL_solid_blockage_corr",
        cd_col: str = "CD_solid_blockage_corr",
    ) -> pd.DataFrame:
        """
        Apply wake blockage correction using the CD fit already stored in self.df.
        Requires fit_cd_polar() to have been run first.
        """
        df = self.df.copy()

        required = [v_col, cl_col, cd_col, "CD0_fit", "k_fit"]
        self.require_columns(df, required, context="PropOff apply_wake_blockage")

        df["CL2_fit"] = df[cl_col] ** 2
        df["CDi_fit"] = df["k_fit"] * df["CL2_fit"]
        df["CDsep_fit"] = df[cd_col] - df["CD0_fit"] - df["CDi_fit"]

        if self.clip_negative_cdsep:
            df["CDsep_fit"] = df["CDsep_fit"].clip(lower=0)

        df["ewb_t_fit"] = (
            (self.s_ref / (4 * self.test_section_area)) * df["CD0_fit"]
            + (5 * self.s_ref / (4 * self.test_section_area)) * df["CDsep_fit"]
        )

        df["q_ratio_wake"] = (
            1
            + (self.s_ref / (2 * self.test_section_area)) * df["CD0_fit"]
            + (5 * self.s_ref / (2 * self.test_section_area)) * df["CDsep_fit"]
        )

        df["V_wake_corr"] = df[v_col] * np.sqrt(df["q_ratio_wake"])

        coeff_cols = [
            "CL_solid_blockage_corr",
            "CD_solid_blockage_corr",
            "CY_solid_blockage_corr",
            "CYaw_solid_blockage_corr",
            "CMroll_solid_blockage_corr",
            "CMpitch_solid_blockage_corr",
            "CMyaw_solid_blockage_corr",
        ]

        for col in coeff_cols:
            if col in df.columns:
                df[f"{col}_wake_corr"] = df[col] / df["q_ratio_wake"]

        self.df = df

        drop_cols = [
            "CL2",
            "fit_used",
            "CD0_fit",
            "k_fit",
            "R2_fit",
            "CD_fit_pred",
            "CDi_fit",
            "CL2_fit",
            "CDsep_fit"
        ]

        new_df = df.drop(columns=drop_cols, errors="ignore")

        if save_csv:
            save_name = filename or "propOff_wake_blockage_corrected.csv"
            self.save_df(new_df, save_name)

        return self.df


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
        
    def _reorder_solid_blockage_columns(self) -> None:
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

    def apply_solid_blockage(
        self,
        save_csv: bool = False,
        filename: Optional[str] = None,
        use_constant_e: bool = True,
        use_e_column: bool = False,
        e_column: str = "e",
    ) -> pd.DataFrame:
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
            save_name = filename or "propOn_solid_blockage_corrected.csv"
            self.save_df(self.df, save_name)

        return self.df

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
        if fit_csv is not None:
            self.load_fit_table(fit_csv)

        if self.fit_df is None:
            raise ValueError("No fit table loaded. Provide fit_csv or call load_fit_table() first.")

        df = self.df.copy()
        fit_df = self.fit_df.copy()

        # build rounded columns if needed
        if "V_round" not in df.columns:
            if "V_solid_blockage_corr" not in df.columns:
                raise ValueError("Need either 'V_round' or 'V_solid_blockage_corr' in dataframe.")
            df["V_round"] = self.round_to_half(df["V_solid_blockage_corr"])

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

    def apply_wake_blockage(
        self,
        fit_csv: Optional[str | Path] = None,
        save_csv: bool = False,
        filename: Optional[str] = None,
        v_col: str = "V_solid_blockage_corr",
        cl_col: str = "CL_solid_blockage_corr",
        cd_col: str = "CD_solid_blockage_corr",
    ) -> pd.DataFrame:
        """
        Apply wake blockage correction using prop-off fit values.
        If CD0_fit and k_fit are not yet attached, provide fit_csv or call attach_fits() first.
        """
        df = self.df.copy()

        if ("CD0_fit" not in df.columns or "k_fit" not in df.columns) and fit_csv is not None:
            self.attach_fits(fit_csv=fit_csv)

        df = self.df.copy()

        required = [v_col, cl_col, cd_col, "CD0_fit", "k_fit", "fit_found"]
        self.require_columns(df, required, context="PropOn apply_wake_blockage")

        df["CL2_fit"] = df[cl_col] ** 2
        df["CDi_fit"] = df["k_fit"] * df["CL2_fit"]
        df["CDsep_fit"] = df[cd_col] - df["CD0_fit"] - df["CDi_fit"]

        if self.clip_negative_cdsep:
            df["CDsep_fit"] = df["CDsep_fit"].clip(lower=0)

        df["ewb_t_fit"] = (
            (self.s_ref / (4 * self.test_section_area)) * df["CD0_fit"]
            + (5 * self.s_ref / (4 * self.test_section_area)) * df["CDsep_fit"]
        )

        df["q_ratio_wake"] = (
            1
            + (self.s_ref / (2 * self.test_section_area)) * df["CD0_fit"]
            + (5 * self.s_ref / (2 * self.test_section_area)) * df["CDsep_fit"]
        )

        df.loc[~df["fit_found"], ["ewb_t_fit", "q_ratio_wake"]] = np.nan

        df["V_wake_corr"] = np.nan
        found_mask = df["fit_found"]
        df.loc[found_mask, "V_wake_corr"] = (
            df.loc[found_mask, v_col] * np.sqrt(df.loc[found_mask, "q_ratio_wake"])
        )

        coeff_cols = [
            "CL_solid_blockage_corr",
            "CD_solid_blockage_corr",
            "CY_solid_blockage_corr",
            "CYaw_solid_blockage_corr",
            "CMroll_solid_blockage_corr",
            "CMpitch_solid_blockage_corr",
            "CMyaw_solid_blockage_corr",
        ]

        for col in coeff_cols:
            if col in df.columns:
                out_col = f"{col}_wake_corr"
                df[out_col] = np.nan
                df.loc[found_mask, out_col] = df.loc[found_mask, col] / df.loc[found_mask, "q_ratio_wake"]

        self.df = df

        if save_csv:
            save_name = filename or "propOn_wake_blockage_corrected.csv"
            self.save_df(self.df, save_name)

        return self.df