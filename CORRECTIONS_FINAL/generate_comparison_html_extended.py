"""
generate_explorer_html.py
--------------------------
Reads propOn_final.csv and propOff_final.csv, merges them on the rounded
index columns, and writes a fully self-contained interactive HTML explorer.

The HTML has three data pages and a Help page:
    1. PropOn vs PropOff  — comparison of FINAL coefficients and corrections,
                            coloured by J value, with prop-off as dashed reference.
    2. PropOff Explorer   — full propOff dataset, colour-by selector, polar
                            and correction metrics, actual/rounded x-axis toggle.
    3. Stability & Control — Cnβ, Cn_dR, Cy_dR, Cr_dR derivatives fitted by
                            linear regression, drag polar, and trim analysis.
    4. Help               — full usage guide and metric glossary.

Direct import usage
-------------------
    from generate_explorer_html import load_and_build, generate_html
    from pathlib import Path

    (cmp_rows, off_rows, meta, j_colors, v_colors,
     cn_beta_on, cn_beta_off, cn_dr_on, cn_dr_off,
     polar_on, polar_off, trim_rows, sc_meta) = load_and_build(
        propon_path  = Path("results_propOn_FINAL/propOn_final.csv"),
        propoff_path = Path("results_propOff_FINAL/propOff_final.csv"),
    )
    generate_html(
        cmp_rows, off_rows, meta, j_colors, v_colors,
        cn_beta_on, cn_beta_off, cn_dr_on, cn_dr_off,
        polar_on, polar_off, trim_rows, sc_meta,
        out_path = Path("results_propOn_FINAL/explorer.html"),
    )

Usage
-----
    python generate_explorer_html.py

    # Custom paths:
    python generate_explorer_html.py \\
        --propon  path/to/propOn_final.csv \\
        --propoff path/to/propOff_final.csv \\
        --out     path/to/explorer.html

Required columns
----------------
Both CSVs must have:
    AoA_round, AoS_round, V_round, dR, dE
    AoA, AoS, V_FINAL
    CL_FINAL, CD_FINAL, CMpitch_FINAL, CYaw_FINAL, CMroll_FINAL, CMyaw_FINAL
    delta_CL_sc, delta_CD_dw, delta_CMpitch_sc, delta_CMpitch_tail
    delta_alpha_dw_deg, delta_alpha_tail_deg
    e_total_blockage, esb, ewb, CLw_tailoff

propOn additionally needs:
    J_round, J
    CFt_FINAL, CT_props_total
    ess, Tc_star_BEM

propOff additionally needs:
    AoA_FINAL, delta_alpha_sc_deg
    CD0_fit, k_fit, R2_fit
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INDEX_COLS    = ["AoA_round", "AoS_round", "V_round", "dR", "dE"]
FINAL_COLS    = ["CL_FINAL", "CD_FINAL", "CMpitch_FINAL",
                 "CYaw_FINAL", "CMroll_FINAL", "CMyaw_FINAL"]
CORR_COLS_ON  = ["delta_CL_sc", "delta_CD_dw", "delta_CMpitch_sc", "delta_CMpitch_tail",
                 "delta_alpha_dw_deg", "delta_alpha_tail_deg",
                 "e_total_blockage", "esb", "ewb", "CLw_tailoff"]
CORR_COLS_OFF = ["delta_CL_sc", "delta_CD_dw", "delta_CMpitch_sc", "delta_CMpitch_tail",
                 "delta_alpha_dw_deg", "delta_alpha_tail_deg",
                 "e_total_blockage", "esb", "ewb", "CLw_tailoff",
                 "CD0_fit", "k_fit", "R2_fit"]
ACTUAL_ON     = ["AoA", "AoS", "V_FINAL", "J"]
ACTUAL_OFF    = ["AoA", "AoS", "V_FINAL"]

J_PALETTE = ["#5b8af5", "#3ecf8e", "#f5a623", "#f05252",
             "#a78bfa", "#fb7185", "#34d399", "#fbbf24"]
V_PALETTE = ["#5b8af5", "#3ecf8e", "#f5a623", "#f05252",
             "#a78bfa", "#fb7185", "#34d399", "#fbbf24"]

DEFAULT_PROPON  = "propOn_final.csv"
DEFAULT_PROPOFF = "propOff_final.csv"
DEFAULT_OUT     = "explorer.html"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def flt(v):
    """Convert to float, return None for NaN/missing."""
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 5)
    except (TypeError, ValueError):
        return None


def opts_html(vals, fmt="{}"):
    """Build <option> tags for a list of values."""
    return "\n".join(
        f'<option value="{v}">{fmt.format(v)}</option>' for v in vals
    )


# ---------------------------------------------------------------------------
# Data loading & building
# ---------------------------------------------------------------------------
def load_and_build(propon_path: Path, propoff_path: Path):
    on_df  = pd.read_csv(propon_path)
    off_df = pd.read_csv(propoff_path)

    # ----------------------------------------------------------------
    # Validate required columns
    # ----------------------------------------------------------------
    required_on  = INDEX_COLS + FINAL_COLS + CORR_COLS_ON + ACTUAL_ON + \
                   ["J_round", "CFt_FINAL", "CT_props_total", "ess", "Tc_star_BEM"]
    required_off = INDEX_COLS + FINAL_COLS + CORR_COLS_OFF + ACTUAL_OFF + \
                   ["AoA_FINAL", "delta_alpha_sc_deg"]

    for col in required_on:
        if col not in on_df.columns:
            print(f"  WARNING: propOn CSV missing column '{col}' — will be null in output")
    for col in required_off:
        if col not in off_df.columns:
            print(f"  WARNING: propOff CSV missing column '{col}' — will be null in output")

    # ----------------------------------------------------------------
    # Build comparison rows (propOn merged with aggregated propOff)
    # ----------------------------------------------------------------
    off_agg_cols = INDEX_COLS + FINAL_COLS + CORR_COLS_OFF + ACTUAL_OFF
    off_agg_cols = [c for c in off_agg_cols if c in off_df.columns]
    off_agg = (
        off_df[off_agg_cols]
        .groupby(INDEX_COLS, as_index=False)
        .mean()
    )

    on_sel_cols = (INDEX_COLS
                   + [c for c in ["J_round", "CFt_FINAL", "CT_props_total"] if c in on_df.columns]
                   + [c for c in FINAL_COLS if c in on_df.columns]
                   + [c for c in CORR_COLS_ON if c in on_df.columns]
                   + [c for c in ACTUAL_ON if c in on_df.columns]
                   + [c for c in ["ess", "Tc_star_BEM"] if c in on_df.columns])
    on_sel = on_df[on_sel_cols].copy()

    merged = on_sel.merge(off_agg, on=INDEX_COLS, how="inner", suffixes=("_on", "_off"))

    print(f"  propOn  rows : {len(on_df)}")
    print(f"  propOff rows : {len(off_df)}  "
          f"(aggregated to {len(off_agg)} unique index combos)")
    print(f"  Merged  rows : {len(merged)}")

    def gcol(row, name):
        """Safe column getter — returns None if column absent."""
        return flt(row[name]) if name in row.index else None

    cmp_rows = []
    for _, r in merged.iterrows():
        cmp_rows.append({
            # rounded index — used for filtering
            "AoA_r": flt(r.AoA_round), "AoS_r": flt(r.AoS_round),
            "V_r":   flt(r.V_round),   "dR":    flt(r.dR), "dE": flt(r.dE),
            "J_r":   gcol(r, "J_round"),
            # actual values — used when "actual x" toggle is on
            "AoA_on":  gcol(r, "AoA_on"),  "AoS_on":  gcol(r, "AoS_on"),
            "V_on":    gcol(r, "V_FINAL_on"),
            "J_act":   gcol(r, "J"),
            "AoA_off": gcol(r, "AoA_off"), "AoS_off": gcol(r, "AoS_off"),
            "V_off":   gcol(r, "V_FINAL_off"),
            # FINAL coefficients
            "CL_on":     gcol(r, "CL_FINAL_on"),
            "CL_off":    gcol(r, "CL_FINAL_off"),
            "CD_on":     gcol(r, "CD_FINAL_on"),
            "CD_off":    gcol(r, "CD_FINAL_off"),
            "CM_on":     gcol(r, "CMpitch_FINAL_on"),
            "CM_off":    gcol(r, "CMpitch_FINAL_off"),
            "CYaw_on":   gcol(r, "CYaw_FINAL_on"),
            "CYaw_off":  gcol(r, "CYaw_FINAL_off"),
            "CMroll_on": gcol(r, "CMroll_FINAL_on"),
            "CMroll_off":gcol(r, "CMroll_FINAL_off"),
            "CMyaw_on":  gcol(r, "CMyaw_FINAL_on"),
            "CMyaw_off": gcol(r, "CMyaw_FINAL_off"),
            "CFt":       gcol(r, "CFt_FINAL"),
            "CT_props":  gcol(r, "CT_props_total"),
            # corrections — propOn
            "dCL_sc_on":   gcol(r, "delta_CL_sc_on"),
            "dCD_dw_on":   gcol(r, "delta_CD_dw_on"),
            "dCM_sc_on":   gcol(r, "delta_CMpitch_sc_on"),
            "dCM_tail_on": gcol(r, "delta_CMpitch_tail_on"),
            "dAoA_dw_on":  gcol(r, "delta_alpha_dw_deg_on"),
            "dAoA_tail_on":gcol(r, "delta_alpha_tail_deg_on"),
            "e_tot_on":    gcol(r, "e_total_blockage_on"),
            "esb_on":      gcol(r, "esb_on"),
            "ewb_on":      gcol(r, "ewb_on"),
            "ess":         gcol(r, "ess"),
            "Tc":          gcol(r, "Tc_star_BEM"),
            "CLw_on":      gcol(r, "CLw_tailoff_on"),
            # corrections — propOff
            "dCL_sc_off":   gcol(r, "delta_CL_sc_off"),
            "dCD_dw_off":   gcol(r, "delta_CD_dw_off"),
            "dCM_sc_off":   gcol(r, "delta_CMpitch_sc_off"),
            "dCM_tail_off": gcol(r, "delta_CMpitch_tail_off"),
            "dAoA_dw_off":  gcol(r, "delta_alpha_dw_deg_off"),
            "dAoA_tail_off":gcol(r, "delta_alpha_tail_deg_off"),
            "e_tot_off":    gcol(r, "e_total_blockage_off"),
            "esb_off":      gcol(r, "esb_off"),
            "ewb_off":      gcol(r, "ewb_off"),
            "CLw_off":      gcol(r, "CLw_tailoff_off"),
            "CD0":          gcol(r, "CD0_fit"),
            "k":            gcol(r, "k_fit"),
            "R2":           gcol(r, "R2_fit"),
        })

    # ----------------------------------------------------------------
    # Build propOff-only rows (all individual rows, no aggregation)
    # ----------------------------------------------------------------
    off_rows = []
    for _, r in off_df.iterrows():
        off_rows.append({
            # rounded index
            "AoA_r": flt(r.AoA_round), "AoS_r": flt(r.AoS_round),
            "V_r":   flt(r.V_round),   "dR":    flt(r.dR), "dE": flt(r.dE),
            # actual values
            "AoA_act": gcol(r, "AoA"),
            "AoS_act": gcol(r, "AoS"),
            "V_act":   gcol(r, "V_FINAL"),
            "AoA_fin": gcol(r, "AoA_FINAL"),
            # FINAL coefficients
            "CL":    gcol(r, "CL_FINAL"),
            "CD":    gcol(r, "CD_FINAL"),
            "CM":    gcol(r, "CMpitch_FINAL"),
            "CYaw":  gcol(r, "CYaw_FINAL"),
            "CMroll":gcol(r, "CMroll_FINAL"),
            "CMyaw": gcol(r, "CMyaw_FINAL"),
            # corrections
            "dCL_sc":   gcol(r, "delta_CL_sc"),
            "dCD_dw":   gcol(r, "delta_CD_dw"),
            "dCM_sc":   gcol(r, "delta_CMpitch_sc"),
            "dCM_tail": gcol(r, "delta_CMpitch_tail"),
            "dAoA_dw":  gcol(r, "delta_alpha_dw_deg"),
            "dAoA_tail":gcol(r, "delta_alpha_tail_deg"),
            "dAoA_sc":  gcol(r, "delta_alpha_sc_deg"),
            "e_tot":    gcol(r, "e_total_blockage"),
            "esb":      gcol(r, "esb"),
            "ewb":      gcol(r, "ewb"),
            "CLw":      gcol(r, "CLw_tailoff"),
            "CD0":      gcol(r, "CD0_fit"),
            "k":        gcol(r, "k_fit"),
            "R2":       gcol(r, "R2_fit"),
        })

    # ----------------------------------------------------------------
    # Build metadata for dropdown options
    # ----------------------------------------------------------------
    meta = {
        "cmp": {
            "J":   sorted([float(x) for x in on_df.J_round.dropna().unique()]),
            "V":   sorted([float(x) for x in on_df.V_round.dropna().unique()]),
            "AoA": sorted([float(x) for x in on_df.AoA_round.dropna().unique()]),
            "AoS": sorted([float(x) for x in on_df.AoS_round.dropna().unique()]),
            "dR":  sorted([float(x) for x in on_df.dR.dropna().unique()]),
            "dE":  sorted([float(x) for x in on_df.dE.dropna().unique()]),
        },
        "off": {
            "V":   sorted([float(x) for x in off_df.V_round.dropna().unique()]),
            "AoA": sorted([float(x) for x in off_df.AoA_round.dropna().unique()]),
            "AoS": sorted([float(x) for x in off_df.AoS_round.dropna().unique()]),
            "dR":  sorted([float(x) for x in off_df.dR.dropna().unique()]),
            "dE":  sorted([float(x) for x in off_df.dE.dropna().unique()]),
        },
    }

    j_colors = {str(j): J_PALETTE[i % len(J_PALETTE)]
                for i, j in enumerate(meta["cmp"]["J"])}
    v_colors = {str(v): V_PALETTE[i % len(V_PALETTE)]
                for i, v in enumerate(meta["off"]["V"])}

    # ----------------------------------------------------------------
    # Stability & Control derivatives
    # ----------------------------------------------------------------
    def linfit(x, y):
        """Linear fit, returns (slope, intercept, r2) or (None,None,None)."""
        x, y = np.array(x, float), np.array(y, float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) < 2:
            return None, None, None
        b, a = np.polyfit(x, y, 1)
        yhat  = a + b * x
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
        return round(float(b), 6), round(float(a), 6), (round(r2, 4) if r2 is not None else None)

    # Cn_beta (dCMyaw/dAoS) — propOn
    cn_beta_on = []
    for (aoa, j, v, dr), g in on_df.groupby(["AoA_round", "J_round", "V_round", "dR"]):
        grp = g.groupby("AoS_round")[["CMyaw_FINAL", "CYaw_FINAL"]].mean().reset_index()
        if grp["AoS_round"].nunique() < 2:
            continue
        slope_cn, _, r2_cn = linfit(grp["AoS_round"], grp["CMyaw_FINAL"])
        slope_cy, _, r2_cy = linfit(grp["AoS_round"], grp["CYaw_FINAL"])
        if slope_cn is None:
            continue
        cn_beta_on.append({"AoA": flt(aoa), "J": flt(j), "V": flt(v), "dR": flt(dr),
                            "Cn_beta": slope_cn, "r2_cn": r2_cn,
                            "Cy_beta": slope_cy, "r2_cy": r2_cy,
                            "n_pts": int(grp["AoS_round"].nunique())})

    # Cn_beta — propOff
    cn_beta_off = []
    for (aoa, v, dr, de), g in off_df.groupby(["AoA_round", "V_round", "dR", "dE"]):
        grp = g.groupby("AoS_round")[["CMyaw_FINAL", "CYaw_FINAL"]].mean().reset_index()
        if grp["AoS_round"].nunique() < 3:
            continue
        slope_cn, _, r2_cn = linfit(grp["AoS_round"], grp["CMyaw_FINAL"])
        slope_cy, _, r2_cy = linfit(grp["AoS_round"], grp["CYaw_FINAL"])
        if slope_cn is None:
            continue
        cn_beta_off.append({"AoA": flt(aoa), "V": flt(v), "dR": flt(dr), "dE": flt(de),
                             "Cn_beta": slope_cn, "r2_cn": r2_cn,
                             "Cy_beta": slope_cy, "r2_cy": r2_cy,
                             "n_pts": int(grp["AoS_round"].nunique())})

    # Cn_dR (dCMyaw/ddR) — propOn
    cn_dr_on = []
    for (aoa, j, v, aos), g in on_df.groupby(["AoA_round", "J_round", "V_round", "AoS_round"]):
        grp = g.groupby("dR")[["CMyaw_FINAL", "CYaw_FINAL", "CMroll_FINAL"]].mean().reset_index()
        if grp["dR"].nunique() < 2:
            continue
        slope_cn, icpt_cn, r2_cn = linfit(grp["dR"], grp["CMyaw_FINAL"])
        slope_cy, _, _            = linfit(grp["dR"], grp["CYaw_FINAL"])
        slope_cr, _, _            = linfit(grp["dR"], grp["CMroll_FINAL"])
        if slope_cn is None:
            continue
        dr_trim = round(-icpt_cn / slope_cn, 2) if abs(slope_cn) > 1e-8 else None
        cn_dr_on.append({"AoA": flt(aoa), "J": flt(j), "V": flt(v), "AoS": flt(aos),
                          "Cn_dR": slope_cn, "Cy_dR": slope_cy, "Cr_dR": slope_cr,
                          "r2_cn": r2_cn, "dR_trim": flt(dr_trim),
                          "n_pts": int(grp["dR"].nunique())})

    # Cn_dR — propOff
    cn_dr_off = []
    for (aoa, v, aos, de), g in off_df.groupby(["AoA_round", "V_round", "AoS_round", "dE"]):
        grp = g.groupby("dR")[["CMyaw_FINAL", "CYaw_FINAL", "CMroll_FINAL"]].mean().reset_index()
        if grp["dR"].nunique() < 2:
            continue
        slope_cn, icpt_cn, r2_cn = linfit(grp["dR"], grp["CMyaw_FINAL"])
        slope_cy, _, _            = linfit(grp["dR"], grp["CYaw_FINAL"])
        slope_cr, _, _            = linfit(grp["dR"], grp["CMroll_FINAL"])
        if slope_cn is None:
            continue
        dr_trim = round(-icpt_cn / slope_cn, 2) if abs(slope_cn) > 1e-8 else None
        cn_dr_off.append({"AoA": flt(aoa), "V": flt(v), "AoS": flt(aos), "dE": flt(de),
                           "Cn_dR": slope_cn, "Cy_dR": slope_cy, "Cr_dR": slope_cr,
                           "r2_cn": r2_cn, "dR_trim": flt(dr_trim),
                           "n_pts": int(grp["dR"].nunique())})

    # Drag polar rows
    polar_on  = [{"CL": flt(r.CL_FINAL), "CD": flt(r.CD_FINAL),
                  "J": flt(r.J_round), "V": flt(r.V_round),
                  "AoA": flt(r.AoA_round), "AoS": flt(r.AoS_round), "dR": flt(r.dR)}
                 for _, r in on_df.iterrows()]
    polar_off = [{"CL": flt(r.CL_FINAL), "CD": flt(r.CD_FINAL),
                  "V": flt(r.V_round), "AoA": flt(r.AoA_round),
                  "AoS": flt(r.AoS_round), "dR": flt(r.dR), "dE": flt(r.dE)}
                 for _, r in off_df.iterrows()]

    # Trim analysis — dR where CMyaw = 0, with interpolated CL, CD, L/D
    trim_rows = []
    for entry in cn_dr_on:
        dr_t = entry.get("dR_trim")
        if dr_t is None or not (-25 <= dr_t <= 5):
            continue
        sub = on_df[(on_df.AoA_round == entry["AoA"]) &
                    (on_df.J_round   == entry["J"])   &
                    (on_df.V_round   == entry["V"])   &
                    (on_df.AoS_round == entry["AoS"])].sort_values("dR")
        if len(sub) < 2:
            cl_t = cd_t = ld_t = None
        else:
            cl_t = float(np.interp(dr_t, sub["dR"].values, sub["CL_FINAL"].values))
            cd_t = float(np.interp(dr_t, sub["dR"].values, sub["CD_FINAL"].values))
            ld_t = cl_t / cd_t if cd_t and cd_t != 0 else None
        trim_rows.append({"AoA": entry["AoA"], "J": entry["J"], "V": entry["V"],
                           "AoS": entry["AoS"], "dR_trim": flt(dr_t),
                           "Cn_dR": entry["Cn_dR"],
                           "CL_trim": flt(cl_t), "CD_trim": flt(cd_t), "LD_trim": flt(ld_t)})

    # SC dropdown metadata
    sc_meta = {
        "on": {
            "J":   sorted(set(r["J"]   for r in cn_beta_on)),
            "V":   sorted(set(r["V"]   for r in cn_beta_on)),
            "AoA": sorted(set(r["AoA"] for r in cn_beta_on)),
            "dR":  sorted(set(r["dR"]  for r in cn_beta_on)),
            "AoS_dr": sorted(set(r["AoS"] for r in cn_dr_on)),
        },
        "off": {
            "V":   sorted(set(r["V"]   for r in cn_beta_off)),
            "AoA": sorted(set(r["AoA"] for r in cn_beta_off)),
            "dR":  sorted(set(r["dR"]  for r in cn_beta_off)),
            "dE":  sorted(set(r["dE"]  for r in cn_beta_off)),
        },
    }

    print(f"  Cn_beta_on : {len(cn_beta_on)} fits")
    print(f"  Cn_beta_off: {len(cn_beta_off)} fits")
    print(f"  Cn_dR_on   : {len(cn_dr_on)} fits")
    print(f"  Cn_dR_off  : {len(cn_dr_off)} fits")
    print(f"  Trim rows  : {len(trim_rows)}")

    return (cmp_rows, off_rows, meta, j_colors, v_colors,
            cn_beta_on, cn_beta_off, cn_dr_on, cn_dr_off,
            polar_on, polar_off, trim_rows, sc_meta)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def generate_html(cmp_rows, off_rows, meta, j_colors, v_colors,
                  cn_beta_on, cn_beta_off, cn_dr_on, cn_dr_off,
                  polar_on, polar_off, trim_rows, sc_meta, out_path: Path):

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AE4115 — Aerodynamic Data Explorer</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

:root{{
  --bg:#0d0f14; --surface:#151821; --surface2:#1d2030; --border:#2a2d3e;
  --text:#e2e4f0; --muted:#6b7094; --accent:#5b8af5; --accent2:#3ecf8e;
  --warn:#f5a623; --danger:#f05252;
  --font:'IBM Plex Sans',sans-serif; --mono:'IBM Plex Mono',monospace;
  --r:6px;
}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:var(--font);font-size:13px;min-height:100vh;}}

.topbar{{display:flex;align-items:center;gap:0;background:var(--surface);
         border-bottom:1px solid var(--border);padding:0 20px;height:48px;}}
.topbar-title{{font-family:var(--mono);font-size:11px;font-weight:600;
               letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
               margin-right:32px;white-space:nowrap;}}
.nav-tab{{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
          padding:0 18px;height:48px;display:flex;align-items:center;cursor:pointer;
          border:none;background:none;color:var(--muted);border-bottom:2px solid transparent;
          transition:all .15s;white-space:nowrap;}}
.nav-tab:hover{{color:var(--text);}}
.nav-tab.active{{color:var(--accent);border-bottom-color:var(--accent);}}

.page{{display:none;padding:16px 20px;}}
.page.active{{display:block;}}

.ctrl-bar{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;
           background:var(--surface);border:1px solid var(--border);
           border-radius:var(--r);padding:10px 14px;margin-bottom:12px;}}
.ctrl-group{{display:flex;align-items:center;gap:6px;}}
.ctrl-label{{font-family:var(--mono);font-size:10px;color:var(--muted);
             text-transform:uppercase;letter-spacing:.08em;white-space:nowrap;}}
select{{font-family:var(--mono);font-size:11px;padding:4px 8px;
        border:1px solid var(--border);border-radius:4px;
        background:var(--surface2);color:var(--text);cursor:pointer;
        outline:none;min-width:70px;}}
select:focus{{border-color:var(--accent);}}
.sep{{width:1px;height:20px;background:var(--border);margin:0 4px;}}

.toggle-wrap{{display:flex;align-items:center;gap:6px;}}
.toggle{{position:relative;width:32px;height:18px;cursor:pointer;}}
.toggle input{{opacity:0;width:0;height:0;}}
.toggle-slider{{position:absolute;inset:0;background:var(--border);border-radius:9px;transition:.2s;}}
.toggle input:checked + .toggle-slider{{background:var(--accent);}}
.toggle-slider:before{{content:'';position:absolute;width:12px;height:12px;
  left:3px;top:3px;background:#fff;border-radius:50%;transition:.2s;}}
.toggle input:checked + .toggle-slider:before{{transform:translateX(14px);}}

.metric-tabs{{display:flex;gap:3px;margin-bottom:10px;flex-wrap:wrap;}}
.mtab{{font-family:var(--mono);font-size:10px;font-weight:600;
       letter-spacing:.06em;text-transform:uppercase;
       padding:5px 12px;border:1px solid var(--border);border-radius:4px;
       cursor:pointer;background:var(--surface);color:var(--muted);transition:all .15s;}}
.mtab:hover{{color:var(--text);border-color:var(--accent);}}
.mtab.active{{background:var(--accent);color:#fff;border-color:var(--accent);}}
.mtab.grp2.active{{background:var(--accent2);border-color:var(--accent2);}}

.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px;margin-bottom:12px;}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:10px 12px;}}
.card-lbl{{font-family:var(--mono);font-size:9px;color:var(--muted);
           text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;
           display:flex;align-items:center;gap:5px;}}
.dot{{width:7px;height:7px;border-radius:2px;flex-shrink:0;}}
.card-val{{font-family:var(--mono);font-size:16px;font-weight:600;}}
.card-sub{{font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:2px;}}
.pos{{color:var(--accent2);}} .neg{{color:var(--danger);}}

.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px;}}
.leg-item{{display:flex;align-items:center;gap:5px;font-family:var(--mono);font-size:10px;color:var(--muted);}}
.lsq{{width:10px;height:10px;border-radius:2px;flex-shrink:0;}}
.ldash{{width:18px;height:0;border-top:2px dashed var(--muted);}}

.chart-box{{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:14px 14px 10px;}}
.chart-wrap{{position:relative;width:100%;height:360px;}}
.chart-meta{{font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:8px;text-align:right;}}

/* HELP PAGE */
.help-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:1100px;}}
.help-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:20px 22px;}}
.help-card h2{{font-family:var(--mono);font-size:11px;font-weight:600;letter-spacing:.1em;
               text-transform:uppercase;color:var(--accent);margin-bottom:14px;
               padding-bottom:8px;border-bottom:1px solid var(--border);}}
.help-card h3{{font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:.06em;
               text-transform:uppercase;color:var(--accent2);margin:14px 0 6px;}}
.help-card p{{font-size:12px;line-height:1.7;color:#c0c4d8;margin-bottom:8px;}}
.help-card ul{{padding-left:16px;margin-bottom:8px;}}
.help-card li{{font-size:12px;line-height:1.8;color:#c0c4d8;}}
.help-tag{{display:inline-block;font-family:var(--mono);font-size:10px;font-weight:600;
           padding:2px 8px;border-radius:3px;margin:2px 2px 2px 0;
           background:var(--surface2);border:1px solid var(--border);color:var(--text);}}
.help-tag.blue{{background:#1e3a5f;border-color:#3b6cb7;color:#93c5fd;}}
.help-tag.green{{background:#1a3d2e;border-color:#2d6a4f;color:#6ee7b7;}}
.help-tag.muted{{color:var(--muted);}}
.help-full{{grid-column:1/-1;}}
.help-formula{{font-family:var(--mono);font-size:11px;background:var(--surface2);
               border:1px solid var(--border);border-radius:4px;padding:10px 14px;
               margin:8px 0;color:#e2e4f0;line-height:1.9;}}
.help-intro{{font-size:13px;line-height:1.7;color:#c0c4d8;max-width:1100px;
             margin-bottom:16px;padding:14px 18px;background:var(--surface);
             border:1px solid var(--border);border-radius:var(--r);}}
</style>
</head>
<body>

<div class="topbar">
  <span class="topbar-title">AE4115 &nbsp;/&nbsp; Data Explorer</span>
  <button class="nav-tab active" onclick="switchPage('cmp',this)">PropOn vs PropOff</button>
  <button class="nav-tab" onclick="switchPage('off',this)">PropOff Explorer</button>
  <button class="nav-tab" onclick="switchPage('sc',this)">Stability &amp; Control</button>
  <button class="nav-tab" onclick="switchPage('help',this)">Help</button>
</div>

<!-- PAGE 1 — COMPARISON -->
<div class="page active" id="page-cmp">
  <div class="ctrl-bar">
    <div class="ctrl-group"><span class="ctrl-label">x-axis</span>
      <select id="c-xax">
        <option value="AoA">AoA</option>
        <option value="AoS">AoS</option>
        <option value="dR">dR</option>
      </select>
    </div>
    <div class="sep"></div>
    <div class="ctrl-group"><span class="ctrl-label">J</span>
      <select id="c-j"><option value="all">all</option>
        {opts_html(meta['cmp']['J'])}
      </select>
    </div>
    <div class="ctrl-group"><span class="ctrl-label">V (m/s)</span>
      <select id="c-v"><option value="all">all</option>
        {opts_html(meta['cmp']['V'])}
      </select>
    </div>
    <div class="ctrl-group"><span class="ctrl-label">fix AoA</span>
      <select id="c-aoa"><option value="all">—</option>
        {opts_html(meta['cmp']['AoA'], "{}°")}
      </select>
    </div>
    <div class="ctrl-group"><span class="ctrl-label">fix AoS</span>
      <select id="c-aos"><option value="all">—</option>
        {opts_html(meta['cmp']['AoS'], "{}°")}
      </select>
    </div>
    <div class="ctrl-group"><span class="ctrl-label">fix dR</span>
      <select id="c-dr"><option value="all">—</option>
        {opts_html(meta['cmp']['dR'], "{}°")}
      </select>
    </div>
    <div class="sep"></div>
    <div class="ctrl-group toggle-wrap">
      <span class="ctrl-label">actual x</span>
      <label class="toggle"><input type="checkbox" id="c-actual">
        <span class="toggle-slider"></span></label>
    </div>
  </div>
  <div class="metric-tabs" id="c-mtabs"></div>
  <div class="cards" id="c-cards"></div>
  <div class="chart-box">
    <div class="legend" id="c-legend"></div>
    <div class="chart-wrap"><canvas id="c-chart"></canvas></div>
    <p class="chart-meta">Indexed on rounded values &nbsp;|&nbsp; PropOff averaged over repeats &nbsp;|&nbsp; Toggle "actual x" to use corrected axis values</p>
  </div>
</div>

<!-- PAGE 2 — PROPOFF EXPLORER -->
<div class="page" id="page-off">
  <div class="ctrl-bar">
    <div class="ctrl-group"><span class="ctrl-label">x-axis</span>
      <select id="o-xax">
        <option value="AoA">AoA</option>
        <option value="AoS">AoS</option>
        <option value="dR">dR</option>
        <option value="dE">dE</option>
        <option value="V">V</option>
      </select>
    </div>
    <div class="ctrl-group"><span class="ctrl-label">colour by</span>
      <select id="o-color">
        <option value="V">V</option>
        <option value="dE">dE</option>
        <option value="dR">dR</option>
        <option value="AoS">AoS</option>
      </select>
    </div>
    <div class="sep"></div>
    <div class="ctrl-group"><span class="ctrl-label">V (m/s)</span>
      <select id="o-v"><option value="all">all</option>
        {opts_html(meta['off']['V'])}
      </select>
    </div>
    <div class="ctrl-group"><span class="ctrl-label">fix AoA</span>
      <select id="o-aoa"><option value="all">—</option>
        {opts_html(meta['off']['AoA'], "{}°")}
      </select>
    </div>
    <div class="ctrl-group"><span class="ctrl-label">fix AoS</span>
      <select id="o-aos"><option value="all">—</option>
        {opts_html(meta['off']['AoS'], "{}°")}
      </select>
    </div>
    <div class="ctrl-group"><span class="ctrl-label">fix dR</span>
      <select id="o-dr"><option value="all">—</option>
        {opts_html(meta['off']['dR'], "{}°")}
      </select>
    </div>
    <div class="ctrl-group"><span class="ctrl-label">fix dE</span>
      <select id="o-de"><option value="all">—</option>
        {opts_html(meta['off']['dE'], "{}°")}
      </select>
    </div>
    <div class="sep"></div>
    <div class="ctrl-group toggle-wrap">
      <span class="ctrl-label">actual x</span>
      <label class="toggle"><input type="checkbox" id="o-actual">
        <span class="toggle-slider"></span></label>
    </div>
  </div>
  <div class="metric-tabs" id="o-mtabs"></div>
  <div class="chart-box">
    <div class="legend" id="o-legend"></div>
    <div class="chart-wrap"><canvas id="o-chart"></canvas></div>
    <p class="chart-meta">PropOff FINAL coefficients &nbsp;|&nbsp; Toggle "actual x" to use AoA_FINAL / V_FINAL on axis</p>
  </div>
</div>

<!-- PAGE 3 — STABILITY & CONTROL -->
<div class="page" id="page-sc">
  <div class="ctrl-bar">
    <div class="ctrl-group"><span class="ctrl-label">plot</span>
      <select id="sc-plot">
        <option value="cn_beta">Cn_beta vs AoA</option>
        <option value="cy_beta">Cy_beta vs AoA</option>
        <option value="cn_dr">Cn_dR vs AoA</option>
        <option value="cy_dr">Cy_dR vs AoA</option>
        <option value="cr_dr">Cr_dR vs AoA (roll coupling)</option>
        <option value="polar">Drag polar (CL vs CD)</option>
        <option value="trim_dr">Trim dR vs AoA</option>
        <option value="trim_ld">L/D at trim vs AoA</option>
        <option value="trim_cl">CL at trim vs AoA</option>
      </select>
    </div>
    <div class="ctrl-group"><span class="ctrl-label">dataset</span>
      <select id="sc-src">
        <option value="both">both</option>
        <option value="on">prop-on only</option>
        <option value="off">prop-off only</option>
      </select>
    </div>
    <div class="sep"></div>
    <div class="ctrl-group"><span class="ctrl-label">V (m/s)</span>
      <select id="sc-v"><option value="all">all</option>
        {opts_html(sorted(set(meta['cmp']['V'] + meta['off']['V'])))}
      </select>
    </div>
    <div class="ctrl-group"><span class="ctrl-label">fix dR</span>
      <select id="sc-dr"><option value="all">all</option>
        {opts_html(sorted(set(meta['cmp']['dR'] + meta['off']['dR'])))}
      </select>
    </div>
    <div class="ctrl-group"><span class="ctrl-label">fix dE (off)</span>
      <select id="sc-de"><option value="all">all</option>
        {opts_html(meta['off']['dE'])}
      </select>
    </div>
    <div class="ctrl-group"><span class="ctrl-label">fix AoS (dR plots)</span>
      <select id="sc-aos"><option value="all">all</option>
        {opts_html(sorted(set(meta['cmp']['AoS'] + meta['off']['AoS'])), "{}°")}
      </select>
    </div>
    <div class="sep"></div>
    <div class="ctrl-group">
      <span class="ctrl-label" id="sc-r2-label">min R²</span>
      <select id="sc-r2min">
        <option value="0">no filter</option>
        <option value="0.7">0.70</option>
        <option value="0.8" selected>0.80</option>
        <option value="0.9">0.90</option>
        <option value="0.95">0.95</option>
      </select>
    </div>
  </div>

  <div class="cards" id="sc-cards"></div>
  <div class="chart-box">
    <div class="legend" id="sc-legend"></div>
    <div class="chart-wrap"><canvas id="sc-chart"></canvas></div>
    <p class="chart-meta" id="sc-meta">Derivatives fitted by linear regression on rounded index values &nbsp;|&nbsp; Filter by min R² to exclude poor fits</p>
  </div>
</div>

<!-- PAGE 4 — HELP -->
<div class="page" id="page-help">
  <p class="help-intro">
    This tool visualises the fully corrected aerodynamic wind-tunnel data from the AE4115 lab.
    All plotted values are <strong>_FINAL</strong> columns — the end result of the complete
    wall-correction pipeline (model-off tare, blockage, streamline-curvature, downwash, and
    tail-plane interference corrections). Use the tabs at the top to switch between the
    comparison view and the propOff explorer.
  </p>

  <div class="help-grid">

    <!-- COMPARISON TAB -->
    <div class="help-card">
      <h2>PropOn vs PropOff — Comparison</h2>
      <p>Overlays prop-on and prop-off data on the same chart, coloured by advance ratio J.
         The prop-off line is always shown as a dashed grey reference.</p>

      <h3>Controls</h3>
      <ul>
        <li><strong>x-axis</strong> — choose what to sweep: AoA, AoS, or dR.</li>
        <li><strong>J</strong> — filter to a single advance ratio, or show all J values at once.</li>
        <li><strong>V (m/s)</strong> — filter to a single tunnel speed.</li>
        <li><strong>fix AoA / fix AoS / fix dR</strong> — lock a dimension to a specific value.
            When set to <span class="help-tag muted">—</span> all values are included and
            <em>averaged together</em> at each x point.</li>
        <li><strong>Actual x toggle</strong> — when OFF, the x-axis uses rounded grid values
            (AoA_round etc.) used for indexing. When ON, the axis shows the actual
            measured/corrected values (AoA, AoA_FINAL, V_FINAL). The grouping and
            averaging always uses rounded values regardless.</li>
      </ul>

      <h3>Metric tabs</h3>
      <p>Two groups separated by a gap:</p>
      <ul>
        <li><span class="help-tag blue">CL CD CMpitch CYaw CMroll CMyaw CFt</span>
            — final aerodynamic coefficients.</li>
        <li><span class="help-tag green">ΔCL sc ΔCD dw ΔCM sc ΔCM tail Δα dw Δα tail ε total ε wake ε slip Tc* CLw ref CD0 fit k fit R² fit</span>
            — correction magnitudes and blockage factors.</li>
      </ul>

      <h3>Summary cards</h3>
      <p>For coefficient metrics (group 1), cards show the mean difference
         (prop-on minus prop-off) per J value across the currently filtered data.
         Green = prop-on higher, red = prop-on lower. Cards are hidden for
         correction metrics since a Δ of a correction does not have a clear sign convention.</p>

      <h3>Averaging behaviour</h3>
      <p>When a fix filter is set to <span class="help-tag muted">—</span>, all values in
         that dimension are averaged together at each x point. For example if fix AoS = —
         and x-axis = AoA, each chart point is the mean over all sideslip angles at that AoA.
         This can smooth out asymmetric effects — fix AoS = 0° for the cleanest symmetric case.</p>
      <p>PropOff data is additionally averaged over repeat runs at the same index before
         being merged with prop-on data.</p>
    </div>

    <!-- PROPOFF EXPLORER -->
    <div class="help-card">
      <h2>PropOff Explorer</h2>
      <p>Shows all individual propOff measurements (4000+ rows) without any aggregation.
         Useful for inspecting the full AoA sweep range, drag polars, and correction
         magnitudes across all test conditions.</p>

      <h3>Controls</h3>
      <ul>
        <li><strong>x-axis</strong> — AoA, AoS, dR, dE, or V. More options than the
            comparison tab because the propOff dataset covers a much wider test matrix.</li>
        <li><strong>colour by</strong> — choose which dimension separates the lines:
            V, dE, dR, or AoS. Each unique value in that dimension gets its own coloured line.</li>
        <li><strong>V / fix AoA / fix AoS / fix dR / fix dE</strong> — same filtering
            logic as the comparison tab. Set to <span class="help-tag muted">—</span> to
            include all values (averaged at each x point).</li>
        <li><strong>Actual x toggle</strong> — same as comparison tab. For AoA this switches
            between AoA_round (grid) and AoA (raw measured). Note that AoA_FINAL is the
            fully corrected angle including streamline-curvature, downwash, and tail corrections
            — it can differ significantly from the raw AoA at high lift.</li>
      </ul>

      <h3>Metric tabs</h3>
      <p>Same two groups as the comparison tab, but with an extra correction metric
         <span class="help-tag green">Δα sc</span> (streamline-curvature AoA increment)
         which is only meaningful for propOff since propOn uses a different CL source column.</p>

      <h3>Tip — inspecting the drag polar</h3>
      <p>Set x-axis = AoA, colour by = dE, fix AoS = 0°, and select CD or CD0 fit.
         This gives you the drag polar family for each elevator setting at symmetric flight.</p>
    </div>

    <!-- STABILITY & CONTROL TAB -->
    <div class="help-card">
      <h2>Stability &amp; Control</h2>
      <p>Extracts and plots aerodynamic stability and control derivatives fitted by linear
         regression from the measurement data. All derivatives are in per-degree units.</p>

      <h3>Plot types</h3>
      <ul>
        <li><strong>Cnβ vs AoA</strong> — directional stability derivative ∂Cn/∂β.
            Fitted from CMyaw vs AoS at each (AoA, J, V, dR) condition.
            More negative = more stable. Prop-on shown solid by J, prop-off dashed by V.</li>
        <li><strong>Cyβ vs AoA</strong> — side-force stability ∂Cy/∂β.
            Always positive (side force in direction of sideslip).</li>
        <li><strong>Cn_dR vs AoA</strong> — rudder control power ∂Cn/∂δR.
            Fitted from CMyaw vs dR. More negative = more effective rudder.</li>
        <li><strong>Cy_dR vs AoA</strong> — side force due to rudder ∂Cy/∂δR.</li>
        <li><strong>Cr_dR vs AoA</strong> — roll coupling from rudder ∂Cl/∂δR
            (adverse/proverse yaw indicator).</li>
        <li><strong>Drag polar</strong> — CL vs CD scatter for all conditions.
            Overlays prop-on (coloured by J) and prop-off (grey). Shows how thrust shifts
            the polar.</li>
        <li><strong>Trim dR vs AoA</strong> — rudder deflection required to achieve
            CMyaw = 0 (directional trim) at each AoA and J, found by linear interpolation.</li>
        <li><strong>L/D at trim</strong> — lift-to-drag ratio at the trim dR condition.
            Shows the aerodynamic performance penalty from trimming.</li>
        <li><strong>CL at trim</strong> — lift coefficient at the trim condition.</li>
      </ul>

      <h3>Controls</h3>
      <ul>
        <li><strong>dataset</strong> — show prop-on, prop-off, or both.</li>
        <li><strong>V</strong> — filter to a single tunnel speed.</li>
        <li><strong>fix dR</strong> — for Cnβ plots, restrict to a specific rudder angle
            (dR=0 gives the clean baseline).</li>
        <li><strong>fix dE</strong> — for prop-off plots, restrict to a specific elevator angle.</li>
        <li><strong>fix AoS</strong> — for dR-sweep plots (Cn_dR etc.), restrict to a specific
            sideslip. AoS=0 gives the symmetric case.</li>
        <li><strong>min R²</strong> — exclude fits with R² below the threshold.
            Default 0.80. Lower values include noisier fits; raise to 0.90+ for publication quality.</li>
      </ul>

      <h3>Interpretation notes</h3>
      <ul>
        <li>PropOn Cnβ is only available at AoA=2.5° (the only AoA with a full AoS sweep
            in the prop-on dataset). PropOff covers the full AoA range.</li>
        <li>Trim dR values outside [−25°, 5°] are excluded as physically implausible
            (outside the rudder deflection range).</li>
        <li>The drag polar scatter plot intentionally shows all raw points rather than
            averaged lines, so you can see the spread across conditions.</li>
      </ul>
    </div>
      <h2>Correction Pipeline</h2>
      <p>All data has passed through the following correction sequence before being
         loaded into this tool. The _FINAL columns are the output of the last step.</p>

      <h3>PropOff sequence</h3>
      <ul>
        <li>Model-off tare subtraction</li>
        <li>Drag polar fit &nbsp;<span class="help-tag muted">CD = CD0 + k·CL²</span></li>
        <li>Solid blockage &nbsp;<span class="help-tag muted">ε_sb = 0.006406642</span></li>
        <li>Wake blockage &nbsp;<span class="help-tag muted">ε_wb = S/(4C)·CD0 + 5S/(4C)·CDsep</span></li>
        <li>Combined blockage correction &nbsp;<span class="help-tag muted">C_corr = C / (1+ε)²</span></li>
        <li>Streamline-curvature correction</li>
        <li>Downwash correction</li>
        <li>Tail-plane interference correction</li>
      </ul>

      <h3>PropOn additional steps</h3>
      <ul>
        <li>BEM thrust separation &nbsp;<span class="help-tag muted">CT_bem(J) polynomial</span></li>
        <li>Slipstream blockage &nbsp;<span class="help-tag muted">ε_ss = −(Tc*/2√(1+2Tc·))·(S_prop/C)</span></li>
      </ul>

      <div class="help-formula">
Blockage correction (velocity):   V_corr = V / (1 + ε_total)
Blockage correction (coeff):      C_corr = C / (1 + ε_total)²
CFt thrust: solid + wake only,    never slipstream (circular)
      </div>
    </div>

    <!-- METRIC GLOSSARY -->
    <div class="help-card">
      <h2>Metric Glossary</h2>

      <h3>Final coefficients</h3>
      <ul>
        <li><strong>CL</strong> — lift coefficient (aerodynamic only after BEM separation for propOn)</li>
        <li><strong>CD</strong> — drag coefficient</li>
        <li><strong>CMpitch</strong> — pitching moment coefficient</li>
        <li><strong>CYaw</strong> — yaw force coefficient</li>
        <li><strong>CMroll</strong> — rolling moment coefficient</li>
        <li><strong>CMyaw</strong> — yawing moment coefficient</li>
        <li><strong>CFt</strong> — propeller thrust force coefficient (propOn only, solid+wake blockage corrected)</li>
      </ul>

      <h3>Correction magnitudes</h3>
      <ul>
        <li><strong>ΔCL sc</strong> — CL increment from streamline-curvature correction</li>
        <li><strong>ΔCD dw</strong> — CD increment from downwash correction</li>
        <li><strong>ΔCM sc</strong> — CMpitch increment from streamline-curvature</li>
        <li><strong>ΔCM tail</strong> — CMpitch increment from tail-plane interference</li>
        <li><strong>Δα dw</strong> — AoA increment from downwash [deg]</li>
        <li><strong>Δα tail</strong> — AoA increment from tail correction [deg]</li>
        <li><strong>Δα sc</strong> — AoA increment from streamline-curvature [deg] (propOff only)</li>
      </ul>

      <h3>Blockage factors</h3>
      <ul>
        <li><strong>ε total</strong> — total blockage factor (sb + wb [+ ss for propOn])</li>
        <li><strong>ε wake</strong> — wake blockage factor ε_wb</li>
        <li><strong>ε slip</strong> — slipstream blockage factor ε_ss (propOn only, negative by design)</li>
        <li><strong>Tc*</strong> — thrust loading T/(q·S_prop) driving slipstream blockage</li>
        <li><strong>CLw ref</strong> — tail-off reference lift used to drive SC, downwash, and tail corrections</li>
      </ul>

      <h3>Drag polar fit</h3>
      <ul>
        <li><strong>CD0 fit</strong> — zero-lift drag from polar fit</li>
        <li><strong>k fit</strong> — induced drag factor from polar fit</li>
        <li><strong>R² fit</strong> — goodness of fit (values below ~0.85 indicate poor polar quality)</li>
      </ul>
    </div>

  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const CMP_DATA      = {json.dumps(cmp_rows,    separators=(',', ':'))};
const OFF_DATA      = {json.dumps(off_rows,     separators=(',', ':'))};
const J_COLORS      = {json.dumps(j_colors)};
const META          = {json.dumps(meta)};
const CN_BETA_ON    = {json.dumps(cn_beta_on,   separators=(',', ':'))};
const CN_BETA_OFF   = {json.dumps(cn_beta_off,  separators=(',', ':'))};
const CN_DR_ON      = {json.dumps(cn_dr_on,     separators=(',', ':'))};
const CN_DR_OFF     = {json.dumps(cn_dr_off,    separators=(',', ':'))};
const POLAR_ON      = {json.dumps(polar_on,     separators=(',', ':'))};
const POLAR_OFF     = {json.dumps(polar_off,    separators=(',', ':'))};
const TRIM_ROWS     = {json.dumps(trim_rows,    separators=(',', ':'))};
const SC_META       = {json.dumps(sc_meta)};
const J_PALETTE_SC  = {json.dumps(J_PALETTE)};
const SC_OFF_PALETTE= ['#94a3b8','#cbd5e1','#64748b','#475569','#e2e8f0'];

const CMP_METRICS = [
  {{key:'CL',        lbl:'CL',       on:'CL_on',      off:'CL_off',      grp:1}},
  {{key:'CD',        lbl:'CD',       on:'CD_on',       off:'CD_off',      grp:1}},
  {{key:'CM',        lbl:'CMpitch',  on:'CM_on',       off:'CM_off',      grp:1}},
  {{key:'CYaw',      lbl:'CYaw',     on:'CYaw_on',    off:'CYaw_off',    grp:1}},
  {{key:'CMroll',    lbl:'CMroll',   on:'CMroll_on',  off:'CMroll_off',  grp:1}},
  {{key:'CMyaw',     lbl:'CMyaw',    on:'CMyaw_on',   off:'CMyaw_off',   grp:1}},
  {{key:'CFt',       lbl:'CFt',      on:'CFt',         off:null,          grp:1}},
  {{key:'dCL_sc',    lbl:'ΔCL sc',   on:'dCL_sc_on',  off:'dCL_sc_off',  grp:2}},
  {{key:'dCD_dw',    lbl:'ΔCD dw',   on:'dCD_dw_on',  off:'dCD_dw_off',  grp:2}},
  {{key:'dCM_sc',    lbl:'ΔCM sc',   on:'dCM_sc_on',  off:'dCM_sc_off',  grp:2}},
  {{key:'dCM_tail',  lbl:'ΔCM tail', on:'dCM_tail_on',off:'dCM_tail_off',grp:2}},
  {{key:'dAoA_dw',   lbl:'Δα dw',    on:'dAoA_dw_on', off:'dAoA_dw_off', grp:2}},
  {{key:'dAoA_tail', lbl:'Δα tail',  on:'dAoA_tail_on',off:'dAoA_tail_off',grp:2}},
  {{key:'e_tot',     lbl:'ε total',  on:'e_tot_on',   off:'e_tot_off',   grp:2}},
  {{key:'ewb',       lbl:'ε wake',   on:'ewb_on',     off:'ewb_off',     grp:2}},
  {{key:'ess',       lbl:'ε slip',   on:'ess',         off:null,          grp:2}},
  {{key:'Tc',        lbl:'Tc*',      on:'Tc',          off:null,          grp:2}},
  {{key:'CLw',       lbl:'CLw ref',  on:'CLw_on',     off:'CLw_off',     grp:2}},
  {{key:'CD0',       lbl:'CD0 fit',  on:null,          off:'CD0',         grp:2}},
  {{key:'k',         lbl:'k fit',    on:null,          off:'k',           grp:2}},
  {{key:'R2',        lbl:'R² fit',   on:null,          off:'R2',          grp:2}},
];

const OFF_METRICS = [
  {{key:'CL',    lbl:'CL',       field:'CL',    grp:1}},
  {{key:'CD',    lbl:'CD',       field:'CD',    grp:1}},
  {{key:'CM',    lbl:'CMpitch',  field:'CM',    grp:1}},
  {{key:'CYaw',  lbl:'CYaw',    field:'CYaw',  grp:1}},
  {{key:'CMroll',lbl:'CMroll',  field:'CMroll',grp:1}},
  {{key:'CMyaw', lbl:'CMyaw',   field:'CMyaw', grp:1}},
  {{key:'dCL_sc',    lbl:'ΔCL sc',    field:'dCL_sc',    grp:2}},
  {{key:'dCD_dw',    lbl:'ΔCD dw',    field:'dCD_dw',    grp:2}},
  {{key:'dCM_sc',    lbl:'ΔCM sc',    field:'dCM_sc',    grp:2}},
  {{key:'dCM_tail',  lbl:'ΔCM tail',  field:'dCM_tail',  grp:2}},
  {{key:'dAoA_dw',   lbl:'Δα dw',     field:'dAoA_dw',   grp:2}},
  {{key:'dAoA_tail', lbl:'Δα tail',   field:'dAoA_tail', grp:2}},
  {{key:'dAoA_sc',   lbl:'Δα sc',     field:'dAoA_sc',   grp:2}},
  {{key:'e_tot',     lbl:'ε total',   field:'e_tot',     grp:2}},
  {{key:'ewb',       lbl:'ε wake',    field:'ewb',       grp:2}},
  {{key:'CLw',       lbl:'CLw ref',   field:'CLw',       grp:2}},
  {{key:'CD0',       lbl:'CD0 fit',   field:'CD0',       grp:2}},
  {{key:'k',         lbl:'k fit',     field:'k',         grp:2}},
  {{key:'R2',        lbl:'R² fit',    field:'R2',        grp:2}},
];

let cChart = null, oChart = null, scChart = null;
let cMetric = 'CL', oMetric = 'CL';

function switchPage(id, btn) {{
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  btn.classList.add('active');
  if (id === 'cmp') renderCmp();
  else if (id === 'off') renderOff();
  else if (id === 'sc') renderSC();
}}

function avg(arr, key) {{
  const v = arr.map(r => r[key]).filter(x => x !== null && x !== undefined && !isNaN(x));
  return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null;
}}
function sel(id) {{ return document.getElementById(id).value; }}
function isAll(v) {{ return v === 'all'; }}

function buildMetricTabs(containerId, metrics, active, onClickFn) {{
  const el = document.getElementById(containerId);
  let html = '', prevGrp = null;
  metrics.forEach(m => {{
    if (m.grp !== prevGrp) {{
      if (prevGrp !== null) html += '<span style="width:8px;display:inline-block"></span>';
      prevGrp = m.grp;
    }}
    const cls = m.key === active
      ? (m.grp === 2 ? 'mtab grp2 active' : 'mtab active')
      : 'mtab';
    html += `<button class="${{cls}}" onclick="${{onClickFn}}('${{m.key}}');this.blur()">${{m.lbl}}</button>`;
  }});
  el.innerHTML = html;
}}

// ── COMPARISON ─────────────────────────────────────────────────────────────
function renderCmp() {{
  buildMetricTabs('c-mtabs', CMP_METRICS, cMetric, 'setCmpMetric');

  const xax    = sel('c-xax');
  const jFil   = sel('c-j');
  const vFil   = sel('c-v');
  const aoaFix = sel('c-aoa');
  const aosFix = sel('c-aos');
  const drFix  = sel('c-dr');
  const useAct = document.getElementById('c-actual').checked;

  let d = [...CMP_DATA];
  if (!isAll(jFil))                         d = d.filter(r => r.J_r   === parseFloat(jFil));
  if (!isAll(vFil))                         d = d.filter(r => r.V_r   === parseFloat(vFil));
  if (xax !== 'AoA' && !isAll(aoaFix))     d = d.filter(r => r.AoA_r === parseFloat(aoaFix));
  if (xax !== 'AoS' && !isAll(aosFix))     d = d.filter(r => r.AoS_r === parseFloat(aosFix));
  if (xax !== 'dR'  && !isAll(drFix))      d = d.filter(r => r.dR    === parseFloat(drFix));

  const met   = CMP_METRICS.find(m => m.key === cMetric);
  const jList = META.cmp.J;
  const xRndKey = r => xax === 'AoA' ? r.AoA_r : xax === 'AoS' ? r.AoS_r : r.dR;
  const xVals   = [...new Set(d.map(xRndKey))].sort((a, b) => a - b);

  function xActual(r, side) {{
    if (xax === 'AoA') return side === 'on' ? r.AoA_on  : r.AoA_off;
    if (xax === 'AoS') return side === 'on' ? r.AoS_on  : r.AoS_off;
    return r.dR;
  }}

  // summary cards
  const cardsEl = document.getElementById('c-cards');
  if (met.on && met.off) {{
    cardsEl.innerHTML = jList.map(j => {{
      const sub = d.filter(r => r.J_r === j);
      if (!sub.length) return '';
      const on_v  = avg(sub, met.on);
      const off_v = avg(sub, met.off);
      if (on_v === null || off_v === null) return '';
      const delta = on_v - off_v;
      const cls   = delta >= 0 ? 'pos' : 'neg';
      return `<div class="card">
        <div class="card-lbl"><span class="dot" style="background:${{J_COLORS[j]}}"></span>J = ${{j}}</div>
        <div class="card-val ${{cls}}">${{delta >= 0 ? '+' : ''}}${{delta.toFixed(4)}}</div>
        <div class="card-sub">mean \u0394${{met.lbl}}</div>
      </div>`;
    }}).join('');
  }} else {{
    cardsEl.innerHTML = '';
  }}

  // legend
  let legHtml = '';
  if (met.off) legHtml += `<span class="leg-item"><span class="ldash"></span>prop-off</span>`;
  jList.forEach(j => {{
    if (d.some(r => r.J_r === j))
      legHtml += `<span class="leg-item"><span class="lsq" style="background:${{J_COLORS[j]}}"></span>prop-on J=${{j}}</span>`;
  }});
  document.getElementById('c-legend').innerHTML = legHtml;

  // datasets
  const datasets = [];
  if (met.off) {{
    const pts = xVals.map(x => {{
      const rows = d.filter(r => xRndKey(r) === x);
      const xv   = useAct ? avg(rows.map(r => ({{v: xActual(r, 'off')}})), 'v') : x;
      const yv   = avg(rows, met.off);
      return yv !== null ? {{x: xv, y: yv}} : null;
    }}).filter(Boolean);
    datasets.push({{
      label: 'prop-off', data: pts, parsing: false,
      borderColor: '#6b7094', borderDash: [6, 4], borderWidth: 2,
      pointRadius: 4, tension: .3, spanGaps: true, backgroundColor: 'transparent',
    }});
  }}
  jList.forEach(j => {{
    if (!met.on) return;
    const pts = xVals.map(x => {{
      const rows = d.filter(r => r.J_r === j && xRndKey(r) === x);
      if (!rows.length) return null;
      const xv = useAct ? avg(rows.map(r => ({{v: xActual(r, 'on')}})), 'v') : x;
      const yv = avg(rows, met.on);
      return (xv !== null && yv !== null) ? {{x: xv, y: yv}} : null;
    }}).filter(Boolean);
    if (!pts.length) return;
    datasets.push({{
      label: `J=${{j}}`, data: pts, parsing: false,
      borderColor: J_COLORS[j], backgroundColor: J_COLORS[j] + '22',
      borderWidth: 2.5, pointRadius: 5, tension: .3, spanGaps: true,
    }});
  }});

  const axLbl = xax + (useAct ? ' (actual) °' : ' (rounded) °');
  if (cChart) cChart.destroy();
  cChart = new Chart(document.getElementById('c-chart'), {{
    type: 'line', data: {{datasets}},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{display: false}},
        tooltip: {{callbacks: {{label: c => `${{c.dataset.label}}: ${{c.parsed.y?.toFixed(4)}}`}}}},
      }},
      scales: {{
        x: {{type:'linear', title:{{display:true,text:axLbl,color:'#6b7094',font:{{size:11,family:"'IBM Plex Mono'"}}}},
             ticks:{{color:'#6b7094',font:{{size:10,family:"'IBM Plex Mono'"}}}},
             grid:{{color:'rgba(255,255,255,0.04)'}}}},
        y: {{title:{{display:true,text:met.lbl,color:'#6b7094',font:{{size:11,family:"'IBM Plex Mono'"}}}},
             ticks:{{color:'#6b7094',font:{{size:10,family:"'IBM Plex Mono'"}},callback:v=>v.toFixed(3)}},
             grid:{{color:'rgba(255,255,255,0.04)'}}}},
      }},
    }},
  }});
}}
function setCmpMetric(k) {{ cMetric = k; renderCmp(); }}

// ── PROPOFF EXPLORER ────────────────────────────────────────────────────────
function renderOff() {{
  buildMetricTabs('o-mtabs', OFF_METRICS, oMetric, 'setOffMetric');

  const xax    = sel('o-xax');
  const colBy  = sel('o-color');
  const vFil   = sel('o-v');
  const aoaFix = sel('o-aoa');
  const aosFix = sel('o-aos');
  const drFix  = sel('o-dr');
  const deFix  = sel('o-de');
  const useAct = document.getElementById('o-actual').checked;

  let d = [...OFF_DATA];
  if (!isAll(vFil))                     d = d.filter(r => r.V_r   === parseFloat(vFil));
  if (xax !== 'AoA' && !isAll(aoaFix)) d = d.filter(r => r.AoA_r === parseFloat(aoaFix));
  if (xax !== 'AoS' && !isAll(aosFix)) d = d.filter(r => r.AoS_r === parseFloat(aosFix));
  if (xax !== 'dR'  && !isAll(drFix))  d = d.filter(r => r.dR    === parseFloat(drFix));
  if (xax !== 'dE'  && !isAll(deFix))  d = d.filter(r => r.dE    === parseFloat(deFix));

  const met = OFF_METRICS.find(m => m.key === oMetric);

  function xRndKey(r) {{
    if (xax==='AoA') return r.AoA_r;
    if (xax==='AoS') return r.AoS_r;
    if (xax==='dR')  return r.dR;
    if (xax==='dE')  return r.dE;
    if (xax==='V')   return r.V_r;
  }}
  function xActVal(r) {{
    if (xax==='AoA') return r.AoA_act;
    if (xax==='AoS') return r.AoS_act;
    if (xax==='dR')  return r.dR;
    if (xax==='dE')  return r.dE;
    if (xax==='V')   return r.V_act;
  }}
  function colorKey(r) {{
    if (colBy==='V')   return String(r.V_r);
    if (colBy==='dE')  return String(r.dE);
    if (colBy==='dR')  return String(r.dR);
    if (colBy==='AoS') return String(r.AoS_r);
  }}

  const palette      = ['#5b8af5','#3ecf8e','#f5a623','#f05252','#a78bfa','#fb7185','#34d399','#fbbf24'];
  const allColorKeys = [...new Set(d.map(colorKey))].sort((a,b) => parseFloat(a) - parseFloat(b));
  const colorMap     = {{}};
  allColorKeys.forEach((k, i) => colorMap[k] = palette[i % palette.length]);

  document.getElementById('o-legend').innerHTML = allColorKeys.map(k =>
    `<span class="leg-item"><span class="lsq" style="background:${{colorMap[k]}}"></span>${{colBy}}=${{k}}°</span>`
  ).join('');

  const xRndVals = [...new Set(d.map(xRndKey))].sort((a,b) => a - b);
  const datasets = [];

  allColorKeys.forEach(ck => {{
    const sub = d.filter(r => colorKey(r) === ck);
    if (!sub.length) return;
    const pts = xRndVals.map(xr => {{
      const rows = sub.filter(r => xRndKey(r) === xr);
      if (!rows.length) return null;
      const actVals = rows.map(xActVal).filter(v => v !== null);
      const xv = useAct && actVals.length
        ? actVals.reduce((s,v)=>s+v,0)/actVals.length
        : xr;
      const yv = avg(rows, met.field);
      return (yv !== null) ? {{x: xv, y: yv}} : null;
    }}).filter(Boolean);
    if (!pts.length) return;
    datasets.push({{
      label: `${{colBy}}=${{ck}}`, data: pts, parsing: false,
      borderColor: colorMap[ck], backgroundColor: colorMap[ck] + '20',
      borderWidth: 2, pointRadius: 3, tension: .3, spanGaps: true,
    }});
  }});

  const axLbl = xax + (useAct ? ' (actual) °' : ' (rounded) °');
  if (oChart) oChart.destroy();
  oChart = new Chart(document.getElementById('o-chart'), {{
    type: 'line', data: {{datasets}},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{display: false}},
        tooltip: {{callbacks: {{label: c => `${{c.dataset.label}}: ${{c.parsed.y?.toFixed(4)}}`}}}},
      }},
      scales: {{
        x: {{type:'linear', title:{{display:true,text:axLbl,color:'#6b7094',font:{{size:11,family:"'IBM Plex Mono'"}}}},
             ticks:{{color:'#6b7094',font:{{size:10,family:"'IBM Plex Mono'"}}}},
             grid:{{color:'rgba(255,255,255,0.04)'}}}},
        y: {{title:{{display:true,text:met.lbl,color:'#6b7094',font:{{size:11,family:"'IBM Plex Mono'"}}}},
             ticks:{{color:'#6b7094',font:{{size:10,family:"'IBM Plex Mono'"}},callback:v=>v.toFixed(3)}},
             grid:{{color:'rgba(255,255,255,0.04)'}}}},
      }},
    }},
  }});
}}
function setOffMetric(k) {{ oMetric = k; renderOff(); }}

// ── STABILITY & CONTROL ────────────────────────────────────────────────────
function renderSC() {{
  const plot   = sel('sc-plot');
  const src    = sel('sc-src');
  const vFil   = sel('sc-v');
  const drFix  = sel('sc-dr');
  const deFix  = sel('sc-de');
  const aosFix = sel('sc-aos');
  const r2min  = parseFloat(sel('sc-r2min')) || 0;

  function fV(d)   {{ return isAll(vFil)  || d.V   === parseFloat(vFil);  }}
  function fDR(d)  {{ return isAll(drFix) || d.dR  === parseFloat(drFix); }}
  function fDE(d)  {{ return isAll(deFix) || d.dE  === parseFloat(deFix); }}
  function fAoS(d) {{ return isAll(aosFix)|| d.AoS === parseFloat(aosFix);}}
  function fR2(d, key) {{ return (d[key] === null || d[key] === undefined) ? false : d[key] >= r2min; }}

  const palette  = ['#5b8af5','#3ecf8e','#f5a623','#f05252','#a78bfa','#fb7185','#34d399','#fbbf24'];
  const offColor = '#94a3b8';

  let datasets = [];
  let xLabel   = 'AoA (°)';
  let yLabel   = plot;
  let metaNote = '';
  let cards_html = '';

  // ── Helper: build one line dataset grouped by groupKey ────────────────
  function buildLines(rows, xKey, yKey, r2Key, groupFn, labelFn, colorFn) {{
    const groups = {{}};
    rows.forEach(r => {{
      const k = groupFn(r);
      if (!groups[k]) groups[k] = [];
      groups[k].push(r);
    }});
    return Object.entries(groups).map(([k, pts]) => {{
      const sorted = pts.sort((a,b) => a[xKey]-b[xKey]);
      const data   = sorted
        .filter(r => r2Key ? fR2(r, r2Key) : true)
        .map(r => ({{x: r[xKey], y: r[yKey]}}))
        .filter(p => p.x !== null && p.y !== null);
      if (!data.length) return null;
      return {{
        label: labelFn(k, pts[0]),
        data, parsing: false,
        borderColor: colorFn(k, pts[0]),
        backgroundColor: colorFn(k, pts[0]) + '25',
        borderWidth: 2.5, pointRadius: 5, tension: .3, spanGaps: true,
      }};
    }}).filter(Boolean);
  }}

  // ── Cn_beta ──────────────────────────────────────────────────────────
  if (plot === 'cn_beta' || plot === 'cy_beta') {{
    const yKey  = plot === 'cn_beta' ? 'Cn_beta' : 'Cy_beta';
    const r2Key = 'r2_cn';
    yLabel = plot === 'cn_beta' ? 'Cnβ = ∂Cn/∂β  [deg⁻¹]' : 'Cyβ = ∂Cy/∂β  [deg⁻¹]';
    metaNote = 'Linear fit of CMyaw vs AoS at each (AoA, J/dE, V, dR) condition. Filter by R² to exclude poor fits.';

    if (src !== 'off') {{
      const d = CN_BETA_ON.filter(r => fV(r) && fDR(r));
      const jList = [...new Set(d.map(r=>r.J))].sort((a,b)=>a-b);
      jList.forEach((j,i) => {{
        const rows = d.filter(r => r.J === j);
        if (!rows.length) return;
        const sorted = rows.sort((a,b)=>a.AoA-b.AoA);
        const pts = sorted.filter(r => r2Key ? fR2(r,r2Key) : true)
                          .map(r => ({{x:r.AoA, y:r[yKey]}})).filter(p=>p.y!==null);
        if (!pts.length) return;
        datasets.push({{label:`prop-on J=${{j}}`, data:pts, parsing:false,
          borderColor:palette[i%palette.length], backgroundColor:palette[i%palette.length]+'25',
          borderWidth:2.5, pointRadius:6, tension:.3}});
      }});
    }}
    if (src !== 'on') {{
      const d = CN_BETA_OFF.filter(r => fV(r) && fDR(r) && fDE(r));
      const vList = [...new Set(d.map(r=>r.V))].sort((a,b)=>a-b);
      vList.forEach((v,i) => {{
        const rows = d.filter(r => r.V === v);
        const sorted = rows.sort((a,b)=>a.AoA-b.AoA);
        const pts = sorted.filter(r => r2Key ? fR2(r,r2Key) : true)
                          .map(r => ({{x:r.AoA, y:r[yKey]}})).filter(p=>p.y!==null);
        if (!pts.length) return;
        datasets.push({{label:`prop-off V=${{v}}`, data:pts, parsing:false,
          borderColor:SC_OFF_PALETTE[i%SC_OFF_PALETTE.length],
          backgroundColor:SC_OFF_PALETTE[i%SC_OFF_PALETTE.length]+'25',
          borderWidth:1.5, pointRadius:3, tension:.3, borderDash:[4,3]}});
      }});
    }}
    // summary cards — mean Cnbeta per J (propOn)
    if (src !== 'off') {{
      const d = CN_BETA_ON.filter(r => fV(r) && fDR(r) && (r2Key ? fR2(r,r2Key) : true));
      const jList = [...new Set(d.map(r=>r.J))].sort((a,b)=>a-b);
      cards_html = jList.map((j,i) => {{
        const vals = d.filter(r=>r.J===j).map(r=>r[yKey]).filter(v=>v!==null);
        if (!vals.length) return '';
        const mean = vals.reduce((s,v)=>s+v,0)/vals.length;
        return `<div class="card">
          <div class="card-lbl"><span class="dot" style="background:${{palette[i%palette.length]}}"></span>J=${{j}} prop-on</div>
          <div class="card-val">${{mean.toFixed(5)}}</div>
          <div class="card-sub">mean ${{plot==='cn_beta'?'Cnβ':'Cyβ'}}</div></div>`;
      }}).join('');
    }}
  }}

  // ── Cn_dR / Cy_dR / Cr_dR ────────────────────────────────────────────
  else if (['cn_dr','cy_dr','cr_dr'].includes(plot)) {{
    const yKey  = plot === 'cn_dr' ? 'Cn_dR' : plot === 'cy_dr' ? 'Cy_dR' : 'Cr_dR';
    const r2Key = 'r2_cn';
    yLabel = plot === 'cn_dr' ? 'Cn_δR = ∂Cn/∂δR  [deg⁻¹]'
           : plot === 'cy_dr' ? 'Cy_δR = ∂Cy/∂δR  [deg⁻¹]'
           : 'Cr_δR = ∂Cl/∂δR  [deg⁻¹]';
    metaNote = 'Linear fit of CMyaw vs dR at each (AoA, J, V, AoS) condition.';

    if (src !== 'off') {{
      const d = CN_DR_ON.filter(r => fV(r) && fAoS(r));
      const jList = [...new Set(d.map(r=>r.J))].sort((a,b)=>a-b);
      jList.forEach((j,i) => {{
        const rows = d.filter(r=>r.J===j).sort((a,b)=>a.AoA-b.AoA);
        const pts  = rows.filter(r=>fR2(r,r2Key)).map(r=>({{x:r.AoA,y:r[yKey]}})).filter(p=>p.y!==null);
        if (!pts.length) return;
        datasets.push({{label:`prop-on J=${{j}}`, data:pts, parsing:false,
          borderColor:palette[i%palette.length], backgroundColor:palette[i%palette.length]+'25',
          borderWidth:2.5, pointRadius:6, tension:.3}});
      }});
    }}
    if (src !== 'on') {{
      const d = CN_DR_OFF.filter(r => fV(r) && fAoS(r) && fDE(r));
      const vList = [...new Set(d.map(r=>r.V))].sort((a,b)=>a-b);
      vList.forEach((v,i) => {{
        const rows = d.filter(r=>r.V===v).sort((a,b)=>a.AoA-b.AoA);
        const pts  = rows.filter(r=>fR2(r,r2Key)).map(r=>({{x:r.AoA,y:r[yKey]}})).filter(p=>p.y!==null);
        if (!pts.length) return;
        datasets.push({{label:`prop-off V=${{v}}`, data:pts, parsing:false,
          borderColor:SC_OFF_PALETTE[i%SC_OFF_PALETTE.length],
          backgroundColor:SC_OFF_PALETTE[i%SC_OFF_PALETTE.length]+'25',
          borderWidth:1.5, pointRadius:3, tension:.3, borderDash:[4,3]}});
      }});
    }}
    if (src !== 'off') {{
      const d = CN_DR_ON.filter(r=>fV(r)&&fAoS(r)&&fR2(r,'r2_cn'));
      const jList = [...new Set(d.map(r=>r.J))].sort((a,b)=>a-b);
      cards_html = jList.map((j,i) => {{
        const vals = d.filter(r=>r.J===j).map(r=>r[yKey]).filter(v=>v!==null);
        if (!vals.length) return '';
        const mean = vals.reduce((s,v)=>s+v,0)/vals.length;
        return `<div class="card">
          <div class="card-lbl"><span class="dot" style="background:${{palette[i%palette.length]}}"></span>J=${{j}}</div>
          <div class="card-val">${{mean.toFixed(5)}}</div>
          <div class="card-sub">mean ${{yKey}}</div></div>`;
      }}).join('');
    }}
  }}

  // ── Drag polar ────────────────────────────────────────────────────────
  else if (plot === 'polar') {{
    xLabel = 'CD  (FINAL)';
    yLabel = 'CL  (FINAL)';
    metaNote = 'Drag polar: CL vs CD. All individual data points.';

    if (src !== 'off') {{
      const d = POLAR_ON.filter(r => fV(r) && (isAll(drFix)||r.dR===parseFloat(drFix)));
      const jList = [...new Set(d.map(r=>r.J))].sort((a,b)=>a-b);
      jList.forEach((j,i) => {{
        const pts = d.filter(r=>r.J===j).map(r=>({{x:r.CD,y:r.CL}})).filter(p=>p.x!==null&&p.y!==null);
        if (!pts.length) return;
        datasets.push({{label:`prop-on J=${{j}}`, data:pts, parsing:false,
          borderColor:palette[i%palette.length], backgroundColor:palette[i%palette.length]+'40',
          borderWidth:0, pointRadius:5, showLine:false}});
      }});
    }}
    if (src !== 'on') {{
      const d = POLAR_OFF.filter(r => fV(r) && fDE(r) && (isAll(drFix)||r.dR===parseFloat(drFix)));
      const pts = d.map(r=>({{x:r.CD,y:r.CL}})).filter(p=>p.x!==null&&p.y!==null);
      if (pts.length) datasets.push({{label:'prop-off', data:pts, parsing:false,
        borderColor:offColor, backgroundColor:offColor+'30',
        borderWidth:0, pointRadius:2, showLine:false}});
    }}
    metaNote = 'Each point is one measurement. Filter by V and dR to isolate conditions.';
  }}

  // ── Trim dR / L/D / CL at trim ───────────────────────────────────────
  else if (['trim_dr','trim_ld','trim_cl'].includes(plot)) {{
    const yKey = plot==='trim_dr' ? 'dR_trim' : plot==='trim_ld' ? 'LD_trim' : 'CL_trim';
    yLabel = plot==='trim_dr' ? 'Trim dR (°)' : plot==='trim_ld' ? 'L/D at trim' : 'CL at trim';
    metaNote = 'Trim condition: dR at which CMyaw = 0, found by linear interpolation. AoS=0 recommended.';

    const d = TRIM_ROWS.filter(r => fV(r) && fAoS(r));
    const jList = [...new Set(d.map(r=>r.J))].sort((a,b)=>a-b);
    jList.forEach((j,i) => {{
      const rows = d.filter(r=>r.J===j).sort((a,b)=>a.AoA-b.AoA);
      const pts  = rows.map(r=>({{x:r.AoA, y:r[yKey]}})).filter(p=>p.x!==null&&p.y!==null);
      if (!pts.length) return;
      datasets.push({{label:`J=${{j}}`, data:pts, parsing:false,
        borderColor:palette[i%palette.length], backgroundColor:palette[i%palette.length]+'25',
        borderWidth:2.5, pointRadius:6, tension:.3}});
    }});
    // cards
    cards_html = jList.map((j,i) => {{
      const vals = d.filter(r=>r.J===j).map(r=>r[yKey]).filter(v=>v!==null);
      if (!vals.length) return '';
      const mean = vals.reduce((s,v)=>s+v,0)/vals.length;
      return `<div class="card">
        <div class="card-lbl"><span class="dot" style="background:${{palette[i%palette.length]}}"></span>J=${{j}}</div>
        <div class="card-val">${{mean.toFixed(3)}}</div>
        <div class="card-sub">mean ${{yLabel}}</div></div>`;
    }}).join('');
  }}

  // legend
  const legEl = document.getElementById('sc-legend');
  legEl.innerHTML = datasets.map(ds => {{
    const isDash = (ds.borderDash && ds.borderDash.length);
    return `<span class="leg-item">` +
      (isDash
        ? `<span style="display:inline-block;width:18px;height:0;border-top:2px dashed ${{ds.borderColor}};"></span>`
        : `<span class="lsq" style="background:${{ds.borderColor}}"></span>`) +
      `${{ds.label}}</span>`;
  }}).join('');

  document.getElementById('sc-cards').innerHTML = cards_html;
  document.getElementById('sc-meta').textContent = metaNote;

  const scType = (plot === 'polar') ? 'scatter' : 'line';
  if (scChart) scChart.destroy();
  scChart = new Chart(document.getElementById('sc-chart'), {{
    type: scType,
    data: {{datasets}},
    options: {{
      responsive:true, maintainAspectRatio:false,
      plugins:{{
        legend:{{display:false}},
        tooltip:{{callbacks:{{label:c=>`${{c.dataset.label}}: (${{c.parsed.x?.toFixed(4)}}, ${{c.parsed.y?.toFixed(5)}}) `}}}},
      }},
      scales:{{
        x:{{type:'linear', title:{{display:true,text:xLabel,color:'#6b7094',font:{{size:11,family:"'IBM Plex Mono'"}}}},
             ticks:{{color:'#6b7094',font:{{size:10,family:"'IBM Plex Mono'"}}}},
             grid:{{color:'rgba(255,255,255,0.04)'}}}},
        y:{{title:{{display:true,text:yLabel,color:'#6b7094',font:{{size:11,family:"'IBM Plex Mono'"}}}},
             ticks:{{color:'#6b7094',font:{{size:10,family:"'IBM Plex Mono'"}},callback:v=>v.toFixed(4)}},
             grid:{{color:'rgba(255,255,255,0.04)'}}}},
      }},
    }},
  }});
}}

// ── EVENT LISTENERS ─────────────────────────────────────────────────────────
['c-xax','c-j','c-v','c-aoa','c-aos','c-dr'].forEach(id =>
  document.getElementById(id).addEventListener('change', renderCmp));
document.getElementById('c-actual').addEventListener('change', renderCmp);

['o-xax','o-color','o-v','o-aoa','o-aos','o-dr','o-de'].forEach(id =>
  document.getElementById(id).addEventListener('change', renderOff));
document.getElementById('o-actual').addEventListener('change', renderOff);

['sc-plot','sc-src','sc-v','sc-dr','sc-de','sc-aos','sc-r2min'].forEach(id =>
  document.getElementById(id).addEventListener('change', renderSC));

renderCmp();
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  Output written -> {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive AE4115 aerodynamic data explorer HTML."
    )
    parser.add_argument("--propon",  default=DEFAULT_PROPON,
                        help=f"Path to propOn_final.csv  (default: {DEFAULT_PROPON})")
    parser.add_argument("--propoff", default=DEFAULT_PROPOFF,
                        help=f"Path to propOff_final.csv (default: {DEFAULT_PROPOFF})")
    parser.add_argument("--out",     default=DEFAULT_OUT,
                        help=f"Output HTML file          (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    propon_path  = Path(args.propon)
    propoff_path = Path(args.propoff)
    out_path     = Path(args.out)

    for p in (propon_path, propoff_path):
        if not p.exists():
            sys.exit(f"ERROR: file not found: {p}")

    print(f"\nLoading data...")
    print(f"  propOn  : {propon_path}")
    print(f"  propOff : {propoff_path}")

    (cmp_rows, off_rows, meta, j_colors, v_colors,
     cn_beta_on, cn_beta_off, cn_dr_on, cn_dr_off,
     polar_on, polar_off, trim_rows, sc_meta) = load_and_build(propon_path, propoff_path)

    generate_html(cmp_rows, off_rows, meta, j_colors, v_colors,
                  cn_beta_on, cn_beta_off, cn_dr_on, cn_dr_off,
                  polar_on, polar_off, trim_rows, sc_meta, out_path)
    print("Done.\n")


if __name__ == "__main__":
    main()