"""
generate_explorer_html.py
--------------------------
Reads propOn_final.csv and propOff_final.csv, merges them on the rounded
index columns, and writes a fully self-contained interactive HTML explorer.

The HTML has two pages:
    1. PropOn vs PropOff  — comparison of FINAL coefficients and corrections,
                            coloured by J value, with prop-off as dashed reference.
    2. PropOff Explorer   — full propOff dataset, colour-by selector, polar
                            and correction metrics, actual/rounded x-axis toggle.

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

    return cmp_rows, off_rows, meta, j_colors, v_colors


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def generate_html(cmp_rows, off_rows, meta, j_colors, v_colors, out_path: Path):

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
</style>
</head>
<body>

<div class="topbar">
  <span class="topbar-title">AE4115 &nbsp;/&nbsp; Data Explorer</span>
  <button class="nav-tab active" onclick="switchPage('cmp',this)">PropOn vs PropOff</button>
  <button class="nav-tab" onclick="switchPage('off',this)">PropOff Explorer</button>
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

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const CMP_DATA  = {json.dumps(cmp_rows, separators=(',', ':'))};
const OFF_DATA  = {json.dumps(off_rows,  separators=(',', ':'))};
const J_COLORS  = {json.dumps(j_colors)};
const META      = {json.dumps(meta)};

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

let cChart = null, oChart = null;
let cMetric = 'CL', oMetric = 'CL';

function switchPage(id, btn) {{
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  btn.classList.add('active');
  if (id === 'cmp') renderCmp(); else renderOff();
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

// ── EVENT LISTENERS ─────────────────────────────────────────────────────────
['c-xax','c-j','c-v','c-aoa','c-aos','c-dr'].forEach(id =>
  document.getElementById(id).addEventListener('change', renderCmp));
document.getElementById('c-actual').addEventListener('change', renderCmp);

['o-xax','o-color','o-v','o-aoa','o-aos','o-dr','o-de'].forEach(id =>
  document.getElementById(id).addEventListener('change', renderOff));
document.getElementById('o-actual').addEventListener('change', renderOff);

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

    cmp_rows, off_rows, meta, j_colors, v_colors = load_and_build(propon_path, propoff_path)
    generate_html(cmp_rows, off_rows, meta, j_colors, v_colors, out_path)
    print("Done.\n")


if __name__ == "__main__":
    main()