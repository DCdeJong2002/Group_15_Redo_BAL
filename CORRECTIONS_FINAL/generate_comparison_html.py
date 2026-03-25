"""
generate_comparison_html.py
----------------------------
Reads the most recent propOn_final.csv and propOff_final.csv, merges them
on (AoA_round, AoS_round, V_round, dR, dE), and writes a fully self-contained
interactive HTML file for comparing _FINAL aerodynamic coefficients.

Usage
-----
    python generate_comparison_html.py

    # Custom paths / output location:
    python generate_comparison_html.py \
        --propon  path/to/propOn_final.csv \
        --propoff path/to/propOff_final.csv \
        --out     path/to/output.html

The script expects both CSVs to have these columns:
    Index  : AoA_round, AoS_round, V_round, dR, dE
    propOn : J_round  (advance ratio)
    Both   : CL_FINAL, CD_FINAL, CMpitch_FINAL,
             CYaw_FINAL, CMroll_FINAL, CMyaw_FINAL
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
INDEX_COLS = ["AoA_round", "AoS_round", "V_round", "dR", "dE"]
FINAL_COLS = [
    "CL_FINAL", "CD_FINAL", "CMpitch_FINAL",
    "CYaw_FINAL", "CMroll_FINAL", "CMyaw_FINAL",
]
DEFAULT_PROPON  = "propOn_final.csv"
DEFAULT_PROPOFF = "propOff_final.csv"
DEFAULT_OUT     = "propon_propoff_comparison.html"


# ---------------------------------------------------------------------------
# Data loading & merging
# ---------------------------------------------------------------------------
def load_and_merge(propon_path: Path, propoff_path: Path) -> list:
    df_on  = pd.read_csv(propon_path)
    df_off = pd.read_csv(propoff_path)

    # Validate required columns
    for col in INDEX_COLS + ["J_round"] + FINAL_COLS:
        if col not in df_on.columns:
            sys.exit(f"ERROR: propOn CSV is missing column '{col}'")
    for col in INDEX_COLS + FINAL_COLS:
        if col not in df_off.columns:
            sys.exit(f"ERROR: propOff CSV is missing column '{col}'")

    # Aggregate propOff over repeats at the same index
    df_off_agg = (
        df_off[INDEX_COLS + FINAL_COLS]
        .groupby(INDEX_COLS, as_index=False)
        .mean()
    )

    df_on_sel = df_on[INDEX_COLS + ["J_round"] + FINAL_COLS].copy()

    merged = df_on_sel.merge(
        df_off_agg,
        on=INDEX_COLS,
        how="inner",
        suffixes=("_on", "_off"),
    )

    if merged.empty:
        sys.exit("ERROR: no matching rows found after merging on index columns.")

    print(f"  propOn  rows : {len(df_on)}")
    print(f"  propOff rows : {len(df_off)}  (aggregated to {len(df_off_agg)} unique index combos)")
    print(f"  Merged  rows : {len(merged)}")

    rows = []
    for _, r in merged.iterrows():
        rows.append({
            "AoA":      round(float(r.AoA_round),  1),
            "AoS":      round(float(r.AoS_round),  1),
            "V":        round(float(r.V_round),     0),
            "dR":       round(float(r.dR),          1),
            "dE":       round(float(r.dE),          1),
            "J":        round(float(r.J_round),     1),
            "CL_off":     round(float(r.CL_FINAL_off),      5),
            "CL_on":      round(float(r.CL_FINAL_on),       5),
            "CD_off":     round(float(r.CD_FINAL_off),      5),
            "CD_on":      round(float(r.CD_FINAL_on),       5),
            "CM_off":     round(float(r.CMpitch_FINAL_off), 5),
            "CM_on":      round(float(r.CMpitch_FINAL_on),  5),
            "CYaw_off":   round(float(r.CYaw_FINAL_off),    5),
            "CYaw_on":    round(float(r.CYaw_FINAL_on),     5),
            "CMroll_off": round(float(r.CMroll_FINAL_off),  5),
            "CMroll_on":  round(float(r.CMroll_FINAL_on),   5),
            "CMyaw_off":  round(float(r.CMyaw_FINAL_off),   5),
            "CMyaw_on":   round(float(r.CMyaw_FINAL_on),    5),
        })

    return rows


def summarise(rows: list) -> None:
    """Print mean deltas by J to stdout as a sanity check."""
    df = pd.DataFrame(rows)
    df["dCL"]   = df.CL_on   - df.CL_off
    df["dCD"]   = df.CD_on   - df.CD_off
    df["dCM"]   = df.CM_on   - df.CM_off
    df["dCYaw"] = df.CYaw_on - df.CYaw_off
    print("\n  Mean deltas by J (prop-on minus prop-off, _FINAL columns):")
    print(
        df.groupby("J")[["dCL", "dCD", "dCM", "dCYaw"]]
        .mean()
        .round(4)
        .to_string()
    )
    print()


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PropOn vs PropOff — FINAL coefficient comparison</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#f8f9fa;color:#1a1a2e;padding:1.5rem 2rem 3rem;}
  h1{font-size:15px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;
     color:#555;margin-bottom:1.5rem;padding-bottom:.6rem;border-bottom:2px solid #e0e0e0;}
  .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;
            background:#fff;border:1px solid #e8e8e8;border-radius:8px;
            padding:.7rem 1rem;margin-bottom:1rem;}
  .controls label{font-size:11px;color:#888;white-space:nowrap;font-weight:500;
                  text-transform:uppercase;letter-spacing:.05em;}
  select{font-size:12px;padding:4px 8px;border:1px solid #d0d0d0;border-radius:5px;
         background:#fafafa;color:#333;cursor:pointer;}
  select:hover{border-color:#999;}
  .tabs{display:flex;gap:3px;margin-bottom:1rem;flex-wrap:wrap;}
  .tab{font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;
       padding:5px 14px;border:1px solid #d8d8d8;border-radius:5px;
       cursor:pointer;background:#fff;color:#777;transition:all .15s;}
  .tab:hover{background:#f0f0f0;color:#333;}
  .tab.active{background:#1a1a2e;color:#fff;border-color:#1a1a2e;}
  .cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-bottom:1rem;}
  .card{background:#fff;border:1px solid #e8e8e8;border-radius:7px;padding:10px 13px;}
  .card-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;
              color:#aaa;margin-bottom:4px;display:flex;align-items:center;gap:5px;}
  .dot{width:8px;height:8px;border-radius:2px;flex-shrink:0;}
  .card-val{font-size:18px;font-weight:700;font-variant-numeric:tabular-nums;}
  .card-sub{font-size:10px;color:#bbb;margin-top:2px;}
  .pos{color:#16a34a;} .neg{color:#dc2626;}
  .legend{display:flex;gap:16px;flex-wrap:wrap;font-size:11px;color:#888;margin-bottom:8px;}
  .legend span{display:flex;align-items:center;gap:5px;}
  .lsq{width:10px;height:10px;border-radius:2px;display:inline-block;flex-shrink:0;}
  .chart-outer{background:#fff;border:1px solid #e8e8e8;border-radius:8px;padding:1rem 1rem .5rem;}
  .chart-wrap{position:relative;width:100%%;height:340px;}
  .meta{font-size:10px;color:#bbb;margin-top:.7rem;text-align:right;}
</style>
</head>
<body>

<h1>PropOn vs PropOff — FINAL aerodynamic coefficients</h1>

<div class="controls">
  <label>x-axis</label>
  <select id="xax">
    <option value="AoA">AoA sweep</option>
    <option value="AoS">AoS sweep</option>
    <option value="dR">dR sweep</option>
    <option value="dE">dE sweep</option>
  </select>
  <label>J</label>
  <select id="jSel">
    <option value="all">all</option>
    %(j_options)s
  </select>
  <label>V (m/s)</label>
  <select id="vSel">
    <option value="all">all</option>
    %(v_options)s
  </select>
  <label>fix AoA</label>
  <select id="aoaFix">
    <option value="all">&#8212;</option>
    %(aoa_options)s
  </select>
  <label>fix AoS</label>
  <select id="aosFix">
    <option value="all">&#8212;</option>
    %(aos_options)s
  </select>
  <label>fix dR</label>
  <select id="drFix">
    <option value="all">&#8212;</option>
    %(dr_options)s
  </select>
  <label>fix dE</label>
  <select id="deFix">
    <option value="all">&#8212;</option>
    %(de_options)s
  </select>
</div>

<div class="tabs" id="tabBar"></div>
<div class="cards" id="summaryCards"></div>

<div class="chart-outer">
  <div class="legend" id="legend"></div>
  <div class="chart-wrap"><canvas id="mainChart"></canvas></div>
  <p class="meta">All values from _FINAL columns &nbsp;|&nbsp; indexed on AoA, AoS, V, dR, dE &nbsp;|&nbsp; prop-off averaged over repeats</p>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const DATA       = %(data_json)s;
const J_COLORS   = %(j_colors_json)s;
const J_LIST     = %(j_list_json)s;
const METRICS = [
  {key:'CL',     label:'CL',      on:'CL_on',      off:'CL_off'},
  {key:'CD',     label:'CD',      on:'CD_on',      off:'CD_off'},
  {key:'CM',     label:'CMpitch', on:'CM_on',       off:'CM_off'},
  {key:'CYaw',   label:'CYaw',    on:'CYaw_on',    off:'CYaw_off'},
  {key:'CMroll', label:'CMroll',  on:'CMroll_on',  off:'CMroll_off'},
  {key:'CMyaw',  label:'CMyaw',   on:'CMyaw_on',   off:'CMyaw_off'},
];

let activeMetric = 'CL';
let chart = null;

function buildTabs() {
  document.getElementById('tabBar').innerHTML = METRICS.map(m =>
    `<button class="tab${m.key === activeMetric ? ' active' : ''}"
       onclick="setMetric('${m.key}')">${m.label}</button>`
  ).join('');
}
function setMetric(k) { activeMetric = k; buildTabs(); render(); }

function buildLegend() {
  const visibleJ = J_LIST.filter(j => DATA.some(r => r.J === j));
  document.getElementById('legend').innerHTML =
    `<span><span style="display:inline-block;width:20px;height:0;
       border-top:2px dashed #999;"></span>&nbsp;prop-off</span>` +
    visibleJ.map(j =>
      `<span><span class="lsq" style="background:${J_COLORS[j]}"></span>prop-on J=${j}</span>`
    ).join('');
}

function getFiltered() {
  const xax    = document.getElementById('xax').value;
  const j      = document.getElementById('jSel').value;
  const v      = document.getElementById('vSel').value;
  const aoaFix = document.getElementById('aoaFix').value;
  const aosFix = document.getElementById('aosFix').value;
  const drFix  = document.getElementById('drFix').value;
  const deFix  = document.getElementById('deFix').value;
  let d = [...DATA];
  if (j !== 'all') d = d.filter(r => r.J === parseFloat(j));
  if (v !== 'all') d = d.filter(r => r.V === parseFloat(v));
  if (xax !== 'AoA' && aoaFix !== 'all') d = d.filter(r => r.AoA === parseFloat(aoaFix));
  if (xax !== 'AoS' && aosFix !== 'all') d = d.filter(r => r.AoS === parseFloat(aosFix));
  if (xax !== 'dR'  && drFix  !== 'all') d = d.filter(r => r.dR  === parseFloat(drFix));
  if (xax !== 'dE'  && deFix  !== 'all') d = d.filter(r => r.dE  === parseFloat(deFix));
  return {data: d, xax};
}

function avg(arr, key) {
  const vals = arr.map(r => r[key]).filter(v => v !== null && !isNaN(v));
  return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
}

function updateCards(data) {
  const met = METRICS.find(m => m.key === activeMetric);
  document.getElementById('summaryCards').innerHTML = J_LIST.map(j => {
    const sub = data.filter(r => r.J === j);
    if (!sub.length) return '';
    const d = avg(sub, met.on) - avg(sub, met.off);
    const cls = d >= 0 ? 'pos' : 'neg';
    return `<div class="card">
      <div class="card-label">
        <span class="dot" style="background:${J_COLORS[j]}"></span>J = ${j}
      </div>
      <div class="card-val ${cls}">${d >= 0 ? '+' : ''}${d.toFixed(4)}</div>
      <div class="card-sub">mean &Delta;${met.label}</div>
    </div>`;
  }).join('');
}

function render() {
  const {data, xax} = getFiltered();
  const met = METRICS.find(m => m.key === activeMetric);
  updateCards(data);

  const xVals = [...new Set(data.map(r => r[xax]))].sort((a, b) => a - b);
  const labels = xVals.map(v => v + '°');
  const datasets = [];

  const offPts = xVals.map(x => {
    const rows = data.filter(r => r[xax] === x);
    return rows.length ? avg(rows, met.off) : null;
  });
  datasets.push({
    label: 'prop-off', data: offPts,
    borderColor: '#999', borderDash: [6, 4], borderWidth: 2,
    pointRadius: 4, pointStyle: 'circle', tension: 0.3,
    spanGaps: true, backgroundColor: 'transparent',
  });

  J_LIST.forEach(j => {
    const pts = xVals.map(x => {
      const rows = data.filter(r => r.J === j && r[xax] === x);
      return rows.length ? avg(rows, met.on) : null;
    });
    if (pts.every(p => p === null)) return;
    datasets.push({
      label: `J=${j}`, data: pts,
      borderColor: J_COLORS[j], backgroundColor: J_COLORS[j] + '22',
      borderWidth: 2.5, pointRadius: 5, tension: 0.3, spanGaps: true,
    });
  });

  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('mainChart'), {
    type: 'line',
    data: {labels, datasets},
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: {display: false},
        tooltip: {callbacks: {label: c => `${c.dataset.label}: ${c.parsed.y?.toFixed(4)}`}},
      },
      scales: {
        x: {title: {display: true, text: xax + ' (deg)', font: {size: 12}},
            ticks: {autoSkip: false}},
        y: {title: {display: true, text: met.label + '  (_FINAL)', font: {size: 12}},
            grid: {color: 'rgba(0,0,0,0.05)'},
            ticks: {callback: v => v.toFixed(3)}},
      },
    },
  });
}

['xax', 'jSel', 'vSel', 'aoaFix', 'aosFix', 'drFix', 'deFix'].forEach(id => {
  document.getElementById(id).addEventListener('change', render);
});

buildTabs();
buildLegend();
render();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def make_options(values: list, fmt: str = "{}") -> str:
    return "\n    ".join(
        f'<option value="{v}">{fmt.format(v)}</option>'
        for v in sorted(values)
    )


def generate_html(rows: list, out_path: Path) -> None:
    df = pd.DataFrame(rows)

    j_vals   = sorted(df["J"].unique())
    v_vals   = sorted(df["V"].unique())
    aoa_vals = sorted(df["AoA"].unique())
    aos_vals = sorted(df["AoS"].unique())
    dr_vals  = sorted(df["dR"].unique())
    de_vals  = sorted(df["dE"].unique())

    palette = ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30",
               "#7F77DD", "#D4537E", "#639922", "#BA7517"]
    j_colors = {j: palette[i % len(palette)] for i, j in enumerate(j_vals)}

    html = HTML_TEMPLATE % dict(
        data_json     = json.dumps(rows, separators=(",", ":")),
        j_colors_json = json.dumps(j_colors),
        j_list_json   = json.dumps(j_vals),
        j_options     = make_options(j_vals),
        v_options     = make_options(v_vals, "{:.0f}"),
        aoa_options   = make_options(aoa_vals, "{}°"),
        aos_options   = make_options(aos_vals, "{}°"),
        dr_options    = make_options(dr_vals,  "{}°"),
        de_options    = make_options(de_vals,  "{}°"),
    )

    out_path.write_text(html, encoding="utf-8")
    print(f"  Output written -> {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate interactive PropOn vs PropOff HTML comparison."
    )
    parser.add_argument(
        "--propon",  default=DEFAULT_PROPON,
        help=f"Path to propOn_final.csv  (default: {DEFAULT_PROPON})"
    )
    parser.add_argument(
        "--propoff", default=DEFAULT_PROPOFF,
        help=f"Path to propOff_final.csv (default: {DEFAULT_PROPOFF})"
    )
    parser.add_argument(
        "--out",     default=DEFAULT_OUT,
        help=f"Output HTML file          (default: {DEFAULT_OUT})"
    )
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

    rows = load_and_merge(propon_path, propoff_path)
    summarise(rows)
    generate_html(rows, out_path)
    print("Done.\n")


if __name__ == "__main__":
    main()