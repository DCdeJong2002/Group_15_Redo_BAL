"""
plot_CT_props_delta.py
----------------------
Plots CT_props_delta (experimental delta method), CT_props_total_BEM, and
CT_props_total_EXP vs J for AoA_round = 2.5 deg at V_round = 20 and 40 m/s.

Three CT estimates are shown for comparison:
  - CT_props_delta    : derived from the prop-on minus prop-off axial force
                        difference, converted to the standard CT definition.
                        This is the most physically direct measurement.
  - CT_props_total_BEM: n_props * CT_bem(J) — the BEM polynomial prediction.
  - CT_props_total_EXP: n_props * CT_exp(J) — interpolated from WebPlotDigitizer
                        experimental CT-J curves.

Usage
-----
    python plot_CT_props_delta.py

The script reads PROPON_PATH (set at the top) which must already contain the
CT_props_delta, CT_props_total_BEM, and CT_props_total_EXP columns.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── User config ───────────────────────────────────────────────────────────────
PROPON_PATH = BASE_DIR / "results_propOn_FINAL" / "propOn_final.csv"
AOA_TARGET  = 2.5          # AoA_round to filter on [deg]
V_TARGETS   = [20, 40]     # V_round values to plot [m/s]
COLORS      = {20: "#4C72B0", 40: "#55A868"}
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLS = [
    "V_round", "AoA_round", "J_round", "J", "dR", "dE",
    "CT_props_delta", "CT_props_total_BEM", "CT_props_total_EXP",
    "propoff_match_found",
]


def load_and_check(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}")
        sys.exit(1)
    return df


def aggregate(df_filt, value_col):
    """Group by (V_round, J_round), return mean J and mean value."""
    return (
        df_filt
        .groupby(["V_round", "J_round"], as_index=False)
        .agg(J_mean=(("J"), "mean"), val=(value_col, "mean"), n=(value_col, "count"))
        .sort_values(["V_round", "J_mean"])
    )


def main():
    df = load_and_check(PROPON_PATH)

    # ------------------------------------------------------------------
    # Delta method filter: dR==0, dE==0, AoA=2.5, propoff matched
    # These are the clean baseline prop-on runs with a prop-off match
    # ------------------------------------------------------------------
    mask_delta = (
        (df["AoA_round"] == AOA_TARGET) &
        (df["V_round"].isin(V_TARGETS)) &
        (df["dR"] == 0) &
        (df["dE"] == 0) &
        (df["propoff_match_found"] == True) &
        (df["CT_props_delta"].notna())
    )

    # ------------------------------------------------------------------
    # BEM / EXP filter: dR==0, dE==0, AoA=2.5
    # These don't need a propoff match — they come from the thrust
    # separation methods which run on every row
    # ------------------------------------------------------------------
    mask_bem_exp = (
        (df["AoA_round"] == AOA_TARGET) &
        (df["V_round"].isin(V_TARGETS)) &
        (df["dR"] == 0) &
        (df["dE"] == 0) &
        (df["CT_props_total_BEM"].notna())
    )

    df_delta   = df[mask_delta].copy()
    df_bem_exp = df[mask_bem_exp].copy()

    if df_delta.empty:
        print(f"No delta data for AoA_round={AOA_TARGET}, V_round in {V_TARGETS}.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Aggregate — one point per (V_round, J_round)
    # ------------------------------------------------------------------
    agg_delta = aggregate(df_delta,   "CT_props_delta")
    agg_bem   = aggregate(df_bem_exp, "CT_props_total_BEM")
    agg_exp   = aggregate(df_bem_exp, "CT_props_total_EXP")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for V in V_TARGETS:
        c = COLORS[V]

        # Delta method — filled circles, solid line
        g_d = agg_delta[agg_delta["V_round"] == V]
        if not g_d.empty:
            ax.plot(g_d["J_mean"], g_d["val"], color=c, lw=2,
                    marker="o", ms=7, zorder=5,
                    label=f"$V_\\infty$ = {V} m/s  (delta method)")

        # BEM — dashed line, no marker
        g_b = agg_bem[agg_bem["V_round"] == V]
        if not g_b.empty:
            ax.plot(g_b["J_mean"], g_b["val"], color=c, lw=1.5,
                    ls="--", zorder=4,
                    label=f"$V_\\infty$ = {V} m/s  (BEM polynomial)")

        # EXP — dotted line, no marker
        g_e = agg_exp[agg_exp["V_round"] == V]
        if not g_e.empty:
            ax.plot(g_e["J_mean"], g_e["val"], color=c, lw=1.5,
                    ls=":", zorder=4,
                    label=f"$V_\\infty$ = {V} m/s  (exp. CT curve)")

    ax.set_xlabel("Advance ratio $J$", fontsize=12)
    ax.set_ylabel("Propeller thrust coefficient $C_{T,\\mathrm{props}}$", fontsize=12)
    ax.set_title(
        f"$C_{{T,\\mathrm{{props}}}}$ vs $J$  —  "
        f"$\\alpha$ = {AOA_TARGET}°,  "
        f"$V_\\infty$ = {V_TARGETS[0]} & {V_TARGETS[1]} m/s\n"
        f"Comparison: delta method vs BEM polynomial vs exp. CT curve",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("CT_props_delta_vs_J.png", dpi=150)
    plt.show()
    print("Saved CT_props_delta_vs_J.png")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\nDelta method values:")
    print(agg_delta[["V_round","J_round","J_mean","val","n"]].rename(
        columns={"val":"CT_props_delta"}).round(5).to_string(index=False))

    print("\nBEM polynomial values (same J points):")
    print(agg_bem[["V_round","J_round","J_mean","val"]].rename(
        columns={"val":"CT_props_total_BEM"}).round(5).to_string(index=False))

    print("\nExp. CT curve values (same J points):")
    print(agg_exp[["V_round","J_round","J_mean","val"]].rename(
        columns={"val":"CT_props_total_EXP"}).round(5).to_string(index=False))


if __name__ == "__main__":
    main()