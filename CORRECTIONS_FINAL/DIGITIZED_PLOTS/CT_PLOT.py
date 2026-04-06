"""
plot_CT_comparison.py
---------------------
Plots the BEM polynomial CT(J) alongside:
  - raw WebPlotDigitizer experimental data
  - linear interpolation within the digitized range
  - linear extrapolation for V=20 and V=40 only, clamped to J=1.6-2.8

V=30 and V=50 are shown interpolation-only (no extrapolation needed for
the prop-on dataset which only contains V=20 and V=40).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

# -- Experimental data (WebPlotDigitizer, Ct_V_exp_data.csv) ------------------
exp_data = {
    20: {
        "J":  [1.000, 1.200, 1.402, 1.598, 1.800, 2.002, 2.201],
        "CT": [0.352, 0.355, 0.339, 0.270, 0.181, 0.121, 0.045],
    },
    30: {
        "J":  [1.400, 1.600, 1.800, 2.000, 2.199],
        "CT": [0.349, 0.302, 0.215, 0.142, 0.065],
    },
    40: {
        "J":  [1.800, 1.901, 2.000, 2.099, 2.199, 2.298],
        "CT": [0.237, 0.197, 0.157, 0.116, 0.075, 0.030],
    },
    50: {
        "J":  [2.101, 2.200, 2.298, 2.399],
        "CT": [0.132, 0.089, 0.045, -0.009],
    },
}

# Velocities where linear extrapolation is applied, and the J range to cover
EXTRAP_VELOCITIES = {20, 40}
J_EXTRAP_MIN = 1.6
J_EXTRAP_MAX = 2.8

_PLOT_DIR = Path(__file__).parent / "RES_PLOTS"
def linear_extrap(J_query, J_arr, CT_arr):
    """
    Linear interpolation inside [J_arr[0], J_arr[-1]].
    Linear extrapolation outside using the two nearest endpoint points.
    J_arr must be sorted ascending.
    """
    if J_query <= J_arr[0]:
        slope = (CT_arr[1] - CT_arr[0]) / (J_arr[1] - J_arr[0])
        return CT_arr[0] + slope * (J_query - J_arr[0])
    elif J_query >= J_arr[-1]:
        slope = (CT_arr[-1] - CT_arr[-2]) / (J_arr[-1] - J_arr[-2])
        return CT_arr[-1] + slope * (J_query - J_arr[-1])
    else:
        return float(np.interp(J_query, J_arr, CT_arr))


# -- BEM polynomial -----------------------------------------------------------
def CT_bem(J):
    return -0.0051*J**4 + 0.0959*J**3 - 0.5888*J**2 + 1.0065*J - 0.1353


J_bem    = np.linspace(0.8, 2.8, 300)
J_propon = np.array([1.6, 2.0, 2.4, 2.8])
colors   = {20: "#4C72B0", 30: "#DD8452", 40: "#55A868", 50: "#C44E52"}

# -- Plot ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(J_bem, CT_bem(J_bem), "k--", lw=2, label="BEM polynomial", zorder=5)

for V, d in exp_data.items():
    J_raw  = np.array(d["J"])
    CT_raw = np.array(d["CT"])
    c      = colors[V]

    # Raw digitized points
    ax.scatter(J_raw, CT_raw, color=c, s=40, zorder=6)

    # Interpolated curve (digitized range only)
    J_interp  = np.linspace(J_raw.min(), J_raw.max(), 200)
    f_interp  = interp1d(J_raw, CT_raw, kind="linear")
    CT_interp = f_interp(J_interp)
    ax.plot(J_interp, CT_interp, color=c, lw=1.8, label=f"Exp. V$_\\infty$ = {V} m/s")

    # Extrapolated extensions (V=20 and V=40 only, clipped to J=1.6-2.8)
    if V in EXTRAP_VELOCITIES:
        if J_raw.min() > J_EXTRAP_MIN:
            J_left  = np.linspace(J_EXTRAP_MIN, J_raw.min(), 80)
            CT_left = np.array([linear_extrap(j, J_raw, CT_raw) for j in J_left])
            ax.plot(J_left, CT_left, color=c, lw=1.8, ls="--")

        if J_raw.max() < J_EXTRAP_MAX:
            J_right  = np.linspace(J_raw.max(), J_EXTRAP_MAX, 80)
            CT_right = np.array([linear_extrap(j, J_raw, CT_raw) for j in J_right])
            ax.plot(J_right, CT_right, color=c, lw=1.8, ls="--")

    # Mark prop-on query points (V=20 and V=40 only)
    if V in EXTRAP_VELOCITIES:
        for J_q in J_propon:
            in_raw = J_raw.min() <= J_q <= J_raw.max()
            in_ext = J_EXTRAP_MIN <= J_q <= J_EXTRAP_MAX

            if in_raw:
                CT_q, marker = float(f_interp(J_q)), "D"
            elif in_ext:
                CT_q, marker = linear_extrap(J_q, J_raw, CT_raw), "s"
            else:
                continue

            ax.scatter(J_q, CT_q, color=c, marker=marker, s=65,
                       edgecolors="k", linewidths=0.8, zorder=8)

# BEM values at prop-on J points
for J_q in J_propon:
    ax.scatter(J_q, CT_bem(J_q), color="k", marker="^", s=60, edgecolors="k", zorder=9)

# Legend proxy handles
ax.scatter([], [], color="grey", marker="D", s=65, edgecolors="k",
           linewidths=0.8, label="Query point (interpolated)")
ax.scatter([], [], color="grey", marker="s", s=65, edgecolors="k",
           linewidths=0.8, label="Query point (extrapolated)")
ax.plot([], [], color="grey", lw=1.8, ls="--", label="Linear extrapolation")
ax.scatter([], [], color="k", marker="^", s=60, label="BEM at prop-on $J$ values")

ax.set_xlabel("Advance ratio $J$", fontsize=12)
ax.set_ylabel("Thrust coefficient $C_T$", fontsize=12)
ax.set_title(
    "BEM polynomial vs. experimental $C_T(J)$ curves\n"
    "with linear extrapolation for $V_\\infty$ = 20 and 40 m/s ($J$ = 1.6–2.8)",
    fontsize=12,
)
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(0.8, 3.0)
ax.set_ylim(-0.3, 0.42)
ax.grid(True, alpha=0.3)
ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)

plt.tight_layout()
out = _PLOT_DIR / "CT_BEM_vs_exp.png"
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved {out.name}")