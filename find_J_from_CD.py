"""
find_J_from_CD.py
-----------------
Given an arbitrary drag coefficient CD (from a trimmed flight condition where T=D),
find the propeller advance ratio J required to deliver that thrust.

Physics
-------
Trim condition:   T = D  =>  T_total = CD * q_inf * S_ref
Single prop:      T_one  = T_total / 2
CT definition:    CT = T_one / (rho * n^2 * D_prop^4)
With n = V/(J*D_prop):
                  CT = (CD * S_ref / (4 * D_prop^2)) * J^2     [load line]

The operating point J* is the intersection of this parabola with the
experimental CT(J) curve at the chosen flight velocity.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Constants (from correction_classes_FINAL_V.py)
# ---------------------------------------------------------------------------
WING_AREA   = 0.2172        # S_ref [m^2]
PROP_DIAM   = 0.2032        # D_prop [m]
K_LOADLINE  = WING_AREA / (4 * PROP_DIAM**2)   # = ~1.3151  [-]

# ---------------------------------------------------------------------------
# Experimental CT(J) data  (from CT_PLOT.py / WebPlotDigitizer)
# ---------------------------------------------------------------------------
EXP_DATA = {
    20: {
        "J":  [1.000, 1.200, 1.402, 1.598, 1.800, 2.002, 2.201],
        "CT": [0.352, 0.355, 0.339, 0.270, 0.181, 0.121, 0.045],
    },
    40: {
        "J":  [1.800, 1.901, 2.000, 2.099, 2.199, 2.298],
        "CT": [0.237, 0.197, 0.157, 0.116, 0.075, 0.030],
    },
}

# J range over which the interpolant/extrapolant is evaluated (same as CT_PLOT.py)
J_EXT_MIN = 1.6
J_EXT_MAX = 2.8


def _linear_extrap(J_query, J_arr, CT_arr):
    """
    Linear interpolation inside [J_arr[0], J_arr[-1]],
    linear extrapolation outside using the two nearest endpoint points.
    Identical logic to CT_PLOT.py.
    """
    if J_query <= J_arr[0]:
        slope = (CT_arr[1] - CT_arr[0]) / (J_arr[1] - J_arr[0])
        return CT_arr[0] + slope * (J_query - J_arr[0])
    elif J_query >= J_arr[-1]:
        slope = (CT_arr[-1] - CT_arr[-2]) / (J_arr[-1] - J_arr[-2])
        return CT_arr[-1] + slope * (J_query - J_arr[-1])
    else:
        return float(np.interp(J_query, J_arr, CT_arr))


def build_CT_interpolant(V: int):
    """
    Returns a callable CT_exp(J) for V=20 or V=40, covering J in [J_EXT_MIN, J_EXT_MAX].
    Uses linear interpolation within the digitised range and linear extrapolation outside,
    consistent with CT_PLOT.py.
    """
    assert V in EXP_DATA, f"V must be 20 or 40, got {V}"
    d = EXP_DATA[V]
    J_raw  = np.array(d["J"])
    CT_raw = np.array(d["CT"])

    J_dense  = np.linspace(J_EXT_MIN, J_EXT_MAX, 2000)
    CT_dense = np.array([_linear_extrap(j, J_raw, CT_raw) for j in J_dense])

    return interp1d(J_dense, CT_dense, kind="linear", bounds_error=False,
                    fill_value="extrapolate")


def find_J_from_CD(CD: float, V: int, verbose: bool = True) -> dict:
    """
    Given a trimmed drag coefficient CD and flight velocity V (20 or 40 m/s),
    find the advance ratio J* at which the propeller delivers T = D.

    Parameters
    ----------
    CD      : aerodynamic drag coefficient (dimensionless)
    V       : flight velocity in m/s (must be 20 or 40)
    verbose : print results to console

    Returns
    -------
    dict with keys: J_star, CT_star, converged
    """
    CT_exp = build_CT_interpolant(V)

    # Load line: CT_load(J) = K * CD * J^2
    def CT_load(J):
        return K_LOADLINE * CD * J**2

    # Residual to zero: CT_exp(J) - CT_load(J) = 0
    def residual(J):
        return CT_exp(J) - CT_load(J)

    # Search bracket: both functions are defined on [J_EXT_MIN, J_EXT_MAX]
    J_lo, J_hi = J_EXT_MIN, J_EXT_MAX
    f_lo, f_hi = residual(J_lo), residual(J_hi)

    result = {"J_star": np.nan, "CT_star": np.nan, "converged": False}

    if f_lo * f_hi > 0:
        if verbose:
            print(f"[CD={CD:.4f}, V={V}] No sign change in [{J_lo}, {J_hi}] — "
                  f"no intersection found. Check CD range.")
        return result

    J_star = brentq(residual, J_lo, J_hi, xtol=1e-8)
    CT_star = float(CT_exp(J_star))

    result.update({"J_star": J_star, "CT_star": CT_star, "converged": True})

    if verbose:
        print(f"V = {V} m/s | CD = {CD:.4f}  =>  J* = {J_star:.4f},  CT* = {CT_star:.4f}")

    return result


# ---------------------------------------------------------------------------
# Full CT_PLOT-style data (all four velocities, same as CT_PLOT.py)
# ---------------------------------------------------------------------------
EXP_DATA_FULL = {
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

COLORS      = {20: "#4C72B0", 30: "#DD8452", 40: "#55A868", 50: "#C44E52"}
EXTRAP_VS   = {20, 40}   # velocities that get extrapolation (same as CT_PLOT.py)
J_BEM_MIN   = 0.8        # BEM polynomial plot range


def CT_bem(J):
    return -0.0051*J**4 + 0.0959*J**3 - 0.5888*J**2 + 1.0065*J - 0.1353


def plot_CT_with_trim(
    CD_cases: list,
    show: bool = True,
    save_path: str | None = None,
):
    """
    Reproduce the full CT_PLOT.py figure (BEM polynomial, all four experimental
    velocity curves with interpolation/extrapolation, digitised scatter points)
    and overlay one load line + intersection marker per (CD, V) pair supplied
    in CD_cases.

    Parameters
    ----------
    CD_cases  : list of (CD, V) tuples, e.g. [(0.06, 40), (0.05, 20)]
    show      : call plt.show() at the end
    save_path : optional file path to save the figure (PNG/PDF/…)
    """
    J_propon = np.array([1.6, 2.0, 2.4, 2.8])   # prop-on query J values

    fig, ax = plt.subplots(figsize=(10, 6))

    # -- BEM polynomial -------------------------------------------------------
    J_bem = np.linspace(J_BEM_MIN, J_EXT_MAX, 300)
    ax.plot(J_bem, CT_bem(J_bem), "k--", lw=2, label="BEM polynomial", zorder=5)

    # -- Experimental curves (identical to CT_PLOT.py) -----------------------
    for V, d in EXP_DATA_FULL.items():
        J_raw  = np.array(d["J"])
        CT_raw = np.array(d["CT"])
        c = COLORS[V]

        # Digitised scatter
        ax.scatter(J_raw, CT_raw, color=c, s=40, zorder=6)

        # Interpolated curve (digitised range)
        J_interp = np.linspace(J_raw.min(), J_raw.max(), 200)
        f_interp = interp1d(J_raw, CT_raw, kind="linear")
        ax.plot(J_interp, f_interp(J_interp), color=c, lw=1.8,
                label=f"Exp. $V_\\infty$ = {V} m/s")

        # Extrapolated extensions (V=20 and V=40 only, clamped to J_EXT range)
        if V in EXTRAP_VS:
            if J_raw.min() > J_EXT_MIN:
                J_left  = np.linspace(J_EXT_MIN, J_raw.min(), 80)
                CT_left = np.array([_linear_extrap(j, J_raw, CT_raw) for j in J_left])
                ax.plot(J_left, CT_left, color=c, lw=1.8, ls="--")

            if J_raw.max() < J_EXT_MAX:
                J_right  = np.linspace(J_raw.max(), J_EXT_MAX, 80)
                CT_right = np.array([_linear_extrap(j, J_raw, CT_raw) for j in J_right])
                ax.plot(J_right, CT_right, color=c, lw=1.8, ls="--")

            # Prop-on query markers (diamond = interpolated, square = extrapolated)
            for J_q in J_propon:
                in_raw = J_raw.min() <= J_q <= J_raw.max()
                in_ext = J_EXT_MIN   <= J_q <= J_EXT_MAX
                if in_raw:
                    CT_q, marker = float(f_interp(J_q)), "D"
                elif in_ext:
                    CT_q, marker = _linear_extrap(J_q, J_raw, CT_raw), "s"
                else:
                    continue
                ax.scatter(J_q, CT_q, color=c, marker=marker, s=65,
                           edgecolors="k", linewidths=0.8, zorder=8)

    # BEM values at prop-on J points
    for J_q in J_propon:
        ax.scatter(J_q, CT_bem(J_q), color="k", marker="^", s=60,
                   edgecolors="k", zorder=9)

    # -- Load lines + intersection markers ------------------------------------
    # Use a visually distinct set of line styles so multiple cases are legible
    load_line_styles = ["-", "--", "-.", ":"]
    load_line_colors = ["#E63946", "#F4A261", "#A8DADC", "#6A0572"]

    J_load = np.linspace(J_EXT_MIN, J_EXT_MAX, 400)

    for i, (CD, V) in enumerate(CD_cases):
        ls  = load_line_styles[i % len(load_line_styles)]
        lc  = load_line_colors[i % len(load_line_colors)]

        CT_load_vals = K_LOADLINE * CD * J_load**2
        ax.plot(J_load, CT_load_vals, color=lc, lw=1.8, ls=ls,
                label=f"Load line $C_D={CD:.4f}$, $V={V}$ m/s", zorder=7)

        res = find_J_from_CD(CD, V, verbose=False)
        if res["converged"]:
            ax.scatter(res["J_star"], res["CT_star"],
                       color=lc, marker="*", s=160,
                       edgecolors="k", linewidths=0.8, zorder=10,
                       label=f"$J^*={res['J_star']:.3f}$ ($C_D={CD:.4f}$, $V={V}$)")
            # Drop a thin vertical guide line to the x-axis
            ax.axvline(res["J_star"], color=lc, lw=0.8, ls=":", alpha=0.7)

    # -- Legend proxy handles (same as CT_PLOT.py) ---------------------------
    ax.scatter([], [], color="grey", marker="D", s=65, edgecolors="k",
               linewidths=0.8, label="Query point (interpolated)")
    ax.scatter([], [], color="grey", marker="s", s=65, edgecolors="k",
               linewidths=0.8, label="Query point (extrapolated)")
    ax.plot([], [], color="grey", lw=1.8, ls="--", label="Linear extrapolation")
    ax.scatter([], [], color="k", marker="^", s=60,
               label="BEM at prop-on $J$ values")

    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax.set_xlabel("Advance ratio $J$", fontsize=12)
    ax.set_ylabel("Thrust coefficient $C_T$", fontsize=12)
    ax.set_title(
        "BEM polynomial vs. experimental $C_T(J)$ curves\n"
        "with trim load lines and operating points ($T = D$)",
        fontsize=12,
    )
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_xlim(J_BEM_MIN, 3.0)
    ax.set_ylim(-0.15, 0.42)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    if show:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # --- Single query ---
    CD_input = 0.05
    V_input  = 40
    result = find_J_from_CD(CD_input, V_input)

    # --- Full CT_PLOT-style figure with load line overlays ------------------
    # Supply as many (CD, V) pairs as needed (up to 4 line styles available)
    plot_CT_with_trim(
        CD_cases=[
            (CD_input, 40),
            (CD_input, 20),
        ],
        save_path="CT_BEM_vs_exp_trim.png",
    )