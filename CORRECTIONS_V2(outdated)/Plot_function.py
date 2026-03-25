import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_propon_diagnostics(df: pd.DataFrame):
    """
    Plot a set of prop-on correction diagnostics in one figure.

    The function tries to use *_FINAL columns if available, otherwise it falls
    back to the base column names.

    Plots shown
    -----------
    1. Drag polar: CD vs CL, colored by V_round
    2. CL vs AoA
    3. CD vs V at one representative AoA_round
    4. CY vs AoA for AoS ~= 0
    5. CMyaw vs dR
    6. CT_prop vs J_round
    7. dCD_net vs dCD_from_dCT
    8. Tc_star vs J_round
    """

    df = df.copy()

    def pick_col(*candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # Prefer FINAL columns if present
    cd_col = pick_col("CD_FINAL", "CD")
    cl_col = pick_col("CL_FINAL", "CL")
    cy_col = pick_col("CY_FINAL", "CY")
    cmyaw_col = pick_col("CMyaw_FINAL", "CMyaw")
    aoa_col = pick_col("AoA_FINAL", "AoA")
    v_col = pick_col("V_FINAL", "V")
    aoa_round_col = pick_col("AoA_round")
    aos_round_col = pick_col("AoS_round")
    v_round_col = pick_col("V_round")
    dr_col = pick_col("dR")
    j_col = pick_col("J_round", "J")
    ct_prop_col = pick_col("CT_prop")
    tc_star_col = pick_col("Tc_star")
    dcd_net_col = pick_col("dCD_net")
    dcd_from_dct_col = pick_col("dCD_from_dCT")

    required_some = [cd_col, cl_col, aoa_col]
    if any(c is None for c in required_some):
        missing = [name for name, val in {
            "CD_FINAL/CD": cd_col,
            "CL_FINAL/CL": cl_col,
            "AoA_FINAL/AoA": aoa_col,
        }.items() if val is None]
        raise ValueError(f"Missing required columns for diagnostics: {missing}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    # ------------------------------------------------------------
    # 1. Drag polar: CD vs CL, colored by V_round if available
    # ------------------------------------------------------------
    ax = axes[0]
    if v_round_col is not None:
        sc = ax.scatter(df[cl_col], df[cd_col], c=df[v_round_col], s=25)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(v_round_col)
    else:
        ax.scatter(df[cl_col], df[cd_col], s=25)
    ax.set_xlabel(cl_col)
    ax.set_ylabel(cd_col)
    ax.set_title("Drag polar")
    ax.grid(True)

    # ------------------------------------------------------------
    # 2. CL vs AoA
    # ------------------------------------------------------------
    ax = axes[1]
    if v_round_col is not None:
        sc = ax.scatter(df[aoa_col], df[cl_col], c=df[v_round_col], s=25)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(v_round_col)
    else:
        ax.scatter(df[aoa_col], df[cl_col], s=25)
    ax.set_xlabel(aoa_col)
    ax.set_ylabel(cl_col)
    ax.set_title("Lift curve")
    ax.grid(True)

    # ------------------------------------------------------------
    # 3. CD vs V at one representative AoA_round
    # ------------------------------------------------------------
    ax = axes[2]
    if aoa_round_col is not None and v_col is not None:
        aoa_vals = sorted(df[aoa_round_col].dropna().unique())
        if len(aoa_vals) > 0:
            target_aoa = 2.5 if 2.5 in aoa_vals else aoa_vals[len(aoa_vals) // 2]
            sub = df[df[aoa_round_col] == target_aoa].copy()
            ax.scatter(sub[v_col], sub[cd_col], s=25)
            ax.set_title(f"{cd_col} vs {v_col} at {aoa_round_col}={target_aoa}")
        else:
            ax.text(0.5, 0.5, "No AoA_round data", ha="center", va="center")
            ax.set_title("CD vs V")
    else:
        ax.text(0.5, 0.5, "Missing AoA_round or V", ha="center", va="center")
        ax.set_title("CD vs V")
    ax.set_xlabel(v_col if v_col is not None else "V")
    ax.set_ylabel(cd_col)
    ax.grid(True)

    # ------------------------------------------------------------
    # 4. CY vs AoA for AoS ~= 0
    # ------------------------------------------------------------
    ax = axes[3]
    if cy_col is not None and aos_round_col is not None:
        sub = df[df[aos_round_col] == 0].copy()
        if not sub.empty:
            ax.scatter(sub[aoa_col], sub[cy_col], s=25)
            ax.set_title(f"{cy_col} vs {aoa_col} at {aos_round_col}=0")
        else:
            ax.text(0.5, 0.5, "No AoS_round = 0 rows", ha="center", va="center")
            ax.set_title("CY symmetry check")
        ax.set_xlabel(aoa_col)
        ax.set_ylabel(cy_col)
    else:
        ax.text(0.5, 0.5, "Missing CY or AoS_round", ha="center", va="center")
        ax.set_title("CY symmetry check")
    ax.grid(True)

    # ------------------------------------------------------------
    # 5. CMyaw vs dR
    # ------------------------------------------------------------
    ax = axes[4]
    if cmyaw_col is not None and dr_col is not None:
        if aoa_round_col is not None:
            aoa_vals = sorted(df[aoa_round_col].dropna().unique())
            target_aoa = 2.5 if 2.5 in aoa_vals else aoa_vals[len(aoa_vals) // 2]
            sub = df[df[aoa_round_col] == target_aoa].copy()
            ax.scatter(sub[dr_col], sub[cmyaw_col], s=25)
            ax.set_title(f"{cmyaw_col} vs {dr_col} at {aoa_round_col}={target_aoa}")
        else:
            ax.scatter(df[dr_col], df[cmyaw_col], s=25)
            ax.set_title(f"{cmyaw_col} vs {dr_col}")
        ax.set_xlabel(dr_col)
        ax.set_ylabel(cmyaw_col)
    else:
        ax.text(0.5, 0.5, "Missing CMyaw or dR", ha="center", va="center")
        ax.set_title("Yawing moment vs rudder")
    ax.grid(True)

    # ------------------------------------------------------------
    # 6. CT_prop vs J_round
    # ------------------------------------------------------------
    ax = axes[5]
    if ct_prop_col is not None and j_col is not None:
        ax.scatter(df[j_col], df[ct_prop_col], s=25)
        ax.set_xlabel(j_col)
        ax.set_ylabel(ct_prop_col)
        ax.set_title("CT_prop vs J")
    else:
        ax.text(0.5, 0.5, "Missing CT_prop or J", ha="center", va="center")
        ax.set_title("CT_prop vs J")
    ax.grid(True)

    # ------------------------------------------------------------
    # 7. dCD_net vs dCD_from_dCT
    # ------------------------------------------------------------
    ax = axes[6]
    if dcd_net_col is not None and dcd_from_dct_col is not None:
        ax.scatter(df[dcd_net_col], df[dcd_from_dct_col], s=25)
        vals = pd.concat([df[dcd_net_col], df[dcd_from_dct_col]], axis=0).dropna()
        if not vals.empty:
            mn, mx = vals.min(), vals.max()
            ax.plot([mn, mx], [mn, mx], linestyle="--")
        ax.set_xlabel(dcd_net_col)
        ax.set_ylabel(dcd_from_dct_col)
        ax.set_title("Direct ΔCD vs ΔCD from ΔCT")
    else:
        ax.text(0.5, 0.5, "Missing dCD columns", ha="center", va="center")
        ax.set_title("ΔCD comparison")
    ax.grid(True)

    # ------------------------------------------------------------
    # 8. Tc_star vs J_round
    # ------------------------------------------------------------
    ax = axes[7]
    if tc_star_col is not None and j_col is not None:
        ax.scatter(df[j_col], df[tc_star_col], s=25)
        ax.set_xlabel(j_col)
        ax.set_ylabel(tc_star_col)
        ax.set_title("Tc_star vs J")
    else:
        ax.text(0.5, 0.5, "Missing Tc_star or J", ha="center", va="center")
        ax.set_title("Tc_star vs J")
    ax.grid(True)

    plt.tight_layout()
    plt.show()