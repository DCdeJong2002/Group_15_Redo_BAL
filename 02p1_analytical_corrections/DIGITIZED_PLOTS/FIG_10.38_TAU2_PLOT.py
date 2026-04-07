"""
Figure 10.38 – τ₂ for closed elliptic jets.

Loads digitized data from CSV, fits analytical model to smooth digitization
noise, then allows lookup by raw tail length lt with named points.

CSV expected next to this script:
    FIG_10_38_closedjet.csv

Plot saved to:
    ../RES_PLOTS/FIG_10_38_tau2.png   (relative to this script)

Fit:  τ₂ = a · (lt/B)^b / (1 + c · (lt/B)^d)
  R² = 0.9995,  RMSE = 0.0084

Usage
-----
Run interactively:   python tau2_lookup.py
Import:              from tau2_lookup import get_tau2, get_tau2_from_lt
                     tau2 = get_tau2(1.2)
                     tau2 = get_tau2_from_lt(2.16)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Constants ──────────────────────────────────────────────────────────────
_B        = 1.8    # tunnel width [m]
_PLOT_DIR = Path(__file__).parent / "RES_PLOTS"
_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load CSV (next to this script) ────────────────────────────────────────
_CSV = Path(__file__).parent / "FIG_10_38_closedjet.csv"


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with open(path) as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        xs.append(float(parts[0].strip()))
        ys.append(float(parts[1].strip()))
    order = np.argsort(xs)
    x = np.array(xs)[order]
    y = np.array(ys)[order]
    return x[x > 0], y[x > 0]   # drop x<=0 for power law fit


x_data, y_data = _load_csv(_CSV)


# ── Fit analytical model ───────────────────────────────────────────────────
def _model(x, a, b, c, d):
    return a * x**b / (1 + c * x**d)


_params, _ = curve_fit(_model, x_data, y_data,
                        p0=[1.5, 0.6, 0.5, 1.0], maxfev=10000)
_A, _B_exp, _C, _D = _params

_LT_B_MIN = 0.0
_LT_B_MAX = 2.0


# ── Lookup functions ───────────────────────────────────────────────────────
def get_tau2(lt_over_B: float) -> float:
    """Return τ₂ for a given lt/B ratio."""
    x = float(lt_over_B)
    if not (_LT_B_MIN <= x <= _LT_B_MAX):
        print(f"  ⚠  lt/B = {x:.4f} is outside the valid range "
              f"[{_LT_B_MIN}, {_LT_B_MAX}].")
        return float("nan")
    if x == 0.0:
        return 0.0
    return float(_model(x, _A, _B_exp, _C, _D))


def get_tau2_from_lt(lt: float, B: float = _B) -> dict:
    """Return τ₂ given raw tail length lt [m] and tunnel width B [m]."""
    lt_over_B = lt / B
    return {"lt_over_B": lt_over_B, "tau2": get_tau2(lt_over_B)}


# ── Plot ───────────────────────────────────────────────────────────────────
def plot_figure(query_points: list[dict] | None = None, B: float = _B,
                save: bool = True):
    """
    Plot the τ₂ curve with named points and a results table.

    Layout: [curve plot (2/3 width)] | [results table (1/3 width)]

    Parameters
    ----------
    query_points : list of dict, each with keys:
        'lt'    : float – tail length [m]
        'label' : str   – name shown on plot and in table
    save : bool
        Save figure to ../RES_PLOTS/FIG_10_38_tau2.png
    """
    x_fine  = np.linspace(0.001, _LT_B_MAX, 600)
    y_fine  = _model(x_fine, _A, _B_exp, _C, _D)
    colours = plt.cm.tab10.colors

    fig = plt.figure(figsize=(14, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)
    ax  = fig.add_subplot(gs[0, :2])
    ax_t = fig.add_subplot(gs[0, 2])

    # ── Curve ──────────────────────────────────────────────────────────────
    ax.plot(x_fine, y_fine, "k-", linewidth=1.8, label="Analytical fit (closed jet)")
    ax.scatter(x_data, y_data, s=6, color="steelblue",
               alpha=0.4, zorder=3, label="Digitized points")

    table_rows = []

    if query_points:
        for i, qp in enumerate(query_points):
            lt   = qp["lt"]
            lbl  = qp.get("label", f"Point {i+1}")
            lt_B = lt / B
            tau2 = get_tau2(lt_B)
            if np.isnan(tau2):
                continue
            col = colours[i % len(colours)]
            ax.scatter([lt_B], [tau2], s=90, color=col, zorder=6, marker="D",
                       label=lbl)
            ax.annotate(
                lbl,
                xy=(lt_B, tau2),
                xytext=(0, 14), textcoords="offset points",
                fontsize=8, fontweight="bold", color=col, ha="center",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.7),
            )
            table_rows.append((lbl, f"{lt:.3f}", f"{lt_B:.4f}", f"{tau2:.5f}", col))

    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel(r"Tail length / Tunnel width,  $l_t\,/\,B$", fontsize=11)
    ax.set_ylabel(r"$\tau_2$", fontsize=13, rotation=0, labelpad=14)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(r"Figure 10.38 – $\tau_2$ for closed elliptic jet  "
                 r"($\tau_2 = a\,x^b/(1+c\,x^d)$,  R²=0.9995)", fontsize=11)

    # ── Table ──────────────────────────────────────────────────────────────
    ax_t.axis("off")
    if table_rows:
        col_labels = ["Component", "lₜ [m]", "lₜ/B", "τ₂"]
        cell_text  = [[r[0], r[1], r[2], r[3]] for r in table_rows]

        tbl = ax_t.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.6)

        for col_idx in range(4):
            tbl[(0, col_idx)].set_facecolor("#2c2c2c")
            tbl[(0, col_idx)].set_text_props(color="white", fontweight="bold")
        for row_idx, (*_, col) in enumerate(table_rows, start=1):
            tbl[(row_idx, 0)].set_facecolor(col)
            tbl[(row_idx, 0)].set_text_props(color="white", fontweight="bold")
            shade = "#f5f5f5" if row_idx % 2 == 0 else "white"
            for col_idx in [1, 2, 3]:
                tbl[(row_idx, col_idx)].set_facecolor(shade)

        ax_t.set_title("Results", fontsize=10, fontweight="bold", pad=8)
    else:
        ax_t.text(0.5, 0.5, "No query points",
                  ha="center", va="center", fontsize=9, color="gray",
                  transform=ax_t.transAxes)

    if save:
        out = _PLOT_DIR / "FIG_10_38_tau2.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved → {out}")

    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  τ₂ lookup  –  Figure 10.38  (closed jet)")
    print(f"  Formula: τ₂ = {_A:.5f}·x^{_B_exp:.5f} / (1 + {_C:.5f}·x^{_D:.5f})")
    print(f"  R² ≈ 0.9995   RMSE ≈ 0.0084   valid lt/B: 0 – 2")
    print(f"  Tunnel width B = {_B} m")
    print("=" * 60)
    print("  Enter points one at a time.")
    print("  Press Enter with no input when done to show the plot.")
    print()

    query_points = []

    while True:
        raw = input("  Enter tail length lt [m] (or Enter to finish): ").strip()
        if raw == "":
            break
        try:
            lt_val = float(raw)
        except ValueError:
            print("  ⚠  Invalid number.")
            continue

        label = input("  Label for this point: ").strip() \
                or f"Point {len(query_points)+1}"

        res = get_tau2_from_lt(lt_val)
        if not np.isnan(res["tau2"]):
            print(f"  ✓  {label}: lt/B = {res['lt_over_B']:.4f},  "
                  f"τ₂ = {res['tau2']:.6f}\n")
            query_points.append({"lt": lt_val, "label": label})

    print(f"\n  Plotting {len(query_points)} point(s)…")
    plot_figure(query_points=query_points if query_points else None)