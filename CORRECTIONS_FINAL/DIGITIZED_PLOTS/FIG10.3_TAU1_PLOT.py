"""
Figure 10.3 – τ₁ for a closed elliptic jet, B/H = 1.43 curve.

x-axis: 2b/B = Model geometric span / Tunnel breadth

CSV expected next to this script:
    FIG_10_3_TAU1.csv

Fit: cubic polynomial in 2b/B  (R² = 0.9994, RMSE = 3.6e-4)

Usage
-----
Run interactively:   python tau1_lookup.py
Import:              from tau1_lookup import get_tau1
                     tau1 = get_tau1(0.5)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────
_B_H      = 1.43   # tunnel aspect ratio B/H for this curve
_PLOT_DIR = Path(__file__).parent / "RES_PLOTS"
_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load CSV (next to this script) ────────────────────────────────────────
_CSV = Path(__file__).parent / "FIG_10_3_TAU1.csv"


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with open(path) as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.strip().split(",")
        if len(parts) < 2 or not parts[0].strip():
            continue
        xs.append(float(parts[0].strip()))
        ys.append(float(parts[1].strip()))
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


x_data, y_data = _load_csv(_CSV)

# Clamp x range to [0, 1] (ignore tiny negative digitisation artefacts)
_X_MIN = 0.0
_X_MAX = x_data[-1]

# ── Fit cubic polynomial ───────────────────────────────────────────────────
_COEFFS = np.polyfit(x_data, y_data, 3)


def get_tau1(two_b_over_B: float) -> float:
    """
    Return τ₁ for B/H = 1.43 at a given 2b/B value.

    Parameters
    ----------
    two_b_over_B : float
        Model geometric span / tunnel breadth = 2b/B  (range: 0 – 1)

    Returns
    -------
    float
        τ₁, or NaN with a warning if out of range.
    """
    x = float(two_b_over_B)
    if not (_X_MIN <= x <= _X_MAX):
        print(f"  ⚠  2b/B = {x:.4f} is outside the valid range "
              f"[{_X_MIN:.3f}, {_X_MAX:.3f}].")
        return float("nan")
    return float(np.polyval(_COEFFS, x))


# ── Plot ───────────────────────────────────────────────────────────────────
def plot_figure(query_points: list[dict] | None = None, save: bool = True):
    """
    Plot the fitted τ₁ curve with named points and a results table.

    Layout: [curve plot (2/3 width)] | [results table (1/3 width)]

    Parameters
    ----------
    query_points : list of dict, each with keys:
        'x'     : float – 2b/B value
        'label' : str   – name shown on plot and in table
    save : bool
        Save figure to ../RES_PLOTS/FIG_10_3_tau1.png
    """
    x_fine  = np.linspace(_X_MIN, _X_MAX, 400)
    y_fine  = np.polyval(_COEFFS, x_fine)
    colours = plt.cm.tab10.colors

    # Layout: plot takes 2 cols, table takes 1 col
    fig = plt.figure(figsize=(14, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)
    ax  = fig.add_subplot(gs[0, :2])   # curve
    ax_t = fig.add_subplot(gs[0, 2])   # table

    # ── Curve ──────────────────────────────────────────────────────────────
    ax.plot(x_fine, y_fine, "k-", linewidth=1.8,
            label=f"Cubic fit  (B/H = {_B_H})")
    ax.scatter(x_data, y_data, s=6, color="steelblue",
               alpha=0.4, zorder=3, label="Digitized points")

    table_rows = []   # collect for table panel

    if query_points:
        for i, qp in enumerate(query_points):
            x_val = qp["x"]
            lbl   = qp.get("label", f"Point {i+1}")
            y_val = get_tau1(x_val)
            if np.isnan(y_val):
                continue
            col = colours[i % len(colours)]

            # Marker
            ax.scatter([x_val], [y_val], s=90, color=col,
                       zorder=6, marker="D", label=lbl)

            # Clean leader line annotation – just the name, values go in table
            ax.annotate(
                lbl,
                xy=(x_val, y_val),
                xytext=(0, 14), textcoords="offset points",
                fontsize=8, fontweight="bold", color=col, ha="center",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.7),
            )

            table_rows.append((lbl, f"{x_val:.4f}", f"{y_val:.5f}", col))

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.7, 1.1)
    ax.set_xlabel(r"Model geometric span / Tunnel breadth,  $2b/B$", fontsize=11)
    ax.set_ylabel(r"$\tau_1$", fontsize=13, rotation=0, labelpad=14)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_title(rf"Figure 10.3 – $\tau_1$  (B/H = {_B_H},  cubic fit,  R²=0.9994)",
                 fontsize=11)

    # ── Table panel ────────────────────────────────────────────────────────
    ax_t.axis("off")

    if table_rows:
        col_labels = ["Component", "2b/B", "τ₁"]
        cell_text  = [[r[0], r[1], r[2]] for r in table_rows]
        cell_cols  = [["white"] * 3 for _ in table_rows]

        tbl = ax_t.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.6)

        # Style header
        for col_idx in range(3):
            tbl[(0, col_idx)].set_facecolor("#2c2c2c")
            tbl[(0, col_idx)].set_text_props(color="white", fontweight="bold")

        # Colour-code first column to match markers
        for row_idx, (_, _, _, col) in enumerate(table_rows, start=1):
            tbl[(row_idx, 0)].set_facecolor(col)
            tbl[(row_idx, 0)].set_text_props(color="white", fontweight="bold")
            # Alternate row shading for data columns
            shade = "#f5f5f5" if row_idx % 2 == 0 else "white"
            for col_idx in [1, 2]:
                tbl[(row_idx, col_idx)].set_facecolor(shade)

        ax_t.set_title("Results", fontsize=10, fontweight="bold", pad=8)
    else:
        ax_t.text(0.5, 0.5, "No query points",
                  ha="center", va="center", fontsize=9, color="gray",
                  transform=ax_t.transAxes)

    if save:
        out = _PLOT_DIR / "FIG_10_3_tau1.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved → {out}")

    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"  τ₁ lookup  –  Figure 10.3  (B/H = {_B_H})")
    c = _COEFFS
    print(f"  Fit: τ₁ = {c[0]:.5f}·x³ + {c[1]:.5f}·x² + {c[2]:.5f}·x + {c[3]:.5f}")
    print(f"  R² ≈ 0.9994   RMSE ≈ 3.6e-4")
    print(f"  Valid range: 2b/B = [{_X_MIN:.3f}, {_X_MAX:.3f}]")
    print("=" * 60)
    print("  Enter points one at a time.")
    print("  Press Enter with no input when done to show the plot.")
    print()

    query_points = []

    while True:
        raw_x = input(f"  Enter 2b/B [{_X_MIN:.3f}–{_X_MAX:.3f}]"
                      f" (or Enter to finish): ").strip()
        if raw_x == "":
            break
        try:
            x_val = float(raw_x)
        except ValueError:
            print("  ⚠  Invalid number.")
            continue

        label = input("  Label for this point: ").strip() \
                or f"Point {len(query_points)+1}"

        tau1 = get_tau1(x_val)
        if not np.isnan(tau1):
            print(f"  ✓  {label}: τ₁ = {tau1:.6f}\n")
            query_points.append({"x": x_val, "label": label})

    print(f"\n  Plotting {len(query_points)} point(s)…")
    plot_figure(query_points=query_points if query_points else None)