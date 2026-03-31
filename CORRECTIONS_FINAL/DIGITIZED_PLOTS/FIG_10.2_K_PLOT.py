"""
Figure 10.2 – K₁ and K₃ for various body types.

Three curves digitized:
  'KS'     → K₃ (streamline body of revolution)
  '4digit' → K₁ (4-digit NACA airfoil series)
  '64'     → K₁ (NACA 64-series airfoil)

x-axis: thickness ratio  t/c  (airfoils) or  d/l  (bodies)

CSV expected next to this script:
    FIG_10_2_ALL.csv

Fits: cubic polynomials in thickness ratio  (R² > 0.9993 for all curves)

Usage
-----
Run interactively:   python k_lookup.py
Import:              from k_lookup import get_K, CURVES
                     k = get_K('4digit', 0.12)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────
_CSV      = Path(__file__).parent / "FIG_10_2_ALL.csv"
_PLOT_DIR = Path(__file__).parent / "RES_PLOTS"
_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Human-readable labels for each CSV column name
_LABELS = {
    "KS":     "K₃ – Streamline body of revolution",
    "4digit": "K₁ – 4-digit NACA series",
    "64":     "K₁ – NACA 64-series",
}


def _load_csv(path: Path) -> dict:
    with open(path) as f:
        lines = f.readlines()
    names_raw = lines[0].strip().split(",")
    names = [n for n in names_raw if n]
    col_pairs = [(i * 2, i * 2 + 1) for i in range(len(names))]

    raw = {n: {"x": [], "y": []} for n in names}
    for line in lines[2:]:
        parts = line.strip().split(",")
        for i, n in enumerate(names):
            cx, cy = col_pairs[i]
            if (cx < len(parts) and cy < len(parts)
                    and parts[cx].strip() and parts[cy].strip()):
                raw[n]["x"].append(float(parts[cx].strip()))
                raw[n]["y"].append(float(parts[cy].strip()))

    curves = {}
    for n in names:
        x = np.array(raw[n]["x"])
        y = np.array(raw[n]["y"])
        order = np.argsort(x)
        x, y = x[order], y[order]
        coeffs = np.polyfit(x, y, 3)
        curves[n] = {
            "x_data": x,
            "y_data": y,
            "coeffs": coeffs,
            "x_min": x[0],
            "x_max": x[-1],
            "label": _LABELS.get(n, n),
        }
    return curves


_CURVES = _load_csv(_CSV)
CURVES = list(_CURVES.keys())   # ['KS', '4digit', '64']


def get_K(curve_name: str, thickness_ratio: float) -> float:
    """
    Return K value for a given curve and thickness ratio.

    Parameters
    ----------
    curve_name : str
        One of: 'KS', '4digit', '64'
    thickness_ratio : float
        t/c (airfoil) or d/l (body)

    Returns
    -------
    float
        K value, or NaN if out of range.
    """
    if curve_name not in _CURVES:
        print(f"  ⚠  Unknown curve '{curve_name}'. Choose from: {CURVES}")
        return float("nan")
    c = _CURVES[curve_name]
    x = float(thickness_ratio)
    if not (c["x_min"] <= x <= c["x_max"]):
        print(f"  ⚠  thickness_ratio={x:.4f} outside digitized range "
              f"[{c['x_min']:.4f}, {c['x_max']:.4f}] for '{curve_name}'.")
        return float("nan")
    return float(np.polyval(c["coeffs"], x))


# ── Plot ──────────────────────────────────────────────────────────────────
# Colour / style per curve
_STYLE = {
    "KS":     {"color": "black",      "ls": "-"},
    "4digit": {"color": "tab:blue",   "ls": "-"},
    "64":     {"color": "tab:orange", "ls": "-"},
}


def plot_figure(query_points: list[dict] | None = None, show_raw: bool = True,
                save: bool = True):
    """
    Plot all fitted K curves with named query points and a results table.

    Layout: [curve plot (2/3 width)] | [results table (1/3 width)]

    Parameters
    ----------
    query_points : list of dict, each with keys:
        'curve'  : str   – curve name ('KS', '4digit', '64')
        'x'      : float – thickness ratio
        'label'  : str   – point label shown on plot and in table
    show_raw : bool
        Whether to show digitized scatter points.
    save : bool
        Save figure to ../RES_PLOTS/FIG_10_2_K.png
    """
    colours = plt.cm.tab10.colors

    fig = plt.figure(figsize=(14, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)
    ax  = fig.add_subplot(gs[0, :2])
    ax_t = fig.add_subplot(gs[0, 2])

    # ── Curves ─────────────────────────────────────────────────────────────
    for name, c in _CURVES.items():
        x_fine = np.linspace(c["x_min"], c["x_max"], 400)
        y_fine = np.polyval(c["coeffs"], x_fine)
        sty = _STYLE.get(name, {"color": "gray", "ls": "--"})
        ax.plot(x_fine, y_fine, color=sty["color"], ls=sty["ls"],
                linewidth=1.8, label=c["label"])
        if show_raw:
            ax.scatter(c["x_data"], c["y_data"], s=8,
                       color=sty["color"], alpha=0.3, zorder=3)

    table_rows = []

    if query_points:
        for i, qp in enumerate(query_points):
            name  = qp["curve"]
            x_val = qp["x"]
            lbl   = qp.get("label", f"Point {i+1}")
            y_val = get_K(name, x_val)
            if np.isnan(y_val):
                continue
            col = colours[i % len(colours)]
            ax.scatter([x_val], [y_val], s=90, color=col,
                       zorder=6, marker="D", label=lbl)
            ax.annotate(
                lbl,
                xy=(x_val, y_val),
                xytext=(0, 14), textcoords="offset points",
                fontsize=8, fontweight="bold", color=col, ha="center",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.7),
            )
            table_rows.append((lbl, _CURVES[name]["label"], f"{x_val:.4f}",
                                f"{y_val:.5f}", col))

    ax.set_xlim(0.04, 0.24)
    ax.set_ylim(0.86, 1.10)
    ax.set_xlabel(r"Thickness ratio  $t/c$  or  $d/l$", fontsize=11)
    ax.set_ylabel(r"$K_1$  and  $K_3$", fontsize=11)
    ax.set_title("Figure 10.2 – Values of $K_1$ and $K_3$", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.35)

    # ── Table ──────────────────────────────────────────────────────────────
    ax_t.axis("off")
    if table_rows:
        col_labels = ["Component", "Curve", "t/c or d/l", "K"]
        cell_text  = [[r[0], r[1], r[2], r[3]] for r in table_rows]

        tbl = ax_t.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
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
        out = _PLOT_DIR / "FIG_10_2_K.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved → {out}")

    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────
def _print_menu():
    print("\n  Available curves:")
    for i, n in enumerate(CURVES):
        print(f"    [{i}] {n:8s}  –  {_LABELS.get(n, n)}")


if __name__ == "__main__":
    print("=" * 60)
    print("  K₁ / K₃ lookup  –  Figure 10.2")
    print("  Enter points one at a time; press Enter with no")
    print("  input when done to show the plot.")
    print("=" * 60)

    query_points = []

    while True:
        _print_menu()
        raw_curve = input("\n  Select curve [0/1/2] (or Enter to finish): ").strip()
        if raw_curve == "":
            break
        try:
            idx = int(raw_curve)
            curve_name = CURVES[idx]
        except (ValueError, IndexError):
            print("  ⚠  Invalid selection.")
            continue

        c = _CURVES[curve_name]
        raw_x = input(f"  Enter thickness ratio "
                      f"[{c['x_min']:.3f} – {c['x_max']:.3f}]: ").strip()
        try:
            x_val = float(raw_x)
        except ValueError:
            print("  ⚠  Invalid number.")
            continue

        label = input("  Label for this point: ").strip() or f"Point {len(query_points)+1}"

        k_val = get_K(curve_name, x_val)
        if not np.isnan(k_val):
            print(f"\n  ✓  {label}: K = {k_val:.6f}")
            query_points.append({"curve": curve_name, "x": x_val, "label": label})

    print(f"\n  Plotting {len(query_points)} point(s)…")
    plot_figure(query_points=query_points if query_points else None)