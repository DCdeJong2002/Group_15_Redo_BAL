"""
Figure 10.28 – Boundary correction factor δ for a wing with uniform loading
               in a closed elliptic jet, λ = 0.7 curve only.

CSV expected next to this script:
    FIG_10_28_lambda0_7.csv   (comma-separated, no header)

Fit: cubic polynomial in k  (R² = 0.9976, RMSE = 9e-5)
Valid range: k ∈ [0.34, 0.83]  (digitized range)

Usage
-----
Run interactively:   python delta_lookup.py
Import:              from delta_lookup import get_delta
                     delta = get_delta(0.6)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────
_CSV      = Path(__file__).parent / "FIG_10_28_lambda0_7.csv"
_PLOT_DIR = Path(__file__).parent / "RES_PLOTS"
_PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            xs.append(float(parts[0].strip()))
            ys.append(float(parts[1].strip()))
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


k_data, delta_data = _load_csv(_CSV)

# ── Fit cubic polynomial ──────────────────────────────────────────────────
_COEFFS = np.polyfit(k_data, delta_data, 3)   # [c3, c2, c1, c0]

_K_MIN = k_data[0]
_K_MAX = k_data[-1]


def get_delta(k: float) -> float:
    """
    Return δ (boundary correction factor) for λ=0.7, closed elliptic jet.

    Uses a cubic polynomial fit to digitized Figure 10.28 data.
    Smooths WebPlotDigitizer noise (R²=0.9976, RMSE=9e-5).

    Parameters
    ----------
    k : float
        Effective span / jet width  (digitized range: 0.34 – 0.83)

    Returns
    -------
    float
        δ, or NaN with a warning if k is outside the digitized range.
    """
    k = float(k)
    if not (_K_MIN <= k <= _K_MAX):
        print(f"  ⚠  k = {k} is outside the digitized range [{_K_MIN:.3f}, {_K_MAX:.3f}].")
        return float("nan")
    return float(np.polyval(_COEFFS, k))


# ── Plot ──────────────────────────────────────────────────────────────────
def plot_figure(query_k: float | None = None, label: str | None = None,
                save: bool = True):
    """
    Plot the δ curve with a named query point and a results table.

    Layout: [curve plot (2/3 width)] | [results table (1/3 width)]

    Parameters
    ----------
    query_k : float | None
        k value to mark on the plot
    label : str | None
        Name shown on the plot and in the table
    save : bool
        Save figure to ../RES_PLOTS/FIG_10_28_delta.png
    """
    k_fine  = np.linspace(_K_MIN, _K_MAX, 400)
    d_fit   = np.polyval(_COEFFS, k_fine)

    fig = plt.figure(figsize=(14, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)
    ax  = fig.add_subplot(gs[0, :2])
    ax_t = fig.add_subplot(gs[0, 2])

    # ── Curve ──────────────────────────────────────────────────────────────
    ax.plot(k_fine, d_fit, "k-", linewidth=1.8,
            label=r"Cubic fit  ($\lambda=0.7$)")
    ax.scatter(k_data, delta_data, s=10, color="steelblue",
               alpha=0.4, zorder=3, label="Digitized points")

    table_rows = []

    if query_k is not None:
        qd = get_delta(query_k)
        if not np.isnan(qd):
            pt_label = label or f"k={query_k:.3f}"
            col = plt.cm.tab10.colors[0]
            ax.scatter([query_k], [qd], s=90, color=col,
                       zorder=6, marker="D", label=pt_label)
            ax.annotate(
                pt_label,
                xy=(query_k, qd),
                xytext=(0, 14), textcoords="offset points",
                fontsize=8, fontweight="bold", color=col, ha="center",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.7),
            )
            table_rows.append((pt_label, f"{query_k:.4f}", f"{qd:.6f}", col))

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.08, 0.16)
    ax.set_xlabel(r"$k = $ Effective span / Jet width", fontsize=11)
    ax.set_ylabel(r"Boundary correction factor,  $\delta$", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(r"Figure 10.28 – $\delta$, uniform loading, closed elliptic jet  "
                 r"($\lambda=0.7$)", fontsize=11)

    # ── Table ──────────────────────────────────────────────────────────────
    ax_t.axis("off")
    if table_rows:
        col_labels = ["Component", "k", "δ"]
        cell_text  = [[r[0], r[1], r[2]] for r in table_rows]

        tbl = ax_t.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.6)

        for col_idx in range(3):
            tbl[(0, col_idx)].set_facecolor("#2c2c2c")
            tbl[(0, col_idx)].set_text_props(color="white", fontweight="bold")
        for row_idx, (*_, col) in enumerate(table_rows, start=1):
            tbl[(row_idx, 0)].set_facecolor(col)
            tbl[(row_idx, 0)].set_text_props(color="white", fontweight="bold")
            shade = "#f5f5f5" if row_idx % 2 == 0 else "white"
            for col_idx in [1, 2]:
                tbl[(row_idx, col_idx)].set_facecolor(shade)

        ax_t.set_title("Results", fontsize=10, fontweight="bold", pad=8)
    else:
        ax_t.text(0.5, 0.5, "No query points",
                  ha="center", va="center", fontsize=9, color="gray",
                  transform=ax_t.transAxes)

    if save:
        out = _PLOT_DIR / "FIG_10_28_delta.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved → {out}")

    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    c = _COEFFS
    print("=" * 60)
    print("  δ lookup  –  Figure 10.28  (λ=0.7, closed elliptic jet)")
    print(f"  Fit: δ = {c[0]:.5f}k³ + {c[1]:.5f}k² + {c[2]:.5f}k + {c[3]:.5f}")
    print(f"  R² ≈ 0.9976   RMSE ≈ 9e-5")
    print(f"  Digitized range: k = [{_K_MIN:.3f}, {_K_MAX:.3f}]")
    print("=" * 60)
    print()
    try:
        raw = input("  Enter k value (or press Enter to just show plot): ").strip()
        if raw == "":
            plot_figure()
        else:
            k_val  = float(raw)
            lbl    = input("  Label for this point: ").strip() or None
            delta  = get_delta(k_val)
            if not np.isnan(delta):
                print(f"\n  δ (λ=0.7) = {delta:.6f}")
            plot_figure(query_k=k_val, label=lbl)
    except ValueError:
        print("  Invalid input – please enter a number.")