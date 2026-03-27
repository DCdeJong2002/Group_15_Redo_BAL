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

# ── Load CSV (next to this script) ───────────────────────────────────────
_CSV = Path(__file__).parent / "FIG_10_28_lambda0_7.csv"


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
def plot_figure(query_k: float | None = None):
    k_fine = np.linspace(_K_MIN, _K_MAX, 400)
    d_fit  = np.polyval(_COEFFS, k_fine)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, show_pts in zip(axes, [False, True]):
        ax.plot(k_fine, d_fit, "k-", linewidth=1.8,
                label=r"Cubic fit  ($\lambda=0.7$)")
        if show_pts:
            ax.scatter(k_data, delta_data, s=14, color="steelblue",
                       alpha=0.7, zorder=3, label="Digitized points")

        if query_k is not None:
            qd = get_delta(query_k)
            if not np.isnan(qd):
                ax.axvline(query_k, color="tab:red", linestyle="--",
                           linewidth=0.9, alpha=0.8)
                ax.axhline(qd, color="tab:red", linestyle="--",
                           linewidth=0.9, alpha=0.8)
                ax.scatter([query_k], [qd], color="tab:red", zorder=5, s=60,
                           label=f"k={query_k:.3f}  →  δ={qd:.5f}")

        ax.set_xlim(0, 1.0)
        ax.set_ylim(0.08, 0.16)
        ax.set_xlabel(r"$k = $ Effective span / Jet width", fontsize=11)
        ax.set_ylabel(r"Boundary correction factor,  $\delta$", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title("Fit only" if not show_pts else "Fit + digitized points",
                     fontsize=10)

    fig.suptitle(
        r"Figure 10.28 – $\delta$ for uniform loading, closed elliptic jet  ($\lambda=0.7$)",
        fontsize=12,
    )
    plt.tight_layout()
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
            k_val = float(raw)
            delta = get_delta(k_val)
            if not np.isnan(delta):
                print(f"\n  δ (λ=0.7) = {delta:.6f}")
            plot_figure(query_k=k_val)
    except ValueError:
        print("  Invalid input – please enter a number.")