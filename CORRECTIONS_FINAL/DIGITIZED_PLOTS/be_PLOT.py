"""
Figure 10.11 – Vortex span / Geometric span (bv/b) for taper ratio λ=0.4
               in a closed elliptic jet.

λ=0.4 is not in the figure directly. This script interpolates linearly in λ
between the digitized λ=0.25 and λ=0.50 curves at each aspect ratio, then
fits a cubic polynomial to the result.

CSV expected next to this script:
    FIG_10_11_TAP.csv   (two-curve file with columns X,Y for each λ)

Fit: cubic polynomial in AR  (R² ≈ 0.9991, RMSE ≈ 2.9e-4)
Valid range: AR ∈ [2.57, 11.89]  (digitized range)

Usage
-----
Run interactively:   python bv_lookup.py
Import:              from bv_lookup import get_bv_ratio
                     bv_b = get_bv_ratio(8.0)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# ── Load CSV (next to this script) ───────────────────────────────────────
_CSV = Path(__file__).parent / "FIG_10_11_TAP.csv"

_LAMBDA_TARGET = 0.4
_LAMBDA_LO     = 0.25
_LAMBDA_HI     = 0.50
_T = (_LAMBDA_TARGET - _LAMBDA_LO) / (_LAMBDA_HI - _LAMBDA_LO)  # = 0.6


def _load_csv(path: Path):
    x25, y25, x50, y50 = [], [], [], []
    with open(path) as f:
        lines = f.readlines()
    for line in lines[2:]:          # skip two header rows
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        x25.append(float(parts[0])); y25.append(float(parts[1]))
        x50.append(float(parts[2])); y50.append(float(parts[3]))
    return (np.array(x25), np.array(y25),
            np.array(x50), np.array(y50))


x25, y25, x50, y50 = _load_csv(_CSV)

# Common AR grid covering the overlap of both curves
_AR_MIN = max(x25[0], x50[0])
_AR_MAX = min(x25[-1], x50[-1])
_ar_grid = np.linspace(_AR_MIN, _AR_MAX, 300)

# PCHIP through each digitized curve, then linearly blend in λ
_interp25 = PchipInterpolator(x25, y25)
_interp50 = PchipInterpolator(x50, y50)
_bv_grid  = (1 - _T) * _interp25(_ar_grid) + _T * _interp50(_ar_grid)

# Fit cubic polynomial to the interpolated λ=0.4 curve
_COEFFS = np.polyfit(_ar_grid, _bv_grid, 3)


def get_bv_ratio(aspect_ratio: float) -> float:
    """
    Return bv/b (vortex span / geometric span) for λ=0.4.

    Derived by linear interpolation in λ between digitized λ=0.25 and
    λ=0.50 curves, then smoothed with a cubic polynomial fit.

    Parameters
    ----------
    aspect_ratio : float
        Wing aspect ratio  (digitized range: ~2.6 – 11.9)

    Returns
    -------
    float
        bv/b, or NaN with a warning if AR is outside the digitized range.
    """
    ar = float(aspect_ratio)
    if not (_AR_MIN <= ar <= _AR_MAX):
        print(f"  ⚠  AR = {ar} is outside the digitized range "
              f"[{_AR_MIN:.2f}, {_AR_MAX:.2f}].")
        return float("nan")
    return float(np.polyval(_COEFFS, ar))


# ── Plot ──────────────────────────────────────────────────────────────────
def plot_figure(query_ar: float | None = None):
    ar_fine = np.linspace(_AR_MIN, _AR_MAX, 400)
    bv_fit  = np.polyval(_COEFFS, ar_fine)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, show_pts in zip(axes, [False, True]):
        # Fitted λ=0.4 curve
        ax.plot(ar_fine, bv_fit, "k-", linewidth=2.0,
                label=r"Cubic fit  ($\lambda=0.4$, interpolated)")

        if show_pts:
            # Raw digitized curves for reference
            ax.plot(x25, y25, "b--", linewidth=1.0, alpha=0.6,
                    label=r"Digitized $\lambda=0.25$")
            ax.plot(x50, y50, "r--", linewidth=1.0, alpha=0.6,
                    label=r"Digitized $\lambda=0.50$")
            # Interpolated (pre-fit) points
            ax.scatter(_ar_grid[::10], _bv_grid[::10], s=18,
                       color="green", alpha=0.7, zorder=3,
                       label=r"Interpolated $\lambda=0.4$ points")

        if query_ar is not None:
            qbv = get_bv_ratio(query_ar)
            if not np.isnan(qbv):
                ax.axvline(query_ar, color="tab:red", linestyle="--",
                           linewidth=0.9, alpha=0.8)
                ax.axhline(qbv, color="tab:red", linestyle="--",
                           linewidth=0.9, alpha=0.8)
                ax.scatter([query_ar], [qbv], color="tab:red", zorder=5, s=60,
                           label=f"AR={query_ar:.2f}  →  bv/b={qbv:.4f}")

        ax.set_xlim(0, 20)
        ax.set_ylim(0.50, 1.00)
        ax.set_xlabel("Aspect ratio", fontsize=11)
        ax.set_ylabel(r"Vortex span / Geometric span,  $b_v/b$", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Fit only" if not show_pts else "Fit + source curves",
                     fontsize=10)

    fig.suptitle(
        r"Figure 10.11 – $b_v/b$ for $\lambda=0.4$  "
        r"(interpolated between $\lambda=0.25$ and $\lambda=0.50$)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


# ── Geometric span and effective span ────────────────────────────────────
_B = 1.397   # geometric span [m]


def get_effective_span(aspect_ratio: float, b: float = _B) -> dict:
    """
    Compute bv (vortex span) and be (effective span) from aspect ratio.

    be = (b + bv) / 2

    Parameters
    ----------
    aspect_ratio : float
        Wing aspect ratio
    b : float
        Geometric span [m]  (default: 1.397 m)

    Returns
    -------
    dict with keys: bv_over_b, bv, be
    """
    bv_over_b = get_bv_ratio(aspect_ratio)
    if np.isnan(bv_over_b):
        return {"bv_over_b": np.nan, "bv": np.nan, "be": np.nan}
    bv = bv_over_b * b
    be = (b + bv) / 2
    return {"bv_over_b": bv_over_b, "bv": bv, "be": be}


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    c = _COEFFS
    print("=" * 62)
    print("  bv/b lookup  –  Figure 10.11  (λ=0.4)")
    print(f"  λ=0.4 interpolated from λ=0.25 & λ=0.50 (t={_T:.2f})")
    print(f"  Fit: bv/b = {c[0]:.5f}·AR³ + {c[1]:.5f}·AR² + {c[2]:.5f}·AR + {c[3]:.5f}")
    print(f"  R² ≈ 0.9991   RMSE ≈ 2.9e-4")
    print(f"  Digitized range: AR = [{_AR_MIN:.2f}, {_AR_MAX:.2f}]")
    print(f"  Geometric span b = {_B} m")
    print("=" * 62)
    print()
    try:
        raw = input("  Enter aspect ratio (or press Enter to just show plot): ").strip()
        if raw == "":
            plot_figure()
        else:
            ar_val = float(raw)
            res = get_effective_span(ar_val)
            if not np.isnan(res["bv_over_b"]):
                print(f"\n  bv/b        = {res['bv_over_b']:.6f}")
                print(f"  bv          = {res['bv']:.6f} m   (= bv/b × b)")
                print(f"  be          = {res['be']:.6f} m   (= (b + bv) / 2)")
            plot_figure(query_ar=ar_val)
    except ValueError:
        print("  Invalid input – please enter a number.")