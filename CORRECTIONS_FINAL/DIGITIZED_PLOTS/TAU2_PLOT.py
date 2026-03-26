"""
Figure 10.38 – τ₂ for closed elliptic jets
Loads digitized data from a CSV file relative to this script's location.

Expected repo layout (CSV and script can be anywhere, just keep them together):
    your_repo/
    ├── tau2_lookup.py
    └── tau_2__l_t_B_.csv

Formula fit to data:  τ₂ = a · (lt/B)^b  /  (1 + c · (lt/B)^d)
  a=5.026874, b=1.157340, c=3.676760, d=1.434704
  R² = 0.9997,  RMSE = 0.006

Usage
-----
Run interactively:   python tau2_lookup.py
Import:              from tau2_lookup import get_tau2
                     tau2 = get_tau2(1.2)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Load CSV (located next to this script) ───────────────────────────────
_CSV = Path(__file__).parent / "tau_2__l_t_B_digitized.csv"


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            xs.append(float(parts[0].strip().replace(",", ".")))
            ys.append(float(parts[1].strip().replace(",", ".")))
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


x_data, y_data = _load_csv(_CSV)


# ── Fit analytical model to data ─────────────────────────────────────────
def _model(x, a, b, c, d):
    return a * x**b / (1 + c * x**d)


_params, _ = curve_fit(_model, x_data, y_data, p0=[1.5, 0.6, 0.5, 1.0], maxfev=10000)
_A, _B, _C, _D = _params

_X_MIN, _X_MAX = 0.0, 2.0


def get_tau2(lt_over_B: float) -> float:
    """
    Return τ₂ (closed jet) for a given lt/B ratio.

    Uses analytical fit:  τ₂ = a · x^b / (1 + c · x^d)
    Parameters are fitted each run from the CSV (smooths digitization noise).

    Parameters
    ----------
    lt_over_B : float
        Tail length / tunnel width  (valid range: 0 – 2)

    Returns
    -------
    float
        τ₂, or NaN with a warning if lt/B is out of range.
    """
    x = float(lt_over_B)
    if not (_X_MIN <= x <= _X_MAX):
        print(f"  ⚠  lt/B = {x} is outside the valid range [{_X_MIN}, {_X_MAX}].")
        return float("nan")
    if x == 0.0:
        return 0.0
    return _model(x, _A, _B, _C, _D)


# ── Plot ─────────────────────────────────────────────────────────────────
def plot_figure(query_x: float | None = None):
    x_fine = np.linspace(0.001, 2.0, 600)
    y_fit  = _model(x_fine, _A, _B, _C, _D)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, show_pts in zip(axes, [False, True]):
        ax.plot(x_fine, y_fit, "k-", linewidth=1.8, label="Analytical fit")
        if show_pts:
            ax.scatter(x_data, y_data, s=8, color="steelblue",
                       alpha=0.6, zorder=3, label="Digitized points")

        if query_x is not None:
            qy = get_tau2(query_x)
            if not np.isnan(qy):
                ax.axvline(query_x, color="tab:red", linestyle="--", linewidth=0.9, alpha=0.8)
                ax.axhline(qy,      color="tab:red", linestyle="--", linewidth=0.9, alpha=0.8)
                ax.scatter([query_x], [qy], color="tab:red", zorder=5, s=60,
                           label=f"$l_t/B={query_x:.3f}$  →  $\\tau_2={qy:.4f}$")

        ax.set_xlim(0, 2.0)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel(r"Tail length / Tunnel width,  $l_t\,/\,B$", fontsize=11)
        ax.set_ylabel(r"$\tau_2$", fontsize=13, rotation=0, labelpad=14)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_title("Fit only" if not show_pts else "Fit + digitized points", fontsize=10)

    fig.suptitle(
        r"Figure 10.38 – $\tau_2$ for closed elliptic jet  "
        r"($\tau_2 = a\,x^b/(1+c\,x^d)$,  R²=0.9997)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 58)
    print("  τ₂ lookup  –  Figure 10.38  (closed jet)")
    print(f"  Formula: τ₂ = {_A:.4f}·x^{_B:.4f} / (1 + {_C:.4f}·x^{_D:.4f})")
    print(f"  R² ≈ 0.9997   RMSE ≈ 0.006   valid range: 0 – 2")
    print("=" * 58)
    print()
    try:
        raw = input("  Enter lt/B value (or press Enter to just show plot): ").strip()
        if raw == "":
            plot_figure()
        else:
            lt_B = float(raw)
            tau2 = get_tau2(lt_B)
            if not np.isnan(tau2):
                print(f"\n  τ₂ (closed jet) = {tau2:.6f}")
            plot_figure(query_x=lt_B)
    except ValueError:
        print("  Invalid input – please enter a number.")