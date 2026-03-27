"""
Figure 10.38 – τ₂ for closed elliptic jets.

Loads digitized data from CSV, fits analytical model to smooth digitization
noise, then allows lookup by either lt/B ratio or raw tail length lt.

CSV expected next to this script:
    FIG_10_38_closedjet.csv

Fit:  τ₂ = a · (lt/B)^b / (1 + c · (lt/B)^d)
  a=4.36978, b=1.09242, c=3.05057, d=1.41271
  R² = 0.9995,  RMSE = 0.0084

Usage
-----
Run interactively:   python tau2_lookup.py
Import:              from tau2_lookup import get_tau2, get_tau2_from_lt
                     tau2 = get_tau2(1.2)          # from lt/B ratio
                     tau2 = get_tau2_from_lt(2.16) # from raw lt [m]
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Constants ─────────────────────────────────────────────────────────────
_B = 1.8      # tunnel width [m]

# ── Load CSV (next to this script) ────────────────────────────────────────
_CSV = Path(__file__).parent / "FIG_10_38_closedjet.csv"


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with open(path) as f:
        lines = f.readlines()
    for line in lines[2:]:          # skip two header rows
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        xs.append(float(parts[0].strip()))
        ys.append(float(parts[1].strip()))
    order = np.argsort(xs)
    x = np.array(xs)[order]
    y = np.array(ys)[order]
    # drop any x <= 0 (can't fit power law there)
    mask = x > 0
    return x[mask], y[mask]


x_data, y_data = _load_csv(_CSV)

# ── Fit analytical model ───────────────────────────────────────────────────
def _model(x, a, b, c, d):
    """Rational power law: a·x^b / (1 + c·x^d)"""
    return a * x**b / (1 + c * x**d)


_params, _ = curve_fit(_model, x_data, y_data,
                        p0=[1.5, 0.6, 0.5, 1.0], maxfev=10000)
_A, _B_exp, _C, _D = _params

_LT_B_MIN = 0.0
_LT_B_MAX = 2.0


def get_tau2(lt_over_B: float) -> float:
    """
    Return τ₂ for a given lt/B ratio (closed jet).

    Parameters
    ----------
    lt_over_B : float
        Tail length / tunnel width  (valid range: 0 – 2)

    Returns
    -------
    float
        τ₂, or NaN with a warning if out of range.
    """
    x = float(lt_over_B)
    if not (_LT_B_MIN <= x <= _LT_B_MAX):
        print(f"  ⚠  lt/B = {x:.4f} is outside the valid range "
              f"[{_LT_B_MIN}, {_LT_B_MAX}].")
        return float("nan")
    if x == 0.0:
        return 0.0
    return float(_model(x, _A, _B_exp, _C, _D))


def get_tau2_from_lt(lt: float, B: float = _B) -> dict:
    """
    Return τ₂ given raw tail length lt and tunnel width B.

    Parameters
    ----------
    lt : float
        Tail length [m]
    B  : float
        Tunnel width [m]  (default: 1.8 m)

    Returns
    -------
    dict with keys: lt_over_B, tau2
    """
    lt_over_B = lt / B
    tau2 = get_tau2(lt_over_B)
    return {"lt_over_B": lt_over_B, "tau2": tau2}


# ── Plot ───────────────────────────────────────────────────────────────────
def plot_figure(query_lt: float | None = None, B: float = _B):
    x_fine = np.linspace(0.001, _LT_B_MAX, 600)
    y_fit  = _model(x_fine, _A, _B_exp, _C, _D)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, show_pts in zip(axes, [False, True]):
        ax.plot(x_fine, y_fit, "k-", linewidth=1.8, label="Analytical fit (closed jet)")
        if show_pts:
            ax.scatter(x_data, y_data, s=8, color="steelblue",
                       alpha=0.6, zorder=3, label="Digitized points")

        if query_lt is not None:
            lt_over_B = query_lt / B
            qt2 = get_tau2(lt_over_B)
            if not np.isnan(qt2):
                ax.axvline(lt_over_B, color="tab:red", linestyle="--",
                           linewidth=0.9, alpha=0.8)
                ax.axhline(qt2, color="tab:red", linestyle="--",
                           linewidth=0.9, alpha=0.8)
                ax.scatter([lt_over_B], [qt2], color="tab:red", zorder=5, s=60,
                           label=(f"$l_t={query_lt:.3f}$ m  →  "
                                  f"$l_t/B={lt_over_B:.3f}$,  "
                                  f"$\\tau_2={qt2:.4f}$"))

        ax.set_xlim(0, 2.0)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel(r"Tail length / Tunnel width,  $l_t\,/\,B$", fontsize=11)
        ax.set_ylabel(r"$\tau_2$", fontsize=13, rotation=0, labelpad=14)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_title("Fit only" if not show_pts else "Fit + digitized points",
                     fontsize=10)

    fig.suptitle(
        r"Figure 10.38 – $\tau_2$ for closed elliptic jet  "
        r"($\tau_2 = a\,x^b/(1+c\,x^d)$,  R²=0.9995)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  τ₂ lookup  –  Figure 10.38  (closed jet)")
    print(f"  Formula: τ₂ = {_A:.5f}·x^{_B_exp:.5f} / (1 + {_C:.5f}·x^{_D:.5f})")
    print(f"  R² ≈ 0.9995   RMSE ≈ 0.0084   valid lt/B: 0 – 2")
    print(f"  Tunnel width B = {_B} m")
    print("=" * 60)
    print()
    try:
        raw = input("  Enter tail length lt [m] (or press Enter to show plot): ").strip()
        if raw == "":
            plot_figure()
        else:
            lt_val = float(raw)
            res = get_tau2_from_lt(lt_val)
            if not np.isnan(res["tau2"]):
                print(f"\n  lt/B  = {res['lt_over_B']:.6f}  (= {lt_val} / {_B})")
                print(f"  τ₂    = {res['tau2']:.6f}")
            plot_figure(query_lt=lt_val)
    except ValueError:
        print("  Invalid input – please enter a number.")