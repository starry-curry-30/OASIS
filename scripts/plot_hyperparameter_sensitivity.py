#!/usr/bin/env python3
"""Generate hyperparameter sensitivity figure (Instance Coverage vs parameter value)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
    }
)

OUT_DIR = Path(__file__).resolve().parent.parent / "resources" / "image"

# value,lambda,alpha,beta,gamma
X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
SERIES = [
    (r"$\lambda$", [89.13, 89.86, 90.62, 91.13, 90.08], "#1f77b4", "o"),
    (r"$\alpha$", [89.98, 90.42, 90.83, 91.13, 90.57], "#d62728", "s"),
    (r"$\beta$", [90.06, 90.47, 91.13, 90.79, 90.18], "#2ca02c", "^"),
    (r"$\gamma$", [89.92, 90.31, 90.74, 91.13, 90.48], "#9467bd", "D"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)

    all_y = []
    for name, y, color, m in SERIES:
        y = np.asarray(y, dtype=float)
        all_y.append(y)
        ax.plot(
            X,
            y,
            color=color,
            linewidth=1.6,
            marker=m,
            markersize=5.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=name,
        )

    ax.set_xlabel("Hyperparameter value")
    ax.set_ylabel("Instance Coverage (%)")
    ax.set_xlim(0.05, 0.95)
    ymin = float(np.min(all_y)) - 0.35
    ymax = float(np.max(all_y)) + 0.35
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(X)
    ax.legend(loc="lower center", ncol=4, frameon=True, fancybox=False, edgecolor="0.6")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "hyperparameter_sensitivity_coverage.pdf"
    png = OUT_DIR / "hyperparameter_sensitivity_coverage.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    print(f"Wrote {pdf}\nWrote {png}")


if __name__ == "__main__":
    main()
