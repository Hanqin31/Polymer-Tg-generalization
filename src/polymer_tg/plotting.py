"""Small plotting helpers so every figure looks the same and is saved to
`figures/` in one consistent way."""

from pathlib import Path

import matplotlib.pyplot as plt

# One fixed colour per test setting, used in every figure, so a reader can
# recognise "orange = cluster split" across the whole notebook.
# These three colours are chosen to stay distinguishable for colour-blind
# readers (checked with a colour-vision-deficiency palette validator).
SETTING_COLORS = {
    "Random CV": "#2a78d6",   # blue
    "Cluster CV": "#eb6834",  # orange
    "External": "#1baf7a",    # aqua/green
}
NEUTRAL = "#8a8985"           # grey for reference lines (diagonals, baselines)


def apply_style():
    """Quiet axes and a light grid so the data, not the frame, stands out."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def save_figure(fig, name, fig_dir="../figures"):
    """Save a figure as PNG (for the README) at a readable resolution."""
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    path = Path(fig_dir) / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    return path
