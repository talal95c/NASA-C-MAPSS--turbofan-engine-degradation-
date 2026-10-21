"""Generate publication-style plots from the saved C-MAPSS result JSONs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
Z90 = 1.6448536
BASELINE = "#888888"
ACCENT = "#2f5bea"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
MODELS = ("ridge", "gbm", "cnn_ensemble")

plt.rcParams.update({"font.size": 11, "savefig.dpi": 200})


def read_json(path):
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=11, colors="#555555", length=3)


def plot_trajectory(predictions, output, cover=False):
    val = predictions.get("val_engine", {})
    try:
        cycle = np.asarray(val["cycle"], dtype=float)
        target = np.asarray(val["target"], dtype=float)
        mu = np.asarray(val["mu"], dtype=float)
        sigma = np.asarray(val["sigma"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return False

    if not len(cycle) or not all(len(values) == len(cycle) for values in (target, mu, sigma)):
        return False

    order = np.argsort(cycle, kind="stable")
    cycle, target, mu, sigma = (values[order] for values in (cycle, target, mu, sigma))
    facecolor = "#f5f5f5" if cover else "white"
    figsize = (8.9, 6.76) if cover else (8, 4.8)
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    ax.set_facecolor(facecolor)
    line_width = 2.5 if cover else 1.7
    ax.plot(cycle, target, color=BASELINE, linestyle="--", linewidth=line_width, label="target")
    ax.plot(cycle, mu, color=ACCENT, linewidth=line_width, label="predicted μ")
    ax.fill_between(cycle, mu - Z90 * sigma, mu + Z90 * sigma, color=ACCENT, alpha=0.15,
                    label="90% interval")
    ax.set_xlabel("cycle")
    ax.set_ylabel("remaining useful life (cycles)")
    style_axes(ax)

    if cover:
        fig.subplots_adjust(left=0.15, right=0.96, bottom=0.15, top=0.95)
    else:
        ax.legend(frameon=False, fontsize=11)
        fig.tight_layout()

    fig.savefig(output, dpi=200, facecolor=facecolor)
    plt.close(fig)
    return True


def plot_test(predictions, output):
    test = predictions.get("test", {})
    try:
        true_capped = np.asarray(test["true_capped"], dtype=float)
        true = np.asarray(test["true"], dtype=float)
        mu = np.asarray(test["mu"], dtype=float)
        sigma = np.asarray(test["sigma"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return False

    if not len(true_capped) or not all(len(values) == len(true_capped) for values in (true, mu, sigma)):
        return False

    order = np.argsort(true_capped, kind="stable")
    x = np.arange(1, len(order) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(x, true[order], color=BASELINE, s=20, label="true", zorder=3)
    ax.errorbar(x, mu[order], yerr=Z90 * sigma[order], fmt="o", color=ACCENT, ecolor=ACCENT,
                markersize=3.5, linewidth=1, capsize=2, label="μ ± 90% interval", zorder=4)
    ax.set_xlabel("test engine (sorted by true capped RUL)")
    ax.set_ylabel("remaining useful life (cycles)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)
    return True


def plot_rmse_by_subset(output):
    subsets = []
    scores = []
    for subset in SUBSETS:
        data = read_json(RESULTS / f"{subset}.json")
        if not data:
            continue
        models = data.get("models", {})
        row = {}
        for model in MODELS:
            try:
                row[model] = float(models[model]["rmse"])
            except (KeyError, TypeError, ValueError):
                continue
        if row:
            subsets.append(subset)
            scores.append(row)

    if not subsets:
        return False

    x = np.arange(len(subsets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor="white")
    ax.set_facecolor("white")
    colors = {"ridge": BASELINE, "gbm": BASELINE, "cnn_ensemble": ACCENT}
    for index, model in enumerate(MODELS):
        values = [row.get(model, np.nan) for row in scores]
        ax.bar(x + (index - 1) * width, values, width, color=colors[model], label=model)
    ax.set_xticks(x, subsets)
    ax.set_xlabel("subset")
    ax.set_ylabel("RMSE (cycles)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)
    return True


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    predictions = read_json(RESULTS / "FD001_predictions.json")
    if predictions and plot_trajectory(predictions, FIGURES / "trajectory_FD001.png"):
        plot_trajectory(predictions, FIGURES / "cover.png", cover=True)
        plot_test(predictions, FIGURES / "test_FD001.png")
    plot_rmse_by_subset(FIGURES / "rmse_by_subset.png")


if __name__ == "__main__":
    main()
