"""Headless plotting helpers for VaR backtests and raw return series."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_var(fc, var_levels, path, title=""):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fc.dates, fc.y_true, label="rendement réalisé", lw=.8)
    ax.plot(fc.dates, var_levels, label="VaR", color="red", lw=1)
    br = fc.y_true < var_levels
    ax.scatter(fc.dates[br], fc.y_true[br], color="black", s=12, label="violations")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_returns(series, path, title=""):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(series.index, series.values, lw=.7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
