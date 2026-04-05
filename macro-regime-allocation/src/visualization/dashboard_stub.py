"""Placeholder chart helpers for regime dashboard outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_composites(composites: pd.DataFrame) -> plt.Figure:
    """Plot composite macro scores."""
    fig, ax = plt.subplots(figsize=(10, 4))
    composites.plot(ax=ax)
    ax.set_title("Macro Composite Scores")
    ax.set_ylabel("z-score")
    ax.grid(True, alpha=0.3)
    return fig
