#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:08:46 2025

@author: natasha
"""

import os
import matplotlib.patches as patches
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.stats import wilcoxon
from mtm_analysis_config import dirname

# ==============================================================================
# load data
# ==============================================================================
file_path = os.path.join(dirname, 'UMAP_of_all_MTMs_plot_df.pkl') # all events from classifier predictions
plot_df = pd.read_pickle(file_path)

final_figures_dir = os.path.join(dirname, "final_figures")


# %% Plot p-value results from divergence test
# ==============================================================================

fig, ax = plt.subplots(figsize=(12, 9))

# Stripplot where dots are p-value results
sns.stripplot(
    data=plot_df,
    x='session_id', y='divergence_p',
    jitter=True, s=10,
    facecolor='dimgray', edgecolor='black', linewidth=1,
    ax=ax
)

# Boxplot overlay to show distribution of results for each session
sns.boxplot(
    data=plot_df,
    x='session_id', y='divergence_p',
    fill=True, color='white', linewidth=2.5, width=.5,
    showfliers=False, 
    ax=ax
)

# Add rainbow background rectangles indicatin sessions belonging to the same rat
# Pick colors in rainbow palette then shuffle so it's not a smooth gradient
rats = sorted(plot_df['rat'].unique())  # stable order of rats
palette = sns.color_palette("rainbow", n_colors=len(rats))
random.seed(55)  # ensure consistent shuffle every run
random.shuffle(palette)
rat_colors = {rat: color for rat, color in zip(rats, palette)}

# Map session_id to its actual x-axis position
session_order = plot_df['session_id'].unique()
session_to_x = {session: i for i, session in enumerate(session_order)}

ylim = ax.get_ylim()  # get correct y-axis limits

for rat, group in plot_df.groupby('rat'):
    x_positions = [session_to_x[s] for s in group['session_id']]
    xmin = min(x_positions) - 0.5
    xmax = max(x_positions) + 0.5
    ax.add_patch(
        patches.Rectangle(
            (xmin, ylim[0]),
            xmax - xmin,
            ylim[1] - ylim[0],
            facecolor=rat_colors[rat],
            alpha=0.25,
            zorder=0
        )
    )
# Horizontal line at p = 0.05
ax.axhline(y=0.05, color='red', linestyle='--', linewidth=4, label='p = 0.05')

# Labels & formatting
ax.set_title("Divergence P-values by Session")
ax.set_ylabel("P-value", fontsize=18, labelpad=10)
ax.set_xlabel("session_id", fontsize=18, labelpad=10)
ax.tick_params(axis='y', labelsize=14)

# Save
for ext in ["svg", "png"]:
    plt.savefig(os.path.join(final_figures_dir, f"divergence_pvalues_plot.{ext}"), 
                format=ext, bbox_inches='tight')

plt.show()


# %% Wilcoxon stats
# ==============================================================================

results = []

for session_id, group in plot_df.groupby("session_id"):
    try:
        stat, p = wilcoxon(group["divergence_p"] - 0.05, alternative="less")
        results.append({"session_id": session_id, "stat": stat, "pval": p})
    except ValueError:
        # happens if group has too few values or all values == 0.05
        results.append({"session_id": session_id, "stat": None, "pval": None})

results_df = pd.DataFrame(results)
print(results_df)

