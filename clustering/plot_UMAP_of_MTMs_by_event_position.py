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

dirname = '/home/natasha/Desktop/clustering_data/'
file_path = os.path.join(dirname, 'UMAP_of_all_MTMs_plot_df.pkl') # all events from classifier predictions
plot_df = pd.read_pickle(file_path)



desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop_path, "divergence_pvalues_plot.svg")


#%%

# Get axis
fig, ax = plt.subplots(figsize=(12, 9))

# Stripplot with gray dots + black outline
sns.stripplot(
    data=plot_df,
    x='session_id', y='divergence_p',
    jitter=True, s=10,
    facecolor='dimgray', edgecolor='black', linewidth=1,
    ax=ax
)

# Boxplot overlay
sns.boxplot(
    data=plot_df,
    x='session_id', y='divergence_p',
    fill=True, color='white', linewidth=2.5, width=.5,
    showfliers=False, ax=ax
)
'''
# Set palette and add background colors by rat
rat_colors = {
    rat: color for rat, color in zip(
        plot_df['rat'].unique(),
        sns.color_palette("rainbow", n_colors=plot_df['rat'].nunique())
    )
}
'''

rats = sorted(plot_df['rat'].unique())  # stable order of rats
palette = sns.color_palette("rainbow", n_colors=len(rats))

random.seed(55)  # ensures consistent shuffle every run
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
# Horizontal line
ax.axhline(y=0.05, color='red', linestyle='--', linewidth=4, label='p = 0.05')

# Labels & formatting
ax.set_title("Divergence P-values by Session")
ax.set_ylabel("P-value", fontsize=18, labelpad=10)
#ax.set_ylim(-0.002, 0.08)
ax.set_xlabel("session_id", fontsize=18, labelpad=10)
#ax.set_xticks(plot_df['session_id'].unique())
#ax.set_xticklabels(plot_df['session_id'], rotation=45, fontsize=14)
ax.tick_params(axis='y', labelsize=14)

# Save
plt.savefig(save_path, format='svg', bbox_inches='tight')
plt.show()


