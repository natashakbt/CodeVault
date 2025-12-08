#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:10:24 2024

@author: natasha
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob


# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/christina_data/'
file_path = os.path.join(dirname, 'christina_all_datasets.pkl')
df = pd.read_pickle(file_path)

# ==============================================================================
# %% plot gapes across time
# ==============================================================================
# Parameters
desired_clust_to_plot = 1  # which cluster_num to plot
window_start = 1500        # start of trial (ms)
window_end = 5000          # end of trial (ms)
n_bins = 1000              # number of time bins for frequency trace
window_size = 20           # smoothing window
filter_licl_conc = '1train_1test_1QHCl'  # set to '0.15M', '0.6M', or None for all



# ==============================================================================
# Filter dataframe by experiment type
# ==============================================================================
if filter_licl_conc is not None:
    df_to_plot = df[df['licl_conc'] == filter_licl_conc]
else:
    df_to_plot = df.copy()


# Group by taste_name
for taste_name, taste_group in df_to_plot.groupby('taste_name'):
    sessions = taste_group['basename'].unique()
    n_sessions = len(sessions)

    fig, axes = plt.subplots(n_sessions, 1, figsize=(12, 3*n_sessions), sharex=True, sharey=False)
    if n_sessions == 1:
        axes = [axes]  # make it iterable

    # Plot each session as a subplot
    for ax, basename in zip(axes, sessions):
        session_group = taste_group[taste_group['basename'] == basename]
        behaviors_to_plot = session_group[session_group['cluster_num'] == desired_clust_to_plot]
        behaviors_to_plot = behaviors_to_plot.drop_duplicates(subset=['segment_bounds'])

        # Time bins
        time_bins = np.linspace(window_start, window_end, n_bins)
        freq_trace = np.zeros_like(time_bins, dtype=float)

        # Build frequency trace
        for _, row in behaviors_to_plot.iterrows():
            start, end = row['segment_bounds']
            start = max(start, window_start)
            end = min(end, window_end)
            freq_trace += ((time_bins >= start) & (time_bins <= end)).astype(float)

        # Smooth frequency trace
        freq_trace_smoothed = np.convolve(freq_trace, np.ones(window_size)/window_size, mode='same')

        # Plot
        ax.step(time_bins, freq_trace_smoothed, color='black', linewidth=2, where='mid')
        ax.axvline(2000, color='r', linestyle='--')
        ax.set_ylabel('Frequency')
        ax.set_title(f"Session: {basename}")

    axes[-1].set_xlabel('Time (ms)')
    plt.suptitle(f"{filter_licl_conc} - Taste: {taste_name}")
    plt.tight_layout()
    plt.show()




# ==============================================================================
# %% Plot all behaviors across all sessions
# ==============================================================================
# Parameters
# ==============================================================================
clusters_to_plot = [0, 1, 2]   # 0 = no movement, 1 = gapes, 2 = MTMs
cluster_labels = {0: "No M.", 1: "Gape", 2: "MTM"}
cluster_colors = {0: '#D3D3D3', 1: '#ff9900', 2: '#4285F4'}
window_start = 1500        # start of trial (ms)
window_end = 5000          # end of trial (ms)
n_bins = 1000              # number of time bins for frequency trace
window_size = 20           # smoothing window
filter_licl_conc = '2train_1test'  # set to '0.15M', '0.6M', or None for all

#['0.15M_LiCl', '0.6M_LiCl', '0.15M_NaCl', '0.6M_NaCl', '1train_1test_0.8QHCl', 
# '1train_1test_1QHCl', '2train_1test']

# ==============================================================================
# Setup directories
# ==============================================================================
plot_dir = os.path.join(dirname, 'movement_freq_by_taste')
os.makedirs(plot_dir, exist_ok=True)

sub_dir = os.path.join(plot_dir, filter_licl_conc)
os.makedirs(sub_dir, exist_ok=True)

# Remove old PNGs
png_files = glob.glob(os.path.join(sub_dir, '*.png'))
for file in png_files:
    os.remove(file)

# ==============================================================================
# Filter dataframe by experiment type
# ==============================================================================
if filter_licl_conc is not None:
    df_to_plot = df[df['licl_conc'] == filter_licl_conc].copy()
else:
    df_to_plot = df.copy()

# ==============================================================================
# Plotting loop
# ==============================================================================
for taste_name, group in df_to_plot.groupby('taste_name'):
    animals = sorted(group['animal_num'].unique())
    all_days = sorted(group['num_of_cta'].unique())
    days = [d for d in all_days if pd.notna(d)]

    print(f'{taste_name}, CTA days: {days}')
    n_rows = len(animals)
    if len(days) > 0:
        n_cols = len(days)
        
        if n_rows == 1 and n_cols ==1:
           fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows),
                                    sharex=True, sharey=True)
        elif n_rows < n_cols:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 5*n_rows),
                                     sharex=True, sharey=True)
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows),
                                     sharex=True, sharey=True)
        # Ensure axes is always 2D array
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if n_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        for i_row, animal in enumerate(animals):
            first_visible_ax = None

            for i_col, day in enumerate(days):
                ax = axes[i_row, i_col]
                session_group = group[
                    (group['animal_num'] == animal) &
                    (group['num_of_cta'] == day)
                ]

                if session_group.empty:
                    ax.axis('off')
                    continue

                total_trials = session_group['trial'].nunique()
                if total_trials == 0:
                    ax.axis('off')
                    print(f"No trials for {animal} {day}??")
                    continue

                if first_visible_ax is None:
                    first_visible_ax = ax

                basename = session_group['basename'].iloc[0]

                # Loop over clusters
                for clust in clusters_to_plot:
                    behaviors_to_plot = (
                        session_group[session_group['cluster_num'] == clust]
                        .drop_duplicates(subset=['segment_bounds'])
                    )
                    if behaviors_to_plot.empty:
                        continue

                    # -----------------------------
                    # Build frequency trace (percent of trials)
                    # -----------------------------
                    time_bins = np.linspace(window_start, window_end, n_bins)
                    trial_nums = sorted(session_group['trial'].unique())
                    freq_matrix = np.zeros((len(trial_nums), n_bins), dtype=bool)

                    for i_trial, trial_num in enumerate(trial_nums):
                        trial_events = session_group[
                            (session_group['trial'] == trial_num) &
                            (session_group['cluster_num'] == clust)
                        ]
                        for _, row in trial_events.iterrows():
                            start, end = row['segment_bounds']
                            start = max(start, window_start)
                            end = min(end, window_end)
                            freq_matrix[i_trial, :] |= ((time_bins >= start) & (time_bins <= end))

                    # Compute % of trials and smooth
                    freq_trace = freq_matrix.sum(axis=0).astype(float)
                    freq_pct = (freq_trace / len(trial_nums)) * 100
                    freq_trace_smoothed = np.convolve(freq_pct, np.ones(window_size)/window_size, mode='same')

                    # Plot
                    ax.step(
                        time_bins, freq_trace_smoothed,
                        color=cluster_colors.get(clust, 'black'),
                        linewidth=2,
                        where='mid',
                        label=cluster_labels.get(clust, f"Cluster {clust}")
                    )

                # -----------------------------
                # Axes lines and limits
                # -----------------------------
                ax.axvline(2000, color='r', linestyle='--')
                ax.set_ylim(0, 100)
                ax.set_xlim(window_start, window_end)

                if i_row == 0:
                    if day == final_test_day_num:
                        ax.set_title("Final test day")
                    else:
                        ax.set_title(f"CTA Num: {day}")
                if i_row == n_rows - 1:
                    ax.set_xlabel('Time (ms)')

            # Set y-label on first visible axis in the row
            if first_visible_ax is not None:
                first_visible_ax.set_ylabel(f"{animal}\n% Trials")

        # -----------------------------
        # Legend
        # -----------------------------
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                handles, labels = axes[i_row, i_col].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, loc='lower center', ncol=len(clusters_to_plot),
                               frameon=True, fontsize=10)
                    break
            else:
                continue
            break

        # -----------------------------
        # Layout and save
        # -----------------------------
        plt.suptitle(f"{taste_name} - {filter_licl_conc}\n", y=0.99)
        plt.tight_layout()
        plot_dir_path_png = os.path.join(sub_dir, f'{filter_licl_conc}_{taste_name}.png')
        plt.savefig(plot_dir_path_png)
        plt.show()



# ==============================================================================
#%% Combined plot:
#    - Top: smoothed % of trials where cluster occurs across time
#    - Bottom: raster of individual trials showing cluster occurrences
# ==============================================================================
# Parameters
# ==============================================================================
clusters_to_plot = [0, 1, 2]   # 0 = no movement, 1 = gapes, 2 = MTMs
animal_to_plot = 'CM30'
taste_name = 'water'
num_of_cta_day= 5.0
#cluster_labels = {0: "No M.", 1: "Gape", 2: "MTM"}
cluster_colors = {0: '#D3D3D3', 1: '#ff9900', 2: '#4285F4'}
window_start = 1500        # start of trial (ms)
window_end = 5000          # end of trial (ms)
n_bins = 1000              # number of time bins for frequency trace
window_size = 20           # smoothing window

for cluster in clusters_to_plot:
    # Filter session data
    session_group = df[
        (df['animal_num'] == animal) &
        (df['taste_name'] == taste_name) &
        (df['num_of_cta'] == day) &
        (df['cluster_num'].isin([cluster]))
    ]
    
    if session_group.empty:
        print(f"No data for {animal}, {taste_name}, day {day}, cluster {cluster}")
        break
    
    total_trials = session_group['trial'].nunique()
    if total_trials == 0:
        print(f"No trials for {animal}, {taste_name}, day {day}")
        break
    
    # ========================
    # Build smoothed % frequency trace
    # ========================
    trial_nums = sorted(session_group['trial'].unique())
    freq_matrix = np.zeros((len(trial_nums), n_bins), dtype=bool)  # <-- bool, not float
    
    for i_trial, trial_num in enumerate(trial_nums):
        trial_events = session_group[session_group['trial'] == trial_num]
        for _, row in trial_events.iterrows():
            start, end = row['segment_bounds']
            start = max(start, window_start)
            end = min(end, window_end)
            freq_matrix[i_trial, :] |= ((time_bins >= start) & (time_bins <= end))
    
    # Sum across trials and convert to %
    freq_trace = freq_matrix.sum(axis=0).astype(float)
    freq_pct = (freq_trace / len(trial_nums)) * 100
    freq_trace_smoothed = np.convolve(freq_pct, np.ones(window_size)/window_size, mode='same')
    
    # ========================
    # Create figure with 2 subplots
    # ========================
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(10, 4 + 0.3*total_trials),  # height scaled by number of trials
        gridspec_kw={'height_ratios': [1, total_trials/10 + 1]}  # top smaller, bottom larger
    )
    
    # Top: smoothed % frequency
    ax_top.step(time_bins, freq_trace_smoothed, color='black', linewidth=3)
    ax_top.set_ylabel('% Trials')
    ax_top.set_xlim(window_start, window_end)
    ax_top.set_ylim(0, 100)
    ax_top.set_title(f'Cluster {cluster} - {animal} - {taste_name} - CTA {day}')
    ax_top.axvline(2000, color='r', linestyle='--')
    
    # Bottom: raster plot
    cluster_events = session_group.copy()
    for trial_num, trial_group in cluster_events.groupby('trial'):
        for _, row in trial_group.iterrows():
            start, end = row['segment_bounds']
            start = max(start, window_start)
            end = min(end, window_end)
            ax_bottom.broken_barh([(start, end-start)], (trial_num-0.4, 0.8), 
                                  facecolor=cluster_colors.get(cluster, 'black'),
                                  )
    ax_bottom.axvline(2000, color='r', linestyle='--')
    ax_bottom.set_ylim(0.5, total_trials+0.5)
    ax_bottom.set_xlim(window_start, window_end)
    ax_bottom.set_xlabel('Time (ms)')
    ax_bottom.set_ylabel('Trial #')
    
    plt.tight_layout()
    plt.show()


