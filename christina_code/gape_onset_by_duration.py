#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 13:36:48 2025

@author: natasha
"""


import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import glob
import matplotlib.colors as mcolors
from matplotlib import gridspec

# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/christina_data/'
file_path = os.path.join(dirname, 'christina_all_datasets.pkl')
df = pd.read_pickle(file_path)

# Create directory 
plot_dir = os.path.join(dirname, 'gape_start_duration_time')
os.makedirs(plot_dir, exist_ok=True)



# ==============================================================================
# %% Finding first gape bout onset and duration
# ==============================================================================

results = []

for basename_name, basename_group in df.groupby('basename'):
   for (trial_num, taste_name), trial_group in basename_group.groupby(['trial', 'taste_name']):
        gapes = trial_group[trial_group['event_type'] == 'gape'].copy()
        gapes['start_time'] = gapes['segment_bounds'].apply(lambda x: x[0])
        gapes['end_time'] = gapes['segment_bounds'].apply(lambda x: x[1])
        
        gapes = gapes[gapes['start_time'] > 2000].copy()
        gapes = gapes.sort_values('start_time').reset_index(drop=True)
        
        licl_conc = basename_group['licl_conc'].iloc[0]
        exp_day_type = basename_group['exp_day_type'].iloc[0]
        exp_day_num = basename_group['exp_day_num'].iloc[0]
        num_of_cta = basename_group['num_of_cta'].iloc[0]

        bout_found = False
        i = 0
        while i <= len(gapes) - 3:  # need at least 3 gapes
            # Start a potential bout
            bout_indices = [i]
            
            # Extend the bout while gap < 250ms
            j = i
            while j < len(gapes) - 1:
                gap = gapes.loc[j+1, 'start_time'] - gapes.loc[j, 'end_time']
                if gap < 250:
                    bout_indices.append(j+1)
                    j += 1
                else:
                    break
            
            # Check if the bout has at least 3 gapes
            if len(bout_indices) >= 3:
                bout_start = gapes.loc[bout_indices[0], 'start_time']
                bout_end = gapes.loc[bout_indices[-1], 'end_time'] # Last gape in the bout
                #bout_end = gapes['end_time'].iloc[-1] # last gape in the trial
                bout_duration = bout_end - bout_start
                
                results.append({
                    'basename': basename_name,
                    'licl_conc': licl_conc,
                    'exp_day_type': exp_day_type,
                    'exp_day_num': exp_day_num,
                    'trial_num': trial_num,
                    'taste_name': taste_name,
                    'num_of_cta': num_of_cta,
                    'first_gape_bout_start': bout_start,
                    'first_gape_bout_duration': bout_duration
                })
                
                bout_found = True
                break  # first bout found, move to next trial/taste
            else:
                # Not enough gapes, move to next starting index
                i += 1
        
        if not bout_found:
            # No valid bout in this trial
            results.append({
                'basename': basename_name,
                'licl_conc': licl_conc,
                'exp_day_type': exp_day_type,
                'exp_day_num': exp_day_num,
                'trial_num': trial_num,
                'taste_name': taste_name,
                'num_of_cta': num_of_cta,
                'first_gape_bout_start': None,
                'first_gape_bout_duration': None
            })


# Convert to dataframe
bout_df = pd.DataFrame(results)

max_y_value = bout_df['first_gape_bout_duration'].max()
max_y_rounded = math.ceil(max_y_value / 100) * 100
max_y_rounded = 5100


#%% NUM OF VALID BOUTS
results = []

for basename_name, basename_group in df.groupby('basename'):
    for (trial_num, taste_name), trial_group in basename_group.groupby(['trial', 'taste_name']):
        gapes = trial_group[trial_group['event_type'] == 'gape'].copy()
        gapes['start_time'] = gapes['segment_bounds'].apply(lambda x: x[0])
        gapes['end_time'] = gapes['segment_bounds'].apply(lambda x: x[1])
        
        gapes = gapes[gapes['start_time'] > 2000].copy()
        gapes = gapes.sort_values('start_time').reset_index(drop=True)
        
        exp_day_type = basename_group['exp_day_type'].iloc[0]
        exp_day_num = basename_group['exp_day_num'].iloc[0]
        
        num_valid_bouts = 0
        first_bout_start = None
        i = 0
        while i <= len(gapes) - 3:  # need at least 3 gapes
            # Start a potential bout
            bout_indices = [i]
            
            # Extend the bout while gap < 250ms
            j = i
            while j < len(gapes) - 1:
                gap = gapes.loc[j+1, 'start_time'] - gapes.loc[j, 'end_time']
                if gap < 250:
                    bout_indices.append(j+1)
                    j += 1
                else:
                    break
            
            # Check if the bout has at least 3 gapes
            if len(bout_indices) >= 3:
                num_valid_bouts += 1
                # Record start of first valid bout
                if first_bout_start is None:
                    first_bout_start = gapes.loc[bout_indices[0], 'start_time']
                # Move i to next gape after current bout to avoid double-counting
                i = bout_indices[-1] + 1
            else:
                i += 1

        # Append results for this trial
        results.append({
            'basename': basename_name,
            'exp_day_type': exp_day_type,
            'exp_day_num': exp_day_num,
            'trial_num': trial_num,
            'taste_name': taste_name,
            'first_gape_bout_start': first_bout_start,
            'first_gape_bout_duration': num_valid_bouts
        })

# Convert to dataframe
bout_df = pd.DataFrame(results)

max_y_rounded = 5

# ==============================================================================
# %% Scatter plot - unique figure by taste + basename
# ==============================================================================

for (basename, taste_name), group_df in bout_df.groupby(['basename', 'taste_name']):

    x = group_df['first_gape_bout_start']
    y = group_df['first_gape_bout_duration']
    plt.xlim(2000, 7000)
    
    # Create the scatter plot
    plt.scatter(x, y)
    plt.xlabel('Gape bout start (ms)\n(2000ms is taste delivery)')
    plt.ylabel('Gape bout duration (ms)')
    plt.suptitle(f"{basename} - {taste_name}")
    plt.show()

# ==============================================================================
# %% Scatter plot - unique figure by basename - taste subplots
# ==============================================================================

for basename, group_df in bout_df.groupby('basename'):
    taste_names = group_df['taste_name'].unique()
    n_tastes = len(taste_names)
    
    # Flexible grid: roughly square
    n_cols = math.ceil(math.sqrt(n_tastes))
    n_rows = math.ceil(n_tastes / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)
    if isinstance(axes, plt.Axes):  # only one subplot
        axes = [axes]
    else:
        axes = axes.flatten()
    for i, taste in enumerate(taste_names):
        ax = axes[i]
        taste_df = group_df[group_df['taste_name'] == taste]
        
        x = taste_df['first_gape_bout_start']
        y = taste_df['first_gape_bout_duration']
        
        ax.scatter(
            x, y,
            c='steelblue',         # soft blue color
            alpha=0.8,             # some transparency
            s=100,                  # point size
            edgecolors='black',    # white outline
            linewidths=0.7
        )
        ax.set_title(f"{taste}")
        ax.set_xlim(2000, 7000)
        ax.set_ylim(0, max_y_rounded)
        ax.set_xlabel('Gape bout start (ms)')
        ax.set_ylabel('Gape bout duration (ms)')
    
    # Remove unused axes
    #for j in range(i+1, len(axes)):
    #    fig.delaxes(axes[j])
    
    fig.suptitle(f"{basename}", fontsize=19)
    plt.tight_layout()
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])  
    
    plt.show()


# ==============================================================================
# %% Scatter plot - unique figure by exp type/day - taste subplots
# ==============================================================================


for (exp_day_num, exp_day_type), group_df in bout_df.groupby(['exp_day_num', 'exp_day_type']):

    taste_names = group_df['taste_name'].unique()
    n_tastes = len(taste_names)
    
    print(taste_names)
    # Flexible grid: roughly square
    n_cols = math.ceil(math.sqrt(n_tastes))
    n_rows = math.ceil(n_tastes / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)
    if isinstance(axes, plt.Axes):  # only one subplot
        axes = [axes]
    else:
        axes = axes.flatten()
    for i, taste in enumerate(taste_names):
        ax = axes[i]
        taste_df = group_df[group_df['taste_name'] == taste]
        
        x = taste_df['first_gape_bout_start']
        y = taste_df['first_gape_bout_duration']
        
        ax.scatter(
            x, y,
            c='steelblue',         # soft blue color
            alpha=0.8,             # some transparency
            s=100,                  # point size
            edgecolors='black',    # white outline
            linewidths=0.7
        )
        ax.set_title(f"{taste}")
        ax.set_xlim(2000, 7000)
        ax.set_ylim(0, max_y_rounded)
        ax.set_xlabel('Gape bout start (ms)')
        ax.set_ylabel('Gape bout duration (ms)')
    
    # Remove unused axes
    #for j in range(i+1, len(axes)):
    #    fig.delaxes(axes[j])
    
    fig.suptitle(f"CTA {exp_day_type} {exp_day_num}", fontsize=19)
    plt.tight_layout()
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])  
    
    plt.show()


    
# ==============================================================================
# %% Scatter plot - unique figure by exp type/day - taste subplots - color by trial_num
# ==============================================================================


# Create subdirectory based on concentration
sub_dir = os.path.join(plot_dir, 'color_by_trial_num')
os.makedirs(sub_dir, exist_ok=True)

# Remove any existing PNGs in that subfolder
png_files = glob.glob(os.path.join(sub_dir, '*.png'))
for file in png_files:
    os.remove(file)

# TODO: Make color map legend accurate 
# TODO: Make color map the same across all figures


for (licl_conc, exp_day_num, exp_day_type), group_df in bout_df.groupby(['licl_conc', 'exp_day_num', 'exp_day_type']):

    taste_names = group_df['taste_name'].unique()
    n_tastes = len(taste_names)
    
    # Flexible grid: roughly square
    n_cols = math.ceil(math.sqrt(n_tastes))
    n_rows = math.ceil(n_tastes / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)
    if isinstance(axes, plt.Axes):  # only one subplot
        axes = [axes]
    else:
        axes = axes.flatten()

    # --- Define color map by trial number ---
    trial_nums = sorted(group_df['trial_num'].unique())
    cmap = cm.get_cmap('cool', len(trial_nums))
    color_map = {t: cmap(i) for i, t in enumerate(trial_nums)}
    norm_ = mcolors.Normalize(vmin=min(trial_nums), vmax=max(trial_nums))

    # --- Plot ---
    for i, taste in enumerate(taste_names):
        ax = axes[i]
        taste_df = group_df[group_df['taste_name'] == taste]
        
        for trial_num, trial_df in taste_df.groupby('trial_num'):
            x = trial_df['first_gape_bout_start']
            y = trial_df['first_gape_bout_duration']

            ax.scatter(
                x, y,
                c=[color_map[trial_num]],
                alpha=0.8,
                s=100,
                edgecolors='black',
                linewidths=0.7,
                label=f"Trial {trial_num}"
            )

        ax.set_title(f"{taste}")
        ax.set_xlim(2000, 7000)
        ax.set_ylim(0, max_y_rounded)
        ax.set_xlabel('Gape bout start (ms)')
        ax.set_ylabel('Gape bout duration (ms)')
        #ax.legend(title="Trial #", fontsize=8, frameon=False)

    fig.suptitle(f"{licl_conc} - CTA {exp_day_type} {exp_day_num}", fontsize=19)
    #plt.tight_layout(rect=[0, 0, 1, 0.96])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.025, pad=0.04)
    cbar.set_label('Trial #', fontsize=12)
    cbar.ax.tick_params(labelsize=10)


    # Remove unused axes if any
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    #if exp_day_type == 'Train':
    #    plt.show()
        
    plot_dir_path_png = os.path.join(sub_dir, f"{licl_conc}_CTA_{exp_day_type}_{exp_day_num}.png")
    plt.savefig(plot_dir_path_png, bbox_inches = "tight")

        
    plt.show()


# ==============================================================================
# %% Scatter plot - Train days only - color by train day #
# ==============================================================================

filter_licl_conc = '0.15M_LiCl'

# Create subdirectory based on concentration
sub_dir = os.path.join(plot_dir, 'color_by_train_day')
os.makedirs(sub_dir, exist_ok=True)

'''
# Remove any existing PNGs in that subfolder
png_files = glob.glob(os.path.join(sub_dir, '*.png'))
for file in png_files:
    os.remove(file)
'''


#train_df = bout_df[bout_df['exp_day_type'] == 'Train']
train_df = bout_df[
    (bout_df['exp_day_type'] == 'Train') &
    (bout_df['licl_conc'] == filter_licl_conc)
]

if not train_df.empty:
    taste_names = train_df['taste_name'].unique()
    n_tastes = len(taste_names)
    
    n_cols = math.ceil(math.sqrt(n_tastes))
    n_rows = math.ceil(n_tastes / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # --- Color by exp_day_num ---
    day_nums = sorted(train_df['exp_day_num'].unique())
    cmap = cm.get_cmap('autumn', len(day_nums))
    color_map = {d: cmap(i) for i, d in enumerate(day_nums)}

    for i, taste in enumerate(taste_names):
        ax = axes[i]
        taste_df = train_df[train_df['taste_name'] == taste]

        for exp_day_num, day_df in taste_df.groupby('exp_day_num'):
            x = day_df['first_gape_bout_start']
            y = day_df['first_gape_bout_duration']

            ax.scatter(
                x, y,
                c=[color_map[exp_day_num]],
                alpha=0.8,
                s=100,
                edgecolors='black',
                linewidths=0.7,
                label=f"Day {exp_day_num}"
            )

        ax.set_title(f"{taste}")
        ax.set_xlim(2000, 7000)
        ax.set_ylim(0, max_y_rounded)
        ax.set_xlabel('First gape bout start (ms)')
        ax.set_ylabel('First gape bout duration (ms)')
        #ax.legend(title="Train Day", fontsize=8, frameon=False)
    legend_patches = [mpatches.Patch(color=color_map[d], label=f"Day {d}") for d in day_nums]
    fig.legend(
        handles=legend_patches,
        title="Train Day",
        loc='center right',
        #bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        title_fontsize=11,
        frameon=False
    )
    fig.suptitle(f"{filter_licl_conc}-CTA Train Days", fontsize=19)
    #plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_dir_path_png = os.path.join(sub_dir, f'{filter_licl_conc}_gape_timing_by_train_day.png')
    plt.savefig(plot_dir_path_png, bbox_inches = "tight")

    
    plt.show()


# ==============================================================================
# %% Bar plot - getting setup
# ==============================================================================
filter_licl_conc = '0.6M_LiCl'

# ==============================================================================
# Set up folders for saving figures
# ==============================================================================
# Create subdirectory based on concentration
sub_dir = os.path.join(plot_dir, 'bar_plot')
os.makedirs(sub_dir, exist_ok=True)

# Remove any existing PNGs in that subfolder
png_files = glob.glob(os.path.join(sub_dir, '*.png'))
for file in png_files:
    os.remove(file)

# Create subdirectory based on concentration
sub_dir = os.path.join(plot_dir, 'barplot')
os.makedirs(sub_dir, exist_ok=True)


# ==============================================================================
# Find session trial 50% split & add to new column 'session_trial_split'
# ==============================================================================
max_trial = bout_df.groupby(
    ["basename", "exp_day_num", "taste_name"]
)["trial_num"].transform("max")

# compute midpoint split value
midpoint = max_trial / 2

bout_df["session_trial_split"] = (bout_df["trial_num"] > midpoint).astype(int)


# ==============================================================================
# Filter dataframes
# ==============================================================================
# --- Filter datasets ---
train_df = bout_df[
    (bout_df['num_of_cta'] < 4.0) &
    (bout_df['licl_conc'] == filter_licl_conc)
]

test_df = bout_df[
    (bout_df['num_of_cta'] == 4.0) &
    (bout_df['licl_conc'] == filter_licl_conc) & 
    (bout_df['taste_name'] == 'highqhcl')
]


# --- Get unique day categories ---
train_days = sorted(train_df['num_of_cta'].unique())
test_days = sorted(test_df['num_of_cta'].unique())



# %% Bar plot - just sessions
# --- Define proportional widths ---
widths = [len(train_days), len(test_days)]

# --- Create figure with custom width ratios ---
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=widths)

# --- Train subplot ---
ax1 = fig.add_subplot(gs[0])
train_box_data = [
    train_df.loc[train_df['num_of_cta'] == c, 'first_gape_bout_start'].dropna().tolist()
    for c in train_days
]

ax1.boxplot(train_box_data, labels=train_days)
ax1.set_xlabel("Number of CTAs")
ax1.set_ylabel("First Gape Onset (ms)")
ax1.set_title("Train Days")


# --- Test subplot ---
ax2 = fig.add_subplot(gs[1], sharey=ax1)
test_box_data = [
    test_df.loc[test_df['num_of_cta'] == c, 'first_gape_bout_start'].dropna().tolist()
    for c in test_days
]


ax2.get_shared_y_axes().join(ax1, ax2)

# ✅ optional: hide duplicate y-tick labels
ax2.tick_params(labelleft=False)

ax2.boxplot(test_box_data, labels=['final test day'])
ax2.set_title("Test Day")

plt.tight_layout()
plt.show()


# %% Bar plot - split sessions (1/2 way)

# --- Define proportional widths for figure ---
widths = [len(train_days), len(test_days)]
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=widths)

# --- Helper function to get split boxplot data ---
def get_split_box_data(df, day_list):
    all_data = []
    labels = []
    for day in day_list:
        day_df = df[df['num_of_cta'] == day]
        for split in [0, 1]:
            split_vals = day_df.loc[day_df['session_trial_split'] == split, 'first_gape_bout_start'].dropna().tolist()
            all_data.append(split_vals)
            labels.append(f"{day} S{split}")
    return all_data, labels

# --- Train subplot ---
ax1 = fig.add_subplot(gs[0])
train_box_data, train_labels = get_split_box_data(train_df, train_days)
ax1.boxplot(train_box_data, labels=train_labels)
ax1.set_xlabel("Number of CTAs / Session")
ax1.set_ylabel("First Gape Bout Start (ms)")
ax1.set_title("Train Days")

# --- Test subplot ---
ax2 = fig.add_subplot(gs[1], sharey=ax1)
test_box_data, test_labels = get_split_box_data(test_df, test_days)
ax2.boxplot(test_box_data, labels=test_labels)
ax2.set_title("Test Day")
ax2.tick_params(labelleft=False)  # hide duplicate y labels

plt.tight_layout()
plt.show()

