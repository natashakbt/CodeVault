#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 11:46:37 2025

@author: natasha
"""
#import numpy as np
import pandas as pd
import os
#import math
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.patches as mpatches
import glob
#import matplotlib.colors as mcolors
from matplotlib import gridspec
import seaborn as sns
from scipy.stats import kruskal
import scikit_posthocs as sp

# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/christina_data/'
file_path = os.path.join(dirname, 'gape_metrics_by_trial.pkl')
bout_df = pd.read_pickle(file_path)


# ==============================================================================
# SET PARAMETERS
# ==============================================================================
filter_licl_conc = '0.6M_LiCl'
var_to_plot_list = ['first_gape_bout_start', 'first_gape_bout_duration']
taste_on_train_days = 'saccharin'
taste_on_final_test_day = 'highqhcl'


if not filter_licl_conc in bout_df['licl_conc'].unique():
    print(f"⚠ Warning: '{filter_licl_conc}' is not valid")
    print(f"valid options for filter_licl_conc: {bout_df['licl_conc'].unique()}")

for var_to_plot in var_to_plot_list:
    if not var_to_plot in bout_df.columns:
        print(f"⚠ Warning: var_to_plot set to '{var_to_plot}' is not valid")
        print(f"Variable should correspond to df column name: {bout_df.columns}")



# ==============================================================================
# %% Bar plot - getting setup
# ==============================================================================
# Setup folders for saving figures
# ==============================================================================
# Create subdirectory based on concentration
sub_dir = os.path.join(dirname, 'bar_plot')
os.makedirs(sub_dir, exist_ok=True)


# ==============================================================================
# Setup plotting dataframes
# ==============================================================================
# --- Filter datasets ---
train_df = bout_df[
    (bout_df['num_of_cta'] < bout_df['num_of_cta'].max()) &
    (bout_df['taste_name'] == taste_on_train_days) &
    (bout_df['licl_conc'] == filter_licl_conc)
]

test_df = bout_df[
    (bout_df['num_of_cta'] == bout_df['num_of_cta'].max()) &
    (bout_df['licl_conc'] == filter_licl_conc) & 
    (bout_df['taste_name'] == taste_on_final_test_day)
]


# --- Get unique day categories ---
train_days = sorted(train_df['num_of_cta'].unique())
test_days = sorted(test_df['num_of_cta'].unique())



# ==============================================================================
# %% Barplot of gape metrics with train days + final test day
# ==============================================================================

for var_to_plot in var_to_plot_list:
    # --- Create figure with custom width ratios ---
    widths = [len(train_days), len(test_days)]
    
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=widths)
    
    
    # --- Train subplot ---
    ax1 = fig.add_subplot(gs[0])
    train_box_data = [
        train_df.loc[train_df['num_of_cta'] == c, var_to_plot].dropna().tolist()
        for c in train_days
    ]
    
    ax1.boxplot(train_box_data, labels=train_days)
    ax1.set_xlabel("Number of CTAs")
    ax1.set_ylabel(var_to_plot)
    ax1.set_title(f"Train Days ({taste_on_train_days})")
    
    
    # --- Test subplot ---
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    test_box_data = [
        test_df.loc[test_df['num_of_cta'] == c, var_to_plot].dropna().tolist()
        for c in test_days
    ]
    
    ax2.get_shared_y_axes().join(ax1, ax2)
    ax2.tick_params(labelleft=False) #hide duplicate y-tick labels
    ax2.boxplot(test_box_data, labels=['final test day'])
    ax2.set_title(f"Test Day ({taste_on_final_test_day})")
    
    fig.suptitle(f"{filter_licl_conc}")
    plt.tight_layout()
    
    plot_dir_path_png = os.path.join(sub_dir, f"{filter_licl_conc}_{var_to_plot}_highqhcl_gapes_by_training.png")
    plt.savefig(plot_dir_path_png, bbox_inches = "tight")
    
    plt.show()


# %% Bar plot - split sessions (1/2 way)
# ==============================================================================
# Barplot of gape metric with train days + final test day
# ==============================================================================

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
            split_vals = day_df.loc[day_df['session_trial_split'] == split, var_to_plot].dropna().tolist()
            all_data.append(split_vals)
            labels.append(f"{day} S{split}")
    return all_data, labels

# --- Train subplot ---
ax1 = fig.add_subplot(gs[0])
train_box_data, train_labels = get_split_box_data(train_df, train_days)
ax1.boxplot(train_box_data, labels=train_labels)
ax1.set_xlabel("Number of CTAs / Session")
ax1.set_ylabel(var_to_plot)
ax1.set_title("Train Days ({taste_on_train_days})")

# --- Test subplot ---
ax2 = fig.add_subplot(gs[1], sharey=ax1)
test_box_data, test_labels = get_split_box_data(test_df, test_days)
ax2.boxplot(test_box_data, labels=test_labels)
ax2.set_title("Test Day")
ax2.tick_params(labelleft=False)  # hide duplicate y labels

plt.tight_layout()
plt.show()

# %%

# --- Create figure with custom width ratios ---
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=widths)

# ---------------------- TRAIN SUBPLOT ----------------------
train_plot_df = train_df[['num_of_cta', var_to_plot]].dropna()
test_plot_df = test_df[['num_of_cta', var_to_plot]].dropna()
test_plot_df['num_of_cta'] = 'final test day'   # collapse to single label if desired

# ---------------------- TRAIN SUBPLOT ----------------------
ax1 = fig.add_subplot(gs[0])

train_plot_df = train_df[['num_of_cta', var_to_plot]].dropna()
sns.stripplot(
    data=train_plot_df,
    x='num_of_cta',
    y=var_to_plot,
    ax=ax1
)

ax1.set_xlabel("Number of CTAs")
ax1.set_ylabel(var_to_plot)
ax1.set_title(f"Train Days ({taste_on_train_days})")

# ---------------------- TEST SUBPLOT ----------------------
ax2 = fig.add_subplot(gs[1], sharey=ax1)

sns.stripplot(
    data=test_plot_df,
    x='num_of_cta',
    y=var_to_plot,
    ax=ax2
)

# share y-axis + hide redundant labels
ax2.get_shared_y_axes().join(ax1, ax2)
ax2.tick_params(labelleft=False)
ax2.set_xlabel("")

ax2.set_title(f"Test Day ({taste_on_final_test_day})")

plt.tight_layout()
plt.show()


# ==============================================================================
# %% Stats: Kruskal wallis + dunn's post-hoc
# ==============================================================================

dunn = {}
kw_result = {}
for var_to_plot in var_to_plot_list:
    groups = []
    # train groups
    for c in train_days:
        g = train_df.loc[train_df['num_of_cta'] == c, var_to_plot].dropna().values
        groups.append(g)
    
    # test group
    test_group = test_df[var_to_plot].dropna().values
    groups.append(test_group)
    
    # Run test
    kw_result[var_to_plot] = kruskal(*groups)
    print(f'\nSTATISTICS FOR: {var_to_plot}')
    print("Kruskal-Wallis H =", kw_result[var_to_plot][0])
    print("p-value =", kw_result[var_to_plot][1])
    
    
    if kw_result[var_to_plot][1] < 0.05:
        # build combined dataframe
        all_df = []
        
        for c in train_days:
            temp = train_df.loc[train_df['num_of_cta'] == c, ['num_of_cta', var_to_plot]].dropna()
            #temp = temp.rename(columns={'first_gape_bout_start': 'Value'})
            all_df.append(temp)
        
        test_temp = test_df[['num_of_cta', var_to_plot]].dropna()
        #test_temp = test_temp.rename(columns={'first_gape_bout_start': 'Value'})
        test_temp['num_of_cta'] = 'final test day'
        all_df.append(test_temp)
        
        plot_df = pd.concat(all_df)
        #plot_df = plot_df.rename(columns={'num_of_cta': 'Condition'})
        
        # Dunn post-hoc test
        dunn[var_to_plot] = sp.posthoc_dunn(plot_df, val_col=var_to_plot, group_col='num_of_cta')
        print("Dunn post-hoc:")
        print(dunn[var_to_plot])
    else:
        print("p > 0.05, no post-hoc test")

# ==============================================================================
# %% 
# ==============================================================================





