#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 09:55:51 2025

@author: natasha
"""


import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import glob
from scipy.stats import chi2_contingency
from scipy.stats import zscore
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
import shutil
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d  # for smoothing
from scipy.optimize import curve_fit
import piecewise_regression
from scipy.stats import ttest_rel
import scipy.stats as stats
import umap
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/clustering_data/'
file_path = os.path.join(dirname, 'clustering_df_update.pkl')
df = pd.read_pickle(file_path)

transition_file_path = os.path.join(dirname, 'scaled_mode_tau_cut.pkl')
transition_df = pd.read_pickle(transition_file_path)

# Remove any data for df that does not have an associated transition time in scaled_mode_tau
df['basename'] = df['basename'].str.lower() # All basenames to lowercase
transition_df['basename'] = transition_df['basename'].str.lower() # All basenames to lowercase
transition_df = transition_df.rename(columns={'taste': 'taste_num'}) # NEW changed column name.
tau_basenames = transition_df.basename.unique() # Find all basenames in transition_df
df = df.loc[df['basename'].isin(tau_basenames)] # Keep only basenames 
# Manually removed this specific data:
df = df[~((df['basename'] == 'km50_5tastes_emg_210911_104510_copy') & (df['taste'] == 1))]
df = df[~((df['basename'] == 'km50_5tastes_emg_210911_104510_copy') & (df['taste'] == 4))]

# ==============================================================================
# Important variables to set
# ==============================================================================
window_len = 500 # Half of the total window
fixed_transition_time = 3000# Set to math.nan or a fixed time from stimulus delivery (2000ms+). If this is not nan it will be used over chosen transition
chosen_transition = 1 # Choose out of 0, 1, or 2 (palatability transition is 1); MAKE SURE TO SET ABOVE TO math.nan


# ==============================================================================
# Define functions
# ==============================================================================
def assign_pal_taste(row):
    # Map taste_name to 1, 0, or -1 based on conditions
    if row['taste_name'] in ['nacl', 'water', 'suc']:
        return 1
    elif row['taste_name'] in ['ca', 'qhcl']:
        return 0
    else:
        print(f"Unknown palatability of taste: {row['taste_name']}")
        return -1

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def convert_taste_num_to_name(basename, taste_num, df):
    result = df.loc[
        (df['basename'] == basename) & (df['taste'] == taste_num), 'taste_name'
    ]
    if not result.empty:
        return result.iloc[0]
    else:
        print("Uh oh. No taste num to name match")
        return None


# %% IMPORTANT DATAFRAME SETUP
# ==============================================================================
# Re-structure transition dataframe
# Create DataFrame of events around the transition
# ==============================================================================
# Initialize lists to store the expanded data
basename_list = []
taste_num_list = []
trial_num_list = []
scaled_mode_tau_list = []


# Iterate over each row in the original dataframe
for i, row in transition_df.iterrows():
    basename = row['basename']
    taste_num = row['taste_num']
    scaled_mode_tau = row['scaled_mode_tau']
    
    # Iterate over the 30 elements in 'scaled_mode_tau'
    for trial_num, tau_array in enumerate(scaled_mode_tau):
        basename_list.append(basename)
        taste_num_list.append(taste_num)
        trial_num_list.append(trial_num)
        scaled_mode_tau_list.append(tau_array[chosen_transition])

# Create the new dataframe
expanded_df = pd.DataFrame({
    'basename': basename_list,
    'taste_num': taste_num_list,
    'trial_num': trial_num_list,
    'scaled_mode_tau': scaled_mode_tau_list
})
expanded_df['basename'] = expanded_df['basename'].str.lower()


# Create DataFrame that only contains events that are whithin the transition window
rows = []

for i in range(len(expanded_df)):
#for i in range(1):
    session_df = df[df['session_ind'] == i]
    for index, row in session_df.iterrows():
        segment_bounds = row['segment_bounds']
        trial = row['trial']
        taste = row['taste']
        basename = row['basename'].lower()
        if math.isnan(fixed_transition_time):
            transition_time_point = expanded_df.loc[
                (expanded_df['trial_num'] == trial) & 
                (expanded_df['taste_num'] == str(taste)) & 
                (expanded_df['basename'] == basename), 
                'scaled_mode_tau'
            ].values[0]
        else:
            transition_time_point = fixed_transition_time # to align to fixed palatability transition
        
        window_start = transition_time_point - window_len
        window_end = transition_time_point + window_len

        # Append wavelength with adjusted start/stops if wavelength is within the window
        if window_start <= segment_bounds[0] <= window_end and window_start <= segment_bounds[1] <= window_end:
            new_row = row.copy()  # Copy row to modify it safely
            new_row['time_from_trial_start'] = (segment_bounds[0] - window_start, segment_bounds[1] - window_start) # Alter segment time to be from trial start
            rows.append(new_row)
        # Adjust wavelength stop time if ends after the window
        elif window_start <= segment_bounds[1] <= window_end:
            new_row = row.copy()
            new_row['segment_bounds'] = (window_start, segment_bounds[1])
            new_row['time_from_trial_start'] = (0, segment_bounds[1] - window_start)
            cut_idx = window_start - segment_bounds[0]
            new_row['segment_raw'] = row['segment_raw'][cut_idx:]
            rows.append(new_row)
        # Adjust wavelength start time if it starts before the window
        elif window_start <= segment_bounds[0] <= window_end:
            new_row = row.copy()
            new_row['segment_bounds'] = (segment_bounds[0], window_end)
            new_row['time_from_trial_start'] = (segment_bounds[0] - window_start, window_len*2)
            cut_idx = window_end - segment_bounds[0]
            new_row['segment_raw'] = row['segment_raw'][:cut_idx]
            rows.append(new_row)


            
# Create a DataFrame from the list of rows
transition_events_df = pd.DataFrame(rows).reset_index(drop=True)
transition_events_df = transition_events_df.drop(columns = ['segment_norm_interp'])




# %% 
# ==============================================================================
# Adding column with before/after event position designation for each waveform
# ==============================================================================

lookup_df = transition_events_df[['taste', 'taste_name', 'basename']].drop_duplicates()
grouped = transition_events_df.groupby(['basename', 'taste'])

transition_events_df['event_position'] = np.nan

for idx, row in transition_events_df.iterrows():
    segment_bounds = row['segment_bounds']

    event_position = "before" if segment_bounds[1] < 2800 else "after"
    transition_events_df.at[idx, 'event_position'] = event_position

     
# %% UMAP of MTM features by test session
# ==============================================================================
# UMAP of waveforms by test session
# ==============================================================================

unique_sessions = transition_events_df['basename'].unique()
for basename in unique_sessions:
    session_df = transition_events_df[transition_events_df['basename'] == basename]
    mtm_df = session_df[session_df['event_type'] == 'MTMs']
    mtm_features = np.stack(mtm_df.raw_features.values)
    
    # UMAP dimmentionality reduction and feature scaling
    reducer = umap.UMAP()
    scaled_mtm_features = StandardScaler().fit_transform(mtm_features) # Scale features
    embedding = reducer.fit_transform(scaled_mtm_features) # UMAP embedding
    
    0
    # Plot the UMAP projections with optimal GMM clusters
    #plt.scatter(embedding[:,0], embedding[:,1], s=5)
    mtm_df = mtm_df.reset_index(drop=True)
    embedding_df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'event_position': mtm_df['event_position'].values
    })
    
    before_values = embedding_df[embedding_df['event_position'] == 'before']


    after_values = embedding_df[embedding_df['event_position'] == 'after']
    
    bin_num = 15
    hb = plt.hist2d(before_values['x'], before_values['y'], bins = bin_num)
    ha = plt.hist2d(after_values['x'], after_values['y'], bins = bin_num)
    
    h_result = hb[0] - ha[0]
    
    plt.clf()
    '''
    plt.imshow(h_result, cmap = 'RdBu', vmin=-12, vmax=12)
    plt.colorbar(label='Before - After Values')
    plt.title(f'{basename} \nbins={bin_num}')
    '''
    g = sns.displot(
        data=embedding_df,
        x='x',
        y='y',
        hue='event_position',
        kind='kde',
        height=6,
        aspect=1
    )

    g.fig.suptitle(f'{basename}')
    plt.show()
    plt.clf()


# %% UMAP of all MTM features
# ==============================================================================
# UMAP of all MTM features
# ==============================================================================

all_mtm_df = transition_events_df[transition_events_df['event_type'] == 'MTMs']
all_mtm_features = np.stack(all_mtm_df.features.values)
all_mtm_df = all_mtm_df.reset_index(drop=True)


reducer = umap.UMAP()
all_scaled_mtm_features = StandardScaler().fit_transform(all_mtm_features) # Scale features
all_embedding = reducer.fit_transform(all_scaled_mtm_features) # UMAP embedding


all_embedding_df = pd.DataFrame({
    'x': all_embedding[:, 0],
    'y': all_embedding[:, 1],
    'event_position': all_mtm_df['event_position'].values
})


sns.displot(
    data=all_embedding_df,
    x='x',
    y='y',
    hue='event_position',
    kind='kde',
    height=6,
    aspect=1
)
plt.show()

# %% UMAP of all MTM features
# ==============================================================================
# UMAP of all MTM features
# ==============================================================================




