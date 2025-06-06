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
from scipy.stats import chi2_contingency, zscore, ttest_rel, chisquare, power_divergence, percentileofscore
import scipy.stats as stats
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
import shutil
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import piecewise_regression
import umap
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm


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


fig_dir = os.path.join(dirname, 'UMAP_of_MTMs_by_event_position')
os.makedirs(fig_dir, exist_ok=True)

files = glob.glob(os.path.join(fig_dir, '*'))
for file in files:
    os.remove(file)  # Remove each file


p_val_dict = {}
n_iterations = 10

unique_sessions = transition_events_df['basename'].unique()
for n in tqdm(range(n_iterations)):
    for basename in unique_sessions:
        session_df = transition_events_df[transition_events_df['basename'] == basename]
        mtm_df = session_df[session_df['event_type'] == 'MTMs']
        mtm_features = np.stack(mtm_df.raw_features.values)
        
        # UMAP dimmentionality reduction and feature scaling
        reducer = umap.UMAP()
        scaled_mtm_features = StandardScaler().fit_transform(mtm_features) # Scale features
        embedding = reducer.fit_transform(scaled_mtm_features) # UMAP embedding
    
        # Plot the UMAP projections
        mtm_df = mtm_df.reset_index(drop=True)
        embedding_df = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'event_position': mtm_df['event_position'].values
        })
        
        if n == 0:
            plt.clf()
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
            fig_path = os.path.join(fig_dir, f'{basename}_countour_plot.png')
            plt.savefig(fig_path)
            
        before_values = embedding_df[embedding_df['event_position'] == 'before']
        after_values = embedding_df[embedding_df['event_position'] == 'after']
        
        bin_num = 10
        hb = plt.hist2d(before_values['x'], 
            before_values['y'], 
            bins = bin_num, 
            density=True
            )
        ha = plt.hist2d(after_values['x'], 
            after_values['y'], 
            bins = bin_num, 
            density = True
            )
        
        h_result = hb[0] - ha[0]
        h_mask = (ha[0] >0 *1) + (hb[0]>0 * 1)
        # plt.imshow(h_mask*1)
        
        hresult_flat = h_result[h_mask].flatten()
        actual_stat = np.abs(hresult_flat).sum()
        if n == 0:
            # For Abu: Does this plot the difference between after and before?
            plt.clf()
            plt.contourf(ha[1][:-1], ha[2][:-1], h_result); plt.colorbar() 
            plt.show()
 
            # Plot h_result as a matrix
            plt.clf()
            plt.imshow(h_result, cmap = 'RdBu')
            plt.colorbar(label='Before - After Values')
            plt.title(f'{basename} \nbins={bin_num}')
            plt.show()
            
            # Plot h_result, but smoothed
            h_result_smooth = gaussian_filter(h_result, sigma=1.0)
            
            plt.clf()
            plt.contourf(
                ha[1][:-1], ha[2][:-1], h_result_smooth,
                levels=100,
                cmap='RdBu_r',
                norm=TwoSlopeNorm(vcenter=0) # Make sure white is 0
            )
            plt.colorbar(label='Smoothed Before - After Density\ncentered on 0')
            plt.title(f'{basename}\nSmoothed Histogram Difference')
            fig_path = os.path.join(fig_dir, f'{basename}_smooth_difference.png')
            plt.savefig(fig_path)

        # chi2_statistic, p_value = chisquare(
        #     hresult_flat,
        #     f_exp = np.ones(len(hresult_flat)*1e-6)
        #     )
        
        # chi2_statistic, p_value = power_divergence(
        #     hresult_flat,
        #     f_exp = np.zeros(len(hresult_flat))
        #     )
    
        #print("Chi-square statistic:", chi2_statistic)
        #print("P-value:", p_value)
        
        n_boot = 1000
        sh_stat = []
        merged_df = pd.concat([before_values, after_values], axis = 0)
        for i_shuff in range(n_boot):
            before_sh = merged_df.sample(n=len(before_values))
            after_sh = merged_df.sample(n=len(after_values))
            
            hb_sh = plt.hist2d(before_sh['x'], 
                before_sh['y'], 
                bins = bin_num, 
                density=True
            )
            ha_sh = plt.hist2d(after_sh['x'], 
                after_sh['y'], 
                bins = bin_num, 
                density = True
            )
            
            h_result_sh = hb_sh[0] - ha_sh[0]
            h_mask_sh = (ha_sh[0] >0 *1) + (hb_sh[0]>0 * 1)

            hresult_flat_sh = h_result_sh[h_mask_sh].flatten()
            sh_stat.append(np.abs(hresult_flat_sh).sum())
            
        divergence_p = 1 - (percentileofscore(sh_stat, actual_stat)/100)
            
        plt.hist(sh_stat);plt.axvline(actual_stat, c = 'red', linestyle = '--')
        # print(divergence_p)
        if basename not in p_val_dict:
            p_val_dict[basename] = []
        p_val_dict[basename].append(divergence_p)
    
        plt.clf()
        

# Convert the dictionary into a tidy DataFrame
plot_df = pd.DataFrame([
    {'basename': key, 'divergence_p': val}
    for key, vals in p_val_dict.items()
    for val in vals
])

# Plot
plt.figure(figsize=(10, 5))
sns.stripplot(data=plot_df, x='basename', y='divergence_p', jitter=True)
plt.axhline(y=0.05, color='red', linestyle='--', label='p = 0.05')  

plt.xticks(rotation=45)
plt.title("Divergence P-values by Session")
plt.ylabel("Divergence P-value")
plt.xlabel("Session (basename)")
plt.tight_layout()
fig_path = os.path.join(fig_dir, 'divergence_p_values.png')
plt.savefig(fig_path)


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




