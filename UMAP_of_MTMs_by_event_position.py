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
file_path = os.path.join(dirname, 'all_datasets_emg_pred.pkl') # all events from classifier predictions
df = pd.read_pickle(file_path)
df = df[~df['laser']]

transition_df = pd.read_pickle(file_path) # Replace with another path if you want to use real transtiion times (something like scaled_mode_tau.pkl)


# ==============================================================================
# Important variables to set
# ==============================================================================
window_len = 500 # Half of the total window to evaluate around transition window
fixed_transition_time = 2800# Set to np.nan or a fixed time from stimulus delivery (2000ms+). If this is not nan it will be used over chosen transition
#fixed_transition_time = np.nan# Set to np.nan or a fixed time from stimulus delivery (2000ms+). If this is not nan it will be used over chosen transition



chosen_transition = 1 # Choose out of 0, 1, or 2 (palatability transition is 1); MAKE SURE TO SET ABOVE TO np.nan


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



# ==============================================================================
# %% IMPORTANT DATAFRAME SETUP
# Re-structure transition dataframe
# Create DataFrame of events around the transition
# ==============================================================================

# Initialize dataframe that will contain every transition time for every trial
expanded_df = df[['taste_num', 'trial_num', 'basename']].drop_duplicates().reset_index(drop=True)

# Set transition times, 
if np.isnan(fixed_transition_time):
    print(f"Using chosen transition #{chosen_transition}")
    for i, row in expanded_df.iterrows():
        basename = row['basename']
        taste_num = str(row['taste_num'])
        scaled_mode_tau_row = transition_df[(transition_df['basename'] == basename) & 
                                        (transition_df['taste_num'] == taste_num)]
        if scaled_mode_tau_row.empty:
            print(f"No match for {basename} and {taste_num}")
        else:
            scaled_mode_tau = scaled_mode_tau_row['scaled_mode_tau'].values[0]
            
            for trial_num, transition_set in enumerate(scaled_mode_tau):
                tau_value = transition_set[chosen_transition]
                
                mask = ((expanded_df['basename'] == basename) &
                    (expanded_df['taste_num'] == int(taste_num)) &
                    (expanded_df['trial_num'] == trial_num))
                
                expanded_df.loc[mask, 'scaled_mode_tau'] = tau_value

elif fixed_transition_time > 2000:
    print(f"Using fixed transition time of {fixed_transition_time}")
    expanded_df['scaled_mode_tau'] = fixed_transition_time
else:
    print("Something's wrong with the chosen transition timing variable")
    
if 'scaled_mode_tau' not in expanded_df.columns:
    raise ValueError("'scaled_mode_tau' column is missing")
elif expanded_df['scaled_mode_tau'].isna().any():
    print("Warning: Some trials are missing a transition time")


# ADDING UNIQUE SESSION INDs RELATED TO BASENAMES
# TODO: MAKE THIS PART OF INITIALIZING THE DATAFRAME
basename_to_index = {name: i for i, name in enumerate(df['basename'].unique())}
df['session_ind'] = df['basename'].map(basename_to_index)

# Create DataFrame that only contains events that are within the transition window
rows = []

for i in tqdm(range(len(expanded_df))):
#for i in range(1):
    session_df = df[df['session_ind'] == i]
    for index, row in session_df.iterrows():
        segment_bounds = row['segment_bounds']
        trial = row['trial_num']
        taste = row['taste_num']
        basename = row['basename'].lower()
        if np.isnan(fixed_transition_time):
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


        # Check segment bounds and adjust
        start_in = window_start <= segment_bounds[0] <= window_end
        end_in = window_start <= segment_bounds[1] <= window_end

        if start_in and end_in:
            new_bounds = (segment_bounds[0] - window_start, segment_bounds[1] - window_start)
            row['time_from_trial_start'] = new_bounds
            rows.append(row)

        elif end_in:
            cut_idx = window_start - segment_bounds[0]
            row['segment_bounds'] = (window_start, segment_bounds[1])
            row['time_from_trial_start'] = (0, segment_bounds[1] - window_start)
            row['segment_raw'] = row['segment_raw'][cut_idx:]
            rows.append(row)

        elif start_in:
            cut_idx = window_end - segment_bounds[0]
            row['segment_bounds'] = (segment_bounds[0], window_end)
            row['time_from_trial_start'] = (segment_bounds[0] - window_start, 2 * window_len)
            row['segment_raw'] = row['segment_raw'][:cut_idx]
            rows.append(row)
            
# Create a DataFrame from the list of rows
transition_events_df = pd.DataFrame(rows).reset_index(drop=True)
transition_events_df = transition_events_df.drop(columns = ['segment_norm_interp'])




# %% 
# ==============================================================================
# Adding column with before/after event position designation for each waveform
# ==============================================================================

lookup_df = transition_events_df[['taste_num', 'taste_name', 'basename']].drop_duplicates()
grouped = transition_events_df.groupby(['basename', 'taste_num'])

transition_events_df['event_position'] = np.nan

for idx, row in transition_events_df.iterrows():
    segment_bounds = row['segment_bounds']

    event_position = "before" if segment_bounds[1] < 2800 else "after"
    transition_events_df.at[idx, 'event_position'] = event_position

     

# ==============================================================================
# %% UMAP of MTM features by test session
# ==============================================================================


fig_dir = os.path.join(dirname, 'UMAP_of_MTMs_by_event_position')
os.makedirs(fig_dir, exist_ok=True)

files = glob.glob(os.path.join(fig_dir, '*'))
for file in files:
    os.remove(file)  # Remove each file

p_val_dict = {}
n_iterations = 1 # Suggest using 10 iterations


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
                aspect=1,
                palette={'before': '#16547e', 'after': '#33a02c'}
            )
 

            # Access the underlying axes
            ax = g.ax  # only works if it's a single plot
            g.set_axis_labels("", "")
            # Get axis limits
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            
            # Remove all axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            
            # Add two small reference lines in bottom-left corner
            offset_x = (xmax - xmin) * 0.04
            offset_y = (ymax - ymin) * 0.05
            corner_x, corner_y = xmin + offset_x, ymin + offset_y
            
            line_length_x = (xmax - xmin) * 0.10
            line_length_y = (ymax - ymin) * 0.10
            
            ax.plot([corner_x, corner_x + line_length_x], [corner_y, corner_y], color='k', lw=2)  # x-axis
            ax.plot([corner_x, corner_x], [corner_y, corner_y + line_length_y], color='k', lw=2)  # y-axis
            
            # Add labels
            ax.text(corner_x - (xmax - xmin)*0.04, corner_y + line_length_y / 2, 
                    'UMAP 2', fontsize=16, ha='right', va='center', rotation='vertical')

            ax.text(corner_x + line_length_x / 2, corner_y - (ymax - ymin)*0.03, 
                    'UMAP 1', fontsize=16, ha='center', va='top')

            # Replace legend labels and remove title
            new_labels = ['Before', 'After']
            for t, l in zip(g._legend.texts, new_labels):
                t.set_text(l)
            g._legend.set_title('')  # remove legend title
            
            g.fig.suptitle(f'{basename}')
            
            fig_path = os.path.join(fig_dir, f'{basename}_countour_plot.png')
            plt.savefig(fig_path)
            fig_path_svg = os.path.join(fig_dir, f'{basename}_countour_plot.svg')
            plt.savefig(fig_path_svg)
            
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
        
        '''
        if n == 0:
            # Extra plots to show the 'before' vs 'after' distributions subtractions
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
        '''

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
plot_df['session'] = pd.factorize(plot_df['basename'])[0]



plot_df['rat'] = plot_df['basename'].str.split('_').str[0]

basename_to_session = {}
prefix_counts = {}

for basename in plot_df['basename'].unique():
    prefix = basename.split('_')[0]
    count = prefix_counts.get(prefix, 1)

    session_name = f"{prefix}_{count}"
    basename_to_session[basename] = session_name
    prefix_counts[prefix] = count + 1
    
plot_df['session'] = plot_df['basename'].map(basename_to_session)

# Sort so sessions stay in order within each rat
plot_df = plot_df.sort_values(['rat', 'basename']).reset_index(drop=True)


# Assign a unique numeric ID to each session
session_map = {s: i for i, s in enumerate(plot_df['session'].unique(), start=1)}
plot_df['session_id'] = plot_df['session'].map(session_map)

# %% Save UMAP data for plotting/stats

desktop_path = os.path.expanduser("~/Desktop/clustering_data/UMAP_of_all_MTMs_plot_df.pkl")

# Save as pickle
plot_df.to_pickle(desktop_path)

print(f"Saved plot_df as a pickle file to {desktop_path}")

