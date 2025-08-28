#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 13:18:26 2025

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

# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/clustering_data/'
file_path = os.path.join(dirname, 'clustering_df_update.pkl')
df = pd.read_pickle(file_path)

transition_file_path = os.path.join(dirname, 'scaled_mode_tau_cut_filtered.pkl')
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
fixed_transition_time = math.nan # Set to math.nan or a fixed time from stimulus delivery (2000ms+). If this is not nan it will be used over chosen transition
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


for i, row in transition_df.iterrows():
    basename = row['basename']
    taste_num = row['taste_num']
    scaled_mode_tau = row['scaled_mode_tau']
    bad_change = row['bad_change']
    
    # Iterate over the 30 elements in 'scaled_mode_tau'
    for trial_num, tau_array in enumerate(scaled_mode_tau):
        # Only append if the corresponding bad_change is False
        if not bad_change[trial_num][chosen_transition]:
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
            # Try to get the corresponding transition time from expanded_df
            match = expanded_df.loc[
                (expanded_df['trial_num'] == trial) & 
                (expanded_df['taste_num'] == str(taste)) & 
                (expanded_df['basename'] == basename), 
                'scaled_mode_tau'
            ]
            # Skip this row if there's no match
            if match.empty:
                continue

            transition_time_point = match.values[0]
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

# %% BEHAVIOR RASTER PLOT
# ==============================================================================
# Raster plot of events around the transition, 1 plot per session per taste
# ==============================================================================

color_mapping = {
    -1: '#ff9900',      # Gapes Color for cluster -1
    -2: '#D3D3D3',      # No mvoement Color for cluster 0
     0: '#4285F4',     # Color for cluster 1
     1: '#88498F',    # Color for cluster 2
     2: '#0CBABA'        # Color for cluster 3
}


lookup_df = transition_events_df[['taste', 'taste_name', 'basename']].drop_duplicates()

clust_dir = os.path.join(dirname, 'cluster_raster_transition')
os.makedirs(clust_dir, exist_ok=True)

# Clear the folder by deleting all files within it
files = glob.glob(os.path.join(clust_dir, '*'))
for file in files:
    os.remove(file)  # Remove each file

# Initialize an empty list to store the rows for the final dictionary
summary_data = []

# Group by basename (session) and taste
grouped = transition_events_df.groupby(['basename', 'taste'])

# Create a figure and populate the dictionary
for basename, taste_group in grouped:

    taste = basename[1]
    basename = basename[0]

    # Create a subplot for each taste
    num_tastes = len(taste_group['taste'].unique())
    fig, axs = plt.subplots(nrows=num_tastes, figsize=(8, 10), sharex=True, sharey=True)
    
    # Ensure axs is always treated as an iterable
    if num_tastes == 1:
        axs = [axs]  # Make axs a list if there's only one subplot

    for ax, (taste, trial_group) in zip(axs, taste_group.groupby('taste')):
        ax.set_title(f"Taste: {taste}")

        # Collect all cluster_num values for the current taste and trial
        all_cluster_nums = []
        # Generate a consistent row index for each trial
        trial_list = list(trial_group['trial'].unique())
        trial_to_row = {trial: i for i, trial in enumerate(trial_list)}


        # Loop through each trial and plot it as a block along time
        for trial, trial_data in trial_group.groupby('trial'):
            row_index = trial_to_row[trial]
            
            if math.isnan(fixed_transition_time):
                transition_time_point = expanded_df.loc[
                    (expanded_df['trial_num'] == trial) & 
                    (expanded_df['taste_num'] == str(taste)) & 
                    (expanded_df['basename'] == basename), 
                    'scaled_mode_tau'
                ].values[0]
            else:
                transition_time_point = fixed_transition_time # to align to fixed palatability transition
            
            # Dictionary to store before/after counts for this trial
            trial_summary = {}

            for _, row in trial_data.iterrows():
                # Set the color using the color mapping based on cluster_num
                cluster_num = row['cluster_num']
                color = color_mapping.get(cluster_num, 'black')  # Default to 'black' if cluster_num not found
                
                # Add cluster_num to the list
                all_cluster_nums.append(cluster_num)

                # Plot each trial as a block
                segment_bounds = row['segment_bounds']
                ax.fill_between([segment_bounds[0] - transition_time_point, segment_bounds[1] - transition_time_point], 
                                row_index - 0.5, row_index + 0.5, 
                                color=color)
                
                # Check if the event occurred before or after the transition
                event_position = "before" if segment_bounds[1] < transition_time_point else "after"
                
                
                taste_name = convert_taste_num_to_name(basename, taste, lookup_df)
                
                # Add to the summary_data list
                summary_data.append({
                    'basename': basename,
                    'taste': taste,
                    'cluster_num': cluster_num,
                    'event_position': event_position,
                    'trial': trial,
                    'taste_name': taste_name
                })

            # Set x-axis limits centered around the transition time
            ax.set_xlim([-(window_len), window_len])  # Set limits from -500 to 500 ms
            ax.axvline(0, color='k', linestyle='--')  # Add a vertical line at transition time (0 ms)

        ax.set_ylabel(f'Trial {trial}')
        ax.set_xlabel('Time (ms)')

        # Print the vector of all cluster_num values for this trial group
        print(f"Taste: {taste_name}, Basename: {basename}:")

    plt.suptitle(f'{basename}')
    plt.tight_layout()
    #clust_all_path = os.path.join(clust_dir, f'{taste_name}_{basename}_cluster_raster.png')
    #plt.savefig(clust_all_path)
    plt.show()
    plt.clf()
    
    
# %%
for ax, (taste, trial_group) in zip(axs, taste_group.groupby('taste')):
    ax.set_title(f"Taste: {taste}")
    
    # Generate a consistent row index for each trial
    trial_list = list(trial_group['trial'].unique())
    trial_to_row = {trial: i for i, trial in enumerate(trial_list)}

    for trial, trial_data in trial_group.groupby('trial'):
        row_index = trial_to_row[trial]

        if math.isnan(fixed_transition_time):
            transition_time_point = expanded_df.loc[
                (expanded_df['trial_num'] == trial) & 
                (expanded_df['taste_num'] == str(taste)) & 
                (expanded_df['basename'] == basename), 
                'scaled_mode_tau'
            ].values[0]
        else:
            transition_time_point = fixed_transition_time

        for _, row in trial_data.iterrows():
            cluster_num = row['cluster_num']
            color = color_mapping.get(cluster_num, 'black')
            segment_bounds = row['segment_bounds']

            ax.fill_between(
                [segment_bounds[0] - transition_time_point, segment_bounds[1] - transition_time_point], 
                row_index - 0.5, row_index + 0.5, 
                color=color
            )

            event_position = "before" if segment_bounds[1] < transition_time_point else "after"
            taste_name = convert_taste_num_to_name(basename, taste, lookup_df)

            summary_data.append({
                'basename': basename,
                'taste': taste,
                'cluster_num': cluster_num,
                'event_position': event_position,
                'trial': trial,
                'taste_name': taste_name
            })

    ax.set_xlim([-(window_len), window_len])
    ax.axvline(0, color='k', linestyle='--')
    ax.set_ylabel('Trial Index')
    ax.set_xlabel('Time (ms)')


    
# %% # BEFORE/AFTER TRANSITION COUNT WITH STATS TESTS

# Convert the summary data to a DataFrame
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df[summary_df['cluster_num'] != -2.0] # REMOVE NO MOVEMENT

unique_combinations = summary_df[['cluster_num', 'taste_name']].drop_duplicates()

# ==============================================================================
# Plot and test where every data point is a trial
# ==============================================================================

count_df = summary_df.groupby(
    ['basename', 'cluster_num', 'event_position', 'taste_name', 'trial']
).size().reset_index(name='movement_count')
# Make sure before values are ordered first, then after values. For plotting
count_df['event_position'] = pd.Categorical(count_df['event_position'], categories=['before', 'after'], ordered=True)

trial_results = []
for _, row in unique_combinations.iterrows():
    cluster = row['cluster_num']
    taste = row['taste_name']

    # Filter count_df for matching rows
    matching_rows = count_df[
        (count_df['cluster_num'] == cluster) &
        (count_df['taste_name'] == taste)
    ]
    
    num_of_rows = len(matching_rows['trial'].unique())
    ttest_df = pd.DataFrame(0, index=range(num_of_rows), columns=['trial', 'before', 'after'])
    ttest_df.trial = matching_rows['trial'].unique()
    
    
    for _, row in matching_rows.iterrows():
        trial = row['trial']
        event_pos = row['event_position']
        move_count = row['movement_count']
        
        ttest_df.loc[ttest_df['trial'] == trial, event_pos] = move_count
        
    before_counts = ttest_df['before'].tolist()
    after_counts = ttest_df['after'].tolist()
    stat, pval = ttest_rel(before_counts, after_counts)   
    res = stats.poisson_means_test(sum(before_counts), len(before_counts), sum(after_counts), len(after_counts))
  
    trial_results.append({
        'taste': taste,
        'cluster_num': cluster,
        'n_trials': len(ttest_df),
        'ttest_pvalue': pval,
        'poisson_pvalue': res.pvalue
    })

# Convert results to DataFrame
trial_results_df = pd.DataFrame(trial_results)

print("\nT-TEST: every data point is a trial")
for _,row in trial_results_df.iterrows():
    if row['ttest_pvalue'] < 0.05:
        taste = row['taste']
        cluster_num= row['cluster_num']
        print(f'Significant: {taste} and cluster {cluster_num}')

        
print("\nPOISSON MEANS TEST: every data point is a trial")
# Print any significant results
for _,row in trial_results_df.iterrows():
    if row['poisson_pvalue'] < 0.05:
        taste = row['taste']
        cluster_num= row['cluster_num']
        print(f'Significant: taste {taste} and cluster {cluster_num}')

# Plot grid of taste num by cluster num
g = sns.FacetGrid(count_df, col='taste_name', row='cluster_num', margin_titles=True, sharey=False)
g.map_dataframe(
    sns.boxplot,
    x='event_position',
    y='movement_count',
    hue='event_position',
    palette='pastel',
    legend=False
)
g.set_axis_labels("", "Movement Count")
plt.tight_layout()
g.fig.suptitle("By trial", y=1.02)  # y controls vertical position

plt.show()
count_df.to_pickle("Desktop/clustering_data/trial_count.pkl")  
trial_results_df.to_pickle("Desktop/clustering_data/trial_results.pkl")  

# ==============================================================================
# Plot and test where every data point is a test session
# ==============================================================================

session_count_df = summary_df.groupby(
    ['basename', 'cluster_num', 'event_position', 'taste_name']
).size().reset_index(name='movement_count')
session_count_df['event_position'] = pd.Categorical(session_count_df['event_position'], categories=['before', 'after'], ordered=True)
session_count_df.to_pickle("Desktop/clustering_data/session_counts.pkl") 
session_results = []
for _, row in unique_combinations.iterrows():
    cluster = row['cluster_num']
    taste = row['taste_name']

    # Filter count_df for matching rows
    matching_rows = session_count_df[
        (session_count_df['cluster_num'] == cluster) &
        (session_count_df['taste_name'] == taste)
    ]
    
    num_of_rows = len(matching_rows['basename'].unique())
    ttest_df = pd.DataFrame(0, index=range(num_of_rows), columns=['basename', 'before', 'after'])
    ttest_df.basename = matching_rows['basename'].unique()
    
    
    for _, row in matching_rows.iterrows():
        basename = row['basename']
        event_pos = row['event_position']
        move_count = row['movement_count']
        
        ttest_df.loc[ttest_df['basename'] == basename, event_pos] = move_count
        
    before_counts = ttest_df['before'].tolist()
    after_counts = ttest_df['after'].tolist()
    stat, pval = ttest_rel(before_counts, after_counts)   
    res = stats.poisson_means_test(sum(before_counts), len(before_counts), sum(after_counts), len(after_counts))

    session_results.append({
        'taste': taste,
        'cluster_num': cluster,
        'n_sessions': len(ttest_df),
        'ttest_pvalue': pval,
        'poisson_pvalue': res.pvalue
    })

# Convert results to DataFrame
session_results_df = pd.DataFrame(session_results)

print("\nT-TEST: every data point is a session")
for _,row in session_results_df.iterrows():
    if row['ttest_pvalue'] < 0.05:
        taste = row['taste']
        cluster_num= row['cluster_num']
        print(f'Significant: {taste} and cluster {cluster_num}')

        
print("\nPOISSON MEANS TEST: every data point is a session")
# Print any significant results
for _,row in session_results_df.iterrows():
    if row['poisson_pvalue'] < 0.05:
        taste = row['taste']
        cluster_num= row['cluster_num']
        print(f'Significant: taste {taste} and cluster {cluster_num}')


# Plot grid of taste num by cluster num
g = sns.FacetGrid(session_count_df, col='taste_name', row='cluster_num', margin_titles=True, sharey=False)
g.map_dataframe(
    sns.stripplot,
    x='event_position',
    y='movement_count',
    hue='event_position',
    palette='pastel',
    legend=False,
    s=10
)
g.set_axis_labels("", "Movement Count")
plt.tight_layout()
g.fig.suptitle("By session", y=1.02)  # y controls vertical position

plt.show()


session_results_df.to_pickle("Desktop/clustering_data/session_results.pkl")  


