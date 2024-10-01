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
dirname = '/home/natasha/Desktop/clustering_data/'
file_path = os.path.join(dirname, 'clustering_df_update.pkl')
df = pd.read_pickle(file_path)

transition_file_path = os.path.join(dirname, 'scaled_mode_tau.pkl')
transition_df = pd.read_pickle(transition_file_path)

# Remove any data for df that does not have an associated transition time in scaled_mode_tau
df['basename'] = df['basename'].str.lower() # All basenames to lowercase
transition_df['basename'] = transition_df['basename'].str.lower() # All basenames to lowercase
tau_basenames = transition_df.basename.unique() # Find all basenames in transition_df
df = df.loc[df['basename'].isin(tau_basenames)] # Keep only basenames 
# Manually removed this sepcific data:
df = df[~((df['basename'] == 'km50_5tastes_emg_210911_104510_copy') & (df['taste'] == 1))]
df = df[~((df['basename'] == 'km50_5tastes_emg_210911_104510_copy') & (df['taste'] == 4))]

window_len = 800

# ==============================================================================
# Re-structure transition dataframe
# Create DataFrame of events around the transition
# ==============================================================================
# Initialize lists to store the expanded data
basename_list = []
taste_num_list = []
trial_num_list = []
scaled_mode_tau_list = []

chosen_transition = 2 # Choose out of 0, 1, or 2

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
        transition_time_point = expanded_df.loc[
            (expanded_df['trial_num'] == trial) & 
            (expanded_df['taste_num'] == str(taste)) & 
            (expanded_df['basename'] == basename), 
            'scaled_mode_tau'
        ].values[0]
        window_start = transition_time_point - window_len
        window_end = transition_time_point + window_len
        #print(window_start, window_end)
        if window_start <= segment_bounds[0] <= window_end and window_start <= segment_bounds[1] <= window_end:
            rows.append(row)
            
# Create a DataFrame from the list of rows
transition_events_df = pd.DataFrame(rows).reset_index(drop=True)


# Define a color mapping for cluster numbers
color_mapping = {
    -1: '#ff9900',      # Gapes Color for cluster -1
    -2: '#D3D3D3',      # No mvoement Color for cluster 0
     0: '#4285F4',     # Color for cluster 1
     1: '#88498F',    # Color for cluster 2
     2: '#0CBABA'        # Color for cluster 3
}


# ==============================================================================
# Plot events around the transition, 1 plot per trial
# ==============================================================================
# Find unique combinations of trial, taste, and session_ind
unique_combinations = transition_events_df.groupby(['trial', 'taste', 'basename'])

emg_dir = os.path.join(dirname, 'EMG_around_transition')
os.makedirs(emg_dir, exist_ok=True)

# Clear the folder by deleting all files within it
files = glob.glob(os.path.join(emg_dir, '*'))  # Get list of all files in the directory
for file in files:
    os.remove(file)  # Remove each file

# =============================================================================
#transition_events_df['event_type'] = transition_events_df['event_type'].astype('category')
#transition_events_df['pred_event_code'] = transition_events_df.event_type.cat.codes
#unique_event_codes = transition_events_df.pred_event_code.unique()

#cmap = plt.cm.get_cmap('tab10')
# =============================================================================

# Group by basename (session), taste, and trial
grouped = transition_events_df.groupby(['basename', 'taste', 'trial'])


for (basename, taste, trial), group in grouped:
    basename_dir = os.path.join(emg_dir, basename)
    os.makedirs(basename_dir, exist_ok=True)  # Ensure the folder is created
 
    plt.figure(figsize=(10, 6))
    
    # Initialize variables to hold the last point of the previous waveform
    prev_end_time = None
    prev_end_value = None
    
    # Retrieve the transition time for this trial
    transition_time = expanded_df.loc[
        (expanded_df['trial_num'] == trial) & 
        (expanded_df['taste_num'] == str(taste)) & 
        (expanded_df['basename'] == basename), 
        'scaled_mode_tau'
    ].values[0]
    
    # Iterate through each row within the group (each segment)
    for _, row in group.iterrows():
        segment_raw = row['segment_raw']
        segment_bounds = row['segment_bounds']
        cluster_num = row['cluster_num']

        # Adjust segment bounds relative to the transition time
        segment_bounds_adjusted = [segment_bounds[0] - transition_time, segment_bounds[1] - transition_time]

        # Create time values using the adjusted segment bounds
        time_values = np.linspace(segment_bounds_adjusted[0], segment_bounds_adjusted[1], len(segment_raw))
        
        # Plot the waveform
        plt.plot(time_values, segment_raw, color='black')
        
        # Add transparent overlay based on cluster number
        color = color_mapping.get(cluster_num, '#FFFFFF')  # Default to white if cluster_num not in mapping
        plt.fill_between(time_values, segment_raw, segment_raw.min(), color=color, alpha=0.3)  # Adjust alpha for transparency
        
        # If there's a previous waveform, plot a light gray line connecting the previous end to the current start
        if prev_end_time is not None:
            plt.plot([prev_end_time, time_values[0]], [prev_end_value, segment_raw[0]], color='lightgray')
        
        # Update prev_end_time and prev_end_value to the last time and value of the current waveform
        prev_end_time = time_values[-1]
        prev_end_value = segment_raw[-1]
    
    # Add a vertical line at the transition time (now at 0 after adjustment)
    plt.axvline(x=0, color='r', linestyle='--', label='Transition Time')
    plt.xlim([-(window_len), window_len])  # Set x-axis limits centered around the transition (0 ms)
    
    # Add titles and labels
    plt.title(f"Waveforms for Trial {trial}, Taste {taste}, {basename}")
    plt.xlabel('Time (ms)')
    plt.ylabel('Waveform Amplitude')
    plt.legend()
    
    # Save the plot
    #emg_all_path = os.path.join(emg_dir, f'trial{trial}_taste{taste}_{basename}_emg.png')
    #plt.savefig(emg_all_path)
    #plt.clf()
    emg_all_path = os.path.join(basename_dir, f'trial{trial}_taste{taste}_{basename}_emg.png')
    plt.savefig(emg_all_path)
    plt.clf()


# ==============================================================================
# Plot events around the transition, 1 plot per trial
# ==============================================================================

clust_dir = os.path.join(dirname, 'cluster_raster_transition')
os.makedirs(clust_dir, exist_ok=True)

# Clear the folder by deleting all files within it
files = glob.glob(os.path.join(clust_dir, '*'))
for file in files:
    os.remove(file)  # Remove each file

# Group by basename (session) and taste
grouped = transition_events_df.groupby(['basename', 'taste'])



# Create a figure
for basename, taste_group in grouped:
    taste = basename[1]
    basename = basename[0]
    
    # Create a subplot for each taste
    num_tastes = len(taste_group['taste'].unique())
    fig, axs = plt.subplots(nrows=num_tastes, figsize=(8, 10), sharex=True, sharey=True)
    
    # Ensure axs is always treated as an iterable
    # TODO This probably needs to be fixed if I want to do suplots for each taste on one graph
    if num_tastes == 1:
        axs = [axs]  # Make axs a list if there's only one subplot

    for ax, (taste, trial_group) in zip(axs, taste_group.groupby('taste')):
        ax.set_title(f"Taste: {taste}")

        # Collect all cluster_num values for the current taste and trial
        all_cluster_nums = []

        # Loop through each trial and plot it as a block along time
        for trial, trial_data in trial_group.groupby('trial'):
            # Retrieve the transition time from expanded_df
            transition_time = expanded_df.loc[
                (expanded_df['trial_num'] == trial) & 
                (expanded_df['taste_num'] == str(taste)) & 
                (expanded_df['basename'] == basename), 
                'scaled_mode_tau'
            ].values[0]

            for _, row in trial_data.iterrows():
                # Set the color using the color mapping based on cluster_num
                cluster_num = row['cluster_num']
                color = color_mapping.get(cluster_num, 'black')  # Default to 'black' if cluster_num not found
                
                # Add cluster_num to the list
                all_cluster_nums.append(cluster_num)

                # Plot each trial as a block
                segment_bounds = row['segment_bounds']
                ax.fill_between([segment_bounds[0] - transition_time, segment_bounds[1] - transition_time], 
                                trial - 0.5, trial + 0.5, color=color)

            # Set x-axis limits centered around the transition time
            ax.set_xlim([-(window_len), window_len])  # Set limits from -500 to 500 ms
            ax.axvline(0, color='k', linestyle='--')  # Add a vertical line at transition time (0 ms)

        ax.set_ylabel(f'Trial {trial}')
        ax.set_xlabel('Time (ms)')

        # Print the vector of all cluster_num values for this trial group
        print(f"Trial Group for Taste: {taste}, Basename: {basename}:")
        print(all_cluster_nums)

    plt.suptitle(f'{basename}')
    plt.tight_layout()
    clust_all_path = os.path.join(clust_dir, f'trial{trial}_taste{taste}_{basename}_cluster_raster.png')
    plt.savefig(clust_all_path)
    plt.clf()


# ==============================================================================
# Plot events around the transition, 1 plot per trial
# ==============================================================================








'''
# Works if there is not a set of numbers in cluster_num
# Create a figure
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

        # Loop through each trial and plot it as a block along time
        for trial, trial_data in trial_group.groupby('trial'):
            # Retrieve the transition time from expanded_df
            transition_time = expanded_df.loc[
                (expanded_df['trial_num'] == trial) & 
                (expanded_df['taste_num'] == str(taste)) & 
                (expanded_df['basename'] == basename), 
                'scaled_mode_tau'
            ].values[0]

            for _, row in trial_data.iterrows():
                # Set the color by cluster_num
                cluster_num = row['cluster_num']
                color = plt.cm.viridis(cluster_num / transition_events_df['cluster_num'].max())  # Normalize by max cluster_num
                
                # Plot each trial as a block
                segment_bounds = row['segment_bounds']
                ax.fill_between([segment_bounds[0] - transition_time, segment_bounds[1] - transition_time], 
                                trial - 0.5, trial + 0.5, color=color)

            # Set x-axis limits centered around the transition time
            ax.set_xlim([-500, 500])  # Set limits from -500 to 500 ms
            ax.axvline(0, color='k', linestyle='--')  # Add a vertical line at transition time (0 ms)

        ax.set_ylabel(f'Trial {trial}')
        ax.set_xlabel('Time (ms)')

    plt.suptitle(f'{basename}')
    plt.tight_layout()
    plt.show()
'''