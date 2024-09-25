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

# No assoicated transition times, so remove this sepcific data:
df = df[~((df['basename'] == 'km50_5tastes_emg_210911_104510_copy') & (df['taste'] == 1))]
df = df[~((df['basename'] == 'km50_5tastes_emg_210911_104510_copy') & (df['taste'] == 4))]


transition_file_path = os.path.join(dirname, 'scaled_mode_tau.pkl')
transition_df = pd.read_pickle(transition_file_path)

window_len = 500


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

# ==============================================================================
# Plot events around the transition, 1 plot per trial
# ==============================================================================
# Find unique combinations of trial, taste, and session_ind
unique_combinations = transition_events_df.groupby(['trial', 'taste', 'basename'])
expanded_df['basename'] = expanded_df['basename'].str.lower()

emg_dir = os.path.join(dirname, 'EMG_around_transition')
os.makedirs(emg_dir, exist_ok=True)

# Clear the folder by deleting all files within it
files = glob.glob(os.path.join(emg_dir, '*'))  # Get list of all files in the directory
for file in files:
    os.remove(file)  # Remove each file

#df.loc[(df.session_ind == session) & (df.event_type == 'MTMs'), 'cluster_num'] = labels
# Iterate over each unique combination of trial, taste, and session_ind
for (trial, taste, basename), group in unique_combinations:
    plt.figure(figsize=(10, 6))
    
    # Initialize a variable to hold the last point of the previous waveform
    prev_end_time = None
    prev_end_value = None
    
    # Iterate through each row within the group
    for _, row in group.iterrows():
        segment_raw = row['segment_raw']
        segment_bounds = row['segment_bounds']
        
        # Create time values using segment_bounds length
        time_values = np.linspace(segment_bounds[0], segment_bounds[1], len(segment_raw))
        
        # Plot the waveform
        #plt.plot(time_values, segment_raw, label=f"Bounds: {segment_bounds}")
        plt.plot(time_values, segment_raw)
        # If there's a previous waveform, plot a light gray line connecting the previous end to the current start
        if prev_end_time is not None:
            plt.plot([prev_end_time, time_values[0]], [prev_end_value, segment_raw[0]], color='lightgray')
        
        # Update prev_end_time and prev_end_value to the last time and value of the current waveform
        prev_end_time = time_values[-1]
        prev_end_value = segment_raw[-1]
    
    
    # Get the transition time for this trial
    transition_time = expanded_df.loc[
        (expanded_df['trial_num'] == trial) & 
        (expanded_df['taste_num'] == str(taste)) & 
        (expanded_df['basename'] == basename), 
        'scaled_mode_tau'
    ].values[0]
        
    # Add a vertical line at the transition time
    plt.axvline(x=transition_time, color='r', linestyle='--', label='Transition Time')
        
    
    # Add titles and labels
    plt.title(f"Waveforms for Trial {trial}, Taste {taste}, {basename}")
    plt.xlabel('Time')
    plt.ylabel('Waveform Amplitude')
    plt.legend()
    
    # Save the plot
    emg_all_path = os.path.join(emg_dir, f'trial{trial}_taste{taste}_{basename}_emg.png')
    plt.savefig(emg_all_path)
    plt.clf()

'''
# ==============================================================================
# Plot events around the transition, 1 plot per trial
# ==============================================================================

cluster_raster_dir = os.path.join(dirname, 'cluster_raster_plots')
os.makedirs(cluster_raster_dir, exist_ok=True)

# Iterate over each unique combination of trial, taste, and session_ind
for (trial, taste, session_ind), group in unique_combinations:
    plt.figure(figsize=(10, 6))
    
    # Initialize a variable to hold the last point of the previous waveform
    prev_end_time = None
    prev_end_value = None
    
    # Iterate through each row within the group
    for _, row in group.iterrows():
        segment_raw = row['segment_raw']
        segment_bounds = row['segment_bounds']
        
        # Create time values using segment_bounds length
        time_values = np.linspace(segment_bounds[0], segment_bounds[1], len(segment_raw))
        
        # Plot the waveform
        #plt.plot(time_values, segment_raw, label=f"Bounds: {segment_bounds}")
        plt.plot(time_values, segment_raw)
        # If there's a previous waveform, plot a light gray line connecting the previous end to the current start
        if prev_end_time is not None:
            plt.plot([prev_end_time, time_values[0]], [prev_end_value, segment_raw[0]], color='lightgray')
        
        # Update prev_end_time and prev_end_value to the last time and value of the current waveform
        prev_end_time = time_values[-1]
        prev_end_value = segment_raw[-1]
    
    # Get the transition time for this trial
    transition_time = expanded_df.loc[expanded_df['trial_num'] == trial, 'scaled_mode_tau'].values[0]
    
    # Add a vertical line at the transition time
    plt.axvline(x=transition_time, color='r', linestyle='--', label='Transition Time')
    
    
    plt.title(f"Waveforms for Trial {trial}, Taste {taste}, Session {session_ind}")
    plt.xlabel('Time')
    plt.ylabel('Waveform Amplitude')
    plt.legend()
    emg_all_path = os.path.join(emg_dir, f'trial{trial}_taste{taste}_session{session_ind}_emg.png')
    plt.savefig(emg_all_path)
    plt.clf()      

'''
