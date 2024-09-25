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


# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/clustering_data/'
file_path = os.path.join(dirname, 'clustering_df_update.pkl')
df = pd.read_pickle(file_path)


transition_file_path = os.path.join(dirname, 'scaled_mode_tau.pkl')
transition_df = pd.read_pickle(transition_file_path)

random_transitions = [np.round(np.random.uniform(low=500, high=2000, size=120), 0).astype(int).tolist() for _ in range(9)]
window_len = 500

rows = []

# Assuming 'transition_df' is already defined

# Initialize lists to store the expanded data
basename_list = []
taste_num_list = []
trial_num_list = []
scaled_mode_tau_list = []

chosen_transition = 2 # Choose out of 1, 2 or 3

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

# Display the resulting dataframe
print(expanded_df)





for i in range(len(random_transitions)):
#for i in range(1):
    session_df = df[df['session_ind'] == i]
    for index, row in session_df.iterrows():
        segment_bounds = row['segment_bounds']
        trial = row['trial']
        window_start = random_transitions[i][trial] - window_len
        window_end = random_transitions[i][trial] + window_len
        if window_start <= segment_bounds[0] <= window_end and window_start <= segment_bounds[1] <= window_end:
            rows.append(row)



# Create a DataFrame from the list of rows
transition_df = pd.DataFrame(rows).reset_index(drop=True)

save_dir = os.path.join(dirname, 'transitioned_aligned_EMG')
os.makedirs(save_dir, exist_ok=True)


# Assuming transitin_df is your DataFrame
# First, convert the string of arrays in 'segment_raw' and 'segment_bounds' to actual lists
'''
def convert_to_array(value):
    """Convert string representation of a list or array into a numpy array, or return the array if it's already in that format."""
    if pd.isna(value):
        # Handle None or NaN values
        return np.array([])
    elif isinstance(value, str):
        # If the value is a string, convert it to an array
        return np.array([float(x) for x in value.strip('[]').split()])
    elif isinstance(value, (np.ndarray, list)):
        # If it's already a numpy array or list, just return it
        return np.array(value)
    else:
        raise ValueError(f"Unexpected data type in conversion function: {type(value)}")

transition_df['segment_raw'] = transition_df['segment_raw'].apply(convert_to_array)
transition_df['segment_bounds'] = transition_df['segment_bounds'].apply(convert_to_array)
'''
# Find unique combinations of trial, taste, and session_ind
unique_combinations = transition_df.groupby(['trial', 'taste', 'session_ind'])


# Iterate over each unique combination of trial, taste, and session_ind
for (trial, taste, session_ind), group in unique_combinations:
    print(trial, taste, session_ind)
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
        plt.plot(time_values, segment_raw, label=f"Bounds: {segment_bounds}")
        
        # If there's a previous waveform, plot a light gray line connecting the previous end to the current start
        if prev_end_time is not None:
            plt.plot([prev_end_time, time_values[0]], [prev_end_value, segment_raw[0]], color='lightgray')
        
        # Update prev_end_time and prev_end_value to the last time and value of the current waveform
        prev_end_time = time_values[-1]
        prev_end_value = segment_raw[-1]
    
    plt.title(f"Waveforms for Trial {trial}, Taste {taste}, Session {session_ind}")
    plt.xlabel('Time')
    plt.ylabel('Waveform Amplitude')
    plt.legend()
    plt.show()    
    
