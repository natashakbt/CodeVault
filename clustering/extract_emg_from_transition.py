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
from scipy.stats import chi2_contingency
from scipy.stats import zscore
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
import shutil
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d  # for smoothing
from scipy.optimize import curve_fit
import piecewise_regression

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

window_len = 200 # Half of the total window



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

chosen_transition = 1 # Choose out of 0, 1, or 2

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
        #transition_time_point = 0 # to align to taste delivery
        window_start = transition_time_point - window_len
        window_end = transition_time_point + window_len
        #print(window_start, window_end)
        if window_start <= segment_bounds[0] <= window_end and window_start <= segment_bounds[1] <= window_end:
            new_row = row.copy()  # Copy row to modify it safely
            new_row['time_from_trial_start'] = (segment_bounds[0] - window_start, segment_bounds[1] - window_start) # Alter segment time to be from trial start
            rows.append(new_row)

        elif window_start <= segment_bounds[1] <= window_end:
            new_row = row.copy()
            new_row['segment_bounds'] = (window_start, segment_bounds[1])
            new_row['time_from_trial_start'] = (0, segment_bounds[1] - window_start)
            cut_idx = window_start - segment_bounds[0]
            new_row['segment_raw'] = row['segment_raw'][cut_idx:]
            rows.append(new_row)
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

# Define a color mapping for cluster numbers
color_mapping = {
    -1: '#ff9900',      # Gapes Color for cluster -1
    -2: '#D3D3D3',      # No mvoement Color for cluster 0
     0: '#4285F4',     # Color for cluster 1
     1: '#88498F',    # Color for cluster 2
     2: '#0CBABA'        # Color for cluster 3
}

# Group by basename (session), taste, and trial
grouped = transition_events_df.groupby(['basename', 'taste', 'trial'])

# %% EMG WAVEFORMS AROUND TRANSITION
# ==============================================================================
# Plot EMG waveforms with behavior label around the transition, 1 plot per trial
# ==============================================================================
# Find unique combinations of trial, taste, and session_ind
unique_combinations = transition_events_df.groupby(['trial', 'taste', 'basename'])

emg_dir = os.path.join(dirname, 'EMG_around_transition')
os.makedirs(emg_dir, exist_ok=True)

# Remove everything in emg_dir
for item in os.listdir(emg_dir):  # List everything inside the directory
    item_path = os.path.join(emg_dir, item)
    if os.path.isfile(item_path):
        os.remove(item_path)  # Delete file
    elif os.path.isdir(item_path):
        shutil.rmtree(item_path)  # Delete subdirectory and its contents
        
# =============================================================================
#transition_events_df['event_type'] = transition_events_df['event_type'].astype('category')
#transition_events_df['pred_event_code'] = transition_events_df.event_type.cat.codes
#unique_event_codes = transition_events_df.pred_event_code.unique()

#cmap = plt.cm.get_cmap('tab10')
# =============================================================================



for (basename, taste, trial), group in grouped:
    
    basename_dir = os.path.join(emg_dir, basename)
    os.makedirs(basename_dir, exist_ok=True)  # Ensure the folder is created
    plt.close()
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
        #print(segment_bounds)
        cluster_num = row['cluster_num']
        #segment_bounds_adjusted = row['time_from_trial_start'] #TODO: EXPERIMENTAL
        #print(segment_bounds_adjusted)
        # Adjust segment bounds relative to the transition time
        segment_bounds_adjusted = [segment_bounds[0] - transition_time, segment_bounds[1] - transition_time]
        #print(segment_bounds_adjusted)
        # Create time values using the adjusted segment bounds
        if segment_bounds_adjusted[0] == segment_bounds_adjusted[1]:
            continue
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
    #plt.axvline(x=0, color='k', linestyle='--', label='Transition Time')
    plt.xlim([-(window_len), window_len])  # Set x-axis limits centered around the transition (0 ms)
    
    # Add titles and labels
    plt.title(f"Waveforms for Trial {trial}, Taste {taste}, {basename}")
    plt.xlabel('Time (ms)')
    plt.ylabel('Waveform Amplitude')
    plt.plot()
    
    # Save the plot
    emg_all_path = os.path.join(basename_dir, f'trial{trial}_taste{taste}_{basename}_emg.png')
    plt.savefig(emg_all_path)
    plt.clf()


# %% BEHAVIOR RASTER PLOT
# ==============================================================================
# Raster plot of events around the transition, 1 plot per session per taste
# ==============================================================================

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

        # Loop through each trial and plot it as a block along time
        for trial, trial_data in trial_group.groupby('trial'):
            # Retrieve the transition time from expanded_df
            transition_time = expanded_df.loc[
                (expanded_df['trial_num'] == trial) & 
                (expanded_df['taste_num'] == str(taste)) & 
                (expanded_df['basename'] == basename), 
                'scaled_mode_tau'
            ].values[0]

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
                ax.fill_between([segment_bounds[0] - transition_time, segment_bounds[1] - transition_time], 
                                trial - 0.5, trial + 0.5, color=color)
                
                # Check if the event occurred before or after the transition
                event_position = "before" if segment_bounds[1] < transition_time else "after"

                # Add to the summary_data list
                summary_data.append({
                    'basename': basename,
                    'taste': taste,
                    'cluster_num': cluster_num,
                    'event_position': event_position,
                    'trial': trial
                })

            # Set x-axis limits centered around the transition time
            ax.set_xlim([-(window_len), window_len])  # Set limits from -500 to 500 ms
            ax.axvline(0, color='k', linestyle='--')  # Add a vertical line at transition time (0 ms)

        ax.set_ylabel(f'Trial {trial}')
        ax.set_xlabel('Time (ms)')

        # Print the vector of all cluster_num values for this trial group
        print(f"Trial Group for Taste: {taste}, Basename: {basename}:")

    plt.suptitle(f'{basename}')
    plt.tight_layout()
    clust_all_path = os.path.join(clust_dir, f'trial{trial}_taste{taste}_{basename}_cluster_raster.png')
    plt.savefig(clust_all_path)
    plt.clf()

# %% # BEFORE/AFTER TRANSITION COUNT - NEED TO MAKE IT LOOP THROUGH WINDOW LENGTH
# Convert the summary data to a DataFrame
summary_df = pd.DataFrame(summary_data)

count_df = pd.crosstab(
    index=[summary_df["basename"], summary_df["trial"], summary_df["taste"], summary_df["event_position"]],
    columns="movement_count"
).reset_index()




# %% # CHISQUARED ANALYSIS
# ==============================================================================
# Chi-squared test
# ==============================================================================
# Convert the summary data to a DataFrame
summary_df = pd.DataFrame(summary_data)
# Pivot the DataFrame to create the before/after summary
summary_pivot = summary_df.groupby(['basename', 'taste', 'cluster_num', 'event_position']).size().unstack(fill_value=0)
# Reset index to make the DataFrame easier to read
summary_pivot.reset_index(inplace=True)

# Add the taste_name by merging with the transition_events_df on basename and taste
# First, create a subset of transition_events_df with only unique (basename, taste) pairs and their taste_name
taste_name_lookup = transition_events_df[['basename', 'taste', 'taste_name']].drop_duplicates()

# Now merge the summary_pivot with this taste_name_lookup DataFrame to add the 'taste_name' column
summary_pivot = pd.merge(summary_pivot, taste_name_lookup, how='left', on=['basename', 'taste'])
# Apply the function to create the 'pal_taste' column
summary_pivot['pal_taste'] = summary_pivot.apply(assign_pal_taste, axis=1)
summary_pivot.drop(columns=['taste'], inplace=True)

collapsed_summary = summary_pivot.groupby(['basename', 'cluster_num', 'pal_taste'], as_index=False).sum()

# TODO: REMOVE NO MOVEMENT
collapsed_summary = collapsed_summary[collapsed_summary['cluster_num'] != -2.0]


# Create a pivot table to ensure 'before' and 'after' values are aligned by 'cluster_num'
pivoted_df = collapsed_summary.pivot(index=['basename', 'pal_taste'], columns='cluster_num', values=['before', 'after']).fillna(0)

# Initialize a list to store the results of the chi-squared tests
chi_squared_results = []

# Iterate through unique combinations of basename and pal_taste
for (basename, pal_taste), group in pivoted_df.groupby(level=['basename', 'pal_taste']):
    # Extract the 'before' and 'after' vectors, ensuring they are in the same order
    before_vector = group['before'].values.flatten()
    after_vector = group['after'].values.flatten()
    print(f'before: {before_vector}')
    print(f'after: {after_vector}')
    # Perform the Chi-squared test
    chi2_stat, p_value, _, _ = chi2_contingency([before_vector, after_vector])

    # Store the results
    chi_squared_results.append({
        'basename': basename,
        'pal_taste': pal_taste,
        'chi2_stat': chi2_stat,
        'p_value': p_value
    })

# Convert the results to a DataFrame
chi_squared_results_df = pd.DataFrame(chi_squared_results)

# ==============================================================================
# Plotting Chi-Squared test and bar graph of behavior frequency before/after 
# ==============================================================================
# Create the scatter plot for the chi-squared p-values
plt.figure(figsize=(6, 8))
plt.axhline(y=0.05, color='k', linestyle='--') # Add a horizontal line at p = 0.05
sns.scatterplot(x='pal_taste', y='p_value', data=chi_squared_results_df, s=200)  # 's' controls dot size

plt.yscale('log') # Set the y-axis to log scale

plt.xticks([0, 1]) # Customize x-axis ticks to show only 0 and 1

# Add labels and title
plt.xlabel('Pal Taste (0 or 1)')
plt.ylabel('p-value')
plt.title('P-Values vs. Pal Taste (Logarithmic Y-Scale)')

plt.xlim([-1, 2]) # Set x-axis limits

plt.show()


# Color indicates basename
# Plotting Chi-Squared test and bar graph of behavior frequency before/after 
plt.figure(figsize=(6, 8))
plt.axhline(y=0.05, color='k', linestyle='--')  # Add a horizontal line at p = 0.05

# Scatter plot for the p-values, using 'basename' as the hue to color-code
sns.scatterplot(x='pal_taste', y='p_value', hue='basename', data=chi_squared_results_df, s=400, palette='Set1', legend=False)

# Set y-axis to log scale
plt.yscale('log')

# Customize x-axis ticks to show only 0 and 1
plt.xticks([0, 1])

# Add labels and title
plt.xlabel('Pal Taste (0 or 1)')
plt.ylabel('p-value')
plt.title('P-Values vs. Pal Taste (Logarithmic Y-Scale)')

# Set x-axis limits
plt.xlim([-1, 2])

# Move the legend outside the plot for better visualization if there are many unique basenames
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Basename')

plt.show()


# %% FREQUENCY PLOTS - by test session



freq_dir = os.path.join(dirname, 'frequency_of_behaviors')
os.makedirs(freq_dir, exist_ok=True)

# Clear the folder by deleting all files within it
files = glob.glob(os.path.join(freq_dir, '*'))
for file in files:
    os.remove(file)  # Remove each file
    

# Group by basename (session) and taste
grouped = transition_events_df.groupby(['basename', 'taste'])
unique_clust_num = transition_events_df['cluster_num'].unique()



for basename, taste_group in grouped:
    behavior_array = np.zeros((len(unique_clust_num), (window_len * 2)))  # Initialize behavior array
    
    for row in taste_group.itertuples():
        cluster_num = row.cluster_num
        start_idx, end_idx = row.time_from_trial_start
        
        # Find index of the cluster
        cluster_idx = np.where(unique_clust_num == cluster_num)[0]

        behavior_array[cluster_idx, start_idx:end_idx + 1] += 1

    num_clusters = len(unique_clust_num[unique_clust_num != -2.0])  # Exclude cluster -2.0
    fig, axes = plt.subplots(num_clusters, 1, figsize=(10, 3 * num_clusters), sharex=True)
    
    if num_clusters == 1:  # If there's only one subplot, put it in a list for consistent indexing
        axes = [axes]
    
    subplot_idx = 0
    for i in range(behavior_array.shape[0]):
        cluster_num = unique_clust_num[i]

        if cluster_num == -2.0:  # Skip cluster -2.0 (no movement)
            continue

        cluster_color = color_mapping.get(cluster_num, '#000000')  # Default to black if missing

        smoothed_line = gaussian_filter1d(behavior_array[i, :], sigma=2)

        axes[subplot_idx].plot(smoothed_line, label=f'Cluster {cluster_num}', color=cluster_color)
        axes[subplot_idx].axvline(x=window_len, color='k', ls='--')
        axes[subplot_idx].set_ylabel('Occurrences')
        axes[subplot_idx].legend()
        subplot_idx += 1

    axes[-1].set_xlabel('Time (in ms; mid-point is transition)')
    fig.suptitle(f'{basename[0]} - {taste_group["taste_name"].iloc[0]}', fontsize=14)
    plt.tight_layout()

    freq_all_path = os.path.join(freq_dir, f'{taste_group["taste_name"].iloc[0]}_{basename[0]}.png')
    plt.savefig(freq_all_path)
    plt.clf()
    plt.close()


# Z-scored values
for basename, taste_group in grouped:
    behavior_array = np.zeros((len(unique_clust_num), window_len * 2))  # Initialize occurrence tracking array
    
    # Populate behavior_array with occurrences
    for row in taste_group.itertuples():
        cluster_num = row.cluster_num
        start_idx, end_idx = row.time_from_trial_start
        cluster_idx = np.where(unique_clust_num == cluster_num)[0]
        behavior_array[cluster_idx, start_idx:end_idx + 1] += 1
    
    # Compute z-score across time for each cluster
    behavior_array_z = np.apply_along_axis(zscore, 1, behavior_array, nan_policy='omit')
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    plot_idx = 0
    
    for i, cluster_num in enumerate(unique_clust_num):
        if cluster_num == -2.0:  # Skip no movement cluster
            continue
        
        x_values = np.arange(behavior_array_z.shape[1])
        y_values = behavior_array_z[i, :]
        cluster_color = color_mapping.get(cluster_num, '#000000')
        
        # Fit sigmoid
        try:
            p0 = [max(y_values), np.median(x_values), 1, 0]  # Initial guess
            popt, _ = curve_fit(sigmoid, x_values[::10], y_values[::10], p0, method='dogbox', maxfev=5000)
            fitted_y = sigmoid(x_values, *popt)
            axes[plot_idx].plot(x_values, fitted_y, color=cluster_color, linestyle='-', linewidth=2, alpha=0.9)
        except RuntimeError:
            print(f"Runtime Error: Skipping cluster {cluster_num}")
        
        # Plot individual data points
        axes[plot_idx].scatter(x_values, y_values, color=cluster_color, alpha=0.7, s=10, label=f'Cluster {cluster_num}')
        axes[plot_idx].axvline(x=window_len, color='k', ls='--')
        axes[plot_idx].set_ylabel('Z-score')
        
        plot_idx += 1
    
    plt.xlabel('Time (ms; mid-point is transition)')
    plt.suptitle(f'Z-score: {basename[0]} - {taste_group["taste_name"].iloc[0]}')
    plt.tight_layout()
    
    # Save the plot
    freq_all_path = os.path.join(freq_dir, f'zscore_{taste_group["taste_name"].iloc[0]}_{basename[0]}.png')
    plt.savefig(freq_all_path)
    plt.clf()
    plt.close()


# %% FREQUENCY PLOTS - ALL COMBINED

# Define a color mapping for cluster numbers
color_mapping = {
    -1: '#ff9900',      # Gapes Color for cluster -1
    -2: '#D3D3D3',      # No mvoement Color for cluster 0
     0: '#4285F4',     # Color for cluster 1
     1: '#88498F',    # Color for cluster 2
     2: '#0CBABA'        # Color for cluster 3
}



# Ensure the directory exists
overlay_dir = os.path.join(freq_dir, 'overlay')
os.makedirs(overlay_dir, exist_ok=True)

# Clear the folder by deleting all files within it
files = glob.glob(os.path.join(overlay_dir, '*'))
for file in files:
    os.remove(file)

# Get unique clusters (excluding -2.0)
unique_clust_num = np.sort(transition_events_df['cluster_num'].unique())
unique_clust_num = unique_clust_num[unique_clust_num != -2.0]  # Remove cluster -2.0 (no movement)

# Group by taste only, then process each separately
taste_groups = transition_events_df.groupby(['taste_name'])

for taste, taste_df in taste_groups:
    fig, axes = plt.subplots(len(unique_clust_num), 1, figsize=(10, 3 * len(unique_clust_num)), sharex=True)

    if len(unique_clust_num) == 1:  # Ensure axes is always iterable
        axes = [axes]

    grouped_sessions = taste_df.groupby(['basename'])

    for (basename, session_df) in grouped_sessions:
        behavior_array = np.zeros((len(unique_clust_num), (window_len * 2)))  # Initialize behavior array
        
        for row in session_df.itertuples():
            cluster_num = row.cluster_num
            start_idx, end_idx = row.time_from_trial_start

            # Find index of the cluster
            cluster_idx = np.where(unique_clust_num == cluster_num)[0]

            behavior_array[cluster_idx, start_idx:end_idx + 1] += 1

        # Overlay each session onto the appropriate cluster subplot
        for i, cluster_num in enumerate(unique_clust_num):
            cluster_color = color_mapping.get(cluster_num, '#000000')
            smoothed_line = gaussian_filter1d(behavior_array[i, :], sigma=2)

            axes[i].plot(smoothed_line, color=cluster_color, alpha=0.7)
            axes[i].axvline(x=window_len, color='k', ls='--')
            axes[i].set_ylabel('Occurrences')

    # Final plot formatting
    axes[-1].set_xlabel('Time (in ms; mid-point is transition)')
    fig.suptitle(f'Overlayed Sessions - {taste_df["taste_name"].iloc[0]}')
    plt.tight_layout()

    # Save the figure for this taste
    overlay_all_path = os.path.join(overlay_dir, f'overlay_{taste}.png')
    plt.savefig(overlay_all_path)
    plt.clf()
    plt.close()

# Z-score

all_behavior_zscores = []

for taste, taste_df in taste_groups:
    fig, axes = plt.subplots(len(unique_clust_num), 1, figsize=(10, 3 * len(unique_clust_num)), sharex=True)

    if len(unique_clust_num) == 1:  # Ensure axes is always iterable
        axes = [axes]
    all_behavior_zscores = []
    # Group by basename (session) within this taste
    grouped_sessions = taste_df.groupby(['basename'])

    for (basename, session_df) in grouped_sessions:
        behavior_array = np.zeros((len(unique_clust_num), (window_len * 2)))  # Initialize behavior array
        
        for row in session_df.itertuples():
            cluster_num = row.cluster_num
            start_idx, end_idx = row.time_from_trial_start

            # Find index of the cluster
            cluster_idx = np.where(unique_clust_num == cluster_num)[0]

            behavior_array[cluster_idx, start_idx:end_idx + 1] += 1
        session_zscores = np.array([zscore(behavior_array[i, :]) for i in range(len(unique_clust_num))])
        session_zscores = np.nan_to_num(session_zscores)  # Handle NaN values
        
        all_behavior_zscores.append(session_zscores)

        # Overlay each session onto the appropriate cluster subplot
        for i, cluster_num in enumerate(unique_clust_num):
            cluster_color = color_mapping.get(cluster_num, '#000000')  # Default to black if missing

            # Compute z-score and smooth it
            behavior_zscore = zscore(behavior_array[i, :])  # Compute z-score across time
            behavior_zscore = np.nan_to_num(behavior_zscore)  # Handle NaN values safely
            #smoothed_line = gaussian_filter1d(behavior_zscore, sigma=2)  # Apply Gaussian smoothing

            axes[i].plot(behavior_zscore, color=cluster_color, alpha=0.7)  # Add session label
            axes[i].axvline(x=window_len, color='k', ls='--')
            axes[i].set_ylabel('Z-score')
    
    all_behavior_zscores = np.array(all_behavior_zscores)  # Convert to NumPy array
    mean_behavior_zscore = np.nanmean(all_behavior_zscores, axis=0)  # Mean across sessions
    
    # Overlay average line
    for i, cluster_num in enumerate(unique_clust_num):
        axes[i].plot(mean_behavior_zscore[i, :], 'k--', linewidth=2, label="Average")
        axes[i].axvline(x=window_len, color='k', ls='--')
        axes[i].set_ylabel('Z-score')
        axes[i].legend()



    # Final plot formatting
    axes[-1].set_xlabel('Time (in ms; mid-point is transition)')
    fig.suptitle(f'Z-score Overlayed Sessions - {taste}', fontsize=14)
    plt.tight_layout()

    # Save the figure for this taste
    overlay_all_path = os.path.join(overlay_dir, f'Zscore_Overlay_{taste}.png')
    plt.savefig(overlay_all_path)
    plt.clf()
    plt.close()



# %% WHAT IS THIS CRAP
# TODO: FIGURE OUT WHAT ALL THIS CRAP BELOW IS DOING. DO I NEED IT?


'''
# Lines Between Basename
# Plotting Chi-Squared test and bar graph of behavior frequency before/after 
plt.figure(figsize=(6, 8))
plt.axhline(y=0.05, color='k', linestyle='--')  # Add a horizontal line at p = 0.05

# Scatter plot for the p-values
sns.scatterplot(x='pal_taste', y='p_value', data=chi_squared_results_df, s=200)

# Add lines between points with the same basename
for basename in chi_squared_results_df['basename'].unique():
    # Filter the data for the current basename
    subset = chi_squared_results_df[chi_squared_results_df['basename'] == basename]
    
    # Check if there are data points for both 0 and 1 pal_taste
    if subset['pal_taste'].nunique() == 2:
        # Sort by pal_taste to ensure proper connection between 0 and 1
        subset = subset.sort_values(by='pal_taste')
        plt.plot(subset['pal_taste'], subset['p_value'], color='grey', linestyle='-', linewidth=1)

# Set y-axis to log scale
plt.yscale('log')

# Customize x-axis ticks to show only 0 and 1
plt.xticks([0, 1])

# Add labels and title
plt.xlabel('Pal Taste (0 or 1)')
plt.ylabel('p-value')
plt.title('P-Values vs. Pal Taste (Logarithmic Y-Scale)')

# Set x-axis limits
plt.xlim([-1, 2])

plt.show()

'''







import seaborn as sns
import matplotlib.pyplot as plt

# Loop through each unique combination of 'basename' and 'pal_taste'
for (basename, pal_taste), group_data in summary_pivot.groupby(['basename', 'pal_taste']):
    # Set up the figure
    plt.figure(figsize=(8, 6))

    # Melt the data to plot 'before' and 'after' as separate bars
    melted_data = group_data.melt(id_vars=['cluster_num'], value_vars=['before', 'after'], 
                                  var_name='Time', value_name='Count')

    # Create the bar plot without error bars
    sns.barplot(x='cluster_num', y='Count', hue='Time', data=melted_data, palette='Paired', errorbar=None)

    # Customize the plot
    plt.title(f'{basename}, Pal Taste {pal_taste}')
    plt.xlabel('Cluster Number')
    plt.ylabel('Count')
    plt.legend(title='Time')
    
    # Show or save the plot as needed
    plt.show()






palette = sns.color_palette("Paired")

# Create a mapping for cluster_num to specific colors
color_mapping = {
    -2: palette[0:2],  # Light blue and dark blue for cluster_num -2
    -1: palette[2:4],  # Light green and dark green for cluster_num -1
    0: palette[4:6],   # Next pair for cluster_num 0
    1: palette[6:8],   # Next pair for cluster_num 1
    2: palette[8:10],  # Next pair for cluster_num 2
}


# Loop through each unique combination of 'basename' and 'pal_taste'
for (basename, pal_taste), group_data in summary_pivot.groupby(['basename', 'pal_taste']):
    # Set up the figure
    plt.figure(figsize=(8, 6))

    # Melt the data to plot 'before' and 'after' as separate bars
    melted_data = group_data.melt(id_vars=['cluster_num'], value_vars=['before', 'after'], 
                                  var_name='Time', value_name='Count')

    # Create the bar plot without error bars
    sns.barplot(x='cluster_num', y='Count', hue='Time', data=melted_data, palette=color_mapping[cluster_num], errorbar=None)

    # Customize the plot
    plt.title(f'{basename}, Pal Taste {pal_taste}')
    plt.xlabel('Cluster Number')
    plt.ylabel('Count')
    plt.legend(title='Time')
    
    # Show or save the plot as needed
    plt.show()
    
    
import matplotlib.pyplot as plt

palette = ['#D3D3D3',
 '#8d8d8d',
 '#fdbf6f',
 '#ff7f00',
 '#a6cee3',
 '#5383EC',
 '#cab2d6',
 '#6a3d9a',
 '#c5e7e7',
 '#55B7B9']







for (basename, pal_taste), group_data in summary_pivot.groupby(['basename', 'pal_taste']):
    
    melted_data = group_data.melt(id_vars=['cluster_num'], value_vars=['before', 'after'], 
                                  var_name='Time', value_name='Count')
    
    ax = sns.barplot(data=melted_data, x='cluster_num', y='Count', hue='Time', palette=palette, errorbar=None,
                     edgecolor='black')  ###errorbar=('ci', 69))
    for bars, colors in zip(ax.containers, (palette[0::2], palette[1::2])):
         for bar, color in zip(bars, colors):
              bar.set_facecolor(color)
    ax.legend(handles=[tuple(bar_group) for bar_group in ax.containers],
              labels=[bar_group.get_label() for bar_group in ax.containers],
              title=ax.legend_.get_title().get_text(),
              handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)})
    # Customize the plot
    plt.title(f'{basename}, Pal Taste {pal_taste}')
    plt.xlabel('Cluster Number', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.legend(title='')

    plt.show()
    
    
    
    


'''
# THIS WORKS - commenting away for now
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




'''




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