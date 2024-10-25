#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:44:39 2024

@author: natasha
"""

import pandas as pd
import numpy as np
import tables
import glob
import os
import scipy.stats
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
#import pingouin as pg
from tqdm import tqdm, trange
import math
from scipy import signal
import scipy.stats as stats
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from tempfile import TemporaryFile
from matplotlib.colors import LinearSegmentedColormap
import json

'''# =========INPUT BASE FOLER, RAT NAME and TEST DAY NUM HERE============'''
base_folder = '/media/natasha/drive2/Natasha_Data' # contains folders of all rats' data

rat_list = ["NB35", "NB32", "NB34"] # List of all rats to be analyzed
#rat_list = ["NB32"]
test_day = [1, 2, 3] # Greatest number of test days
scorer_initials = ['YW', 'NBT']  # ONLY TWO! Initials of scorers (as indicated in csv file names)
'''# ===================================================================='''

'''
#==============================================================================
# Importing data and getting setup
#==============================================================================
'''
#Finding path to test day folder inside of base_folder
dirs = []
for rat_name in rat_list:
    for day in test_day:
        #Finding path to test day folder inside of base_folder
        dirname = (os.path.join(os.path.join(base_folder, rat_name), f"Test{day}"))
        if not os.path.exists(dirname):
            dirname = os.path.join(os.path.join(base_folder, rat_name), f"test{day}")
            if not os.path.exists(dirname):
                print(f"Error: Directory '{dirname}' does not exist.")
        dirs.append(dirname)

path_scoring_dict = []
for dirname in dirs:
    # Load scoring dictionary within npz file
    scoring_file = glob.glob(os.path.join(dirname,'*scoring_dict.npz'))
    if scoring_file:
        for file in scoring_file:
            path_scoring_dict.append(file)
    #scoring_dict = dict(np.load(path_scoring_dict, allow_pickle=True))
    
vid_trial_start = {}
behavior_dict ={}
for path in path_scoring_dict:
    loaded_npz = np.load(path, allow_pickle=True)
    # Convert the loaded data back to a dictionary of DataFrames
    scoring_dict = {key: pd.DataFrame(loaded_npz[key].item()) for key in loaded_npz}
    vid_trial_start[path] = scoring_dict['trial_start']
    behavior_dict[path] = scoring_dict['behavior']

keys_list = list(vid_trial_start.keys())

trial_info = {}
for dirname in dirs:
    # First digit is which taste #. Second digit is taste presentation #
    path_trial_info = glob.glob(os.path.join(dirname,'*trial_info.npy'))
    if path_trial_info:
        loaded_npy_file = np.load(path_trial_info[0])
        trial_info[dirname] = loaded_npy_file


trial_pre_stim = 500 # in ms
trial_post_stim = 5000 # in ms

'''
#==============================================================================
# Putting stuff into dataframes
#==============================================================================
'''

# Initialize an empty list to hold each modified dataframe
dataframes = []

# Iterate over each key-value pair in the dictionary
for path, df in vid_trial_start.items():
    
    basename = path.split('/')[-1].replace('.npz', '')
    # Add a new column for the basename
    df['basename'] = basename
    if 'NBT' in basename:
        df['scorer'] = 'NBT'
    elif 'YW' in basename:
        df['scorer'] = 'YW'
    # Append the modified dataframe to the list
    dataframes.append(df)
# Concatenate all dataframes into one large dataframe
vid_df = pd.concat(dataframes, ignore_index=True)

# Extract 'rat' (string before the first '_')
vid_df['rat'] = vid_df['basename'].str.split('_').str[0]
# Extract 'test' (string after the first '_' but before the second '_')
vid_df['test'] = vid_df['basename'].str.split('_').str[1]


# Create empty dataframe for the final results
final_vid_df = pd.DataFrame(columns=vid_df.columns)

# Iterate through vid_df rows
for index, row in vid_df.iterrows():
    # Find rows in vid_df that match Modifier #1, rat, and test
    matches = vid_df[(vid_df['Modifier #1'] == row['Modifier #1']) &
                     (vid_df['rat'] == row['rat']) &
                     (vid_df['test'] == row['test'])]

    # Check for Time value matches
    if len(matches['Time'].unique()) > 1:
        print(f"Time mismatch found for Modifier #1={row['Modifier #1']}, rat={row['rat']}, test={row['test']}")
        
    final_vid_df = pd.concat([final_vid_df, matches.iloc[[0]]], ignore_index=True)

final_vid_df = final_vid_df[final_vid_df['Modifier #2'] == 'in view']


indices_to_drop = []

for index, row in final_vid_df.iterrows():
    rat = row['rat']
    mod_1_value = int(row['Modifier #1'])
    
    for key in trial_info.keys():
        if rat in key:
            # Check if mod_1_value is within the range of trial_info[key]
            if mod_1_value < len(trial_info[key]):
                a = trial_info[key][mod_1_value]
                print(a[0])
                final_vid_df.loc[index, 'tastant'] = a[0]
                final_vid_df.loc[index, 'delivery_num'] = a[1]
            else:
                # Add index to the list if out of range
                indices_to_drop.append(index)
                break  # Break to avoid multiple checks for the same rat

# Drop the rows from final_vid_df based on the collected indices
final_vid_df.drop(indices_to_drop, inplace=True)


# Initialize an empty list to hold each modified dataframe
dataframes = []

# Iterate over each key-value pair in the dictionary
for path, df in behavior_dict.items():
    
    basename = path.split('/')[-1].replace('.npz', '')
    # Add a new column for the basename
    df['basename'] = basename
    if 'NBT' in basename:
        df['scorer'] = 'NBT'
    elif 'YW' in basename:
        df['scorer'] = 'YW'
    # Append the modified dataframe to the list
    dataframes.append(df)
# Concatenate all dataframes into one large dataframe
behavior_df = pd.concat(dataframes, ignore_index=True)


# Extract 'rat' (string before the first '_')
behavior_df['rat'] = behavior_df['basename'].str.split('_').str[0]
# Extract 'test' (string after the first '_' but before the second '_')
behavior_df['test'] = behavior_df['basename'].str.split('_').str[1]

# Display the resulting dataframe
print(behavior_df)

# Filter out rows where 'Behavior' is 'other' or 'to discuss'
behavior_df =behavior_df[~behavior_df['Behavior'].isin(['other', 'to discuss'])]



# Extract the 'Behavior type' column
behavior_sequence = behavior_df['Behavior type'].values

# Initialize a list to store the indices of the problematic rows
problematic_indices = []

# Check for alternation and store problematic indices
for i in range(len(behavior_sequence) - 1):
    if (behavior_sequence[i] == 'START' and behavior_sequence[i + 1] != 'STOP') or \
       (behavior_sequence[i] == 'STOP' and behavior_sequence[i + 1] != 'START'):
        problematic_indices.append(i + 1)  # Add the index of the next row

# Print the problematic rows if any were found
if problematic_indices:
    print("Problematic rows where 'Behavior type' does not alternate correctly:")
    print(behavior_df.iloc[problematic_indices])
else:
    print("All rows alternate correctly between 'START' and 'STOP'.")




# Step 1: Separate START and STOP rows
start_df = behavior_df[behavior_df['Behavior type'] == 'START'].copy().reset_index(drop=True)
stop_df = behavior_df[behavior_df['Behavior type'] == 'STOP'].copy().reset_index(drop=True)

# Step 2: Rename the 'Time' columns
start_df = start_df.rename(columns={'Time': 'Start'}).drop(columns='Behavior type')
stop_df = stop_df.rename(columns={'Time': 'Stop'}).drop(columns='Behavior type')

# Step 3: Concatenate start and stop dataframes side by side
# This assumes each START has a corresponding STOP in the same order.
merged_df = pd.concat([start_df, stop_df['Stop']], axis=1)

# Display the result
merged_df


unique_behaviors = merged_df['Behavior'].unique()


# Display the updated dataframe
print(merged_df)


'''
#==============================================================================
# Making heatmaps
#==============================================================================
'''



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrames (use your actual DataFrames)
# behavior_df = pd.DataFrame(...)  # Your behavior DataFrame
# final_vid_df = pd.DataFrame(...)  # Your final video DataFrame

# Ensure the 'Time' columns are floats
behavior_df['Time'] = behavior_df['Time'].astype(float)
final_vid_df['Time'] = final_vid_df['Time'].astype(float)

# Get unique combinations of rat, test, tastant, and scorer
unique_combinations = final_vid_df[['rat', 'test', 'scorer']].drop_duplicates()

'''
# Create bins
bin_edges = pd.interval_range(start=start_time, end=end_time, freq=bin_size)


    # Create a DataFrame from the heatmap data
    heatmap_df = pd.DataFrame(heatmap_data)

    # Check if heatmap_df is empty
    if heatmap_df.empty:
        print(f"No behaviors recorded for Rat: {rat}, Test: {test}, Tastant: {tastant}, Scorer: {scorer}.")
        continue  # Skip to the next combination

    # Pivot the DataFrame to create a matrix suitable for a heatmap
    heatmap_pivot = heatmap_df.pivot_table(index='Behavior', 
                                            columns='Time Binned', 
                                            values='Count', 
                                            aggfunc='sum', 
                                            fill_value=0)

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_pivot, cmap='YlGnBu', annot=False)  # Set annot=False to remove numbers
    plt.title(f'Heatmap of Behaviors for Rat: {rat}, Test: {test}, Tastant: {tastant}, Scorer: {scorer}')
    plt.xlabel('Time (s)')
    plt.ylabel('Behaviors')
    plt.show()

'''

# Define the bin size (10ms)
# Define the bin size (10ms)
bin_size = 0.01  # 10ms in seconds
start_time = 0.5  # Start 500ms before trial
end_time = 3.0  # End 3000ms after trial
num_bins = int((end_time + start_time) / bin_size)

# Create a list of zeros with length equal to the number of bins
zero_list = [0] * num_bins
zero_dict = {}
for i in unique_behaviors:
    zero_dict[i] = zero_list
zero_df = pd.DataFrame(zero_dict)


for c, combo in unique_combinations.iterrows():
    rat = combo[0]
    test = combo[1]
    #tastant = combo[2]
    scorer = combo[2]
    trial_times = final_vid_df[
    (final_vid_df['rat'] == rat) & 
    (final_vid_df['test'] == test) & 
    (final_vid_df['scorer'] == scorer)]['Time']
    for time in trial_times:
        last_time = time + end_time
        zero_df[:] = 0
        for i, row in merged_df.iterrows():
            if (row['Start'] >= time) & (row['Stop'] <= last_time) & (row['rat'] == rat) & (row['scorer'] ==scorer) & (row['test'] == test):
                print("cool")
            
 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure the 'Time' columns are floats
behavior_df['Time'] = behavior_df['Time'].astype(float)
final_vid_df['Time'] = final_vid_df['Time'].astype(float)

# Get unique combinations of rat, test, tastant, and scorer
unique_combinations = final_vid_df[['rat', 'test', 'tastant', 'scorer']].drop_duplicates()

# Define the bin size (10ms)
bin_size = 0.01  # 10ms in seconds
start_time = 0.5  # Start 500ms before trial
end_time = 3.0  # End 3000ms after trial
num_bins = int((end_time + start_time) / bin_size)

# Create a DataFrame with zeros for each behavior
unique_behaviors = merged_df['Behavior'].unique()

# Nested dictionary to store data by tastant, scorer, and rat
plot_data = {}

# Iterate over unique combinations
for _, combo in unique_combinations.iterrows():
    rat = combo['rat']
    test = combo['test']
    tastant = combo['tastant']
    scorer = combo['scorer']
    
    # Initialize dictionary structure
    if tastant not in plot_data:
        plot_data[tastant] = {}
    if scorer not in plot_data[tastant]:
        plot_data[tastant][scorer] = {}
    if rat not in plot_data[tastant][scorer]:
        plot_data[tastant][scorer][rat] = []

    # Filter the times for the current combination
    trial_times = final_vid_df[
        (final_vid_df['rat'] == rat) & 
        (final_vid_df['test'] == test) & 
        (final_vid_df['tastant'] == tastant) & 
        (final_vid_df['scorer'] == scorer)]['Time']
    
    # Iterate over each trial time
    for time in trial_times:
        last_time = time + end_time
        zero_df = pd.DataFrame(0, index=unique_behaviors, columns=range(num_bins))  # Reset zero_df for each trial time
        
        # Iterate over merged_df rows
        for i, row in merged_df.iterrows():
            if (row['Start'] >= time) and (row['Stop'] <= last_time) and \
               (row['rat'] == rat) and (row['scorer'] == scorer) and (row['test'] == test):
                behavior = row['Behavior']
                
                # Calculate start and stop bins relative to the trial start time
                start_bin = int((row['Start'] - time + start_time) / bin_size)
                stop_bin = int((row['Stop'] - time + start_time) / bin_size)
                
                # Update the bins for this behavior
                zero_df.loc[behavior, start_bin:stop_bin] = 1
        
        # Store the zero_df for this trial in the plot_data dictionary
        plot_data[tastant][scorer][rat].append(zero_df)

# Define the folder path on the desktop
desktop_path = os.path.expanduser("~/Desktop")
folder_path = os.path.join(desktop_path, "Heatmaps")

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)



# Plot heatmaps for each combination of tastant, scorer, and rat
for tastant, scorer_dict in plot_data.items():
    for scorer, rat_dict in scorer_dict.items():
        for rat, trials in rat_dict.items():
            # Combine trials for this combination into a single DataFrame
            #combined_df = sum(trials) / len(trials)  # Averaging across trials
            combined_df = sum(trials)
            # Create heatmap
            plt.figure(figsize=(15, 10))
            sns.heatmap(combined_df, cmap="YlGnBu", cbar=True)
            plt.title(f"Heatmap of Behavior Bins for Tastant {tastant}, Scorer {scorer}, Rat {rat}")
            plt.xlabel("Time Bin (10ms each)")
            plt.ylabel("Behavior")
            
            filename = f"{rat}_{tastant}_{scorer}.png"
            file_path = os.path.join(folder_path, filename)
            plt.savefig(file_path, format='png', dpi=300, bbox_inches='tight')
            
            # Close the plot to free up memory
            plt.close()



# Plot heatmaps for each combination of tastant and scorer, combining all rats
for tastant, scorer_dict in plot_data.items():
    for scorer, rat_dict in scorer_dict.items():
        # Combine data across all rats
        combined_df = sum([sum(trials) for trials in rat_dict.values()])  # Summing across all rat trials
        
        # Create heatmap
        plt.figure(figsize=(15, 10))
        sns.heatmap(combined_df, cmap="viridis", cbar=True)
        plt.title(f"Heatmap of Raw Behavior Counts for Tastant {tastant}, Scorer {scorer} (Combined Rats)")
        plt.xlabel("Time Bin (10ms each)")
        plt.ylabel("Behavior")
 # Save the plot
        filename = f"Heatmap_Tastant_{tastant}_Scorer_{scorer}.png"
        file_path = os.path.join(folder_path, filename)
        plt.savefig(file_path, format='png', dpi=600, bbox_inches='tight')
        
        # Close the plot to free up memory
        plt.close()

    