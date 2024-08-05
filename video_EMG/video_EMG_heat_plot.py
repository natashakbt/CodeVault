#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:54:16 2024

@author: natasha
"""
### CODE DOES NOT WORK

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
NEW PLOT IN PROGRESS
'''

'''
#figuring out like-trials 
to create heat map of intensity for pal/non-pal
'''
'''
# My code
unique_behaviors = list(set(behavior_table['Behavior']))
behavior_names= unique_behaviors


#table_with_trial_info
in_view_mask = vid_trial_start.iloc[:,2] == 'in view'
good_trial_table = vid_trial_start[in_view_mask]

suc_trials = [] 
nacl_trials = []
ca_trials = []
qhcl_trials = []
for index, row in good_trial_table.iterrows():
    modifier_value = row['Modifier #1']
    temp_tastant = trial_info[int(modifier_value)][0]
    if temp_tastant == 0:
        suc_trials.append(int(modifier_value))
    elif temp_tastant == 1:
        nacl_trials.append(int(modifier_value))
    elif temp_tastant == 2:
        ca_trials.append(int(modifier_value))
    elif temp_tastant == 3:
        qhcl_trials.append(int(modifier_value))

pal_trials = suc_trials
unpal_trials =  qhcl_trials

desired_order = ['gape', 'mouth or tongue movements', 'lateral tongue protrusion', 'unknown mouth movement', 'other']

'''
#Input condition you want to plot here
'''
plotting_condition = 'unpal' #'unapl' or 'pal' only!


# Define the time bin parameters (adjust these as needed)
time_bin_width_ms = 100  # Width of each time bin in milliseconds
#num_time_bins = int((trial_pre_stim + trial_post_stim) / time_bin_width_ms)
num_time_bins = int((trial_post_stim) / time_bin_width_ms)

# Create a matrix to store behavior occurrences (initialize to zeros)
behavior_matrix = np.zeros((len(behavior_names), num_time_bins))

# Iterate through either unpalatable trials or palatable
for trial in tqdm(unpal_trials if plotting_condition == 'unpal' else pal_trials):
    trial_start_time = vid_trial_start.iloc[trial - 1, 4]
    #trial_start_time -= (trial_pre_stim / 1000)

    trial_behaviors = behavior_table[(behavior_table['Time'] >= trial_start_time) & (behavior_table['Time'] <= trial_start_time + (trial_post_stim / 1000))]

    trial_behaviors = trial_behaviors[trial_behaviors['Behavior'] != 'out of view']
    
    
    tmp_behaviors_dict = {i: None for i in unique_behaviors}
    print(tmp_behaviors_dict)
    for index,row in trial_behaviors.iterrows(): 
        if tmp_behaviors_dict[row[0]] == None:
            tmp_behaviors_dict[row[0]] = [(row[4]-trial_start_time)*1000]
        else:
            tmp_behaviors_dict[row[0]].extend([(row[4]-trial_start_time)*1000])
            
            
    for key, value in tmp_behaviors_dict.items():
        if isinstance(value, list):
            tmp_behaviors_dict[key] = [(value[i], value[i + 1]) 
                                   for i in range(0, len(value) - 1, 2)]
    #re-arranging into desired order

    rearranged_tmp_dict = {key: tmp_behaviors_dict[key] for key in desired_order if key in tmp_behaviors_dict}

    # Add any keys not in the desired order
    for key in tmp_behaviors_dict:
        if key not in rearranged_tmp_dict:
            rearranged_tmp_dict[key] = tmp_behaviors_dict[key]
            
    #Figuring out how may bins each behavior belongs in        
    for behavior, intervals in rearranged_tmp_dict.items():
        if intervals is None:
            continue  # Skip over intervals that are None
        
        behavior_idx = desired_order.index(behavior)  # Get the index of the behavior
        
        for interval in intervals:
            start_time, end_time = interval
            
            # Calculate the time bin indices for the interval
            start_bin = int((start_time) / time_bin_width_ms)
            end_bin = int((end_time) / time_bin_width_ms)
            
            # Increment the corresponding bins in behavior_matrix
            for bin_idx in range(start_bin, end_bin):
                behavior_matrix[behavior_idx, bin_idx] += 1 #### THIS PART IS MESSED UP
    print(behavior_matrix[0])
print('final!')
print(behavior_matrix[0])
# Create a heatmap of behavior occurrences
plt.figure(figsize=(12, 8))
cax = plt.imshow(behavior_matrix, cmap='viridis', aspect='auto', interpolation='none') #try magma or viridis
plt.colorbar(cax, label='Behavior Occurrences')
plt.xlabel('Time (ms)')
plt.ylabel('Behaviors')

plt.yticks(np.arange(len(desired_order)), desired_order)
#plt.xticks(np.arange(0, num_time_bins, num_time_bins // 10), np.arange(-trial_pre_stim, trial_post_stim + 1, (trial_pre_stim + trial_post_stim) // 10))


if plotting_condition == 'unpal':
    title_suffix = f'Unpalatable tastants (total trials = {len(unpal_trials)})'
else:
    title_suffix = f'Palatable tastants (total trials = {len(pal_trials)})'

plt.title(
    f'{rat_name}, Test Day {test_day}\n{title_suffix}')

plt.show()



'''


# Extract unique behaviors and define desired order
unique_behaviors = list(set(scoring_dict['behavior']))
behavior_names = unique_behaviors
desired_order = ['gape', 'mouth or tongue movements', 'lateral tongue protrusion', 'unknown mouth movement', 'other']

# Define the time bin parameters
time_bin_width_ms = 100  # Width of each time bin in milliseconds
num_time_bins = int(trial_post_stim / time_bin_width_ms)



# Initialize dictionaries to store aggregated data
aggregate_data = {
    'YW_unpal': {'mtm': [], 'ltm': [], 'gape': []},
    'YW_pal': {'mtm': [], 'ltm': [], 'gape': []},
    'NBT_unpal': {'mtm': [], 'ltm': [], 'gape': []},
    'NBT_pal': {'mtm': [], 'ltm': [], 'gape': []}
}


every_scored_trials = []
for key in keys_list:
    df = vid_trial_start[key]
    # Filter for trials where modifier #1 is 'in view'
    mask = df.iloc[:, 2] == 'in view'
    every_scored_trials.append(df.loc[mask, df.columns[1]].values)

# Find the intersection of scored trials between the two scorers
common_trials = []
for i in range(0, len(vid_trial_start)-1, 2):
    common_trial = sorted(list(set(every_scored_trials[i]).intersection(set(every_scored_trials[i+1]))), key = int)
    common_trial = [int(element) for element in common_trial] # Converting to integers
    common_trials.append(common_trial)




#for behavior_key, behavior_table in behavior_dict.items():
#    for trial_key, trial_table in vid_trial_start.items():

key_num = 0
for trial_set in common_trials:
    i = 0
    for i in range(2):
        for trial in trial_set:
            row_of_current_trial = vid_trial_start[keys_list[key_num]][vid_trial_start[keys_list[key_num]]['Modifier #1'] == str(trial)]
            print(row_of_current_trial)
            trial_start_time = row_of_current_trial.loc[:, 'Time'].values[0]
            
            mod_key = os.path.dirname(key)
            taste_indx = trial_info[mod_key][trial-1][0]
            
            if taste_indx == 0 or taste_indx == 1 or taste_indx == 2: #pal
                
        
        
        
        key_num += 1









'''
# ChatGPT code that is not working
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import os

# Assuming the dictionary is named behavior_dict and contains keys as file paths and values as behavior tables
# Also assuming vid_trial_start is a dictionary of DataFrames
# behavior_dict = { ... }
# vid_trial_start = { ... }

# Extract unique behaviors and define desired order
unique_behaviors = list(set(scoring_dict['behavior']))
behavior_names = unique_behaviors
desired_order = ['gape', 'mouth or tongue movements', 'lateral tongue protrusion', 'unknown mouth movement', 'other']

# Define the time bin parameters
time_bin_width_ms = 100  # Width of each time bin in milliseconds
num_time_bins = int(trial_post_stim / time_bin_width_ms)

# Initialize dictionaries to store aggregated data
aggregate_data = {
    'YW_unpal': [],
    'YW_pal': [],
    'NBT_unpal': [],
    'NBT_pal': []
}

# Regular expression to extract the rat name, test day, and scorer from the file path
pattern = re.compile(r'/media/natasha/drive2/Natasha_Data/(?P<rat_name>NB\d+)/(?P<test_day>Test\d+)/.*_(?P<scorer>\w+)_scoring_dict\.npz')

# Function to update behavior matrix
def update_behavior_matrix(behavior_matrix, behavior_table, trial_list, trial_start_col_idx, trial_start_time_offset_ms, key):
    for trial in trial_list:
        print(key)
        
        trial_start_time = vid_trial_start[key].iloc[trial - 1, trial_start_col_idx]
        trial_behaviors = behavior_table[(behavior_table['Time'] >= trial_start_time) & 
                                         (behavior_table['Time'] <= trial_start_time + (trial_post_stim / 1000))]
        trial_behaviors = trial_behaviors[trial_behaviors['Behavior'] != 'out of view']
        
        tmp_behaviors_dict = {i: None for i in unique_behaviors}
        
        for index, row in trial_behaviors.iterrows():
            if tmp_behaviors_dict[row[0]] is None:
                tmp_behaviors_dict[row[0]] = [(row[4] - trial_start_time) * 1000]
            else:
                tmp_behaviors_dict[row[0]].extend([(row[4] - trial_start_time) * 1000])
        
        for key, value in tmp_behaviors_dict.items():
            if isinstance(value, list):
                tmp_behaviors_dict[key] = [(value[i], value[i + 1]) for i in range(0, len(value) - 1, 2)]
        
        rearranged_tmp_dict = {key: tmp_behaviors_dict[key] for key in desired_order if key in tmp_behaviors_dict}
        for key in tmp_behaviors_dict:
            if key not in rearranged_tmp_dict:
                rearranged_tmp_dict[key] = tmp_behaviors_dict[key]

        for behavior, intervals in rearranged_tmp_dict.items():
            if intervals is None:
                continue
            
            behavior_idx = desired_order.index(behavior)
            
            for interval in intervals:
                start_time, end_time = interval
                
                start_bin = int(start_time / time_bin_width_ms)
                end_bin = int(end_time / time_bin_width_ms)
                
                for bin_idx in range(start_bin, min(end_bin, num_time_bins)):
                    behavior_matrix[behavior_idx, bin_idx] += 1

# Iterate through the dictionary and aggregate data for each scorer and condition
for key, behavior_table in behavior_dict.items():
    print(key)
    match = pattern.search(key)
    if match:
        rat_name = match.group('rat_name')
        test_day = match.group('test_day')
        scorer = match.group('scorer')
        print(scorer)
        trial_start_data = vid_trial_start[key]
        
        # Filter good trials based on 'in view' condition
        in_view_mask = trial_start_data['Modifier #2'] == 'in view'
        good_trial_table = trial_start_data[in_view_mask]
        
        # Categorize trials based on tastant type
        suc_trials = []
        nacl_trials = []
        ca_trials = []
        qhcl_trials = []
        
        for index, row in good_trial_table.iterrows():
            mod_key = os.path.dirname(key)
            
            modifier_value = row['Modifier #1']
            temp_tastant = trial_info[mod_key][int(modifier_value)-1][0]
            if temp_tastant == 0:
                suc_trials.append(int(modifier_value))
            elif temp_tastant == 1:
                nacl_trials.append(int(modifier_value))
            elif temp_tastant == 2:
                ca_trials.append(int(modifier_value))
            elif temp_tastant == 3:
                qhcl_trials.append(int(modifier_value))
        
        pal_trials = suc_trials
        unpal_trials = qhcl_trials

        if scorer == 'YW':
            aggregate_data['YW_unpal'].extend(unpal_trials)
            aggregate_data['YW_pal'].extend(pal_trials)
        elif scorer == 'NBT':
            aggregate_data['NBT_unpal'].extend(unpal_trials)
            aggregate_data['NBT_pal'].extend(pal_trials)

# Generate and plot behavior matrices for each condition
for key, trial_list in aggregate_data.items():
    behavior_matrix = np.zeros((len(behavior_names), num_time_bins))
    update_behavior_matrix(behavior_matrix, behavior_table, trial_list, 4, trial_post_stim, key)
    
    scorer, condition = key.split('_')
    plt.figure(figsize=(12, 8))
    cax = plt.imshow(behavior_matrix, cmap='viridis', aspect='auto', interpolation='none')
    plt.colorbar(cax, label='Behavior Occurrences')
    plt.xlabel('Time (ms)')
    plt.ylabel('Behaviors')
    plt.yticks(np.arange(len(desired_order)), desired_order)
    
    plt.title(f'{scorer} - {condition} trials (total trials = {len(trial_list)})')
    plt.show()
'''
