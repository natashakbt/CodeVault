#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:19:23 2024

@author: natasha
"""


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
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr


'''# =========INPUT BASE FOLER, RAT NAME and TEST DAY NUM HERE============'''
base_folder = '/media/natasha/drive2/Natasha_Data' # contains folders of all rats' data

rat_name = "NB32"
test_day = [1, 2, 3]
scorer_initials = ['YW', 'NBT']  # Initials of scorers. ONLY TWO.
'''# ===================================================================='''


'''
#==============================================================================
#Creating scoring_dict: processed scored behavior data from BORIS csv files
#==============================================================================
'''
dirs = []
for day in test_day:
    #Finding path to test day folder inside of base_folder
    dirname = (os.path.join(os.path.join(base_folder, rat_name), f"Test{day}"))
    if not os.path.exists(dirname):
        dirname = os.path.join(os.path.join(base_folder, rat_name), f"test{day}")
        if not os.path.exists(dirname):
            print(f"Error: Directory '{dirname}' does not exist.")
    dirs.append(dirname)

csv_paths = []
for dirname in dirs:
    # Transforming csv files exported from BORIS into a dictionary used for analysis
    csv_path = glob.glob(os.path.join(dirname, '**', '*.csv'), recursive=True) #finding all csv files
    for path in csv_path.copy(): # Removing the electrode_layout.csv
        if 'electrode_layout' in path:
            csv_path.remove(path)
    if len(csv_path) <2 : 
        print('One or zero scoring csv files found!')
    else :
        print('scoring csv files:\n', csv_path)
    csv_paths.append(csv_path)

scoring_dict = {} # dictionary containing two tables: trial starts only, and behaviors only.  
for csv_path in csv_paths:
    # Data from each csv file appended into scoring_data
    scoring_data = []
    for csv in csv_path:
        scoring_data.append(pd.read_csv(csv))
        
    # Only keeping essential columns
    columns_to_keep = ['Behavior', 'Modifier #1', 'Modifier #2', 'Behavior type', 'Time']
    for table in scoring_data:
        columns_to_drop = [col for col in table.columns if col not in columns_to_keep]
        table.drop(columns=columns_to_drop, inplace=True)
    
    
    for csv, table in zip(csv_path, scoring_data):
        mask = table.iloc[:, 0] == 'trial start'
        trial_start = table[mask] # Select rows indicating 'trial start'
        behavior = table[~mask] # All other rows will be scored behaviors
        # Store both DataFrames in the dictionary
        scoring_dict[csv] = {
            'trial_start': trial_start,
            'behavior': behavior
        }
        
keys_list = list(scoring_dict.keys())

'''
## Saving scoring table as an .npy file to dirname
os.chdir(dirname)
new_filename = f'{rat_name}_test{test_day}_comparison_dict.npz'
np.savez(new_filename, scoring_dict)
'''


'''
#==============================================================================
# INTER-RATER RELIABILITY ANALYSES
#==============================================================================
'''
# Finding the trials scored in common by both scorers (assumed by label 'in view')
every_scored_trials = []
for key in keys_list:
    df = scoring_dict[key]['trial_start']
    # Filter for trials where modifier #1 is 'in view'
    mask = df.iloc[:, 2] == 'in view'
    every_scored_trials.append(df.loc[mask, df.columns[1]].values)

common_trials = []
for i in range(0, len(scoring_dict), 2):
    # Finding the intersection of scored trials between two raters
    common_trial = sorted(list(set(every_scored_trials[i]).intersection(set(every_scored_trials[i+1]))), key = int)
    common_trial = [int(element) for element in common_trial] # Converting to integers
    common_trials.append(common_trial)
# day 1 = [23, 27, 28, 29, 30, 60, 102, 113]


'''
# === Analyzing frequency (correlation) and presence/absense (Cohen's Kappa) of a behavior between scorers  ===
'''
unique_behaviors = scoring_dict[keys_list[0]]['behavior']['Behavior'].unique()
unique_behaviors = unique_behaviors.tolist()
unique_behaviors.append('no movement') 
# TODO: THIS IS CHEATING. ACTUALLY SCAN THROUGH ALL THE BEHAVIORS


# Initialize behavior frequency dictionary for each scorer
behavior_frequency = {}
for key in keys_list:
    behavior_frequency[key] = {i: [0] * len(common_trials[int(keys_list.index(key) * 0.5)]) for i in unique_behaviors}
    

# Count behaviors for each trial and scorer
day_loop = 0
for key in keys_list:
    trial_count = 0
    for row_index in range(len(scoring_dict[key]['trial_start'])):
        
        if int(scoring_dict[key]['trial_start'].iloc[row_index, 1]) in common_trials[int(day_loop)]: 
            trial_start_time = scoring_dict[key]['trial_start'].iloc[row_index, 4]
            if row_index < len(scoring_dict[key]['trial_start'])-1:
                trial_end_time = scoring_dict[key]['trial_start'].iloc[row_index+1, 4]
            else:
                trial_end_time = float(scoring_dict[key]['behavior'].iloc[-1:, 4])

            for behavior_index, behavior_row in scoring_dict[key]['behavior'].iterrows():
                behavior_time = behavior_row['Time']
                behavior_type = behavior_row['Behavior']
                start_or_stop = behavior_row['Behavior type']


                # Check if behavior falls within the trial time range
                if (trial_start_time <= behavior_time) and (behavior_time <= trial_end_time) and (start_or_stop == 'START'):
                    behavior_frequency[key][behavior_type][trial_count] += 1

            
            # Append the list of counts for this trial to behavior_frequency
            trial_count += 1
    day_loop += 0.5


# Initialize dictionaries for Pearson's correlation and Cohen's kappa
pearsons_behavior_frequency = {behavior: {scorer: [] for scorer in scorer_initials} for behavior in unique_behaviors}
cohens_kappa_present_absent = {behavior: {scorer: [] for scorer in scorer_initials} for behavior in unique_behaviors}

# Calculate frequencies and presence/absence for each behavior and scorer
for behavior in unique_behaviors:
    for key in keys_list:
        scorer = scorer_initials[0] if scorer_initials[0] in key else scorer_initials[1]
        behavior_data = behavior_frequency[key][behavior]
        pearsons_behavior_frequency[behavior][scorer].extend(behavior_data)
        behavior_present_absent = ['y' if count >= 1 else 'n' for count in behavior_data]
        cohens_kappa_present_absent[behavior][scorer].extend(behavior_present_absent)

# Get behaviors of interest for Pearson's correlation
behaviors_i_care_about = ['mouth or tongue movement', 'gape', 'lateral tongue movement']

# Calculate and print Pearson's correlation and Cohen's kappa for each behavior of interest
for behavior in behaviors_i_care_about: 
    y1 = pearsons_behavior_frequency[behavior][scorer_initials[0]]
    y2 = pearsons_behavior_frequency[behavior][scorer_initials[1]]
    corr, p = pearsonr(y1, y2)
    print('Pearsons correlation for', behavior, 'is: %.3f' % corr, p)

    y1_presence = cohens_kappa_present_absent[behavior][scorer_initials[0]]
    y2_presence = cohens_kappa_present_absent[behavior][scorer_initials[1]]
    k = cohen_kappa_score(y1_presence, y2_presence) 
    print('Cohens kappa score for', behavior, 'is', k)



'''
# === Cohen's Kappa of behavior sequence and timing - BROKEN ===
'''

'''
#Once None error is fixed. Make below into one big for loop through common_trials
'''
#for trial in common_trials:
trial = 6 #trial num out of 120 with first trial=1
trial_pre_stim = 500 # in ms
trial_post_stim = 10000 # in ms

trial_len = trial_pre_stim + trial_post_stim

trial_start_time = scoring_dict[keys_list[0]]['trial_start'].iloc[trial-1, 4] ###TRYING TO FIX THIS

for behavior_index, behavior_row in scoring_dict[keys_list[2]]['trial_start'].iterrows():
    if behavior_row['Modifier #1'] == str(trial):
        trial_start_time = behavior_row['Time']




trial_end_time = trial_start_time + (trial_post_stim/1000) #determine end of trial in video time 
trial_start_time -= (trial_pre_stim/1000) #original: 0.5


trial_behaviors = []
for key in keys_list :   
    temp_table = scoring_dict[key]['behavior']
    #behavior_table = scoring_dict[key]['trial_start']['Time']
    trial_behaviors.append(temp_table[(temp_table['Time'] >= trial_start_time) 
                                    & (temp_table['Time'] <= trial_end_time)])



# Find the index with the maximum length
# Get the lengths of each index
behaviors_scored_num = [len(df) for df in trial_behaviors]
# Find the index with the maximum length
max_index = behaviors_scored_num.index(max(behaviors_scored_num))
min_index = behaviors_scored_num.index(min(behaviors_scored_num))
y1_behaviors = list(trial_behaviors[max_index]['Behavior'][::2])
### Add mid-point value to this
y1_times = []

#finding the mid-point time of each of the behaviors
for i in range(0, len(trial_behaviors[max_index]), 2):
    start_time = trial_behaviors[max_index]['Time'].iloc[i]
    stop_time = trial_behaviors[max_index]['Time'].iloc[i+1]
    mid_time_point = (stop_time - start_time) + start_time
    y1_times.append(mid_time_point)
'''
#TODO: all behaviors comming out as None on y2
#WORK ON THIS!!!!!
'''

y2_behaviors = []

# Iterate through each time value in y1_times
for time_point in y1_times:
    # Initialize a variable to keep track of whether a match is found
    match_found = False
    
    # Iterate through each row in the table
    for index, row in trial_behaviors[min_index].iterrows():
        if index + 1 < len(trial_behaviors[min_index]['Time']):
        # Check if the time_point falls within the range defined by START and STOP times
            if row['Time'] <= time_point <= trial_behaviors[min_index]['Time'][index+1]:
                # If it does, append the behavior to the behaviors_at_times list
                y2_behaviors.append(row['Behavior'])
                match_found = True
                break  # Break the loop after finding the behavior for the current time point
        
    # If no match is found, append 'None' to the behaviors_at_times list
    if not match_found:
        y2_behaviors.append('None')
            
        
print(y1_behaviors)
print(y2_behaviors)

cohen_kappa_score(y1_behaviors, y2_behaviors)


'''
#==============================================================================
# Plotting - setup
#==============================================================================
'''

###SETTING UP THE DATA TO PLOT
## I THINK THIS WORKS OK???
day_to_plot = 1
trial = 23 #trial num out of 120 with first trial=1
trial_pre_stim = 500 # in ms
trial_post_stim = 20000 # in ms
# day 1 = [23, 27, 28, 29, 30, 60, 102, 113]
trial_len = trial_pre_stim + trial_post_stim

key_index_needed = day_to_plot*2-2

trial_start_time = scoring_dict[keys_list[key_index_needed]]['trial_start'].iloc[trial-1, 4]


for behavior_index, behavior_row in scoring_dict[keys_list[key_index_needed]]['trial_start'].iterrows():
    if behavior_row['Modifier #1'] == str(trial):
        trial_start_time = behavior_row['Time']



trial_end_time = trial_start_time + (trial_post_stim/1000) #determine end of trial in video time 
trial_start_time -= (trial_pre_stim/1000) #original: 0.5

trial_behaviors = []
for key in keys_list :   
    temp_table = scoring_dict[key]['behavior']
    #behavior_table = scoring_dict[key]['trial_start']['Time']
    trial_behaviors.append(temp_table[(temp_table['Time'] >= trial_start_time) 
                                    & (temp_table['Time'] <= trial_end_time)])

# Find unique values in the 'Behavior' column
#unique_behaviors = pd.concat(trial_behaviors)['Behavior'].unique()
#behaviors_dict = {i: None for i in unique_behaviors}


behaviors_dict = {key: {i: None for i in unique_behaviors} for key in keys_list}

# loop through all behaviors in dict
# append time any time the behavior starts or stops
for i in range(len(keys_list)):
    for index,row in trial_behaviors[i].iterrows():  #TODO: Works with trial_behaviors[0].iterrows - add a parent for loop?
        #converting video time to ephys time
        temp_time = [((row[4]-trial_start_time)/(trial_end_time - trial_start_time))*trial_len]
        if behaviors_dict[keys_list[i]][row[0]] == None:
            behaviors_dict[keys_list[i]][row[0]] = temp_time
        else:
            behaviors_dict[keys_list[i]][row[0]].extend(temp_time)

#converts the list within each index
# into a list of tuples containing pairs of consecutive values
# [(first_start_time, first_stop_time), (etc.)}]
for key in keys_list:
    for k, value in behaviors_dict[key].items():
        if isinstance(value, list):
            behaviors_dict[key][k] = [(value[i], value[i + 1]) 
                                   for i in range(0, len(value) - 1, 2)]


behavior_names = unique_behaviors # could be changed out for rearranging stuff. see below
# #re-arranging into desired order
# desired_order = ['gape', 'mouth movements', 'tongue protrusion', 'lateral tongue protrusion', 'unknown mouth movement']

# rearranged_dict = {key: behaviors_dict[key] for key in desired_order if key in behaviors_dict}

# # Add any keys not in the desired order
# for key in behaviors_dict:
#     if key not in rearranged_dict:
#         rearranged_dict[key] = behaviors_dict[key]

# print(rearranged_dict)
# # Making list of behavior names and time intervals
# # Needed for ease of plotting
# behavior_names = list(rearranged_dict.keys())
# time_intervals = list(rearranged_dict.values())


'''
# === Actually creating figure ===
'''


# Create a figure and axis
fig, (ax1, ax2) = plt.subplots(2, 1,
                                    figsize=(8, 4),
                                    dpi=600,
                                    sharex=True,
                                    sharey=True)

### Subplot 1 stuff (behavior)
# Set the y-axis ticks and labels
ax1.set_yticks(range(len(behavior_names)))
ax1.set_yticklabels(behavior_names)

# Define the linewidth of the bars
bar_linewidth = 25  # Adjust the value as needed

# Create a colormap with the number of behaviors
colors = cm.get_cmap('tab10', len(behavior_names))

# Iterate over the behavior names and corresponding time intervals
for i, intervals in enumerate(behaviors_dict[keys_list[0]]):
    temp_times = behaviors_dict[keys_list[0]][intervals]
    if temp_times is not None:
        for interval in temp_times: # Iterate over the time intervals and plot horizontal bars
            start_time, end_time = interval
            ax1.hlines(i, start_time, end_time,
                       linewidth=bar_linewidth, color=colors(i))
            
for i, intervals in enumerate(behaviors_dict[keys_list[1]]):
    temp_times = behaviors_dict[keys_list[1]][intervals]
    if temp_times is not None:
        for interval in temp_times:  # Iterate over the time intervals and plot horizontal bars
            start_time, end_time = interval
            ax2.hlines(i, start_time, end_time,
                       linewidth=bar_linewidth, color=colors(i))

plt.xlabel('Time (ms)')
plt.ylabel('Behaviors')


subplottitle1 = keys_list[2][keys_list[0].rfind('/') + 1:]
subplottitle2 = keys_list[3][keys_list[1].rfind('/') + 1:]

ax1.set_title(subplottitle1)
ax2.set_title(subplottitle2)

#plt.title('Trial: ', trial)
# Show the plot
plt.show()
