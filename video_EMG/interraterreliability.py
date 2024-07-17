#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:17:26 2024

@author: natasha

This script analyzes the interrater-reliability between two scorers 
that used the BORIS program to label oral/facial movements of rats responding to taste.

Prerequisites:
- Base folder must contain subfolders for each rat.
- Each rat's folder should have subfolders named "Test#" or "test#".
- Each test folder must contain two CSV files exported from BORIS, named with the scorer's initials.
- CSV file must have behaviors in pairs and alternating START and STOP. 
- In CSV, all trials with 'in view' modifier must be scored

Structure:
1. Import and organize data into a large dictionary.
2. Analyze Pearson's correlation of frequency of each behavior scored within a trial
3. Analyze Cohen's kappa of whether a behavior was scored or not within a trial
4. Analyze Cohen's kappa for the alignment of every behavior
5. Calculate the percent overlap of timed scored between two scorers
6. Plot and save scoring sequence for each trial.


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

rat_list = ["NB35", "NB32", "NB34"] # List of all rats to be analyzed
#rat_list = ["NB32"]
test_day = [1, 2, 3] # Greatest number of test days
scorer_initials = ['YW', 'NBT']  # ONLY TWO! Initials of scorers (as indicated in csv file names)
'''# ===================================================================='''

# Validate that the DataFrame for interreader-reliability analyses is setup properly.
def validate_dataframe(df):
    # Check if the indices of pairs of behaviors increment by one step
    indices = df.index
    for i in range(1, len(indices), 2): # Iterate in steps of 2
        if indices[i] != indices[i-1] + 1: # Check if the pair is one step apart
            raise ValueError("Indices are not one apart in the DataFrame.")
            
    # Check if 'Behavior type' alternates between 'START' and 'STOP'
    behavior_types = df['Behavior type'].tolist()
    if not all(a == 'START' and b == 'STOP' for a, b in zip(behavior_types[::2], behavior_types[1::2])):
        raise ValueError("Behavior types are not alternating 'START' and 'STOP'.")


# Add corresponding elements of two lists together.
def add_lists(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    result = []
    
    for elem1, elem2 in zip(list1, list2):
        result.append(elem1 + elem2)
    # Return a new list with the results
    
    return result

# =============================================================================
# Import and process csv files exported from BORIS containing scored behavior data.
# Main output is scoring_dict. 
# Keys are the paths to all csv files. Associated value is a dataframe of csv.
# =============================================================================

# Find all paths to test day folders within rat name folders
# Assume test day folders are named as 'Test#' or 'test#'
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

# Find all paths to csv files within dirs
# Remove electrode_layout.csv. BUT NOT OTHER CSV FILES (COULD CAUSE ERRORS if there are "extra" CSV files)
csv_paths = []
for dirname in dirs:
    # Transform csv files exported from BORIS into a dictionary used for analysis
    csv_path = glob.glob(os.path.join(dirname, '**', '*.csv'), recursive=True) # Find all csv files
    csv_path = [path for path in csv_path if 'electrode_layout' not in path] # Remove the electrode_layout.csv
    if len(csv_path) <2 : 
        print("WARNING Too few (<2) csv files found in:", dirname) # Won't be added
    elif len(csv_path) >2:
        print("WARNING Too many (>2) csv files found in:", dirname) # Won't be added
    else: # If csv_path is equal to 2, append to csv_paths
        csv_paths.append(csv_path) 

print("Analyzing the following csv files:", csv_paths)

# Set up the scoring data from found CSV files into a dictionary
scoring_dict = {} # Initialize the dictionary
for csv_path in csv_paths:
    # Data from each csv file appended into scoring_data
    scoring_data = []
    for csv in csv_path:
        scoring_data.append(pd.read_csv(csv))
    columns_to_keep = ['Behavior', 'Modifier #1', 'Modifier #2', 'Behavior type', 'Time'] # Essential columns to keep
    for table in scoring_data:
        columns_to_drop = [col for col in table.columns if col not in columns_to_keep]
        table.drop(columns=columns_to_drop, inplace=True) # Remove unwanted columns
    for csv, table in zip(csv_path, scoring_data):
        scoring_dict[csv] = table  # Convert DataFrame to list of lists

 
keys_list = list(scoring_dict.keys())

# ## Saving scoring table as an .npy file to dirname
# os.chdir(dirname)
# new_filename = f'{rat_name}_test{test_day}_comparison_dict.npz'
# np.savez(new_filename, scoring_dict)


# =============================================================================
# Create lists, behaviors_i_care_about, unique_behaviors, and common_trials
# that are important for the inter-rater reliability analyses below
# =============================================================================

# Set behaviors of interest for analyses
behaviors_i_care_about = ['mouth or tongue movement', 'gape', 'lateral tongue movement']

# Create a list of all unique behaviors in scoring_dict
unique_behaviors = []
for key in scoring_dict:
    behaviors = scoring_dict[key]['Behavior'].unique()
    unique_behaviors.extend(behaviors.tolist())
# Convert the list to a set to remove duplicates and then convert it back to a list
unique_behaviors = list(set(unique_behaviors))

# Find the trials scored in common by both scorers (by trial label 'in view')
every_scored_trials = []
for key in keys_list:
    df = scoring_dict[key]
    # Filter for trials where modifier #1 is 'in view'
    mask = df.iloc[:, 2] == 'in view'
    every_scored_trials.append(df.loc[mask, df.columns[1]].values)

# Find the intersection of scored trials between the two scorers
common_trials = []
for i in range(0, len(scoring_dict), 2):
    common_trial = sorted(list(set(every_scored_trials[i]).intersection(set(every_scored_trials[i+1]))), key = int)
    common_trial = [int(element) for element in common_trial] # Converting to integers
    common_trials.append(common_trial)


# =============================================================================
# INTER-RATER RELIABILITY ANALYSES
# =============================================================================

# === Analyzing frequency (correlation) and presence/absense (Cohen's Kappa) of a behavior between scorers  ===

# Initialize behavior frequency dictionary for each scorer with zeroes
behavior_frequency = {}
for key in keys_list:
    behavior_frequency[key] = {i: [0] * len(common_trials[int(keys_list.index(key) * 0.5)]) for i in unique_behaviors}

# Count behaviors for each trial and scorer
day_loop = 0
for key in keys_list:
    trial_count = 0
    for row_index in range(len(scoring_dict[key])):
        if scoring_dict[key].iloc[row_index, 0] == 'trial start': # Check that the row is a trial start
            if int(scoring_dict[key].iloc[row_index, 1]) in common_trials[int(day_loop)]: # Check if the trial # is in common_trials
                trial_start_time = scoring_dict[key].iloc[row_index, 4]     
              
                # Find the next 'trial start' to determine end of current trial
                next_trial_start_index = row_index + 1
                while next_trial_start_index < len(scoring_dict[key]): #loop through rows to find next instance of 'trial start'
                    if scoring_dict[key].iloc[next_trial_start_index, 0] == 'trial start':
                        break
                    next_trial_start_index += 1
                
                # If 'trial start' is found, save its time as trial_end_time
                if next_trial_start_index < len(scoring_dict[key]):
                    trial_end_time = scoring_dict[key].iloc[next_trial_start_index, 4]
                # If 'trial start' is not found (i.e. you're on the last trial), set end time to the last value of the table
                else:
                    trial_end_time = float(scoring_dict[key].iloc[-1:, 4]) #if you're on the last trial. Make trial end time the last value of the table
                
                #trial_end_time = trial_start_time + 10000 #in ms. IF YOU WANT A FIXED TRIAL LENGTH, UNCOMMENT THIS.
                
                for behavior_index, behavior_row in scoring_dict[key].iterrows():
                    behavior_time = behavior_row['Time']
                    behavior_type = behavior_row['Behavior']
                    start_or_stop = behavior_row['Behavior type']

    
                    # Check if behavior falls within the trial time range
                    if (trial_start_time <= behavior_time) and (behavior_time <= trial_end_time) and (start_or_stop == 'START'):
                        behavior_frequency[key][behavior_type][trial_count] += 1 #add to count of 
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


# Calculate and print Pearson's correlation and Cohen's kappa for each behavior of interest
for behavior in behaviors_i_care_about: 
    y1 = pearsons_behavior_frequency[behavior][scorer_initials[0]]
    y2 = pearsons_behavior_frequency[behavior][scorer_initials[1]]
    corr, p = pearsonr(y1, y2)
    print(behavior, 'pearsons correlation: %.3f' % corr, ', p = ',p)
for behavior in behaviors_i_care_about: 
    y1_presence = cohens_kappa_present_absent[behavior][scorer_initials[0]]
    y2_presence = cohens_kappa_present_absent[behavior][scorer_initials[1]]
    k = cohen_kappa_score(y1_presence, y2_presence) 
    print(behavior, 'cohens kappa: ', round(k,3))
    
 

# === Cohen's Kappa of behavior sequence and timing ===
#TODO: See if I can evaluate whether a behavior is aligned to mid-point with more wiggle room?

cohens_kappa_seq = {scorer: [] for scorer in scorer_initials}

#for key_index in range(0, len(keys_list), 2): # Loop through test days
for key_index in [2]: # For troubleshooting with just one test day for one scorer
    day_index = math.floor(key_index/2)
    for trial in common_trials[day_index]: 
    #for trial in [102]:# For troubleshooting just one trial
        for behavior_index, behavior_row in scoring_dict[keys_list[key_index]].iterrows(): #Find the trial start time
            if behavior_row['Modifier #1'] == str(trial):
                trial_start_time = behavior_row['Time']
        trial_end_time = trial_start_time + 10 # Set trial end time for how long to compare, in s.

        trial_behaviors = []

        key_pair = [keys_list[key_index]] + [keys_list[key_index+1]] # To find CSV file pairs to compare

        for key in key_pair :  

            temp_table = scoring_dict[key]
            temp_table = temp_table[temp_table['Behavior'].isin(behaviors_i_care_about)] # Only copy behaviors I care about
            # List of two dataframes, for each scorer, with behaviors within a given trial
            trial_behaviors.append(temp_table[(temp_table['Time'] > trial_start_time) 
                                            & (temp_table['Time'] <= trial_end_time)])
                
        # Find the scorer with the most number of behaviors. Becomes 'leading' scorer for the trial
        behaviors_scored_num = [len(df) for df in trial_behaviors] # Get the lengths of each index
        max_index = behaviors_scored_num.index(max(behaviors_scored_num))
        min_index = 1 - max_index #doing it this way to get around condition where number of behaviors scored is equal
        
        # If an uneven number of behaviors captured (i.e. missing the STOP of a START/STOP pair), remove the last row
        # TODO: Can I add the stop pair instead of deleting the start?
        if len(trial_behaviors[max_index]) % 2 != 0: 
            trial_behaviors[max_index] = trial_behaviors[max_index][:-1]
        if len(trial_behaviors[min_index]) % 2 != 0: 
            trial_behaviors[min_index] = trial_behaviors[min_index][:-1]
        
        for i, df in enumerate(trial_behaviors):
            try:
                validate_dataframe(df) # Check that dataframe is correctly setup for comparison
            except ValueError as e:
                print(f"Error: {str(e)}")
                print("Check behavior sequence of trial", trial, "in", key_pair[i], "\n")
        
        y1_behaviors = list(trial_behaviors[max_index]['Behavior'][::2])  # Set y1 as behaviors sequence from the 'leading' scorer

        # Find the mid-point time of each of the y1 behaviors
        y1_times = [] 
        for i in range(0, len(trial_behaviors[max_index]), 2):
            start_time = trial_behaviors[max_index]['Time'].iloc[i]
            stop_time = trial_behaviors[max_index]['Time'].iloc[i+1]
            mid_time_point = ((stop_time - start_time))/2 + start_time
            y1_times.append(mid_time_point)

        # Find whether each midpoint time aligns to a behavior scored by the 'lagging' scorer
        y2_behaviors = []
        for time_point in y1_times: # Iterate through each time value in y1_times
            match_found = False # Initialize a variable to keep track of whether a match is found
            # Iterate through each row in the behavior table
            for index, row in trial_behaviors[min_index].iloc[::2].iterrows():
                # Check if time_point falls within the range defined by START and STOP times 
                if row['Time'] <= time_point <= trial_behaviors[min_index]['Time'][index+1]:
                    # If it does, append the behavior to the behaviors_at_times list
                    y2_behaviors.append(row['Behavior'])
                    match_found = True
                    break  # Break the loop after finding the behavior for the current time point
                
            # If no match is found, append 'none' to the behaviors_at_times list
            if not match_found:
                y2_behaviors.append('none')
        
        # Add behavior sequence to dictionary      
        cohens_kappa_seq[scorer_initials[max_index]].extend(y1_behaviors)
        cohens_kappa_seq[scorer_initials[min_index]].extend(y2_behaviors)


# Find all indices of 'none' in first list, delete 'none's and associated behavior in second list
first_indices_of_none = [i for i, value in enumerate(cohens_kappa_seq[scorer_initials[0]]) if value == 'none']
for index in reversed(first_indices_of_none):
    del cohens_kappa_seq[scorer_initials[1]][index]
    del cohens_kappa_seq[scorer_initials[0]][index]
    
# Find all indices of 'none' in second list, delete 'none's and associated behavior in first list
second_indices_of_none = [i for i, value in enumerate(cohens_kappa_seq[scorer_initials[1]]) if value == 'none']
for index in reversed(second_indices_of_none):
    del cohens_kappa_seq[scorer_initials[1]][index]
    del cohens_kappa_seq[scorer_initials[0]][index]


# Calculate Cohen's kappa score between behavior sequences of both scorers
k = cohen_kappa_score(cohens_kappa_seq[scorer_initials[1]], cohens_kappa_seq[scorer_initials[0]])
print("Cohen's Kappa score: ", round(k,3))


# === Percent of total scoring time that overlaps between the two scorers ===

# Find the trial with the longest time scored
max_time = 0
for key in keys_list:
    if float(scoring_dict[key]['Time'][-1:]) > max_time:
        max_time = float(scoring_dict[key]['Time'][-1:])

# Create an array that is the length of the longest trial times sixty, which is frames per second of the camera.
behavior_array = {behavior: {key: [0] * (math.ceil(max_time) * 60) for key in keys_list} for behavior in behaviors_i_care_about}

start_found = False
start_index = None
stop_index = None
for key in  keys_list:
    for behavior in behaviors_i_care_about:
        for behavior_index, behavior_row in scoring_dict[key].iterrows():
            if (behavior_row['Behavior'] == behavior):
                if behavior_row["Behavior type"] == 'START':
                    start_found = True
                    start_index = round(float(behavior_row['Time'])*60)
                elif behavior_row["Behavior type"] == 'STOP' and start_found:
                    stop_index = round(float(behavior_row['Time'])*60)
                    for i in range(start_index, stop_index):
                        behavior_array[behavior][key][i] = 1
                    # Reset for next pair
                    start_found = False
                    start_index = None
                    stop_index = None

overlap_array = {behavior: [] for behavior in behaviors_i_care_about}

for behavior in behaviors_i_care_about:
    for key_index in range(0, len(keys_list), 2):
        key_pair = [keys_list[key_index]] + [keys_list[key_index+1]]
        sum_list = add_lists(behavior_array[behavior][key_pair[0]], behavior_array[behavior][key_pair[1]])
        overlap_array[behavior].extend(sum_list)
    
for behavior in behaviors_i_care_about:
    my_list = overlap_array[behavior]
    percent_overlap = my_list.count(2)/(my_list.count(1) + my_list.count(2))
    print("fraction of overlap for", behavior, "is: ", round(percent_overlap, 4))
        
    
# TODO: add plotting stuff here. Have it run through all common_trials and save the graphs

