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
- CSV file must have 
    - trial start times coded as point events, with a modifier including 'in view'
    - behaviors coded as state events with alternating START and STOP times. 
- In CSV, all trials with 'in view' modifier must be scored

Structure:
1. Import and organize data into a large dictionary.
2. Analyze Pearson's correlation of frequency of each behavior scored within a trial
3. Analyze Cohen's kappa of whether a behavior was scored or not within a trial
4. Analyze Cohen's kappa for the alignment of every behavior
5. Calculate the percent overlap of timed scored between two scorers
6. Plot and save scoring sequence for each trial.


"""

import glob
import os
import pandas as pd
import math
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import random

'''# =========INPUT BASE FOLER, RAT NAME and TEST DAY NUM HERE============'''
base_folder = '/media/natasha/drive2/Natasha_Data' # contains folders of all rats' data

rat_list = ["NB35", "NB32", "NB34"] # List of all rats to be analyzed
test_day = [1, 2, 3] # Greatest number of test days. If not all animals have this number of test days, you'll get a warning but it can be ignored
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
        filename_last = os.path.basename(csv)
        last_part = os.path.splitext(filename_last)[0]
        scoring_dict[last_part] = table  # Convert DataFrame to list of lists

 
keys_list = list(scoring_dict.keys())

# =============================================================================
# Create lists, behaviors_i_care_about, unique_behaviors, and common_trials
# that are important for the inter-rater reliability analyses below
# =============================================================================

# Set behaviors of interest for analyses, everything else will not be analyzed
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

# %% Presence/absence cohen's kappa
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
    
 
# %% Cohen's kappa behavior sequence/timing
# === I THINK THE CODE BELOW DOESN'T WORK - DON'T USE IT ===
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
        
# %% New inter-rater reliability
# =============================================================================
# INTER-RATER RELIABILITY ANALYSES (USE THIS ONE!)
# =============================================================================

fps = 60 
trial_length = 5 # in seconds, how long to evaluate the interrater reliability

# =============================================================================
# DEFINE IMPORTANT FUNCTIONS
# =============================================================================

def clean_dataframe(index, df):
    new_rows = []
    good_trials = common_trials[index]
    trial_num = None
    trial_end = None
    i = 0

    while i < len(df) - 1:
        row = df.iloc[i]
        
        if row['Behavior'] == 'trial start':
            trial_num = int(row['Modifier #1'])
            trial_end = int(row['Time']) + trial_length
            i += 1
            continue

        # Make sure we are working with a START/STOP pair
        if (row['Behavior'] in behaviors_i_care_about and 
            row['Behavior type'] == 'START' and
            i + 1 < len(df)):

            next_row = df.iloc[i + 1]

            # Confirm this is a proper pair
            if (next_row['Behavior'] == row['Behavior'] and 
                next_row['Behavior type'] == 'STOP'):

                # Check if both START and STOP fall within the trial
                if (trial_num in good_trials and 
                    int(row['Time']) < trial_end and 
                    int(next_row['Time']) < trial_end):

                    # Append both START and STOP rows
                    for this_row in (row, next_row):
                        filtered_row = this_row[['Behavior', 'Behavior type', 'Time']].copy()
                        filtered_row['Trial'] = trial_num
                        new_rows.append(filtered_row)
                
                i += 2  # Skip to the next pair
            else:
                i += 1  # Skip this unpaired START
        else:
            i += 1  # Not a START or behavior we care about

    new_df = pd.DataFrame(new_rows)
    new_df = new_df.reset_index(drop=True)
    validate_dataframe(new_df)
    return new_df
      
def get_bouts(array):
    bouts = []
    in_bout = False
    start = None
    for i, val in enumerate(array):
        if val == 1 and not in_bout:
            start = i
            in_bout = True
        elif val == 0 and in_bout:
            bouts.append((start, i))  # [start, end)
            in_bout = False
    if in_bout:
        bouts.append((start, len(array)))
    return bouts

def bouts_overlap(b1, b2):
    # Return True if b1 and b2 overlap
    return not (b1[1] <= b2[0] or b1[0] >= b2[1])


def align_bouts(bouts0, bouts1):
    matched0 = []
    matched1 = []
    for i, b0 in enumerate(bouts0):
        # Try to match with the first available overlapping bout from scorer1
        found_match = False
        for j, b1 in enumerate(bouts1):
            if bouts_overlap(b0, b1):
                found_match = True
                matched0.append('y')
                matched1.append('y')
                break
        if not found_match:
            matched0.append('y')
            matched1.append('n')

    return matched0, matched1

# =============================================================================
# BUILDING DATAFRAME NEEDED FOR COMPARISON
# dataframe to be built is to_compare_df
# For every day/scorer/behavior - there is an array that is trial_length in sec * fps long
# This array is 0s and 1s for when the behavior occured per frame, concatinating for every trial
# =============================================================================

rows = []

for key in keys_list:
    prefix, suffix = key.rsplit("_", 1)
    index = math.floor(keys_list.index(key)/2)
    
    new_df = clean_dataframe(index, scoring_dict[key])
    
    rows.append({
        'test_day': prefix,
        'scorer': suffix,
        'dataframe': new_df
    })

eval_scoring_df = pd.DataFrame(rows)

unique_days = eval_scoring_df['test_day'].unique()

to_compare = []
for days in unique_days:
    matching_rows = eval_scoring_df[eval_scoring_df['test_day'] == days]
    if len(matching_rows) != 2:
        print("something is wrong...") 

    for behavior in behaviors_i_care_about:
        for df in matching_rows.iterrows():
            scorer = df[1]['scorer']
            behavior_df = df[1]['dataframe']
            behavior_df = behavior_df[behavior_df['Behavior'] == behavior]
            if behavior_df.empty:
                continue  # Skip if none of specific behaviors were labelled in a given test session
            end_time = behavior_df['Time'].max()  # Last time point in seconds
            num_frames = int(np.ceil(end_time * fps))  # total number of frames
            
            # Initialize binary time series
            movement_array = np.zeros(num_frames, dtype=int)
            
            # 3. Iterate through START-STOP pairs and set 1s
            
            starts = behavior_df[behavior_df['Behavior type'] == 'START']['Time'].values
            stops = behavior_df[behavior_df['Behavior type'] == 'STOP']['Time'].values
            
            # Sanity check
            assert len(starts) == len(stops), "Unmatched START/STOP pairs"
            
            # Change binary time series to 1s if behavior is occuring
            for start, stop in zip(starts, stops):
                start_idx = int(np.floor(start * fps))
                stop_idx = int(np.ceil(stop * fps))
                movement_array[start_idx:stop_idx] = 1
            print(len(movement_array), scorer, days, behavior)
            to_compare.append({
                'days': days,
                'scorer': scorer,
                'behavior': behavior,
                'movement_array': movement_array
                })

to_compare_df = pd.DataFrame(to_compare) 

# =============================================================================
# COMPARE ARRAYS TO EACH OTHER WITH ACCURACY SCORES
# A IS ONE WAY, B IS THE OTHER WAY
# =============================================================================

results_a = {}
results_b = {}
grouped = to_compare_df.groupby(['days', 'behavior'])

# Iterate through each group and extract movement arrays
for (day, behavior), group in grouped:
    arrays = group['movement_array'].tolist()  # list of arrays (one per scorer)
    if len(arrays) < 2:
        continue
    
    scorers = group['scorer'].tolist()
    min_len = min(arr.shape[0] for arr in arrays)

    trimmed_arrays = [arr[:min_len] for arr in arrays] # Trim arrays to match the shortest
    
    # Map scorer to trimmed array
    scorer_to_array = dict(zip(scorers, trimmed_arrays))
    
    # Ensure consistement assign of a0 to 'YW' and a1 to 'NBT'
    a0 = scorer_to_array['YW']
    a1 = scorer_to_array['NBT']
    
        
    # Elementwise comparison
    comparison = a0 != a1
    if np.any(comparison) == False:
        print(f"No difference between scorers' arrays? {day} {behavior}")
    
    
    bouts0 = get_bouts(a0)
    bouts1 = get_bouts(a1)
    
    results0, results1 = align_bouts(bouts0, bouts1)
    results2, results3 = align_bouts(bouts1, bouts0)
    
    if behavior not in results_a:
        results_a[behavior] = {scorers[0]: [], scorers[1]: []}
    if behavior not in results_b:
        results_b[behavior] = {scorers[0]: [], scorers[1]: []}
    # Assign using real scorer names
    results_a[behavior][scorers[0]].extend(results0)
    results_a[behavior][scorers[1]].extend(results1)    
    
    # Assign using real scorer names
    results_b[behavior][scorers[0]].extend(results2)
    results_b[behavior][scorers[1]].extend(results3)


results_bar_chart_a = []
results_bar_chart_b = []
for behavior in behaviors_i_care_about: 
    y0_presence_nested = results_a[behavior][scorer_initials[0]]
    y1_presence_nested = results_a[behavior][scorer_initials[1]]
    
    y2_presence_nested = results_b[behavior][scorer_initials[0]]
    y3_presence_nested = results_b[behavior][scorer_initials[1]]
    
    
    ka = accuracy_score(y0_presence_nested, y1_presence_nested) 
    kb = accuracy_score(y2_presence_nested, y3_presence_nested) 
    results_bar_chart_a.append(ka)
    results_bar_chart_b.append(kb)

# =============================================================================
# CREATE SHUFFLED DATA
# =============================================================================

# %%
iterations = 100 # how many times to shuffle
sh_results_bar_chart_a = []
sh_results_bar_chart_b = []
for i in range(iterations):
    sh_to_compare = []
    for days in unique_days:
        matching_rows = eval_scoring_df[eval_scoring_df['test_day'] == days]
        if len(matching_rows) != 2:
            print("something is wrong...") 
    
        for behavior in behaviors_i_care_about:
            for df in matching_rows.iterrows():
                scorer = df[1]['scorer']
                behavior_df_to_shuffle = df[1]['dataframe']
                
                for i in range(0, len(behavior_df_to_shuffle), 2):
                    rand_behavior = random.choice(behaviors_i_care_about)
                    
                    behavior_df_to_shuffle.at[i, 'Behavior'] = rand_behavior
                    behavior_df_to_shuffle.at[i + 1, 'Behavior'] = rand_behavior
                
                behavior_df = behavior_df_to_shuffle[behavior_df_to_shuffle['Behavior'] == behavior]
                if behavior_df.empty:
                    continue  # Skip if none of specific behaviors were labelled in a given test session
                end_time = behavior_df['Time'].max()  # Last time point in seconds
                num_frames = int(np.ceil(end_time * fps))  # total number of frames
                
                # Initialize binary time series
                movement_array = np.zeros(num_frames, dtype=int)
                
                # 3. Iterate through START-STOP pairs and set 1s
                starts = behavior_df_to_shuffle[behavior_df_to_shuffle['Behavior type'] == 'START']['Time'].values
                stops = behavior_df_to_shuffle[behavior_df_to_shuffle['Behavior type'] == 'STOP']['Time'].values
                
                # Sanity check
                assert len(starts) == len(stops), "Unmatched START/STOP pairs"
                
                # Change binary time series to 1s if behavior is occuring
                for start, stop in zip(starts, stops):
                    start_idx = int(np.floor(start * fps))
                    stop_idx = int(np.ceil(stop * fps))
                    movement_array[start_idx:stop_idx] = 1
                #print(len(movement_array), scorer, days, behavior)
                sh_to_compare.append({
                    'days': days,
                    'scorer': scorer,
                    'behavior': behavior,
                    'movement_array': movement_array
                    })
    
    sh_to_compare_df = pd.DataFrame(sh_to_compare)
    
    shuffle_df_a = pd.concat([
        to_compare_df[to_compare_df['scorer'] == 'YW'],
        sh_to_compare_df[sh_to_compare_df['scorer'] == 'NBT']
    ], ignore_index=True)
    
    # For shuffle_df_b: keep original 'NBT' + shuffled 'YW'
    shuffle_df_b = pd.concat([
        to_compare_df[to_compare_df['scorer'] == 'NBT'],
        sh_to_compare_df[sh_to_compare_df['scorer'] == 'YW']
    ], ignore_index=True)
    
    sh_results_a = {}
    sh_results_b = {}

    for i in range(2):
        if i == 0:
            sh_grouped = shuffle_df_a.groupby(['days', 'behavior'])
        elif i == 1:
            sh_grouped = shuffle_df_b.groupby(['days', 'behavior'])
                
        # Iterate through each group and extract movement arrays
        for (day, behavior), group in sh_grouped:
            arrays = group['movement_array'].tolist()  # list of arrays (one per scorer)
            if len(arrays) < 2:
                continue
            
            scorers = group['scorer'].tolist()
            min_len = min(arr.shape[0] for arr in arrays)
    
            trimmed_arrays = [arr[:min_len] for arr in arrays] # Trim arrays to match the shortest
            
            # Map scorer to trimmed array
            scorer_to_array = dict(zip(scorers, trimmed_arrays))
            
            if i == 0:
                # Ensure consistement assign of a0 to 'NBT' and a1 to 'YW'
                a0 = scorer_to_array['NBT']
                a1 = scorer_to_array['YW']
            elif i == 1:
                # Ensure consistement assign of a0 to 'YW' and a1 to 'NBT'
                a0 = scorer_to_array['YW']
                a1 = scorer_to_array['NBT']
            
            # Compare frame-by-frame 0/1s arrays to double check
            # that there is no weird error 
            comparison = a0 != a1
            if np.any(comparison) == False:
                print(f"No difference between scorers' arrays? {day} {behavior}")
            
            bouts0 = get_bouts(a0)
            bouts1 = get_bouts(a1)
            
            results0, results1 = align_bouts(bouts0, bouts1)

            if i == 0:
                if behavior not in sh_results_a:
                    sh_results_a[behavior] = {scorers[0]: [], scorers[1]: []}
                # Assign using real scorer names
                sh_results_a[behavior][scorers[0]].extend(results0)
                sh_results_a[behavior][scorers[1]].extend(results1)    
                        
                    
            elif i == 1:        
                if behavior not in sh_results_b:
                    sh_results_b[behavior] = {scorers[0]: [], scorers[1]: []}

                # Assign using real scorer names
                sh_results_b[behavior][scorers[0]].extend(results0)
                sh_results_b[behavior][scorers[1]].extend(results1)

    for behavior in behaviors_i_care_about: 
        y0_presence_nested = sh_results_a[behavior][scorer_initials[0]]
        y1_presence_nested = sh_results_a[behavior][scorer_initials[1]]
        
        y2_presence_nested = sh_results_b[behavior][scorer_initials[0]]
        y3_presence_nested = sh_results_b[behavior][scorer_initials[1]]
        
        ka = accuracy_score(y0_presence_nested, y1_presence_nested) 
        kb = accuracy_score(y2_presence_nested, y3_presence_nested) 
        
        sh_results_bar_chart_a.append((behavior, ka))
        sh_results_bar_chart_b.append((behavior, kb))

sh_results_bar_chart_a_df = pd.DataFrame(sh_results_bar_chart_a, columns=['behavior', 'result'])
sh_results_bar_chart_b_df = pd.DataFrame(sh_results_bar_chart_b, columns=['behavior', 'result'])

final_shuffle_results_bar_df = pd.concat(
    [sh_results_bar_chart_a_df, sh_results_bar_chart_b_df],
    ignore_index=True
)

shuffle_means = final_shuffle_results_bar_df.groupby('behavior').mean()
print(shuffle_means)

shuffle_errors = (
    final_shuffle_results_bar_df
    .groupby('behavior')['result']
    .std()
    .reindex(['mouth or tongue movement', 'gape', 'lateral tongue movement'])
    .to_numpy()
)


shuffle_means = (
    final_shuffle_results_bar_df
    .groupby('behavior')['result']
    .mean()
    .reindex(['mouth or tongue movement', 'gape', 'lateral tongue movement'])
    .to_numpy()
)

# %%
# =============================================================================
# PLOT FIGURE
# =============================================================================

averages = [(a + b) / 2 for a, b in zip(results_bar_chart_a, results_bar_chart_b)]

# Set up figure
#x = np.arange(len(behaviors_i_care_about))
x = np.array([0, 0.6, 1.2])
width = 0.4  # width of the bar

fig, ax = plt.subplots()

# Plot average bars
bars = ax.bar(x, averages, width=width, color='0.8',
              linewidth=2,
              edgecolor='k')

'''
plt.errorbar(x, shuffle_means, shuffle_errors, 
             linestyle='None', marker='^', markersize=0,
             ecolor='black',      # error bar color
             elinewidth=2)
'''
for xpos, yval in zip(x, shuffle_means):
    ax.hlines(y=yval, xmin=xpos - width/2, xmax=xpos + width/2,
              colors='red', linestyles='dashed', linewidth=2)

# Plot dots for individual results
ax.scatter(x - width/4, results_bar_chart_a, 
           color='k', zorder=5, s=120,
           marker = '+')
ax.scatter(x + width/4, results_bar_chart_b, 
           color='k', zorder=5, s=120,
           marker = 'x')

# Labels and formatting
ax.set_xticks(x)
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(0, 1)
ax.set_xticklabels(['MTM', 'Gape', 'LTM'])
ax.set_ylabel('Accuracy Score')

plt.tight_layout()
plt.show()

