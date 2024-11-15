#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:22:22 2024

@author: natasha
"""

import numpy as np
import tables
import glob
import os
import pandas as pd
from tqdm import tqdm
import json



# =============================================================================
# Ensure info input below is correct
# =============================================================================
base_folder = '/media/natasha/drive2/Natasha_Data' # Parent folder containing all rats' data
columns_to_keep = ['Behavior', 'Modifier #1', 'Modifier #2', 'Behavior type', 'Time']
rat_list = ['NB32', 'NB34', 'NB35']
test_day = [1, 2, 3] # Greatest number of test days 
scorer_initials = ['YW', 'NBT']  # Initials of all scorers, as found in [filename]_SCORER.csv

# =============================================================================
# Necessary functions
# =============================================================================

def check_trial_dfs(temp_trial_dict):
    df_list = list(temp_trial_dict.values())
    
    # Assuming all dataframes are aligned, we'll compare based on the 'in view' condition
    matching_dfs = []
    
    # Process the DataFrames
    for i, df in enumerate(df_list):
        # Filter for 'in view' trial starts
        df_filtered = df[df['Modifier #2'] == 'in view']
        
        if i == 0:
            # For the first dataframe, initialize the base set of 'in view' trial starts
            matching_dfs.append(df_filtered)
        else:
            # Merge subsequent dataframes on 'Behavior' and 'Time' for 'in view' trials
            matching_dfs[0] = pd.merge(matching_dfs[0], df_filtered[['Behavior', 'Modifier #1', 'Time']], on=['Behavior', 'Modifier #1'], suffixes=('_prev', '_curr'), how='outer')
    
    # Ensure matching time values
    merged_df = matching_dfs[0]
    match_condition = True
    for i in range(1, len(df_list)):
        match_condition &= (merged_df['Time_prev'] == merged_df['Time_curr'])
    
    # Filter to keep only the matching rows
    matching_df = merged_df[match_condition]
    
    if len(merged_df[~match_condition]) > 0:
        print('Trial start mismatches found:\n', merged_df[~match_condition])

    return matching_df

def add_taste_names(matching_df, dirname):
    info_file = glob.glob(os.path.join(dirname, '**', '*.info'))[0]
    with open(info_file, 'r') as file:
        info_dict = json.load(file)
    tastes = info_dict.get('taste_params', {}).get('tastes', [])
    
    h5_path = glob.glob(os.path.join(dirname, '*', '*.h5'))[0]
    if not h5_path:
        print("Path to H5 file not found!")
    h5 = tables.open_file(h5_path, 'r') 

    #getting the names of the dig_ins
    big_dig_in = h5.get_node("/digital_in")
    dig_in_names = [i for i in big_dig_in._v_children]

    trial_time_index = [] #a list (of length = # of tastants) of each time stamp where solanoid was opened
    for name in tqdm(dig_in_names):
        dig_in = h5.get_node("/digital_in", name)[:]
        ones_indices = np.where(dig_in == 1)[0]
        filtered_indices = ones_indices[(dig_in[ones_indices - 1] == 0)]
        trial_time_index.append(filtered_indices)
        
    all_trial_indices = sorted(np.concatenate(trial_time_index)) #making taste delivery time stamps into one sorted list
    
    table_with_trial_info = [] #final shape: column 1 = taste index; column 2 = presentation # of that taste; rows = total trial number
    counters = [0] * len(dig_in_names) #counter for the number of times a tastant has been delivered
    for i in all_trial_indices:
        for j, valve_index in enumerate(trial_time_index):
            if i in valve_index:
                table_with_trial_info.append([j, counters[j]])
                counters[j] += 1
                break
        else:
            print("Mismatch between trial_time_index and all_trial_index")
  
    for idx, row in matching_df.iterrows():
        trial_num = int(row['Trial_num']) -1
        tst = tastes[table_with_trial_info[trial_num][0]]
        delivery_num = table_with_trial_info[trial_num][1]
        matching_df.loc[idx, 'Tastant'] = tst
        matching_df.loc[idx, 'DelivNum'] = delivery_num
        matching_df.loc[idx, 'TastantNum'] = table_with_trial_info[0][0]
        
    return(matching_df)


# =============================================================================
# Create scored_behaviors: Contains a dataframe from each scorer's BORIS CSV file
# Create scored_trials: Contains info about scored trials
# =============================================================================
dirs = []
for rat_name in rat_list:
    for day in test_day:
        #Finding path to test day folder inside of base_folder
        dirname = (os.path.join(os.path.join(base_folder, rat_name), f"Test{day}"))
        if not os.path.exists(dirname):
            dirname = os.path.join(os.path.join(base_folder, rat_name), f"test{day}")
            if not os.path.exists(dirname):
                print(f"Error: Directory '{dirname}' does not exist.")
                break
        dirs.append(dirname)

csv_paths = []
for dirname in tqdm(dirs):
    # Transforming csv files exported from BORIS into a dictionary used for analysis
    csv_path = glob.glob(os.path.join(dirname, '**', '*.csv'), recursive=True) #finding all csv files
    for path in csv_path.copy(): # Removing the electrode_layout.csv
        if 'electrode_layout' in path:
            csv_path.remove(path)
    if len(csv_path) <2 : 
        print('One or zero scoring csv files found!')
    else :
        print('\nScoring csv files:\n', csv_path)
    scoring_dict = {}
    temp_trial_dict = {}
    for csv in csv_path:
          
        boris_df = pd.read_csv(csv)
        columns_to_drop = [col for col in boris_df.columns if col not in columns_to_keep]
        boris_df.drop(columns=columns_to_drop, inplace=True)

        mask = boris_df.iloc[:, 0] == 'trial start'
        temp_trial_df = boris_df[mask]
        scoring_df = boris_df[~mask]
        
        filename_scorer = csv.split('_')[-1].split('.csv')[0]
        if filename_scorer not in scorer_initials:
            print('Caution:', filename_scorer, 'is not in the list of scorers')
            print('Skipping:', csv)
        else:
            scoring_dict[filename_scorer] = scoring_df
            temp_trial_dict[filename_scorer] = temp_trial_df
    
    
    matching_df = check_trial_dfs(temp_trial_dict)
    matching_df.drop(['Behavior', 'Behavior type', 'Time_curr'], axis = 1, inplace = True)
    matching_df.rename(columns={'Modifier #1': 'Trial_num', 'Time_prev': 'Time'}, inplace = True)
    matching_df[['Tastant', 'DelivNum', 'TastantNum']] = None
    
    final_trial_df = add_taste_names(matching_df, dirname)
    

    # TODO: CHECK THAT EVERY SCORING_DF IS PROPERLY CONFIGURED (ALT. START/STOP)
    
    
    new_dir = os.path.join(dirname, 'processed_scoring')
    os.makedirs(new_dir, exist_ok=True)
    
    # Process and save the scoring dictionary
    basename = os.path.basename(csv)
    new_dict_filename = basename.rsplit('_', 1)[0] + '_scored_behaviors.npz'
    np.savez(os.path.join(new_dir, new_dict_filename), scoring_dict)
    print('\nIn', new_dir, 'saving:\n', new_dict_filename)
    
    # Process and save the final trial DataFrame as a pickle file
    new_pkl_filename = basename.rsplit('_', 1)[0] + '_scored_trials.pkl'
    filepath = os.path.join(new_dir, new_pkl_filename)
    final_trial_df.to_pickle(filepath)
    print(' ', new_pkl_filename)


