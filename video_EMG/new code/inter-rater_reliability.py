#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:58:46 2024

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
import pickle


# =============================================================================
# Ensure info input below is correct
# =============================================================================
base_folder = '/media/natasha/drive2/Natasha_Data' # Parent folder containing all rats' data
rat_list = ['NB32', 'NB34', 'NB35']
test_day = [1, 2, 3] # Greatest number of test days 
scorer_initials = ['YW', 'NBT']  # Initials of all scorers, as found in [filename]_SCORER.csv
fps = 60

def check_alternation(this_session_df):
    # Check alternation
    behavior_column = this_session_df['Behavior type']
    alternates_correctly = all(
        (behavior_column.iloc[i] == 'START' and behavior_column.iloc[i + 1] == 'STOP') or
        (behavior_column.iloc[i] == 'STOP' and behavior_column.iloc[i + 1] == 'START')
        for i in range(len(behavior_column) - 1)
    )
    
    if alternates_correctly:
        print("The rows alternate CORRECTLY")
    else:
        print("The rows DO NOT alternate correctly.")




    
# =============================================================================
# Loading data
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
                print(f"Warning: Directory '{dirname}' does not exist.")
                break
        dir_to_append = os.path.join(dirname, "processed_scoring")
        dirs.append(dir_to_append)

scoring_dict = {}
trial_dict = {}
scorer_initials = []
observation_id = []
for dirname in dirs:
    npy_path = glob.glob(os.path.join(dirname, '*.npy'))[0]
    pkl_path = glob.glob(os.path.join(dirname, '*.pkl'))[0]
    
    session_scoring_dict = np.load(npy_path, allow_pickle=True).item()
    
    scorer_lookup = session_scoring_dict.keys()
    for scorer in scorer_lookup:
        if scorer not in scorer_initials:
            scorer_initials.append(scorer)
        scorer_df = session_scoring_dict[scorer]
        id_to_add = scorer_df['Observation id'].unique()
        
        ## TODO Move code to check len to preprocessing step
        if len(id_to_add) == 1 and id_to_add not in observation_id:
            observation_id.append(id_to_add[0])
            
        
            
        if scorer not in scoring_dict:
            scoring_dict[scorer] = scorer_df
        else:   
            scoring_dict[scorer] = pd.concat([scoring_dict[scorer], scorer_df], ignore_index = True)
    with open(pkl_path, 'rb') as file:
        data = pd.read_pickle(file)
        
    trial_dict[observation_id[-1]] = data

# =============================================================================
# 
# =============================================================================

# Set behaviors of interest for analyses, everything else will not be analyzed
behaviors_i_care_about = {'mouth or tongue movement': 1, 'gape': 2, 'lateral tongue movement': 3}

# Create a list of all unique behaviors in scoring_dict
max_sess_len = 0
unique_behaviors = []
for key in scoring_dict:
    behaviors = scoring_dict[key]['Behavior'].unique()
    unique_behaviors.extend(behaviors.tolist())
    max_time = scoring_dict[key]['Time'].max()
    if max_time > max_sess_len:
        max_sess_len = int(math.ceil(max_time))
# Convert the list to a set to remove duplicates and then convert it back to a list
unique_behaviors = list(set(unique_behaviors))

cohens_dict = {}
for scorer in scorer_initials:
    cohens_dict[scorer] = [0]

for scorer in scorer_initials:
    for o_id in [observation_id[0]]:
        behav_occur_list = np.zeros(max_sess_len*fps)
        
        mask = scoring_dict[scorer]['Observation id'] == o_id
        this_session_df = scoring_dict[scorer][mask]
        
        check_alternation(this_session_df)
        for idx, row in this_session_df.iterrows():
            if row['Behavior'] in behaviors_i_care_about and row['Behavior type']=='START':
                start_frame = int(row['Time'] * fps)
                stop_frame = int(this_session_df.loc[idx +1]['Time'] * fps)
                behav_occur_list[start_frame:stop_frame] = behaviors_i_care_about[row['Behavior']]
        cohens_dict[scorer].extend(behav_occur_list)
                

        

# Filter lists by removing elements if either list has a 0 at that index
filtered_list1 = []
filtered_list2 = []

for val1, val2 in zip(cohens_dict['YW'], cohens_dict['NBT']):
    if val1 != 0 and val2 != 0:  # Keep if neither is 0
        filtered_list1.append(val1)
        filtered_list2.append(val2)

k = cohen_kappa_score(filtered_list1, filtered_list2)
print("Cohen's Kappa score of all behaviors: ", round(k,3))

#TODO: DOBULE CHECK THAT THIS PART WORKS

# Loop through behaviors of interest and filter lists based on each value (1, 2, 3)
for behavior, behavior_code in behaviors_i_care_about.items():
    print(f"Testing for behavior: {behavior} (Code: {behavior_code})")

    # Filter lists based on the current behavior code (1, 2, or 3)
    filtered_list1 = []
    filtered_list2 = []

    for val1, val2 in zip(cohens_dict['YW'], cohens_dict['NBT']):
        # Only keep values where either list has the behavior of interest and is not 0
        if (val1 == behavior_code and val2 != 0) or (val2 == behavior_code and val1 != 0):
            filtered_list1.append(val1)
            filtered_list2.append(val2)
    k = cohen_kappa_score(filtered_list1, filtered_list2)
    print(f"Cohen's Kappa score for {behavior}: ", round(k,3))
    print()
#TODO: THIS PART IS NOT WORKING






