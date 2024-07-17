#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:43:31 2023

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


"""
######################################
IMPORTING DATA AND GETTING SETUP
######################################
"""
#functions to search through base folder
def find_test_day_folder(test_subject, test_day):
    base_folder = "/media/natasha/drive2/Natasha_Data"  # Path to drive where sorted data is stored

    test_subject_folder = os.path.join(base_folder, test_subject)
    test_day_folder = os.path.join(test_subject_folder, f"Test{test_day}")
    
    if os.path.exists(test_day_folder):
        return test_day_folder
    else:
        return None


def search_for_file(root_folder, file_extension):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                return file_path  # Return the file path if found

    return None  # Return None if file is not found

"""
INPUT RAT NAME, TEST DAY, AND DIG_INs HERE
"""
rat_name = "NB27"
test_day = 2
dig_in_numbers = [12, 13, 14, 15] #for taste only
taste_names = ['sucrose', 'nacl', 'citric acid', 'qhcl'] #IN SEQUENCE with DIG-INs

#Finding path to test day folder inside of extradrive (i.e. drive2)
dirname = find_test_day_folder(rat_name, test_day)
if dirname:
    print(f"dirname is: {dirname}")
else:
    print("Data folder not found.")

NBT_scoring_dir = os.path.join(dirname, "other_scoring_results/NB27_test2.csv")
TRG_scoring_dir = os.path.join(dirname, "other_scoring_results/trgtest2.csv")
CM_scoring_dir = os.path.join(dirname, "other_scoring_results/NB27_test2_CM.csv")
YW_scoring_dir = os.path.join(dirname, "other_scoring_results/yixi_nb27_test2.csv")
# === Setting up csv data from scored video into tables ===
#Searching and reading video scoring results in csv format

NBT_scoring_data = pd.read_csv(NBT_scoring_dir)
TRG_scoring_data = pd.read_csv(TRG_scoring_dir)
CM_scoring_data = pd.read_csv(CM_scoring_dir)
YW_scoring_data = pd.read_csv(YW_scoring_dir)
#deleting superflous columns from csv files
columns_to_delete = ['Observation id', 'Observation date', 'Description', 'Observation duration', 'Observation type', 
                     'Source', 'Media duration (s)', 'FPS', 'date', 'test num', 'Subject', 
                     'Behavioral category', 'Media file name', 'Image index', 'Image file path', 'Comment']
for column in columns_to_delete:
    del NBT_scoring_data[column]
    del TRG_scoring_data[column]
    del CM_scoring_data[column]
    del YW_scoring_data[column]
    
 
#seperating scoring table into two tables.
#trial_table is only all trial starts (120 rows)
#behavior_table is all scored behavior of rat
mask = NBT_scoring_data.iloc[:, 0] == 'trial start'
vid_trial_table = NBT_scoring_data[mask]
NBT_scoring_data = NBT_scoring_data[~mask]

#another_mask = CM_scoring_data.iloc[:,0] == 'trial start'
#CM_scoring_data = CM_scoring_data[~mask]

del NBT_scoring_data['Modifier #2']
del NBT_scoring_data['Modifier #1']
del CM_scoring_data['Modifier #2']
del CM_scoring_data['Modifier #1']
del YW_scoring_data['Modifier #2']
del YW_scoring_data['Modifier #1']

"""
######################################
Creating figure with 3 suplots
top subplot is scored behavior
middle is AD EMG, bottom is STY EMG
Across a few seconds of a given trial
######################################
NECESSARY PLOTTING INPUT HERE:
Input the single trial (first trial = 1)
and time pre-stimulus and post-stimulus delivery
"""



trial = 74 #trial num out of 120 with first trial =1
trial_pre_stim = 500 #original: 500
trial_post_stim = 50000 # original: 5000

trial_len = trial_pre_stim + trial_post_stim

# === Preparing behavior data for plotting ===
# determine start and end time of the trial based on video time
#trial_start_time = vid_trial_table.iloc[trial-1, 3]
trial_start_time = vid_trial_table.iloc[trial-1, 1]
trial_end_time = trial_start_time + (trial_post_stim/1000) #determine end of trial in video time 
trial_start_time -= (trial_pre_stim/1000) #original: 0.5

#table of all behavior data within trial start and stop time
NBT_trial_behaviors = NBT_scoring_data[(NBT_scoring_data['Time'] >= trial_start_time) 
                                       & (NBT_scoring_data['Time'] <= trial_end_time)]
TRG_trial_behaviors = TRG_scoring_data[(TRG_scoring_data['Time'] >= trial_start_time)
                                       & (TRG_scoring_data['Time'] <= trial_end_time)] 
CM_trial_behaviors = CM_scoring_data[(CM_scoring_data['Time'] >= trial_start_time)
                                     & (CM_scoring_data['Time'] <= trial_end_time)]
YW_trial_behaviors = YW_scoring_data[(YW_scoring_data['Time'] >= trial_start_time)
                                 & (YW_scoring_data['Time'] <= trial_end_time)]

NBT_trial_behaviors = NBT_trial_behaviors[NBT_trial_behaviors['Behavior'] != 'out of view']


#Creating dictionary where key is all unique behaviors
unique_behaviors = list(set(NBT_scoring_data['Behavior']))
NBT_behaviors_dict = {i: None for i in unique_behaviors}
TRG_behaviors_dict = {i: None for i in unique_behaviors}
CM_behaviors_dict = {i: None for i in unique_behaviors}
YW_behaviors_dict = {i: None for i in unique_behaviors}

# loop through all behaviors in dict
# append time any time the behavior starts or stops
for index,row in NBT_trial_behaviors.iterrows(): 
    #converting video time to ephys time
    temp_time = [((row[2]-trial_start_time)/(trial_end_time - trial_start_time))*trial_len]
    if NBT_behaviors_dict[row[0]] == None:
        NBT_behaviors_dict[row[0]] = temp_time
    else:
        NBT_behaviors_dict[row[0]].extend(temp_time)

for index,row in TRG_trial_behaviors.iterrows(): 
    #converting video time to ephys time
    temp_time = [((row[2]-trial_start_time)/(trial_end_time - trial_start_time))*trial_len]
    if TRG_behaviors_dict[row[0]] == None:
        TRG_behaviors_dict[row[0]] = temp_time
    else:
        TRG_behaviors_dict[row[0]].extend(temp_time)

for index,row in CM_trial_behaviors.iterrows(): 
    #converting video time to ephys time
    temp_time = [((row[2]-trial_start_time)/(trial_end_time - trial_start_time))*trial_len]
    if CM_behaviors_dict[row[0]] == None:
        CM_behaviors_dict[row[0]] = temp_time
    else:
        CM_behaviors_dict[row[0]].extend(temp_time)

for index,row in YW_trial_behaviors.iterrows(): 
    #converting video time to ephys time
    temp_time = [((row[2]-trial_start_time)/(trial_end_time - trial_start_time))*trial_len]
    if YW_behaviors_dict[row[0]] == None:
        YW_behaviors_dict[row[0]] = temp_time
    else:
        YW_behaviors_dict[row[0]].extend(temp_time)

#converts the list within each index
# into a list of tuples containing pairs of consecutive values
# [(first_start_time, first_stop_time), (etc.)}]
for key, value in NBT_behaviors_dict.items():
    if isinstance(value, list):
        NBT_behaviors_dict[key] = [(value[i], value[i + 1]) 
                               for i in range(0, len(value) - 1, 2)]

for key, value in TRG_behaviors_dict.items():
    if isinstance(value, list):
        TRG_behaviors_dict[key] = [(value[i], value[i + 1]) 
                               for i in range(0, len(value) - 1, 2)]
        
for key, value in CM_behaviors_dict.items():
    if isinstance(value, list):
        CM_behaviors_dict[key] = [(value[i], value[i + 1]) 
                               for i in range(0, len(value) - 1, 2)]

for key, value in YW_behaviors_dict.items():
    if isinstance(value, list):
        YW_behaviors_dict[key] = [(value[i], value[i + 1]) 
                               for i in range(0, len(value) - 1, 2)]
    
#removing behavior "out of view"
#behaviors_dict.pop('out of view')

# Making list of behavior names and time intervals
# Needed for ease of plotting
NBT_behavior_names = list(NBT_behaviors_dict.keys())
NBT_time_intervals = list(NBT_behaviors_dict.values())

TRG_behavior_names = list(TRG_behaviors_dict.keys())
TRG_time_intervals = list(TRG_behaviors_dict.values())

CM_behavior_names = list(CM_behaviors_dict.keys())
CM_time_intervals = list(CM_behaviors_dict.values())

YW_behavior_names = list(YW_behaviors_dict.keys())
YW_time_intervals = list(YW_behaviors_dict.values())


'''
# === Actually creating figure ===
'''
# Create a figure and axis
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, 
                               figsize=(16,18), #FIX THIS!!!!!!!!!!!!!!! 
                               sharex=True,
                               sharey=True)

### Subplot 1 stuff (behavior)
# Set the y-axis ticks and labels
ax1.set_yticks(range(len(NBT_behavior_names)))
ax1.set_yticklabels(NBT_behavior_names)


# Define the linewidth of the bars
bar_linewidth = 25  # Adjust the value as needed

# Create a colormap with the number of behaviors
colors = cm.get_cmap('Accent', len(NBT_behavior_names))

# Iterate over the behavior names and corresponding time intervals
for i, intervals in enumerate(NBT_time_intervals):
    if intervals is not None:
        # Iterate over the time intervals and plot horizontal bars
        for interval in intervals:
            start_time, end_time = interval
            ax1.hlines(i, start_time, end_time, linewidth=bar_linewidth, color=colors(i))

for i, intervals in enumerate(TRG_time_intervals):
    if intervals is not None:
        # Iterate over the time intervals and plot horizontal bars
        for interval in intervals:
            start_time, end_time = interval
            ax2.hlines(i, start_time, end_time, linewidth=bar_linewidth, color=colors(i))

for i, intervals in enumerate(CM_time_intervals):
    if intervals is not None:
        # Iterate over the time intervals and plot horizontal bars
        for interval in intervals:
            start_time, end_time = interval
            ax3.hlines(i, start_time, end_time, linewidth=bar_linewidth, color=colors(i))

for i, intervals in enumerate(YW_time_intervals):
    if intervals is not None:
        # Iterate over the time intervals and plot horizontal bars
        for interval in intervals:
            start_time, end_time = interval
            ax4.hlines(i, start_time, end_time, linewidth=bar_linewidth, color=colors(i))

ax1.axvline(trial_pre_stim, linestyle='--', color='gray') #original = 500
ax2.axvline(trial_pre_stim, linestyle='--', color='gray') #original = 500
ax3.axvline(trial_pre_stim, linestyle='--', color='gray') #original = 500
ax4.axvline(trial_pre_stim, linestyle='--', color='gray') #original = 500
ax1.set_title("Natasha")
ax2.set_title("Thomas")
ax3.set_title("Christina")
ax4.set_title("Yixi")
fig.suptitle(f'{rat_name}, Test Day {test_day}: Trial {trial}', y =0.95)
plt.subplots_adjust(hspace=0.5)
plt.show()
