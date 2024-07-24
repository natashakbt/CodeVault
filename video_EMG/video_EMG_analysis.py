#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:57:39 2023

@author: natasha


This script creates npz and npy files necessary for future analyses and plots.
- [rat]_[test]_[scorer]_scoring_dict.npz
    ->  dictionary where first level of keys either 'trial_start' or 'behavior'
    -> 'trial start' then has keys 
- [testname]_emg_dict.npz  

Prerequisites:
- Base folder must contain subfolders for each rat.
- Each rat's folder should have subfolders named "Test#" or "test#".
- Each test folder must contain the following files:
    - .CSV file(s) exported from BORIS
    - [testname].h5
    - emg_env.npy and emg_filt.npy inside folder "emgad", "emgAD", "emgsty", and/or "emgSTY"

Structure:
1. 


"""
# TODO: make dictionaries into data frames?
# TODO: save into folders

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
base_folder = '/media/natasha/drive2/Natasha_Data' # Must contain folder with rat's name

rat_name = "NB34"
test_day = 3
scorers = ['YW', 'NBT']
'''# ===================================================================='''

# Find path to test day folder inside of base_folder
dirname = os.path.join(os.path.join(base_folder, rat_name), f"Test{test_day}")
if not os.path.exists(dirname):
    dirname = os.path.join(os.path.join(base_folder, rat_name), f"test{test_day}")
    if not os.path.exists(dirname):
        print(f"Error: Directory '{dirname}' does not exist.")


'''
#==============================================================================
#Creating scoring_dict: processed scored behavior data from BORIS csv files
#==============================================================================
'''
## Transform csv files exported from BORIS into a dictionary used for analysis
csv_path = glob.glob(os.path.join(dirname, '**', '*.csv'), recursive=True) # Find all csv files
for path in csv_path.copy(): # Remove the electrode_layout.csv file from paths
    if 'electrode_layout' in path:
        csv_path.remove(path)
 
if len(csv_path) > 2:
    # Check if any path contains 'final'
    final_path = None
    for path in csv_path:
        if 'final' in path:
            final_path = path
            break
    
    if final_path is not None:
        # If a path containing 'final' is found, keep only that path
        csv_path = [final_path]
    else:
        # If no path contains 'final', keep the first path
        csv_path = csv_path[0]
# TODO: FIX THE ABOVE TO ANALYZE ALL SCORER'S RESULTS   
if not csv_path : 
    print('No scoring csv files found!')
else :
    print('scoring csv files:\n', csv_path)

for csv in csv_path:
    print(csv)
    
    scoring_data = pd.read_csv(csv)
    
    #only keeping essential columns
    columns_to_keep = ['Behavior', 'Modifier #1', 'Modifier #2', 'Behavior type', 'Time']
    
    columns_to_drop = [col for col in scoring_data.columns if col not in columns_to_keep]
    scoring_data.drop(columns=columns_to_drop, inplace=True)
    
    mask = scoring_data.iloc[:, 0] == 'trial start'
    trial_start = scoring_data[mask] # Select rows indicating 'trial start'
    behavior = scoring_data[~mask] # All other rows will be scored behaviors
        # Store both DataFrames in the dictionary
    scoring_dict = {
        'trial_start': trial_start.to_dict(),
        'behavior': behavior.to_dict()
    }
    #scoring_dict['trial_start'] = trial_start
    #scoring_dict['behavior'] = behavior
    
    
    if scorers[0] in csv:
       new_filename = f'{rat_name}_test{test_day}_YW_scoring_dict.npz'
    elif scorers[1] in csv:
        new_filename = f'{rat_name}_test{test_day}_NBT_scoring_dict.npz'
    ## Saving scoring table as an .npy file to dirname
    os.chdir(dirname)
    np.savez(new_filename, **scoring_dict)
    print('\nSaving', new_filename, 'to', dirname)


'''
for table in scoring_data:
    columns_to_drop = [col for col in table.columns if col not in columns_to_keep]
    table.drop(columns=columns_to_drop, inplace=True)


#data from each csv file appended into scoring_data
scoring_data = []
for csv in csv_path:
    scoring_data.append(pd.read_csv(csv))



scoring_dict = {} #dictionary containing two tables. One of trial starts only. Other of behaviors only.
for csv, table in zip(csv_path, scoring_data):
    mask = table.iloc[:, 0] == 'trial start'
    trial_start = table[mask] # Select rows indicating 'trial start'
    behavior = table[~mask] # All other rows will be scored behaviors
    # Store both DataFrames in the dictionary
    scoring_dict[csv] = {
        'trial_start': trial_start,
        'behavior': behavior
    }
'''



'''
#============================================================================== 
#Creating trial_info: information about presentation identity and number for each trial
#==============================================================================
'''
h5_path = glob.glob(os.path.join(dirname, '*', '*.h5'))[0]

if not h5_path:
    print("Path to H5 file not found!")
h5 = tables.open_file(h5_path, 'r') 

#getting the names of the dig_ins
big_dig_in = h5.get_node("/digital_in")
dig_in_names = [i for i in big_dig_in._v_children]
print(dig_in_names)

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
        print("Match not found between trial_time_index and all_trial_index")

os.chdir(dirname)
new_filename = f'{rat_name}_test{test_day}_trial_info.npy'
np.save(new_filename, table_with_trial_info)
print('\nSaving', new_filename, 'to', dirname)

'''
#==============================================================================
# Creating emg_dict: processed results from EMG analysis within blech_clust 
#==============================================================================
'''

'''
# Old code. Works better below I think
emg_output_path = None
for root, dirs, files in os.walk(dirname): #finding path of 'emg_output' in base folder
    if 'emg_output' in dirs:
        emg_output_path = os.path.join(root, 'emg_output')
        break
if not emg_output_path:
    print("emg_output folder not found within", dirname)

#finding all npy files in emg_ouput folderSWS
long_emg_path = glob.glob(os.path.join(emg_output_path, '*', '*.npy'), recursive=True) #has extra npy files you don't want!

# removing all duplicate 'ad' and 'sty' (styloglossus) emg analysis files from the list 
emg_path = [] 
seen_ad_filt = False; seen_ad_env = False; seen_sty_filt = False; seen_sty_env = False;
for path in long_emg_path.copy():
    if ('AD' in path or 'ad' in path or 'Ad' in path) and 'filt' in path and not seen_ad_filt:
        emg_path.append(path)
        seen_ad_filt = True
    elif ('AD' in path or 'ad' in path or 'Ad' in path) and 'env' in path and not seen_ad_env:
        emg_path.append(path)
        seen_ad_env = True
    elif ('STY' in path or 'sty' in path or 'Sty' in path) and 'env' in path and not seen_sty_filt:
        emg_path.append(path)
        seen_sty_filt = True
    elif ('STY' in path or 'sty' in path or 'Sty' in path) and 'filt' in path and not seen_sty_env:
        emg_path.append(path)
        seen_sty_env = True
        

if not emg_path : #print statement of results
    print('No processed EMG npy files found!')
elif len(emg_path) > 4:
    print('Many EMG data files found. Double check that they should all be processed:\n', emg_path)
else :
    print('\nProcessing EMG data:', emg_path)

emg_data_key = [(path.split('/')[-2], path.split('/')[-1]) for path in emg_path] #keys for emg_dict

emg_dict = {} #final shape: emg_dict[key][taste][trial]
for key, emg in zip(emg_data_key, emg_path):
    emg_dict[key] = np.load(emg)

os.chdir(dirname)
new_filename = f'{rat_name}_test{test_day}_emg_dict.npz'
np.savez(new_filename, **emg_dict)
print('\nSaving', new_filename, 'to', dirname)

'''


# Assuming emg_dict is already created with keys and values as described
emg_dict = {}  # Populate this dictionary as per your logic

# Example saving logic
emg_output_path = None
for root, dirs, files in os.walk(dirname):  # finding path of 'emg_output' in base folder
    if 'emg_output' in dirs:
        emg_output_path = os.path.join(root, 'emg_output')
        break
if not emg_output_path:
    print("emg_output folder not found within", dirname)

# Finding all npy files in emg_output folder
long_emg_path = glob.glob(os.path.join(emg_output_path, '*', '*.npy'), recursive=True)  # has extra npy files you don't want!

# Removing all duplicate 'ad' and 'sty' (styloglossus) emg analysis files from the list
emg_path = []
seen_ad_filt = False; seen_ad_env = False; seen_sty_filt = False; seen_sty_env = False;
for path in long_emg_path.copy():
    if ('AD' in path or 'ad' in path or 'Ad' in path) and 'filt' in path and not seen_ad_filt:
        emg_path.append(path)
        seen_ad_filt = True
    elif ('AD' in path or 'ad' in path or 'Ad' in path) and 'env' in path and not seen_ad_env:
        emg_path.append(path)
        seen_ad_env = True
    elif ('STY' in path or 'sty' in path or 'Sty' in path) and 'env' in path and not seen_sty_filt:
        emg_path.append(path)
        seen_sty_filt = True
    elif ('STY' in path or 'sty' in path or 'Sty' in path) and 'filt' in path and not seen_sty_env:
        emg_path.append(path)
        seen_sty_env = True

if not emg_path:  # print statement of results
    print('No processed EMG npy files found!')
elif len(emg_path) > 4:
    print('Many EMG data files found. Double check that they should all be processed:\n', emg_path)
else:
    print('\nProcessing EMG data:', emg_path)

emg_data_key = [(path.split('/')[-2], path.split('/')[-1]) for path in emg_path]  # keys for emg_dict

emg_dict = {}  # final shape: emg_dict[key][taste][trial]
for key, emg in zip(emg_data_key, emg_path):
    emg_dict[key] = np.load(emg)

# Convert tuple keys to strings
string_keyed_emg_dict = {"_".join(key): value for key, value in emg_dict.items()}

# Save the emg_dict
new_filename = f'{rat_name}_test{test_day}_emg_dict.npz'
np.savez(new_filename, **string_keyed_emg_dict)
print('\nSaving', new_filename, 'to', dirname)


'''
###############################################################################
# CODE SPLIT HERE
###############################################################################
'''
'''
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

trial = 6 #trial num out of 120 with first trial =1
trial_pre_stim = 500 # in ms
trial_post_stim = 5000 # in ms

trial_len = trial_pre_stim + trial_post_stim

# === Preparing behavior data for plotting ===
# determine start and end time of the trial based on video time
trial_start_time = vid_trial_table.iloc[trial-1, 4]
trial_end_time = trial_start_time + (trial_post_stim/1000) #determine end of trial in video time 
trial_start_time -= (trial_pre_stim/1000) #original: 0.5

#table of all behavior data within trial start and stop time
trial_behaviors = behavior_table[(behavior_table['Time'] >= trial_start_time) 
                                 & (behavior_table['Time'] <= trial_end_time)]

trial_behaviors = trial_behaviors[trial_behaviors['Behavior'] != 'out of view']


#Creating dictionary where key is all unique behaviors
unique_behaviors = list(set(behavior_table['Behavior']))
behaviors_dict = {i: None for i in unique_behaviors}

# loop through all behaviors in dict
# append time any time the behavior starts or stops
for index,row in trial_behaviors.iterrows(): 
    #converting video time to ephys time
    temp_time = [((row[4]-trial_start_time)/(trial_end_time - trial_start_time))*trial_len]
    if behaviors_dict[row[0]] == None:
        behaviors_dict[row[0]] = temp_time
    else:
        behaviors_dict[row[0]].extend(temp_time)

#converts the list within each index
# into a list of tuples containing pairs of consecutive values
# [(first_start_time, first_stop_time), (etc.)}]
for key, value in behaviors_dict.items():
    if isinstance(value, list):
        behaviors_dict[key] = [(value[i], value[i + 1]) 
                               for i in range(0, len(value) - 1, 2)]

#re-arranging into desired order
desired_order = ['gape', 'mouth movements', 'tongue protrusion', 'lateral tongue protrusion', 'unknown mouth movement']

rearranged_dict = {key: behaviors_dict[key] for key in desired_order if key in behaviors_dict}

# Add any keys not in the desired order
for key in behaviors_dict:
    if key not in rearranged_dict:
        rearranged_dict[key] = behaviors_dict[key]

print(rearranged_dict)
# Making list of behavior names and time intervals
# Needed for ease of plotting
behavior_names = list(rearranged_dict.keys())
time_intervals = list(rearranged_dict.values())


'''
# === Actually creating figure ===
'''

# Create a figure and axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
                                    figsize=(8, 8),
                                    dpi=600,
                                    sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1, 1]})

### Subplot 1 stuff (behavior)
# Set the y-axis ticks and labels
ax1.set_yticks(range(len(behavior_names)))
ax1.set_yticklabels(behavior_names)

# Define the linewidth of the bars
bar_linewidth = 25  # Adjust the value as needed

# Create a colormap with the number of behaviors
colors = cm.get_cmap('tab10', len(behavior_names))

# Iterate over the behavior names and corresponding time intervals
for i, intervals in enumerate(time_intervals):
    if intervals is not None:
        for interval in intervals: # Iterate over the time intervals and plot horizontal bars
            start_time, end_time = interval
            ax1.hlines(i, start_time, end_time,
                       linewidth=bar_linewidth, color=colors(i))

### Subplot 2 + 3 stuff (EMG)
emg_trial = table_with_trial_info[trial-1][1]
emg_taste = table_with_trial_info[trial-1][0]  # suc=0 ; nacl=1 ; ca=2 ; qhcl=3

# making xlims for EMG data based on input in section above
# EMG trial is 0>7000ms, 2000 pre-stim + 5000 post-stim
my_xlims = np.arange((-1*trial_pre_stim+2000), (trial_post_stim+2000))

#Putting color on subplot 2+3 that match with behavior intervals
for index, i in enumerate(time_intervals):
    if i is not None:
        for j in i:
            ax2.axvspan(j[0], j[1], color=colors(index), alpha=0.5)
            ax3.axvspan(j[0], j[1], color=colors(index), alpha=0.5)

#plotting AD and STY EMG
# #4285F4 = nice blue/ #DB4437 = nice red

emg_data_plot = [filt_ad, filt_sty]
ax2.plot(env_ad[emg_taste, emg_trial, my_xlims], '0.4', color='#DB4437')
ax2.set_ylabel('Envoleopped\nAnterior Digastric')

ax3.plot(env_sty[emg_taste, emg_trial, my_xlims], '0.4', color='#4285F4')
ax3.set_ylabel('Envelopped\nStyloglossus')


### Putting dashed line at trial delviery,
#as set in section above.
ax1.axvline(trial_pre_stim, linestyle='--', color='gray')  # original = 500
ax2.axvline(trial_pre_stim, linestyle='--', color='gray')
ax3.axvline(trial_pre_stim, linestyle='--', color='gray')

# Set title
#fig.suptitle(
#    f'{rat_name}, Test Day {test_day}\nTrial {trial}: delivery#{emg_trial} of {taste_names[emg_taste]}',
#    y=0.95)

# Set x-axis label, and limits
ax3.set_xlabel('Time (ms)')
ax1.set_ylim(-0.5, len(behavior_names) - 0.1)  # Adjust the limits as needed
#ax3.get_shared_y_axes().join(ax2, ax3) #AD and STY share same y-axis range

plt.delaxes(ax2)
plt.delaxes(ax3)


# Show the plot
plt.show()




'''
#NEW PLOT IN PROGRESS
'''

'''
#figuring out like-trials 
#to create heat map of intensity for pal/non-pal
'''

#table_with_trial_info
in_view_mask = vid_trial_table.iloc[:,2] == 'in view'
good_trial_table = vid_trial_table[in_view_mask]

suc_trials = [] 
nacl_trials = []
ca_trials = []
qhcl_trials = []
for index, row in good_trial_table.iterrows():
    modifier_value = row['Modifier #1']
    temp_tastant = table_with_trial_info[int(modifier_value)][0]
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

desired_order = ['gape', 'mouth movements', 'tongue protrusion', 'lateral tongue protrusion', 'unknown mouth movement']

'''#Input condition you want to plot here
'''
plotting_condition = 'unpal' #'unapl' or 'pal' only!
''''''''

# Define the time bin parameters (adjust these as needed)
time_bin_width_ms = 100  # Width of each time bin in milliseconds
#num_time_bins = int((trial_pre_stim + trial_post_stim) / time_bin_width_ms)
num_time_bins = int((trial_post_stim) / time_bin_width_ms)

# Create a matrix to store behavior occurrences (initialize to zeros)
behavior_matrix = np.zeros((len(behavior_names), num_time_bins))

# Iterate through either unpalatable trials or palatable
for trial in tqdm(unpal_trials if plotting_condition == 'unpal' else pal_trials):
    trial_start_time = vid_trial_table.iloc[trial - 1, 4]
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
                behavior_matrix[behavior_idx, bin_idx] += 1
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
#ANOTHER PLOT FOR VIDEO
'''

# Create a figure and axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
                                    figsize=(8, 8),
                                    sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1, 1]})

### Subplot 1 stuff (behavior)
# Set the y-axis ticks and labels
ax1.set_yticks(range(len(behavior_names)))
ax1.set_yticklabels(behavior_names)

# Define the linewidth of the bars
bar_linewidth = 25  # Adjust the value as needed

# Create a colormap with the number of behaviors
colors = cm.get_cmap('tab10', len(behavior_names))

# Iterate over the behavior names and corresponding time intervals
for i, intervals in enumerate(time_intervals):
    if intervals is not None:
        # Iterate over the time intervals and plot horizontal bars
        for interval in intervals:
            start_time, end_time = interval
            ax1.hlines(i, start_time, end_time,
                       linewidth=bar_linewidth, color=colors(i))

### Subplot 2 + 3 stuff (EMG)
emg_trial = table_with_trial_info[trial-1][1]
emg_taste = table_with_trial_info[trial-1][0]  # suc=0 ; nacl=1 ; ca=2 ; qhcl=3

# making xlims for EMG data based on input in section above
# EMG trial is 0>7000ms, 2000 pre-stim + 5000 post-stim
my_xlims = np.arange((-1*trial_pre_stim+2000), (trial_post_stim+2000))

#Putting color on subplot 2+3 that match with behavior intervals
for index, i in enumerate(time_intervals):
    if i is not None:
        for j in i:
            ax2.axvspan(j[0], j[1], color=colors(index), alpha=0.5)
            ax3.axvspan(j[0], j[1], color=colors(index), alpha=0.5)

#plotting AD and STY EMG
# #4285F4 = nice blue/ #DB4437 = nice red

emg_data_plot = [filt_ad, filt_sty]
ax2.plot(env_ad[emg_taste, emg_trial, my_xlims], '0.4', color='#DB4437')
ax2.set_ylabel('Envoleopped\nAnterior Digastric')

ax3.plot(env_sty[emg_taste, emg_trial, my_xlims], '0.4', color='#4285F4')
ax3.set_ylabel('Envelopped\nStyloglossus')


### Putting dashed line at trial delviery,
#as set in section above.
ax1.axvline(trial_pre_stim, linestyle='--', color='gray')  # original = 500
ax2.axvline(trial_pre_stim, linestyle='--', color='gray')
ax3.axvline(trial_pre_stim, linestyle='--', color='gray')

# Set title

taste = info_dict['taste_params']['tastes'][emg_taste]
fig.suptitle(
    f'{rat_name}, Test Day {test_day}\nTrial {trial}: delivery#{emg_trial} of {taste}',
    y=0.95)

# Set x-axis label, and limits
ax1.set_xlabel('Time (ms)')
ax1.set_ylim(-0.5, len(behavior_names) - 0.1)  # Adjust the limits as needed
#ax3.get_shared_y_axes().join(ax2, ax3) #AD and STY share same y-axis range
# Show the plot

plt.delaxes(ax2)
plt.delaxes(ax3)

plt.show()
'''