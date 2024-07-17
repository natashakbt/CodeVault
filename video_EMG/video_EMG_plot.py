#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:08:49 2024

@author: natasha
"""

### FIX THIS TO SPECIFY WHICH SCORER YOU WANT TO ANALYZE!!!!!



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

'''===================================================================='''
'''INPUT BASE FOLER, RAT NAME and TEST DAY NUM HERE'''

base_folder = '/media/natasha/drive2/Natasha_Data' #contains folders of all rats' data

rat_name = 'NB34'
test_day = 3
trial = 64 #trial num out of 120 with first trial =1
'''===================================================================='''


#TO LOAD THIS DICTIONARY IN THE FUTURE:
#import numpy as np
#loaded_dict = np.load('NB32_test1_scoring_dict.npz', allow_pickle=True)
#scoring_dict = dict(loaded_dict)

'''
# Load info file
info_file_path = glob.glob(os.path.join(dirname, '*', '*.info'))[0]
info_dict = json.load(open(info_file_path,'r'))
'''




'''
#==============================================================================
# Importing data and getting setup
#==============================================================================
'''
#Finding path to test day folder inside of base_folder

dirname = os.path.join(os.path.join(base_folder, rat_name), f"Test{test_day}")
if not os.path.exists(dirname):
    dirname = os.path.join(os.path.join(base_folder, rat_name), f"test{test_day}")
    if not os.path.exists(dirname):
        print(f"Error: Directory '{dirname}' does not exist.")

# Load info file
path_info_file = glob.glob(os.path.join(dirname, '*', '*.info'))[0]
info_dict = json.load(open(path_info_file,'r'))

# Load scoring dictionary within npz file
path_scoring_dict = glob.glob(os.path.join(dirname,'*scoring_dict.npz'))[0]
#scoring_dict = dict(np.load(path_scoring_dict, allow_pickle=True))
loaded_npz = np.load(path_scoring_dict, allow_pickle=True)
# Convert the loaded data back to a dictionary of DataFrames
scoring_dict = {key: pd.DataFrame(loaded_npz[key].item()) for key in loaded_npz}
vid_trial_start = scoring_dict['trial_start']
behavior_table = scoring_dict['behavior']

'''
path_emg_dict = glob.glob(os.path.join(dirname,'*emg_dict.npz'))[0]
emg_dict = dict(np.load(path_emg_dict, allow_pickle=True))
'''

## CODE TO IMPORT EMG DATA BELOW - IN PROGRESS
# Load the npz file
path_emg_dict = glob.glob(os.path.join(dirname, '*emg_dict.npz'))[0]
loaded_npz = np.load(path_emg_dict, allow_pickle=True)

# Convert the loaded data back to a dictionary with tuple keys
loaded_emg_dict = {tuple(key.split('_')): loaded_npz[key] for key in loaded_npz}
print(loaded_npz[key] for key in loaded_npz)
# Now you should be able to access the data within loaded_emg_dict
# Example to access specific EMG data (assuming you know the keys)
emg_ad_filt = loaded_emg_dict[('emgad', 'emg', 'filt.npy')]
emg_ad_env = loaded_emg_dict[('emgad', 'emg', 'env.npy')]
#emg_sty_filt = loaded_emg_dict[('STY', 'emg', 'filt.npy')]
#emg_sty_env = loaded_emg_dict[('STY', 'emg', 'env.npy')]

# Print statements to verify access
print('EMG AD Filtered Data:', emg_ad_filt)
print('EMG AD Envelope Data:', emg_ad_env)
#print('EMG STY Filtered Data:', emg_sty_filt)
#print('EMG STY Envelope Data:', emg_sty_env)


## End of EMG data importing



# First digit is which taste #. Second digit is taste presentation #
path_trial_info = glob.glob(os.path.join(dirname,'*trial_info.npy'))[0]
trial_info = np.load(path_trial_info)


# === Importing EMG blech_clust results ===
#finding path to EMG data
#h5_path = glob.glob(os.path.join(dirname, '*', '*.h5'))[0]

#if not h5_path:
#    print("Path to H5 file not found!")
#h5 = tables.open_file(h5_path, 'r')    

'''
#==============================================================================
# PLOTTING: NEED TO UPDATE
#==============================================================================
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


trial_pre_stim = 500 # in ms
trial_post_stim = 5000 # in ms

trial_len = trial_pre_stim + trial_post_stim

# === Preparing behavior data for plotting ===
# determine start and end time of the trial based on video time
#trial_start_time = vid_trial_table.iloc[trial-1, 4]
trial_start_time = None
for index, row in vid_trial_start.iterrows():
    if row[1] == str(trial):
        trial_start_time = row[4]
#trial_start_time = vid_trial_start.iloc[trial-1, 4]
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
desired_order = ['gape', 'mouth movements', 'tongue protrusion', 'lateral tongue movement', 'unknown mouth movement', 'other']

rearranged_dict = {key: behaviors_dict[key] for key in desired_order if key in behaviors_dict}

# Add any keys not in the desired order
for key in behaviors_dict:
    if key not in rearranged_dict:
        rearranged_dict[key] = behaviors_dict[key]

keys_to_keep = ['mouth or tongue movement', 'lateral tongue movement', 'gape']
new_dict = {key: rearranged_dict[key] for key in keys_to_keep if key in rearranged_dict}

rearranged_dict = new_dict

print(rearranged_dict)
# Making list of behavior names and time intervals
# Needed for ease of plotting
behavior_names = list(rearranged_dict.keys())
time_intervals = list(rearranged_dict.values())




# Update this list with the hex color codes you want to use
custom_colors = ['#3B75AF', '#AFC7E8', '#EF8636']  # Example hex color codes

plt.rcParams.update({'font.size': 14})  # You can adjust the size as needed

# Create a figure and axis
fig, (ax1, ax2) = plt.subplots(2, 1,
                               figsize=(8, 6),
                               dpi=600,
                               sharex=True,
                               gridspec_kw={'height_ratios': [1, 1]})

### Subplot 1 stuff (behavior)
# Set the y-axis ticks and labels
ax1.set_yticks(range(len(behavior_names)))
ax1.set_yticklabels(behavior_names)

# Define the linewidth of the bars
bar_linewidth = 25  # Adjust the value as needed

# Iterate over the behavior names and corresponding time intervals
for i, intervals in enumerate(time_intervals):
    if intervals is not None:
        for interval in intervals:  # Iterate over the time intervals and plot horizontal bars
            start_time, end_time = interval
            ax1.hlines(i, start_time, end_time,
                       linewidth=bar_linewidth, color=custom_colors[i % len(custom_colors)])

### Subplot 2 stuff (EMG) 
emg_trial = trial_info[trial-1][1]
emg_taste = trial_info[trial-1][0]  # suc=0; nacl=1; ca=2; qhcl=3

# Making xlims for EMG data based on input in section above
# EMG trial is 0>7000ms, 2000 pre-stim + 5000 post-stim
my_xlims = np.arange((-1 * trial_pre_stim + 2000), (trial_post_stim + 2000))

# Putting color on subplot 2 that matches with behavior intervals
for index, i in enumerate(time_intervals):
    if i is not None:
        for j in i:
            ax2.axvspan(j[0], j[1], color=custom_colors[index % len(custom_colors)], alpha=0.5)

# Plotting AD EMG
# emg_data_plot = [filt_ad, filt_sty] #original
emg_data_plot = [emg_ad_filt]
ax2.plot(emg_ad_env[emg_taste, emg_trial, my_xlims], color='0.4')
ax2.set_ylabel('EMG signal envelope (mV)')

### Putting dashed line at trial delivery, as set in section above.
ax1.axvline(trial_pre_stim, linestyle='--', color='gray')  
ax2.axvline(trial_pre_stim, linestyle='--', color='gray')

# Set title
# fig.suptitle(
#     f'{rat_name}, Test Day {test_day}\nTrial {trial}: delivery#{emg_trial} of {taste_names[emg_taste]}',
#     y=0.95)

# x_ticks = np.arange(-1.5, 5, 1)
# Set x-axis label, and limits

ax1.set_ylim(-0.5, len(behavior_names) - 0.1)  # Adjust the limits as needed
# ax3.get_shared_y_axes().join(ax2, ax3) #AD and STY share same y-axis range
# ax2.set_xticklabels(x_ticks)
# plt.delaxes(ax2)

tick_positions = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
tick_labels = ['-0.5', '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5']

ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels)
ax2.set_xlabel('Time (s)')
ax1.set_yticklabels(['MTM', 'LTM', 'Gape'])

# Show the plot
plt.show()







'''

'''
# === Actually creating figure ===
'''
plt.rcParams.update({'font.size': 14})  # You can adjust the size as needed

# Create a figure and axis
fig, (ax1, ax2) = plt.subplots(2, 1,
                                    figsize=(8, 6),
                                    dpi=600,
                                    sharex=True,
                                    gridspec_kw={'height_ratios': [1, 1]})

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
emg_trial = trial_info[trial-1][1]
emg_taste = trial_info[trial-1][0]  # suc=0 ; nacl=1 ; ca=2 ; qhcl=3

# making xlims for EMG data based on input in section above
# EMG trial is 0>7000ms, 2000 pre-stim + 5000 post-stim
my_xlims = np.arange((-1*trial_pre_stim+2000), (trial_post_stim+2000))

#Putting color on subplot 2+3 that match with behavior intervals
for index, i in enumerate(time_intervals):
    if i is not None:
        for j in i:
            ax2.axvspan(j[0], j[1], color=colors(index), alpha=0.5)

#plotting AD and STY EMG
# #4285F4 = nice blue/ #DB4437 = nice red
#emg_data_plot = [filt_ad, filt_sty] #original
emg_data_plot = [emg_ad_filt]
#ax2.plot(emg_ad_env[emg_taste, emg_trial, my_xlims], '0.4', color='#DB4437')
ax2.plot(emg_ad_env[emg_taste, emg_trial, my_xlims], '0.4')
ax2.set_ylabel('EMG signal envelope (mV)')

#ax3.plot(env_sty[emg_taste, emg_trial, my_xlims], '0.4', color='#4285F4')
#ax3.set_ylabel('Envelopped\nStyloglossus')


### Putting dashed line at trial delviery,
#as set in section above.
ax1.axvline(trial_pre_stim, linestyle='--', color='gray')  
ax2.axvline(trial_pre_stim, linestyle='--', color='gray')


# Set title
#fig.suptitle(
#    f'{rat_name}, Test Day {test_day}\nTrial {trial}: delivery#{emg_trial} of {taste_names[emg_taste]}',
#    y=0.95)

#x_ticks = np.arange(-1.5, 5, 1)
# Set x-axis label, and limits

ax1.set_ylim(-0.5, len(behavior_names) - 0.1)  # Adjust the limits as needed
#ax3.get_shared_y_axes().join(ax2, ax3) #AD and STY share same y-axis range
#ax2.set_xticklabels(x_ticks)
#plt.delaxes(ax2)

tick_positions = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
tick_labels = ['-0.5', '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5']

ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels)
ax2.set_xlabel('Time (s)')
ax1.set_yticklabels(['MTM', 'LTM', 'Gape'])
# Show the plot
plt.show()


'''

# '''
# NEW PLOT IN PROGRESS
# '''

# '''
# #figuring out like-trials 
# to create heat map of intensity for pal/non-pal
# '''

# #table_with_trial_info
# in_view_mask = vid_trial_start.iloc[:,2] == 'in view'
# good_trial_table = vid_trial_start[in_view_mask]

# suc_trials = [] 
# nacl_trials = []
# ca_trials = []
# qhcl_trials = []
# for index, row in good_trial_table.iterrows():
#     modifier_value = row['Modifier #1']
#     temp_tastant = trial_info[int(modifier_value)][0]
#     if temp_tastant == 0:
#         suc_trials.append(int(modifier_value))
#     elif temp_tastant == 1:
#         nacl_trials.append(int(modifier_value))
#     elif temp_tastant == 2:
#         ca_trials.append(int(modifier_value))
#     elif temp_tastant == 3:
#         qhcl_trials.append(int(modifier_value))

# pal_trials = suc_trials
# unpal_trials =  qhcl_trials

# desired_order = ['gape', 'mouth or tongue movement', 'lateral tongue movement', 'unknown mouth movement', 'other', 'to discuss']

# '''
# #Input condition you want to plot here
# '''
# plotting_condition = 'unpal' #'unapl' or 'pal' only!


# # Define the time bin parameters (adjust these as needed)
# time_bin_width_ms = 100  # Width of each time bin in milliseconds
# #num_time_bins = int((trial_pre_stim + trial_post_stim) / time_bin_width_ms)
# num_time_bins = int((trial_post_stim) / time_bin_width_ms)

# # Create a matrix to store behavior occurrences (initialize to zeros)
# behavior_matrix = np.zeros((len(behavior_names), num_time_bins))

# # Iterate through either unpalatable trials or palatable
# for trial in tqdm(unpal_trials if plotting_condition == 'unpal' else pal_trials):
#     trial_start_time = vid_trial_start.iloc[trial - 1, 4]
#     #trial_start_time -= (trial_pre_stim / 1000)

#     trial_behaviors = behavior_table[(behavior_table['Time'] >= trial_start_time) & (behavior_table['Time'] <= trial_start_time + (trial_post_stim / 1000))]

#     trial_behaviors = trial_behaviors[trial_behaviors['Behavior'] != 'out of view']
    
    
#     tmp_behaviors_dict = {i: None for i in unique_behaviors}
#     print(tmp_behaviors_dict)
#     for index,row in trial_behaviors.iterrows(): 
#         if tmp_behaviors_dict[row[0]] == None:
#             tmp_behaviors_dict[row[0]] = [(row[4]-trial_start_time)*1000]
#         else:
#             tmp_behaviors_dict[row[0]].extend([(row[4]-trial_start_time)*1000])
            
            
#     for key, value in tmp_behaviors_dict.items():
#         if isinstance(value, list):
#             tmp_behaviors_dict[key] = [(value[i], value[i + 1]) 
#                                    for i in range(0, len(value) - 1, 2)]
#     #re-arranging into desired order

#     rearranged_tmp_dict = {key: tmp_behaviors_dict[key] for key in desired_order if key in tmp_behaviors_dict}

#     # Add any keys not in the desired order
#     for key in tmp_behaviors_dict:
#         if key not in rearranged_tmp_dict:
#             rearranged_tmp_dict[key] = tmp_behaviors_dict[key]
            
#     #Figuring out how may bins each behavior belongs in        
#     for behavior, intervals in rearranged_tmp_dict.items():
#         if intervals is None:
#             continue  # Skip over intervals that are None
        
#         behavior_idx = desired_order.index(behavior)  # Get the index of the behavior
        
#         for interval in intervals:
#             start_time, end_time = interval
            
#             # Calculate the time bin indices for the interval
#             start_bin = int((start_time) / time_bin_width_ms)
#             end_bin = int((end_time) / time_bin_width_ms)
            
#             # Increment the corresponding bins in behavior_matrix
#             for bin_idx in range(start_bin, end_bin):
#                 behavior_matrix[behavior_idx, bin_idx] += 1 #### THIS PART IS MESSED UP
#     print(behavior_matrix[0])
# print('final!')
# print(behavior_matrix[0])
# # Create a heatmap of behavior occurrences
# plt.figure(figsize=(12, 8))
# cax = plt.imshow(behavior_matrix, cmap='viridis', aspect='auto', interpolation='none') #try magma or viridis
# plt.colorbar(cax, label='Behavior Occurrences')
# plt.xlabel('Time (ms)')
# plt.ylabel('Behaviors')

# plt.yticks(np.arange(len(desired_order)), desired_order)
# #plt.xticks(np.arange(0, num_time_bins, num_time_bins // 10), np.arange(-trial_pre_stim, trial_post_stim + 1, (trial_pre_stim + trial_post_stim) // 10))


# if plotting_condition == 'unpal':
#     title_suffix = f'Unpalatable tastants (total trials = {len(unpal_trials)})'
# else:
#     title_suffix = f'Palatable tastants (total trials = {len(pal_trials)})'

# plt.title(
#     f'{rat_name}, Test Day {test_day}\n{title_suffix}')

# plt.show()



# '''
# ANOTHER PLOT FOR VIDEO
# HAVEN'T TESTED WITH NEW CODE SPLIT"
# '''

# # Create a figure and axis
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
#                                     figsize=(8, 8),
#                                     sharex=True,
#                                     gridspec_kw={'height_ratios': [2, 1, 1]})

# ### Subplot 1 stuff (behavior)
# # Set the y-axis ticks and labels
# ax1.set_yticks(range(len(behavior_names)))
# ax1.set_yticklabels(behavior_names)

# # Define the linewidth of the bars
# bar_linewidth = 25  # Adjust the value as needed

# # Create a colormap with the number of behaviors
# colors = cm.get_cmap('tab10', len(behavior_names))

# # Iterate over the behavior names and corresponding time intervals
# for i, intervals in enumerate(time_intervals):
#     if intervals is not None:
#         # Iterate over the time intervals and plot horizontal bars
#         for interval in intervals:
#             start_time, end_time = interval
#             ax1.hlines(i, start_time, end_time,
#                        linewidth=bar_linewidth, color=colors(i))

# ### Subplot 2 + 3 stuff (EMG)
# emg_trial = table_with_trial_info[trial-1][1]
# emg_taste = table_with_trial_info[trial-1][0]  # suc=0 ; nacl=1 ; ca=2 ; qhcl=3

# # making xlims for EMG data based on input in section above
# # EMG trial is 0>7000ms, 2000 pre-stim + 5000 post-stim
# my_xlims = np.arange((-1*trial_pre_stim+2000), (trial_post_stim+2000))

# #Putting color on subplot 2+3 that match with behavior intervals
# for index, i in enumerate(time_intervals):
#     if i is not None:
#         for j in i:
#             ax2.axvspan(j[0], j[1], color=colors(index), alpha=0.5)
#             ax3.axvspan(j[0], j[1], color=colors(index), alpha=0.5)

# #plotting AD and STY EMG
# # #4285F4 = nice blue/ #DB4437 = nice red

# emg_data_plot = [filt_ad, filt_sty]
# ax2.plot(env_ad[emg_taste, emg_trial, my_xlims], '0.4', color='#DB4437')
# ax2.set_ylabel('Envoleopped\nAnterior Digastric')

# ax3.plot(env_sty[emg_taste, emg_trial, my_xlims], '0.4', color='#4285F4')
# ax3.set_ylabel('Envelopped\nStyloglossus')


# ### Putting dashed line at trial delviery,
# #as set in section above.
# ax1.axvline(trial_pre_stim, linestyle='--', color='gray')  # original = 500
# ax2.axvline(trial_pre_stim, linestyle='--', color='gray')
# ax3.axvline(trial_pre_stim, linestyle='--', color='gray')

# # Set title

# taste = info_dict['taste_params']['tastes'][emg_taste]
# fig.suptitle(
#     f'{rat_name}, Test Day {test_day}\nTrial {trial}: delivery#{emg_trial} of {taste}',
#     y=0.95)

# # Set x-axis label, and limits
# ax1.set_xlabel('Time (ms)')
# ax1.set_ylim(-0.5, len(behavior_names) - 0.1)  # Adjust the limits as needed
# #ax3.get_shared_y_axes().join(ax2, ax3) #AD and STY share same y-axis range
# # Show the plot

# plt.delaxes(ax2)
# plt.delaxes(ax3)

# plt.show()
