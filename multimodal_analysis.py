#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:13:11 2024

@author: natasha

This script will overwrie the original data file and remove any multimodal waveforms.
Save a backup of your raw data first.

"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import umap
import glob
from scipy import signal
from scipy.signal import find_peaks

# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/clustering_data/'
#file_path = os.path.join(dirname, 'mtm_clustering_df.pkl')
file_path = os.path.join(dirname, 'all_datasets_emg_pred.pkl')
df = pd.read_pickle(file_path)


unique_basenames = df['basename'].unique()
'''
df.rename_axis("obsolete_idx", inplace=True) # Default Index does not increment by 1. Renaming it.
df.reset_index(inplace=True) # Make new Index that increments by 1 per row
'''
# Make a dataframe of just mouth or tongue movement events
#mtm_bool = df.event_type.str.contains('mouth or tongue movement')
mtm_bool = df.event_type.str.contains('MTMs')
mtm_df_all = df.loc[mtm_bool]


# ==============================================================================
# %% Test which waveforms are bimodal (using non-normalized waveforms)
# Compute the cross-correlation of each wavweform with itself
# Examine the cross-correlation function (at only positive lag values) for peaks
# -> Multi-modal waveforms have 1 or more peaks 
# ==============================================================================
df['multimodal'] = 'n/a' # Create column called 'multimodal' with n/a as default

multi_segments = []
uni_segments = []

for index, row in mtm_df_all.iterrows():
#for index, row in mtm_df_all.head(10).iterrows(): # Use to plot a few uni/multimodal waveforms
    segment = row['segment_raw']
    corr = signal.correlate(segment, segment) # Correlate segment against itself
    lags = signal.correlation_lags(len(segment), len(segment))
    pos_corr = corr[ lags >=0] # Keeping only positive lags
    
    corr_peaks, _ = find_peaks(pos_corr) # Test if cross-correlation has 1+ peaks, indicating multimodality
    
    # Put categorized segments into respective lists
    # Add 'yes' or 'no' value to 'multimodal' column
    if len(corr_peaks) > 0:
        multi_segments.append(row['segment_raw'])
        df.loc[index, 'multimodal'] = 'yes' 
    else:
        uni_segments.append(row['segment_raw'])
        df.loc[index, 'multimodal'] = 'no'
    
    ## Use code below to plot each waveform and its cross-correlation. 
    ## WARNING: So many waveforms that it takes hours and then crashes 
    '''
    fig, (ax_seg, ax_corr) = plt.subplots(2,1, figsize=(7,10))
    ax_seg.plot(row['segment_raw'])
    if len(corr_peaks) > 0:
        ax_corr.plot(pos_corr, color='red')
    else:
        ax_corr.plot(pos_corr)
    ax_corr.plot(corr_peaks, pos_corr[corr_peaks], "x", markersize=20)
    '''
perc_multimodal = len(multi_segments)/(len(multi_segments) + len(uni_segments))*100
print(f"Percent of multimodal waveforms removed of MTMs: {perc_multimodal}")

# ==============================================================================
# %% Overerwrite data file with new dataframe without multimodal segments
# ==============================================================================
df_filter_multimodal = df[df['multimodal'] != 'yes'] # Remove multimodal from df
#df_filter_multimodal = df # Un-comment this line to KEEP multimodal waveforms in the dataframe.
df_filter_multimodal.to_pickle(file_path) # Overwrite and save dataset


# ==============================================================================
# %% Plots for visualizing uni/multi-modal waveforms
# ==============================================================================
# Create folder for saving plots
multimodal_dir = os.path.join(dirname, 'multimodal_analysis')
os.makedirs(multimodal_dir, exist_ok=True)
# Remove any files in new folder
all_files = glob.glob(os.path.join(multimodal_dir, '*'))
for file in all_files:
    os.remove(file)
    
# ==============================================================================
# Plot all uni/multimodal segments overlapped
# ==============================================================================
## NB: Using normalized segments just for plotting
fig, (ax_uni, ax_multi) = plt.subplots(2, 1, figsize=(8, 8))

# Plot unimodal segments
for segment in uni_segments:
    ax_uni.plot(segment, alpha=0.1, label="Unimodal Segments", color='cornflowerblue')
ax_uni.set_title("Unimodal Segments")
ax_uni.set_xlabel("Time")
ax_uni.set_ylabel("Amplitude")
fig.suptitle("Normalized Segments")

# Plot multimodal segments
for segment in multi_segments:
    ax_multi.plot(segment, alpha=0.1, label="Multimodal Segments", color='lightcoral')
ax_multi.set_title("Multimodal Segments")
ax_multi.set_xlabel("Time")
ax_multi.set_ylabel("Amplitude")

fig.tight_layout()

overlap_plt_path = os.path.join(multimodal_dir, 'overlap_waveforms_by_modality.png')
plt.savefig(overlap_plt_path)
plt.show()


# ==============================================================================
# Plot waveforms in UMAP space to see if multi/unimodal waveforms cluster
# ==============================================================================
# Initialize UMAP reduction
reducer = umap.UMAP()
new_mtm_bool = df.event_type.str.contains('MTMs')
new_mtm_df_all = df.loc[new_mtm_bool]
waveforms = new_mtm_df_all['segment_norm_interp'].tolist()
#scaled_waveforms = StandardScaler().fit_transform(waveforms)
embedding = reducer.fit_transform(waveforms) # UMAP embedding

# Plot UMAP
color_map = {'yes': 'lightcoral', 'no': 'cornflowerblue'}
colors = new_mtm_df_all['multimodal'].map(color_map)
plt.figure(figsize=(10, 8))
plt.scatter(
    embedding[:, 0], embedding[:, 1], 
    c=colors, s=5)
for value, color in color_map.items(): # Add legend for 'multimodal' categories ('yes' or 'no')
    plt.scatter([], [], c=color, label=value)  # Invisible points for legend
plt.legend(title='Multimodal', loc='upper right')

umap_plt_path = os.path.join(multimodal_dir, 'UMAP_waveforms_by_modality.png')
plt.savefig(umap_plt_path)
plt.show()
