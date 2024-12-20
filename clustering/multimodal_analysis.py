#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:13:11 2024

@author: natasha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import os
import umap
import glob
import diptest
from scipy import stats, signal
from scipy.signal import find_peaks

# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/clustering_data/'
#file_path = os.path.join(dirname, 'mtm_clustering_df.pkl') # only labeled stuff?
file_path = os.path.join(dirname, 'all_datasets_emg_pred.pkl') #everything with predictions?
df = pd.read_pickle(file_path)
df = df.rename(columns={'pred_event_type': 'event_type'})

unique_basenames = df['basename'].unique()
basename_to_num = {name: idx for idx, name in enumerate(unique_basenames)}
df['session_ind'] = df['basename'].map(basename_to_num)
df.rename_axis("obsolete_idx", inplace=True) # Default Index does not increment by 1. Renaming it.
df.reset_index(inplace=True) # Make new Index that increments by 1 per row

# Make a dataframe of just mouth or tongue movement events
#mtm_bool = df.event_type.str.contains('mouth or tongue movement')
mtm_bool = df.event_type.str.contains('MTMs')
mtm_df_all = df.loc[mtm_bool]


# ==============================================================================
# Test which segments are bimodal, add to array
# ==============================================================================
df['multimodal'] = 'n/a'

multi_segments = []
uni_segments = []

for index, row in mtm_df_all.iterrows():
#for idx in mtm_df_all.index: #ChatGPT not working
    #row = mtm_df_all.loc[idx] #ChatGPT not working
    segment = row['segment_raw']
    corr = signal.correlate(segment, segment) # Correlate segment against itself
    lags = signal.correlation_lags(len(segment), len(segment))
    pos_corr = corr[ lags >=0] # Keeping only positive lags
    
    corr_peaks, _ = find_peaks(pos_corr) # Test if cross-correlation has 1+ peaks, indicating multimodality
    
    # Put categorized segments into respective lists
    if len(corr_peaks) > 0:
        multi_segments.append(row['segment_norm_interp'])
        df.loc[index, 'multimodal'] = 'yes' 
    else:
        uni_segments.append(row['segment_norm_interp'])
        df.loc[index, 'multimodal'] = 'no'
    
    #fig, (ax_seg, ax_corr) = plt.subplots(2,1, figsize=(7,10))
    #ax_seg.plot(row['segment_raw'])
    #if len(corr_peaks) > 0:
    #    ax_corr.plot(pos_corr, color='red')
    #else:
    #    ax_corr.plot(pos_corr)
    #ax_corr.plot(corr_peaks, pos_corr[corr_peaks], "x", markersize=20)

# Plot all uni/multimodal segments overlapped
fig, (ax_uni, ax_multi) = plt.subplots(2, 1, figsize=(8, 8))

# Plot multimodal segments
for segment in multi_segments:
    ax_multi.plot(segment, alpha=0.1, label="Multimodal Segments", color='cornflowerblue')

ax_multi.set_title("Multimodal Segments")
ax_multi.set_xlabel("Time")
ax_multi.set_ylabel("Amplitude")

# Plot unimodal segments
for segment in uni_segments:
    ax_uni.plot(segment, alpha=0.1, label="Unimodal Segments", color='cornflowerblue')

ax_uni.set_title("Unimodal Segments")
ax_uni.set_xlabel("Time")
ax_uni.set_ylabel("Amplitude")
fig.suptitle("Normalized Segments")

fig.tight_layout()
plt.show()  




# UMAP dimmentionality reduction and feature scaling
reducer = umap.UMAP()

new_mtm_bool = df.event_type.str.contains('MTMs')
new_mtm_df_all = df.loc[new_mtm_bool]

waveforms = new_mtm_df_all['segment_norm_interp'].tolist()
#scaled_waveforms = StandardScaler().fit_transform(waveforms)
embedding = reducer.fit_transform(waveforms) # UMAP embedding

color_map = {'yes': 'blue', 'no': 'orange'}  # Define colors for 'yes' and 'no'
colors = new_mtm_df_all['multimodal'].map(color_map)
# Create the UMAP scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(
    embedding[:, 0], embedding[:, 1], 
    c=colors, s=5
)

# Add legend for 'multimodal' categories
for value, color in color_map.items():
    plt.scatter([], [], c=color, label=value)  # Invisible points for legend

plt.legend(title='Multimodal', loc='upper right')
plt.show()



    
    