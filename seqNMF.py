#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 11:03:25 2025

@author: natasha
"""


from seqnmf import seqnmf, plot, example_data

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from mtm_analysis_config import dirname
from scipy import stats

# ==============================================================================
# Load data and get setup
# ==============================================================================
file_path = os.path.join(dirname, 'clustering_df_update.pkl') # all events from classifier predictions
df = pd.read_pickle(file_path)


fs = 30000  # Hz
trial_window = (-2000, 5000)  # ms
samples_per_trial = int((trial_window[1] - trial_window[0]) / 1000 * fs)


# %%
results = {}

for session_ind in df['session_ind'].unique():
    session_df = df[(df['session_ind'] == session_ind) & (df['cluster_num'] != -2)]
    
    cluster_nums = session_df['cluster_num'].unique()
    trial_nums = session_df['trial_num'].unique()
    
    # build global time axis across all trials
    total_samples = samples_per_trial * len(trial_nums)
    time = np.arange(total_samples) / fs  # seconds

    # initialize binary matrix
    mat = np.zeros((len(cluster_nums), total_samples), dtype=int)

    for i, trial in enumerate(trial_nums):
        trial_df = session_df[session_df['trial_num'] == trial]
        offset = i * samples_per_trial
        
        for _, row in trial_df.iterrows():
            start_ms, end_ms = row['segment_bounds']
            start_idx = int((start_ms - trial_window[0]) / 1000 * fs)
            end_idx = int((end_ms - trial_window[0]) / 1000 * fs)
            mat[np.where(cluster_nums == row['cluster_num'])[0][0],
                offset + start_idx : offset + end_idx] = 1

    results[session_ind] = pd.DataFrame(mat, index=cluster_nums, columns=time)


# %%
#[W, H, cost, loadings, power] = seqnmf(example_data, K=20, L=100, Lambda=0.001, plot_it=True)
[W, H, cost, loadings, power] = seqnmf(mat, K=20, L=9000, Lambda=0.001, plot_it=True)

plot(W, H).show()



k = 0  # sequence index you want to plot
seq = W[:, k, :]  # shape: (4, 100)

plt.figure(figsize=(10, 4))
for i in range(seq.shape[0]):
    plt.plot(seq[i, :] + i*2, label=f'Feature {i+1}')  
    # '+ i*2' offsets each feature for clarity; remove if you want them on top of each other

plt.xlabel('Timepoints')
plt.ylabel('Amplitude (offset for clarity)')
plt.title(f'SeqNMF pattern {k}')
plt.legend()
plt.show()



plt.imshow(seq, aspect='auto', cmap='viridis')
plt.xlabel('Timepoints')
plt.ylabel('Features')
plt.title(f'SeqNMF pattern {k}')
plt.colorbar(label='Amplitude')
plt.show()


threshold = 0.01  # adjust if your data scale is different

active_patterns = []
for k in range(H.shape[0]):
    if np.any(H[k, :] > threshold):
        active_patterns.append(k)

print(f"Number of active patterns: {len(active_patterns)}")
print(f"Active pattern indices: {active_patterns}")


