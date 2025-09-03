#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 22:05:04 2025

@author: natasha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy import stats
import seaborn as sns
from scipy.stats import spearmanr

file_path = '/home/natasha/Desktop/clustering_data/non_fixed_clust_data.pkl'
 # all events from classifier predictions
non_fixed_clust_data = pd.read_pickle(file_path)
# Load the saved data

optimal_cluster_list = non_fixed_clust_data['optimal_cluster_list']
session_size_list = non_fixed_clust_data['session_size_list']

dirname = '/home/natasha/Desktop/clustering_data/'
pca_dir = os.path.join(dirname, 'extra_plots')
os.makedirs(pca_dir, exist_ok=True)
# Remove any png files in plots folder
png_files = glob.glob(os.path.join(pca_dir, '*.png'))
for file in png_files:
    os.remove(file)
    
iterations = 50
rounded_numbers = [round(num) for num in optimal_cluster_list]

# %% Scatter plot of optimal cluster size vs sessions size (i.e. number of MTMs)
# ==============================================================================
# Scatter plot of optimal cluster size vs sessions size (i.e. number of MTMs)
# ==============================================================================
jitter_strength = 0.05

# Add jitter to the session size and optimal clusters
session_size_jittered = np.array(session_size_list) + np.random.normal(0, jitter_strength, len(session_size_list))
optimal_cluster_jittered = np.array(optimal_cluster_list) + np.random.normal(0, jitter_strength, len(optimal_cluster_list))

# Make scatter plot
plt.scatter(session_size_list, rounded_numbers, c='cornflowerblue', marker='o')

m, b = np.polyfit(session_size_list, rounded_numbers, 1)
plt.plot(session_size_list, m * session_size_list + b, color='red', label=f'Regression Line: y = {m:.2f}x + {b:.2f}')

plt.xlabel('Session Size')
plt.ylabel('Optimal Number of Clusters')
plt.title(f'Optimal Cluster Number vs Session Size ({iterations} iterations)')
#scatter_plot_path = os.path.join(pca_dir, 'optimal_clusters_vs_session_size.png')
#plt.savefig(scatter_plot_path)
plt.show()

#
rho, p_val = spearmanr(session_size_list, rounded_numbers)
print(f"Spearman rho = {rho:.3f}, p = {p_val:.4f}")


# %% HISTOGRAM PLOTTING VARIATION + STATS FOR OPTIMAL CLUSTER DISTRIBUTION
# ==============================================================================
# 
# ==============================================================================

## Histogram of ditribution of optimal cluster sizes across all iterations
mode_result = stats.mode(rounded_numbers, keepdims=False)
print(f"The mode is: {mode_result[0]}")
    

# Optimal # of cluster histogram
rounded_numbers = [round(num) for num in optimal_cluster_list]
#plt.hist(optimal_cluster_list, bins=len(n_components_range), 

plt.figure(figsize=(6, 8))
plt.hist(rounded_numbers,
         bins = np.arange(0, 16),
         color='0.8', 
         edgecolor='black',
         linewidth=2)
plt.axvline(x=mode_result[0]+0.5, #add 0.5 to shift value because of bin sizes
            color='red', 
            linestyle='--', 
            linewidth=3.5,
            label='Mode')
plt.xlabel('Optimal Number of Clusters')
plt.ylabel('Frequency')
plt.title(f'Frequency of Optimal Cluster Number ({iterations} iterations)')
plt.xlim(0.5, 14.5)
plt.legend()
# Center ticks on each bar with correct labels
plt.xticks(ticks=np.arange(0.5, 15.0, 1), labels=np.arange(0, 15))
plt.savefig("/home/natasha/Desktop/final_figures/optimal_cluster_freq.svg", format="svg")  # Save before show
plt.savefig("/home/natasha/Desktop/final_figures/optimal_cluster_freq.png", format="png")  # Save before show
plt.show()
    
