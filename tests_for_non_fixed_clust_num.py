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
from scipy.stats import spearmanr
import statistics
from mtm_analysis_config import dirname, iterations

# ==============================================================================
# Load and setup data
# ==============================================================================
file_path = os.path.join(dirname, 'non_fixed_clust_data.pkl')
non_fixed_clust_data = pd.read_pickle(file_path)

optimal_cluster_list = non_fixed_clust_data['optimal_cluster_list']
session_size_list = non_fixed_clust_data['session_size_list']

rounded_numbers = [round(num) for num in optimal_cluster_list]

# Set folders for where to save plots
final_figures_dir = os.path.join(dirname, "final_figures")
extra_plots_dir = os.path.join(dirname, 'extra_plots')
os.makedirs(extra_plots_dir, exist_ok=True)


# %% Stats + plot optimal cluster size vs sessions size (i.e. number of MTMs)
# ==============================================================================
# Spearman rank correlation to test session size on cluster size
# ==============================================================================
rho, p_val = spearmanr(session_size_list, rounded_numbers)
if p_val > 0.05:
    print("No effet of session size on optimal number of clusters")
    print(f"Spearman rho = {rho:.3f}, p = {p_val:.4f}")
else:
    print("⚠ Warning: effect of session size on optimal number of clusters")
    print(f"Spearman rho = {rho:.3f}, p = {p_val:.4f}")

# ==============================================================================
# Scatter plot of optimal cluster size vs sessions size
# ==============================================================================
jitter_strength = 0.05

# Add jitter to the session size and optimal clusters
session_size_jittered = np.array(session_size_list) + np.random.normal(0, jitter_strength, len(session_size_list))
optimal_cluster_jittered = np.array(optimal_cluster_list) + np.random.normal(0, jitter_strength, len(optimal_cluster_list))

# Make scatter plot
plt.scatter(session_size_list, rounded_numbers, c='cornflowerblue', marker='o')

m, b = np.polyfit(session_size_list, rounded_numbers, 1)
plt.plot(session_size_list, m * session_size_list + b, color='red', 
         label=f'Regression Line: y = {m:.2f}x + {b:.2f}\nSpearman ρ={rho:.3f}, p={p_val:.2f}')

plt.legend()
plt.xlabel('Session Size')
plt.ylabel('Optimal Number of Clusters')
plt.title(f'Optimal Cluster Number vs Session Size ({iterations} iterations)')

scatter_plot_path = os.path.join(extra_plots_dir, 'optimal_clusters_vs_session_size.png')
plt.savefig(scatter_plot_path)
plt.close()


# %% Stat result on optimal cluster number + histogram plot
# ==============================================================================

mode_result = statistics.mode(rounded_numbers)
mean_result = round(statistics.mean(rounded_numbers))
median_result = statistics.median(rounded_numbers)

if mode_result == mean_result == median_result:
    print(f"\n*** Across all sessions and {iterations} iterations")
    print(f"THE OPTIMAL CLUSTER NUMBER IS {mode_result} ***")
    print(f"Please manually update fixed_cluster_num = {mode_result} in clustering_config and rerun clustering_MTM.py")
else:
    print("Optimal cluster number:")
    print(f"The mode is: {mode_result}")
    print(f"The mean is: {mean_result}")
    print(f"The median is: {median_result}")
    print("Please manually update fixed_cluster_num in clustering_config to desired number and rerun clustering_MTM.py")

# Histogram of ditribution of optimal cluster sizes across all iterations
plt.figure(figsize=(6, 8))
plt.hist(rounded_numbers,
         bins = np.arange(0, 16),
         color='0.8', 
         edgecolor='black',
         linewidth=2)
plt.axvline(x=mode_result+0.5, #add 0.5 to shift value because of bin sizes
            color='red', 
            linestyle='--', 
            linewidth=3.5,
            label='Mode')
plt.legend()
plt.xlim(0.5, 14.5)
plt.xticks(ticks=np.arange(0.5, 15.0, 1), labels=np.arange(0, 15)) # Center ticks on each bar with correct labels
plt.xlabel('Optimal Number of Clusters')
plt.ylabel('Frequency')
plt.title(f'Frequency of Optimal Cluster Number ({iterations} iterations)') # Iteration values is set in clustering_config file

for ext in ["svg", "png"]:
    plt.savefig(os.path.join(final_figures_dir, f"optimal_cluster_freq.{ext}"), format=ext)

plt.show()
    
