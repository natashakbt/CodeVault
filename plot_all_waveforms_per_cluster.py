#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 12:00:22 2025

@author: natasha
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from mtm_analysis_config import dirname

# ==============================================================================
# Load data and get setup
# ==============================================================================
file_path = os.path.join(dirname, 'clustering_df_update_with_laser.pkl')
df = pd.read_pickle(file_path)

# ==============================================================================
# Setup folder structure and clear any .png files in folders
# ============================================================================== 
# Create folder for saving plots
label_dir = os.path.join(dirname, 'waveforms_by_cluster')
gapes_dir = os.path.join(label_dir, 'gapes')
nothing_dir = os.path.join(label_dir, 'nothing')


for folder in [label_dir, gapes_dir, nothing_dir]:
    os.makedirs(folder, exist_ok=True)
    all_files = glob.glob(os.path.join(folder, '*.png'))
    for file in all_files:
        os.remove(file)


# %% Plot waveforms per cluster label 
# ==============================================================================
# Setup color map
# ==============================================================================

# Colors for gapes and no movement is set
color_mapping = {
    -1.0: '#ff9900',  # Gapes Color for cluster -1
    -2.0: '#D3D3D3'   # No movement Color for cluster -2
}

# Generate unique colors by basenames for MTMs
basename_list = df['basename'].unique()
basename_colors = plt.cm.viridis_r(np.linspace(0, 1, len(basename_list)))
basename_color_map = dict(zip(basename_list, basename_colors))



# ==============================================================================
# Plot waveforms divided cluster and basename
# ==============================================================================
cluster_basename_groups = df.groupby(['cluster_num', 'basename'])
for (cluster, basename), group in tqdm(cluster_basename_groups):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use predefined colors for -1.0 and -2.0 clusters, otherwise assign colors by basename
    color = color_mapping.get(cluster, basename_color_map.get(basename, 'black'))
    
    for segment in group['segment_raw']:
        ax.plot(segment, alpha=0.1, color=color)
    
    ax.set_title(f'Cluster {cluster} - {basename} Waveforms')
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    
    # Save plot
    plot_filename = f'{basename}_cluster{cluster}.png'
    if cluster == -1.0:
        plot_path = os.path.join(label_dir, 'gapes', plot_filename)
        plt.savefig(plot_path)
        plt.close(fig) 
    elif cluster == -2.0:
        plot_path = os.path.join(label_dir, 'nothing', plot_filename)
        plt.savefig(plot_path)
        plt.close(fig)
    else:
        plot_path = os.path.join(label_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close(fig)
