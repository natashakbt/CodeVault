#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:10:42 2025

@author: natasha




"""

import numpy as np


# ==============================================================================
# Paths and input files
# ==============================================================================

# Base directory containing prediction files and where outputs will be saved
# (plots, processed dataframes, clustering results, etc.)
dirname='/home/natasha/Desktop/christina_data'

# A pickle file containing all XGBoost EMG segment predictions combined into one dataframe
xgb_predictions_filename = 'christina_all_datasets.pkl'


# ==============================================================================
# Clustering parameters (used by clustering_MTM script)
# ==============================================================================

# Fixed number of clusters:
#   - np.nan  → test 1–14 clusters to find the best fit
#   - integer → force clustering to that number for all sessions
#   (Previously determined optimal clustering # for MTMs: 3)
fixed_cluster_num = 3

# Number of GMM repeats per session:
#   - suggested: 50 when fixed_cluster_num = np.nan (for estimating optimal cluster number)
#   - suggested: 1 when fixed_cluster_num = integer
iterations = 1


# ==============================================================================
# Visualization parameters
# ==============================================================================

# Mapping from cluster label → color (hex)
color_mapping = {
    -1: '#ff9900',    # Color for gapes cluster -1
    -2: '#D3D3D3',    # Color for no movement cluster 0
     0: '#4285F4',    # Color for MTM cluster 1
     1: '#88498F',    # Color for MTM cluster 2
     2: '#0CBABA'     # Color for MTM cluster 3
}    

# Optional additional hex codes for dynamically assigned clusters (when fixed_cluster_num = np.nan)
extra_colors = []


# ==============================================================================
# Feature configuration (do not modify - pre-determined by XGB pipeline)
# ==============================================================================

# Feature names in the order they appear in the 'features' column
feature_names = [
    "duration",
    "left_interval",
    "right_interval",
    "max_freq",
    "amplitude_norm",
    "pca_0",
    "pca_1",
    "pca_2",
]
