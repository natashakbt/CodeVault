#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:10:42 2025

@author: natasha
"""
import numpy as np

# Set location and file name of predictions generated from XGB
# dirname will also be the base folder in which plots and additional dataframes will be saved
dirname='/home/natasha/Desktop/clustering_data/'
xgb_predictions_filename = 'all_datasets_emg_pred-TEST.pkl'



# Parameters used by clustering_MTM code
iterations = 50
fixed_cluster_num = np.nan




# Define a color mapping for cluster numbers
color_mapping = {
    -1: '#ff9900',    # Color for gapes cluster -1
    -2: '#D3D3D3',    # Color for no movement cluster 0
     0: '#4285F4',    # Color for cluster 1
     1: '#88498F',    # Color for cluster 2
     2: '#0CBABA'     # Color for cluster 3
}    

extra_colors = []


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
