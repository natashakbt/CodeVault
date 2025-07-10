#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 11:14:45 2025

@author: natasha
"""

import pandas as pd
import os

# ==============================================================================
# Load data
# ==============================================================================

dirname = '/home/natasha/Desktop/clustering_data/'
#file_path = os.path.join(dirname, 'all_datasets_emg_pred.pkl') # all events from classifier predictions
file_path = os.path.join(dirname, 'all_datasets_emg_pred-TEST.pkl') # all events from classifier predictions
df = pd.read_pickle(file_path)

# ==============================================================================
# Cleanup data
# ==============================================================================

df = df.rename(columns={'pred_event_type': 'event_type'})
df = df.rename(columns={'taste': 'taste_num'})
df = df.rename(columns={'trial': 'trial_num'})
df['basename'] = df['basename'].str.lower() # All basenames to lowercase


df = df[["features", "segment_raw", "segment_norm_interp", 
         "segment_bounds", "taste_num", "trial_num", 
         "basename", "animal_num", "taste_name", 
         "raw_features", "event_type", "session_ind", 
         "multimodal"]] # Keep only needed columns

df.reset_index(drop=True, inplace=True)

# ==============================================================================
# Save new dataframe
# ==============================================================================
## Save the new dataframe into a pickle file
output_file_path = os.path.join(dirname, 'all_datasets_emg_pred.pkl')
df.to_pickle(output_file_path)

print(f"DataFrame successfully saved to {output_file_path}")


