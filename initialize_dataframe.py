#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 11:14:45 2025

@author: natasha
"""

import pandas as pd
import os
from mtm_analysis_config import dirname, xgb_predictions_filename

# ==============================================================================
# Load data
# ==============================================================================
#file_path = os.path.join(dirname, 'all_datasets_emg_pred.pkl') # all events from classifier predictions
file_path = os.path.join(dirname, xgb_predictions_filename) # all events from classifier predictions
df = pd.read_pickle(file_path)

# ==============================================================================
# Create a backup of the original data
# ==============================================================================


answer = input("Do you want to create a backup of the original data? (y/n): ").strip().lower()


if answer in ("y", "yes"):
    
    # Make sure backup folder exists
    backup_dir = os.path.join(dirname, '_original_data_backup')
    os.makedirs(backup_dir, exist_ok=True)

    # Insert '_backup' before the file extension
    base, ext = os.path.splitext(xgb_predictions_filename)
    backup_filename = f"{base}_backup{ext}"
    backup_file_path = os.path.join(backup_dir, backup_filename)
    '''
    df.to_pickle(backup_file_path)
    '''
    print(f"✅ Backup saved to: {backup_file_path}")
    
else:
    print("⚠️ Backup skipped.")


# ==============================================================================
# Cleanup data
# ==============================================================================

df = df.rename(columns={'pred_event_type': 'event_type'})
df = df.rename(columns={'taste': 'taste_num'})
df = df.rename(columns={'trial': 'trial_num'})
df['basename'] = df['basename'].str.lower() # All basenames to lowercase

# Make sure each basename has an associated session index
basename_to_index = {name: i for i, name in enumerate(df['basename'].unique())}
df['session_ind'] = df['basename'].map(basename_to_index)

df.event_type = df.event_type.replace('mouth or tongue movement', 'MTMs')

df = df[["features", "segment_raw", "segment_norm_interp", 
         "segment_bounds", "taste_num", "trial_num", 
         "basename", "animal_num", "taste_name", 
         "raw_features", "event_type", "session_ind", 
         "multimodal", "laser_duration_ms", "laser_lag_ms",
         "laser"]] # Keep only needed columns

df.reset_index(drop=True, inplace=True)

# ==============================================================================
# Test rows for duplicates -> could be caused by incorrect duplicate waveforms
# ==============================================================================
test_df = df.drop(["segment_raw", "features", "segment_norm_interp", "raw_features"], axis=1)

if test_df.duplicated().sum() > 0:
    print("Warning: There are some duplicate rows in the dataframe")
if df["features"].apply(lambda x: tuple(x)).duplicated().sum() > 0:
    print("Warning: Some of the rows in the DataFrame have duplicate feature vectors")


# ==============================================================================
# Save new dataframe
# ==============================================================================
## Save the new dataframe into a pickle file
output_file_path = os.path.join(dirname, 'all_datasets_emg_pred.pkl')
df.to_pickle(output_file_path)

print(f"DataFrame successfully saved to {output_file_path}")


# ==============================================================================
# Make sure there is a folder final_figures - generally publication figures
# ==============================================================================
final_figures_dir = os.path.join(dirname, "final_figures")
os.makedirs(final_figures_dir, exist_ok=True)

