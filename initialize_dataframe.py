#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 11:14:45 2025

@author: natasha

Preprocess and clean combined EMG classifier output.

This script:
1. Loads the XGB EMG predictions (assumes each session has been combined into one dataframe)
2. Optionally creates a backup of the original file
3. Renames and standardizes columns
4. Adds session indices
5. Removes unused columns
6. Performs duplicate segments checks
7. Saves a cleaned dataframe that OVERWRITES the original file

Inputs:
- Pickle file defined by `xgb_predictions_filename` in mtm_analysis_config

Outputs:
- Cleaned pickle file saved to:
  `{dirname}/all_datasets_emg_pred.pkl`
"""


import pandas as pd
import os
# REQUIRED: Run file mtm_analysis_config.py (F5) in Spyder after each restart
# enables import of required variables into any downstream script
from mtm_analysis_config import dirname, xgb_predictions_filename
from datetime import datetime


# ==============================================================================
# Load data
# ==============================================================================

file_path = os.path.join(dirname, xgb_predictions_filename) # all events from classifier predictions
df = pd.read_pickle(file_path)


# ==============================================================================
# Create a backup of the original data
# ==============================================================================

answer = input(f"Do you want to create a backup of {xgb_predictions_filename} data? (y/n): ").strip().lower()

if answer in ("y", "yes"):
    
    # Make sure backup folder exists
    backup_dir = os.path.join(dirname, '_original_data_backup')
    os.makedirs(backup_dir, exist_ok=True)

    # Insert '_backup' before the file extension
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(xgb_predictions_filename)
    backup_filename = f"{base}_backup{timestamp}{ext}"
    backup_file_path = os.path.join(backup_dir, backup_filename)
    
    df.to_pickle(backup_file_path)
    
    print(f"✅ Backup saved to: {backup_file_path}")
    
else:
    print("⚠️ Backup skipped.")


# ==============================================================================
# Data cleanup and standardization
# ==============================================================================

df = df.rename(columns={'pred_event_type': 'event_type'})
df = df.rename(columns={'taste': 'taste_num'})
df = df.rename(columns={'trial': 'trial_num'})
df['basename'] = df['basename'].str.lower() # All basenames to lowercase

# Make sure each basename has an associated session index
basename_to_index = {name: i for i, name in enumerate(df['basename'].unique())}
df['session_ind'] = df['basename'].map(basename_to_index)

df.event_type = df.event_type.replace('mouth or tongue movement', 'MTMs')

# Columns that must exist for downstream analysis
required_columns = [
    "features",
    "segment_raw",
    "segment_norm_interp",
    "segment_bounds",
    "taste_num",
    "trial_num",
    "basename",
    "animal_num",
    "taste_name",
    "raw_features",
    "event_type",
    "session_ind",
]

# Columns that can be included if present but are optional
optional_columns = [
    "multimodal",
    "laser_duration_ms",
    "laser_lag_ms",
    "laser",
    "exp_day_type",
    "exp_day_num",
    "licl_conc",
    "num_of_cta"
]

# Check that all required columns exist
missing = set(required_columns) - set(df.columns)
assert not missing, f"Missing required columns: {missing}"

# Keep required columns + any optional columns that exist
cols_to_keep = required_columns + [c for c in optional_columns if c in df.columns]

df = df[cols_to_keep]

df.reset_index(drop=True, inplace=True)


# ==============================================================================
# Test rows for duplicates -> every row should correspond to a unique waveform
# Duplicates are likely be caused by an error when combining XGB segments files into one pkl file
# ==============================================================================

test_df = df.drop(["segment_raw", "features", "segment_norm_interp", "raw_features"], axis=1)

if test_df.duplicated().sum() > 0:
    print("Warning: There are some duplicate rows in the dataframe. Something is wrong with your dataframe!")
if df["features"].apply(lambda x: tuple(x)).duplicated().sum() > 0:
    print("Warning: Some of the rows in the DataFrame have duplicate feature vectors. Something is wrong with your dataframe!")


# ==============================================================================
# Save new dataframe
# ==============================================================================
## Save the new dataframe into a pickle file
output_file_path = os.path.join(dirname, 'all_datasets_emg_pred.pkl')
df.to_pickle(output_file_path)

print(f"DataFrame successfully saved to {output_file_path}")


# ==============================================================================
# Ensure final_figure directory exists - where publication-quality figures are saved to
# ==============================================================================
final_figures_dir = os.path.join(dirname, "final_figures")
os.makedirs(final_figures_dir, exist_ok=True)



