#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 13:04:06 2025

@author: natasha
"""

import pandas as pd
import os

# ==============================================================================
# Load data and add Narendra's data
# ==============================================================================
dirname = '/home/natasha/Desktop/clustering_data/'
file_path = os.path.join(dirname, 'all_datasets_emg_pred_noNM.pkl') # all events from classifier predictions
df = pd.read_pickle(file_path)

NM_dirpath = os.path.join(dirname, 'predictions')
metadata_dir = os.path.join(NM_dirpath, 'emg_env_and_metadata_frame')

df_list = []

for filename in os.listdir(NM_dirpath):
    if not filename.endswith('.pkl'):
        continue

    full_path = os.path.join(NM_dirpath, filename)
    try:
        new_df = pd.read_pickle(full_path)
        file_name = os.path.splitext(filename)[0]
        basename = file_name.removesuffix('_segments')
        new_df['basename'] = basename
        new_df['animal_num'] = basename.split('_')[0]
        new_df = new_df.rename(columns={'pred_names': 'event_type'})
        #new_df = new_df.rename(columns={'taste': 'taste_num'})
        # Look at corresponding metadata CSV for taste_name info
        matching_csv = next(
            (f for f in os.listdir(metadata_dir) if basename in f and f.endswith('.csv')), None
        )

        if matching_csv:
            metadata_path = os.path.join(metadata_dir, matching_csv)
            try:
                metadata_df = pd.read_csv(metadata_path)
                metadata_df = metadata_df.rename(columns={'taste': 'taste_name'})
                #mapping_df = metadata_df[['dig_in_num_taste', 'taste']].drop_duplicates()
                #taste_map = dict(zip(mapping_df['dig_in_num_taste'], mapping_df['taste']))
                #new_df['metadata_path'] = metadata_path
                # Keep only the relevant columns
                metadata_cols = metadata_df[['taste_rel_trial_num', 'dig_in_num_taste', 
                                             'taste_name', 'laser_duration_ms', 'laser_lag_ms']].drop_duplicates()
                '''
                # Merge metadata into new_df based on 'taste_rel_trial_num'
                new_df = pd.merge(
                    new_df,
                    metadata_df[['taste_rel_trial_num', 'laser_duration_ms', 'laser_lag_ms', 'taste_name']],
                    left_on='trial',
                    right_on='taste_rel_trial_num',
                    how='left'
                )
                new_df = new_df.drop(columns=['taste_rel_trial_num'])
                '''
                new_df = pd.merge(
                    new_df,
                    metadata_df[['taste_rel_trial_num', 'dig_in_num_taste', 
                                 'laser_duration_ms', 'laser_lag_ms', 'taste_name']],
                    left_on=['trial', 'taste'],
                    right_on=['taste_rel_trial_num', 'dig_in_num_taste'],
                    how='left'
                )
                new_df = new_df.drop(columns=['taste_rel_trial_num', 'dig_in_num_taste'])
            except Exception as e:
                print(f"Error reading metadata CSV for {metadata_path}: {e}")
                
        else:
            print(f"No metadata CSV found for {basename}")

        df_list.append(new_df)

    except Exception as e:
        print(f"Error reading {filename}: {e}")
    
# Concatenate all the new dataframes onto the original df
if df_list:
    df = pd.concat([df] + df_list, ignore_index=True)

# Repalce nan in laser columns with 0
df[['laser_duration_ms', 'laser_lag_ms']] = df[['laser_duration_ms', 'laser_lag_ms']].fillna(0)
df['laser'] = (df['laser_duration_ms'] != 0) | (df['laser_lag_ms'] != 0)


# ==============================================================================
# Save integrated dataframe
# ==============================================================================
## Save the new dataframe into a pickle file
output_file_path = os.path.join(dirname, 'all_datasets_emg_pred-TEST.pkl')
df.to_pickle(output_file_path)

print(f"DataFrame successfully saved to {output_file_path}")

