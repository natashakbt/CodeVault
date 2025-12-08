#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 14:25:23 2025

@author: natasha
"""
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
import re
from tqdm import tqdm
# ==============================================================================
# Load data
# ==============================================================================
dirname = '/home/natasha/Desktop/christina_data/raw_data'

segment_folders = ['0.15M_LiCl_segments', '0.6M_LiCl_segments', 
                   '0.15M_NaCl_segments', '0.6M_NaCl_segments',
                   '1train_1test_0.8QHCl_segments', '1train_1test_1QHCl_segments',
                   '2train_1test_segments'
                   ]
metadata_dir = os.path.join(dirname, 'all_info_files')
metadata_dir = Path(metadata_dir)   # convert string to Path object

df_list = []

# ==============================================================================
# Load data
# ==============================================================================
print("Searching through folders for data files")
for folder in tqdm(segment_folders):
    CM_dirpath = os.path.join(dirname, folder)
    
    for filename in os.listdir(CM_dirpath):
        if not filename.endswith('.pkl'):
            continue
    
        full_path = os.path.join(CM_dirpath, filename)
        try:
            new_df = pd.read_pickle(full_path)
            file_name = os.path.splitext(filename)[0]
            basename = file_name.removesuffix('_emg_classifier_segments')
            new_df['basename'] = basename
            new_df['animal_num'] = basename.split('_')[0]
            
            new_df['exp_day_type'] = new_df['basename'].apply(lambda x: 'Test' if 'Test' in x else 'Train')
            new_df['exp_day_num'] = new_df['basename'].apply(lambda x: int(re.search(r'(\d+)$', x).group(1)) if re.search(r'(\d+)$', x) else 1)

            
            #new_df['licl_conc'] = folder.split('_')[0]  # tag by folder name
            new_df['licl_conc'] = folder.removesuffix('_segments')
            new_df = new_df.rename(columns={'pred_names': 'event_type'})
            new_df = new_df.rename(columns={'taste': 'taste_num'})
    
            metadata_path = False  # flag to track if we find a match
            
            for f in metadata_dir.iterdir():
                name_lc = f.name
                result = '_'.join(name_lc.split('_')[:2])
    
                if result.strip() == basename.strip():
                    #print(f"MATCH: {basename} ↔ {result}")
                    metadata_path = f
                    break  # stop checking others once a match is found
    
            if not metadata_path:
                print(f"⚠️ WARNING: No metadata file found for {basename}")
    
            if metadata_path:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Extract the dig_ins, tastes, and concentrations from taste_params
                taste_params = metadata.get("taste_params", {})
                dig_ins = taste_params.get("dig_ins") or taste_params.get("dig_in_nums") or []
                tastes = [t.lower() for t in taste_params.get("tastes", [])]
                concs = taste_params.get("concs", [])
                
                taste_nums = sorted(new_df['taste_num'].unique())
                if len(new_df['taste_num'].unique()) != len(tastes):
                    print(
                        f"Mismatch: {len(taste_nums)} taste_num values vs {len(tastes)} infofile"
                    )
    
                
                taste_num_to_taste = dict(zip(taste_nums, tastes))
                taste_num_to_conc = dict(zip(taste_nums, concs))
                
                # Map to DataFrame
                new_df['taste_name'] = new_df['taste_num'].map(taste_num_to_taste)
                new_df['taste_conc'] = new_df['taste_num'].map(taste_num_to_conc)

            df_list.append(new_df)
    
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
df = pd.concat(df_list, ignore_index=True)


# ==============================================================================
# Assigning column 'num_of_cta' where 1st day = number of train days +1
# ==============================================================================
def assign_num_of_cta(group):
    # Filter train days
    train_days = group[group['exp_day_type'] == 'Train'].sort_values('exp_day_num')
    
    train_days = sorted(group.loc[group['exp_day_type'] == 'Train', 'exp_day_num'].dropna().unique())
    # Create mapping for train days
    train_map = {day: i for i, day in enumerate(train_days)}

    # find last CTA number from Train days
    last_cta = len(train_map) - 1 if len(train_map) > 0 else np.nan
    result = []
    for _, row in group.iterrows():
        if row['exp_day_type'] == 'Train':
            result.append(train_map.get(row['exp_day_num'], np.nan))
        elif row['exp_day_type'] == 'Test':
            if row['exp_day_num'] == 1:
                result.append(last_cta + 1 if not np.isnan(last_cta) else np.nan)
            else:
                result.append(np.nan)
        else:
            result.append(np.nan)
    group['num_of_cta'] = result
    return group

df = df.groupby('animal_num', group_keys=False).apply(assign_num_of_cta)



# ==============================================================================
# Fix up some parts of the dataframe so that it's standardized
# ==============================================================================

df['session_ind'] = df['basename'].astype('category').cat.codes
df = df.rename(columns={'pred': 'cluster_num'})
df['taste_name'] = df['taste_name'].replace('sac', 'saccharin')
df['taste_name'] = df['taste_name'].replace('quinine', 'qhcl')
#df['taste_name'] = df['taste_name'].replace('quinc', 'qhcl')

final_test_day_num = df['num_of_cta'].max() + 1
df['num_of_cta'] = df['num_of_cta'].replace(np.nan, final_test_day_num)
print(f"Final test day is numbered as {final_test_day_num} under 'num_of_cta' column")


mask = (df['animal_num'] == 'CM42') & (df['taste_name'] == 'highqhcl')

# Step 1: check if taste_conc only has one unique value
unique_concs = df.loc[mask, 'taste_conc'].dropna().unique()
if len(unique_concs) == 1:
    # Step 2: rename highqhcl → qhcl
    df.loc[mask, 'taste_name'] = 'qhcl'
    print("For CM42, renamed highqhcl → qhcl")
else:
    print("Multiple qhcl concentrations found for CM42 — not renaming.")



# ==============================================================================
# Save the new dataframe into a pickle file
# ==============================================================================
parent_dir = os.path.dirname(dirname)
output_file_path = os.path.join(parent_dir, 'christina_all_datasets.pkl')
df.to_pickle(output_file_path)

print(f"DataFrame successfully saved to {output_file_path}")       
