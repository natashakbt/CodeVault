#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 11:43:34 2025

@author: natasha
"""
import pandas as pd
import os

# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/christina_data/'
file_path = os.path.join(dirname, 'christina_all_datasets.pkl')
df = pd.read_pickle(file_path)


# ==============================================================================
# %% Finding gaping metrics by trial
# A gape bout is defined as 3 or more gapes in a row, and no inter-gape interval is >250ms
# METRICS:
# first_gape_bout_start: first gape bout onset time (ms)
# first_gape_bout_duration: length of time of the first gape bout (ms)
# total_gape_time_1s: How long the rat gaping for within a 1 sec window after the first gape bout start time
# ==============================================================================

results = []

for basename_name, basename_group in df.groupby('basename'):
   for (trial_num, taste_name), trial_group in basename_group.groupby(['trial', 'taste_name']):
        gapes = trial_group[trial_group['event_type'] == 'gape'].copy()
        gapes['start_time'] = gapes['segment_bounds'].apply(lambda x: x[0])
        gapes['end_time'] = gapes['segment_bounds'].apply(lambda x: x[1])
        
        gapes = gapes[gapes['start_time'] > 2000].copy()
        gapes = gapes.sort_values('start_time').reset_index(drop=True)
        
        licl_conc = basename_group['licl_conc'].iloc[0]
        exp_day_type = basename_group['exp_day_type'].iloc[0]
        exp_day_num = basename_group['exp_day_num'].iloc[0]
        num_of_cta = basename_group['num_of_cta'].iloc[0]
        animal_num = basename_group['animal_num'].iloc[0]
        bout_found = False
        i = 0
        while i <= len(gapes) - 3:  # need at least 3 gapes in a row
            # Start a potential bout
            bout_indices = [i]
            
            # Extend the bout while gap < 250ms
            j = i
            while j < len(gapes) - 1:
                gap = gapes.loc[j+1, 'start_time'] - gapes.loc[j, 'end_time']
                if gap < 250:
                    bout_indices.append(j+1)
                    j += 1
                else:
                    break
            
            # Check if the bout has at least 3 gapes
            if len(bout_indices) >= 3:
                bout_start = gapes.loc[bout_indices[0], 'start_time']
                bout_end = gapes.loc[bout_indices[-1], 'end_time'] # Last gape in the bout
                #bout_end = gapes['end_time'].iloc[-1] # last gape in the trial
                bout_duration = bout_end - bout_start
                
                # ---- compute gape time in first 1-second window ----
                win_start = bout_start
                win_end = bout_start + 1000
                
                total_gape_time_1s = 0
                for _, row in gapes.iterrows():
                    seg_start = row['start_time']
                    seg_end = row['end_time']
                
                    overlap = max(0, min(seg_end, win_end) - max(seg_start, win_start))
                    total_gape_time_1s += overlap
                
                results.append({
                    'basename': basename_name,
                    'animal_num': animal_num,
                    'licl_conc': licl_conc,
                    'exp_day_type': exp_day_type,
                    'exp_day_num': exp_day_num,
                    'trial_num': trial_num,
                    'taste_name': taste_name,
                    'num_of_cta': num_of_cta,
                    'first_gape_bout_start': bout_start,
                    'first_gape_bout_duration': bout_duration,
                    'total_gape_time_1s': total_gape_time_1s
                })
                
                bout_found = True
                break  # first bout found, move to next trial/taste
            else:
                # Not enough gapes, move to next starting index
                i += 1
        
        if not bout_found:
            # No valid bout in this trial
            results.append({
                'basename': basename_name,
                'animal_num': animal_num,
                'licl_conc': licl_conc,
                'exp_day_type': exp_day_type,
                'exp_day_num': exp_day_num,
                'trial_num': trial_num,
                'taste_name': taste_name,
                'num_of_cta': num_of_cta,
                'first_gape_bout_start': None,
                'first_gape_bout_duration': None,
                'total_gape_time_1s': None
            })


# Convert to dataframe
bout_df = pd.DataFrame(results)

# ==============================================================================
# Find session trial 50% split & add to new column 'session_trial_split'
# ==============================================================================
max_trial = bout_df.groupby(
    ["basename", "exp_day_num", "taste_name"]
)["trial_num"].transform("max")

# compute midpoint split value
midpoint = max_trial / 2

bout_df["session_trial_split"] = (bout_df["trial_num"] > midpoint).astype(int)

# ==============================================================================
# Save dataframe
# ==============================================================================
output_file_path = os.path.join(dirname, 'gape_metrics_by_trial.pkl')
bout_df.to_pickle(output_file_path)

print(f"DataFrame 'gape_metrics_by_trial.pkl' successfully saved to {output_file_path}")    
