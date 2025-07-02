#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 13:04:06 2025

@author: natasha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import os
import umap
import glob
from scipy import stats
from sklearn.cluster import KMeans
import piecewise_regression
from scipy.spatial.distance import mahalanobis

# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/clustering_data/'
#file_path = os.path.join(dirname, 'mtm_clustering_df.pkl') # only events labelled by video scoring
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

        # Look at corresponding metadata CSV for taste_name info
        matching_csv = next(
            (f for f in os.listdir(metadata_dir) if basename in f and f.endswith('.csv')), None
        )

        if matching_csv:
            metadata_path = os.path.join(metadata_dir, matching_csv)
            try:
                metadata_df = pd.read_csv(metadata_path)
                mapping_df = metadata_df[['dig_in_num_taste', 'taste']].drop_duplicates()
                taste_map = dict(zip(mapping_df['dig_in_num_taste'], mapping_df['taste']))
                #print(f"Mapping for {basename}:", taste_map)
                new_df['metadata_path'] = metadata_path  # Optional, if you want to keep the path
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


df = df.rename(columns={'pred_event_type': 'event_type'})


# TODO: DOUBLE CHECK THAT DATAFRAME IS ALL GOOD
# TODO: DO I HAVE NARENDRA'S CHANGEPOINT TIMES??

unique_basenames = df['basename'].unique()
basename_to_num = {name: idx for idx, name in enumerate(unique_basenames)}
df['session_ind'] = df['basename'].map(basename_to_num)

df.event_type = df.event_type.replace('mouth or tongue movement', 'MTMs')

# Make a dataframe of just mouth or tongue movement events
mtm_bool = df.event_type.str.contains('MTMs')
mtm_df = df.loc[mtm_bool]



# ==============================================================================
# TODO: SAVE FILES
# ==============================================================================
