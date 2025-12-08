# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
from mtm_analysis_config import dirname, feature_names, color_mapping
from scipy.signal import find_peaks
from scipy.stats import chisquare

# ==============================================================================
# Load data and get setup
# ==============================================================================

file_path = os.path.join(dirname, 'clustering_df_update.pkl')
df = pd.read_pickle(file_path)

ltp_scored_file_path = os.path.join(dirname, 'scored_df.pkl')
ltp_df_scored = pd.read_pickle(ltp_scored_file_path)
ltp_df_scored = ltp_df_scored.rename(columns={'updated_event_type': 'scored_event'})


# ==============================================================================
# Merge dataframes
# ==============================================================================

# Convert segment_raw into tuples to allow for comparison between dataframes
df['segment_raw'] = df['segment_raw'].apply(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x)
ltp_df_scored['segment_raw'] = ltp_df_scored['segment_raw'].apply(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x)

# Check that there are no duplicate values of segment_raw -> it should be a unique identifier
print("Duplicates in df:", df['segment_raw'].duplicated().sum())
print("Duplicates in ltp_df_scored:", ltp_df_scored['segment_raw'].duplicated().sum())

# Merge dataframes
df = df.merge(
    ltp_df_scored[['segment_raw', 'scored_event']],
    on='segment_raw',
    how='left'
)

# Check that all ltm_scored_df rows found a match and got merged into df
matched = df['scored_event'].notna().sum()
total_ltp = len(ltp_df_scored)
print(f"Matched {matched} out of {total_ltp} scored rows ({matched/total_ltp:.2%})")


# ==============================================================================
# Check for LTM enrichment in one of the MTM clusters
# ==============================================================================
ltp_df = df[df['scored_event'] == 'lateral tongue protrusion']


# All clusters
counts_all = ltp_df['cluster_num'].value_counts().sort_index()
chi2_all, p_all = chisquare(counts_all)
print("All clusters:", chi2_all, p_all)

# Only clusters 0, 1, 2
counts_subset = ltp_df[ltp_df['cluster_num'].isin([0, 1, 2])]['cluster_num'].value_counts().sort_index()
chi2_subset, p_subset = chisquare(counts_subset)
print("Clusters 0,1,2:", chi2_subset, p_subset)


# x-axis labels
labels = ['no movement', 'gapes', '0', '1', '2']

# Create the bar plot
ax = counts_all.plot(kind='bar', title='Cluster distribution')

# Set new x-axis labels
ax.set_xticks(range(len(labels))) 
ax.set_xticklabels(labels)
ax.set_ylabel('counts')
plt.show()
