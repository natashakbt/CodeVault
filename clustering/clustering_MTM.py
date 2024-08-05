# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import umap
import glob
import diptest



# Load data
file_path = '/home/natasha/Desktop/clustering_data/mtm_clustering_df.pkl'
df = pd.read_pickle(file_path)

# Make a dataframe of just mouth or tongue movement events
mtm_bool = df.event_type.str.contains('mouth or tongue movement')
mtm_df = df.loc[mtm_bool]

# Says what number session
mtm_df.session_ind

# Array of every MTM events, and value for each of the 8 features
mtm_features = np.stack(mtm_df.features.values)


# Make UMAP plot of all MTM events
reducer = umap.UMAP()

scaled_mtm_features = StandardScaler().fit_transform(mtm_features)

embedding = reducer.fit_transform(scaled_mtm_features)
plt.scatter(embedding[:,0], embedding[:,1])
plt.title('UMAP projection of MTM events')

# Make UMAP plot of all MTM events for EACH SESSION

for session in df.session_ind.unique():
    mtm_session_bool = mtm_df.session_ind.astype(str).str.contains(str(session))
    mtm_session_df = mtm_df.loc[mtm_session_bool]
    mtm_session_features = np.stack(mtm_session_df.features.values)
    reducer = umap.UMAP()
    scaled_mtm_session= StandardScaler().fit_transform(mtm_session_features)
    embedding = reducer.fit_transform(scaled_mtm_session)
    plt.scatter(embedding[:,0], embedding[:,1])
    plt.title(f'UMAP projections of MTM events for session {session}')
    plt.show()
    
# Test which segments are bimodal
p_values = []
segment_raw = mtm_df['segment_raw']
for index, segment in enumerate(segment_raw):
    dip, pval = diptest.diptest(segment)
    p_values.append(pval)

mtm_df.loc[:,'p_value'] = p_values

mtm_df_multi = mtm_df[mtm_df['p_value'] < 0.005]
# mtm_df_uni = mtm_df[mtm_df['p_value'] >= 0.005]

# Create plots folder
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)
# Remove any files in plots folder
png_files = glob.glob(os.path.join(output_dir, '*.png'))
for file in png_files:
    os.remove(file)

# Plot all bimodal waveforms. Double check that the look ok
for index, row in mtm_df_multi.iterrows():
    segment = row['segment_raw']
    plt.plot(segment)
    plt_title = f"plot_{index}.png"
    plt.savefig(os.path.join(output_dir, plt_title))
    plt.clf()

percent_multi = len(mtm_df_multi)/len(mtm_df)*100
print('Percent of multimodal waveforms out of total:', percent_multi)
#TODO: 
