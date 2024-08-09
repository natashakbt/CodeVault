# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import os
import umap
import glob
import diptest
from scipy import stats


# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/clustering_data/'
file_path = os.path.join(dirname, 'mtm_clustering_df.pkl')
df = pd.read_pickle(file_path)

# Make a dataframe of just mouth or tongue movement events
mtm_bool = df.event_type.str.contains('mouth or tongue movement')
mtm_df_all = df.loc[mtm_bool]

# ==============================================================================
# Remove any bimodal-shaped EMG segments
# ==============================================================================
# Test which segments are bimodal using diptest statistic
p_values = []
segment_raw = mtm_df_all['baseline_scaled_segments']
for index, segment in enumerate(segment_raw):
    dip, pval = diptest.diptest(segment)
    p_values.append(pval)

# Add p-value to dataframe
mtm_df_all.loc[:,'p_value'] = p_values
# Select MTMs that have a significant p-value
mtm_df_multi = mtm_df_all[mtm_df_all['p_value'] < 0.005]


# Create folder for plots of multimodal segments
output_dir = os.path.join(dirname, 'multimodal_segments')
os.makedirs(output_dir, exist_ok=True)
# Remove any png files in plots folder
png_files = glob.glob(os.path.join(output_dir, '*.png'))
for file in png_files:
    os.remove(file)

# Plot all bimodal waveforms. You should double check that they look OK
for index, row in mtm_df_multi.iterrows():
    segment = row['segment_raw']
    plt.plot(segment)
    plt_title = f"segment_{index}.png"
    plt.savefig(os.path.join(output_dir, plt_title))
    plt.clf()

percent_multi = len(mtm_df_multi)/len(mtm_df_all)*100
print(f'Percent of multimodal waveforms of total: {percent_multi:.2f}%')


# ==============================================================================
# Create UMAP projection of all MTM events and by session. 
# Fit a GMM to the UMAP projection
# Calculate BIC score to determine best number of GMM clusters
# ==============================================================================
# Create directory for saving UMAP clustering results
umap_dir = os.path.join(dirname, 'UMAP_results')
os.makedirs(umap_dir, exist_ok=True)
# Remove any png files in plots folder
png_files = glob.glob(os.path.join(umap_dir, '*.png'))
for file in png_files:
    os.remove(file)

# Use the unimodal MTM waveforms for the rest of the analyses
mtm_df = mtm_df_all[mtm_df_all['p_value'] >= 0.005]
# Array of every MTM event and their values for each of the 8 features
mtm_features = np.stack(mtm_df.features.values)

# UMAP dimmentionality reduction and feature scaling
reducer = umap.UMAP()
n_components_range = range(1, 15)  # Define a range of cluster numbers to test
scaled_mtm_features = StandardScaler().fit_transform(mtm_features) # Scale features
embedding = reducer.fit_transform(scaled_mtm_features) # UMAP embedding

# Determine the optimal number of clusters using BIC
bic_scores = []
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(embedding)
    bic = gmm.bic(embedding)
    bic_scores.append(bic)

# Find the number of components with the lowest BIC
optimal_n_components = n_components_range[np.argmin(bic_scores)]
print(f'All sessions: Optimal number of clusters is {optimal_n_components}')

# Fit the GMM with the optimal number of clusters
optimal_gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
optimal_gmm.fit(embedding)
labels = optimal_gmm.predict(embedding)
   
# Plot the UMAP projections with optimal GMM clusters
plt.scatter(embedding[:,0], embedding[:,1], c=labels, cmap='viridis', s=5)
plt.title(f'All Sessions: UMAP projection with GMM ({optimal_n_components} clusters)')
umap_all_path = os.path.join(umap_dir, 'all_sessions_umap.png')
plt.savefig(umap_all_path)
plt.clf()

# Plot BIC values for a range of cluster sizes
plt.plot(n_components_range, bic_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('BIC')
plt.title('All Sessions: BIC Scores')
bic_all_path = os.path.join(umap_dir, 'all_sessions_bic.png')
plt.savefig(bic_all_path)
plt.clf()   

# Initialize lists to save relevant calculated values
optimal_cluster_list = []
session_size_list = []

iterations = 20

# UMAP with GMM on a session-by-session basis
for session in df.session_ind.unique():
    for i in range(iterations):

        # Filter data for the current session
        mtm_session_bool = mtm_df.session_ind.astype(str).str.contains(str(session))
        mtm_session_df = mtm_df.loc[mtm_session_bool]
        mtm_session_features = np.stack(mtm_session_df.features.values)
    
        scaled_mtm_session = StandardScaler().fit_transform(mtm_session_features) # Scale features
        embedding = reducer.fit_transform(scaled_mtm_session) # UMAP embedding
        
        # Determine the optimal number of clusters using BIC
        bic_scores = []
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(embedding)
            bic = gmm.bic(embedding)
            bic_scores.append(bic)
        
        # Find the number of components with the lowest BIC
        optimal_n_components = n_components_range[np.argmin(bic_scores)]
        print(f'Session {session}: Optimal number of clusters is {optimal_n_components}')
        
        # Store the optimal clusters number and the number of MTMs within a session
        optimal_cluster_list.append(optimal_n_components)
        session_size_list.append(len(mtm_session_df))
        
        # Fit the GMM with the optimal number of clusters
        optimal_gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
        optimal_gmm.fit(embedding)
        labels = optimal_gmm.predict(embedding)
        
        # For speed, only create individual session plots for the first iteration
        if i == 0:
            # Plot the UMAP projections with optimal GsMM clusters
            plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=5)
            plt.title(f'Session {session}: UMAP projection with GMM ({optimal_n_components} clusters)')
            plt.colorbar(label='GMM Cluster')
            umap_session_path = os.path.join(umap_dir, f'session_{session}_umap.png')
            plt.savefig(umap_session_path)
            plt.clf()
            
            # Plot BIC values for a range of cluster sizes
            plt.plot(n_components_range, bic_scores, marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('BIC')
            plt.title(f'Session {session}: BIC Scores')
            bic_session_path = os.path.join(umap_dir, f'session_{session}_bic.png')
            plt.savefig(bic_session_path)
            plt.clf()

## Scatter plot of optimal cluster size vs sessions size (i.e. number of MTMs)
# Define jitter amount
jitter_strength = 0.5  # Adjust the strength of jitter as needed

# Add jitter to the session size and optimal clusters
session_size_jittered = np.array(session_size_list) + np.random.normal(0, jitter_strength, len(session_size_list))
optimal_cluster_jittered = np.array(optimal_cluster_list) + np.random.normal(0, jitter_strength, len(optimal_cluster_list))

# Make scatter plot
plt.scatter(session_size_jittered, optimal_cluster_jittered, c='cornflowerblue', marker='o')
plt.xlabel('Session Size')
plt.ylabel('Optimal Number of Clusters')
plt.title('Optimal Cluster Number vs Session Size')
scatter_plot_path = os.path.join(umap_dir, 'optimal_clusters_vs_session_size.png')
plt.savefig(scatter_plot_path)
plt.show()
    
## Histogram of ditribution of optimal cluster sizes across all iterations
plt.hist(optimal_cluster_list, bins=len(n_components_range), color='cornflowerblue', edgecolor='black')
plt.xlabel('Optimal Number of Clusters')
plt.ylabel('Frequency')
plt.title('Histogram of Optimal Cluster Numbers Across All Iterations')
histogram_path = os.path.join(umap_dir, 'optimal_clusters_histogram.png')
plt.savefig(histogram_path)
plt.show()
    
mode_result = stats.mode(optimal_cluster_list, keepdims=False)
print(f"The mode is: {mode_result[0]}")
    
    
    