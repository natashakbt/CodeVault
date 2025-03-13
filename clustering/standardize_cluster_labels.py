#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:06:52 2025

@author: natasha
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from scipy.stats import chi2_contingency
from scipy.stats import zscore
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
import shutil
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d  # for smoothing
from scipy.optimize import curve_fit
import piecewise_regression
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/clustering_data/'
file_path = os.path.join(dirname, 'clustering_df_update.pkl')
df = pd.read_pickle(file_path)

# Remove any data for df that does not have an associated transition time in scaled_mode_tau
df['basename'] = df['basename'].str.lower() # All basenames to lowercase

# Manually removed this sepcific data:
df = df[~((df['basename'] == 'km50_5tastes_emg_210911_104510_copy') & (df['taste'] == 1))]
df = df[~((df['basename'] == 'km50_5tastes_emg_210911_104510_copy') & (df['taste'] == 4))]

    
# ==============================================================================
# Setup color map
# ==============================================================================

# Define a color mapping for specific cluster numbers
color_mapping = {
    -1.0: '#ff9900',  # Gapes Color for cluster -1
    -2.0: '#D3D3D3'   # No movement Color for cluster -2
}

# Generate unique colors for basenames
basename_list = df['basename'].unique()
basename_colors = plt.cm.viridis_r(np.linspace(0, 1, len(basename_list)))
basename_color_map = dict(zip(basename_list, basename_colors))


# ==============================================================================
# Setup folder structure and clear any .png files in folders
# ============================================================================== 
# Create folder for saving plots
label_dir = os.path.join(dirname, 'cluster_label_standardization')
gapes_dir = os.path.join(label_dir, 'gapes')
nothing_dir = os.path.join(label_dir, 'nothing')

for folder in [label_dir, gapes_dir, nothing_dir]:
    os.makedirs(folder, exist_ok=True)
    all_files = glob.glob(os.path.join(folder, '*.png'))
    for file in all_files:
        os.remove(file)
        
# ==============================================================================
# Function
# ==============================================================================       
def standardize_labels(this_vector_df, next_basename):

    for cluster in cluster_range:

        filtered_df = df[(df['basename'] == next_basename) & (df['cluster_num'] == cluster)]
        feature_matrix = np.vstack(filtered_df['features'].values)  # Stack rows into a matrix
        next_avg_vector = np.mean(feature_matrix, axis=0) # Compute the average vector
        next_vector_df.at[next_vector_df[next_vector_df['clust_num'] == cluster].index[0], 'avg_vector'] = next_avg_vector.tolist()
    
    this_vectors = np.vstack(this_vector_df['avg_vector'].values)
    next_vectors = np.vstack(next_vector_df['avg_vector'].values)
    
    # Compute pairwise cosine similarity between every vector in `this_vectors` and `next_vectors`
    similarity_matrix = cosine_similarity(this_vectors, next_vectors)

    similarity_df = pd.DataFrame(similarity_matrix, 
                             index=this_vector_df['clust_num'], 
                             columns=next_vector_df['clust_num'])
    
    plt.figure(figsize=(8, 6))  # Optional: adjust the size of the figure
    sns.heatmap(similarity_df, annot=True, cmap='coolwarm', cbar=True, fmt=".2f", 
                xticklabels=next_vector_df['clust_num'], yticklabels=this_vector_df['clust_num'])
    
    # Add labels and title
    plt.title('Cosine Similarity Heatmap')
    plt.xlabel('Next Vector')
    plt.ylabel('This Vector')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    similarity_list = similarity_matrix.flatten().tolist()
    
    this_used = []
    next_used = []
    print("new comparison")
    while len(this_used) < len(cluster_range):
        
        max_num = np.max(similarity_list)

        max_index = np.argwhere(similarity_matrix == max_num)[0]

        # Step 3: Use these indices to find the corresponding cluster numbers
        this_cluster = this_vector_df['clust_num'].iloc[max_index[0]]
        next_cluster = next_vector_df['clust_num'].iloc[max_index[1]]
        
        if this_cluster not in this_used and next_cluster not in next_used:
            df.loc[(df['basename'] == next_basename) & (df['cluster_num'] == next_cluster), 'new_cluster_num'] = this_cluster
            print(this_cluster, next_cluster)
        
            this_used.append(this_cluster)
            next_used.append(next_cluster)
        similarity_list.remove(max_num)

    
    for cluster in cluster_range:
        one_filtered_df = df[(df['basename'] == basename) & (df['cluster_num'] == cluster)]
        two_filtered_df = df[(df['basename'] == next_basename) & (df['cluster_num'] == cluster)]
        combined_df = pd.concat([one_filtered_df, two_filtered_df])
        
        # Stack rows into a matrix
        feature_matrix = np.vstack(combined_df['features'].values)  
        
        # Compute the average vector
        this_avg_vector = np.mean(feature_matrix, axis=0).astype(object)
        this_vector_df.at[this_vector_df[this_vector_df['clust_num'] == cluster].index[0], 'avg_vector'] = this_avg_vector.tolist()

    
    return this_vector_df

# %%%
# ==============================================================================
# Plot waveforms by cluster and basename
# ==============================================================================
cluster_basename_groups = df.groupby(['cluster_num', 'basename'])
for (cluster, basename), group in cluster_basename_groups:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use predefined colors for -1.0 and -2.0 clusters, otherwise assign colors by basename
    color = color_mapping.get(cluster, basename_color_map.get(basename, 'black'))
    
    for segment in group['segment_norm_interp']:
        ax.plot(segment, alpha=0.1, color=color)
    
    ax.set_title(f'Cluster {cluster} - {basename} Waveforms')
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    
    # Save plot
    plot_filename = f'{basename}_cluster{cluster}.png'
    if cluster == -1.0:
        plot_path = os.path.join(label_dir, 'gapes', plot_filename)
        plt.savefig(plot_path)
        plt.close(fig) 
    elif cluster == -2.0:
        plot_path = os.path.join(label_dir, 'nothing', plot_filename)
        plt.savefig(plot_path)
        plt.close(fig)
    else:
        plot_path = os.path.join(label_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close(fig)


# %%%
# ==============================================================================
# Plot waveforms by cluster and basename
# ==============================================================================

df['new_cluster_num'] = np.nan

unique_basename = df['basename'].unique().tolist()
cluster_range = df['cluster_num'].unique()[df['cluster_num'].unique() >= 0].tolist()
cluster_range.sort()


initialize_this_vector = {'clust_num': cluster_range, 'avg_vector': np.nan}
this_vector_df = pd.DataFrame(data=initialize_this_vector, dtype=object)

initialize_next_vector = {'clust_num': cluster_range, 'avg_vector': np.nan}
next_vector_df = pd.DataFrame(data=initialize_next_vector, dtype=object)



### CODE BELOW IS EXPERIMENTAL
basename = unique_basename[0]
next_basename = unique_basename[1]

filtered_df = df[(df['basename'] == basename) & (df['cluster_num'] == cluster)]
feature_matrix = np.vstack(filtered_df['features'].values)  # Stack rows into a matrix
this_avg_vector = np.mean(feature_matrix, axis=0).astype(object) # Compute the average vector
this_vector_df.at[this_vector_df[this_vector_df['clust_num'] == cluster].index[0], 'avg_vector'] = this_avg_vector.tolist()

standardize_labels(this_vector_df, next_basename)



'''
### THIS CODE WORKS
for basename in unique_basename[:1]:
    idx = unique_basename.index(basename)
    next_basename = unique_basename[idx+1]
    for cluster in cluster_range:
        filtered_df = df[(df['basename'] == basename) & (df['cluster_num'] == cluster)]
        feature_matrix = np.vstack(filtered_df['features'].values)  # Stack rows into a matrix
        this_avg_vector = np.mean(feature_matrix, axis=0).astype(object) # Compute the average vector
        this_vector_df.at[this_vector_df[this_vector_df['clust_num'] == cluster].index[0], 'avg_vector'] = this_avg_vector.tolist()



        filtered_df = df[(df['basename'] == next_basename) & (df['cluster_num'] == cluster)]
        feature_matrix = np.vstack(filtered_df['features'].values)  # Stack rows into a matrix
        next_avg_vector = np.mean(feature_matrix, axis=0) # Compute the average vector
        next_vector_df.at[next_vector_df[next_vector_df['clust_num'] == cluster].index[0], 'avg_vector'] = next_avg_vector.tolist()
    
    this_vectors = np.vstack(this_vector_df['avg_vector'].values)
    next_vectors = np.vstack(next_vector_df['avg_vector'].values)
    
    # Compute pairwise cosine similarity between every vector in `this_vectors` and `next_vectors`
    similarity_matrix = cosine_similarity(this_vectors, next_vectors)

    similarity_df = pd.DataFrame(similarity_matrix, 
                             index=this_vector_df['clust_num'], 
                             columns=next_vector_df['clust_num'])
    
    plt.figure(figsize=(8, 6))  # Optional: adjust the size of the figure
    sns.heatmap(similarity_df, annot=True, cmap='coolwarm', cbar=True, fmt=".2f", 
                xticklabels=next_vector_df['clust_num'], yticklabels=this_vector_df['clust_num'])
    
    # Add labels and title
    plt.title('Cosine Similarity Heatmap')
    plt.xlabel('Next Vector')
    plt.ylabel(f'This Vector (basename #{idx})')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    similarity_list = similarity_matrix.flatten().tolist()
    this_used = []
    next_used = []
    print("new comparison")
    while len(this_used) < len(cluster_range):
        
        max_num = np.max(similarity_list)

        max_index = np.argwhere(similarity_matrix == max_num)[0]

        # Step 3: Use these indices to find the corresponding cluster numbers
        this_cluster = this_vector_df['clust_num'].iloc[max_index[0]]
        next_cluster = next_vector_df['clust_num'].iloc[max_index[1]]
        
        if this_cluster not in this_used and next_cluster not in next_used:
            df.loc[(df['basename'] == next_basename) & (df['cluster_num'] == next_cluster), 'new_cluster_num'] = this_cluster
            print(this_cluster, next_cluster)
        
            this_used.append(this_cluster)
            next_used.append(next_cluster)
        similarity_list.remove(max_num)



for cluster in cluster_range:
    one_filtered_df = df[(df['basename'] == basename) & (df['cluster_num'] == cluster)]
    two_filtered_df = df[(df['basename'] == next_basename) & (df['cluster_num'] == cluster)]
    combined_df = pd.concat([one_filtered_df, two_filtered_df])
    
    # Stack rows into a matrix
    feature_matrix = np.vstack(combined_df['features'].values)  
    
    # Compute the average vector
    this_avg_vector = np.mean(feature_matrix, axis=0).astype(object)
    this_vector_df.at[this_vector_df[this_vector_df['clust_num'] == cluster].index[0], 'avg_vector'] = this_avg_vector.tolist()

'''






# %%
for basename in unique_basename[2:]:
    print(basename)
    this_vector_df =  standardize_labels(this_vector_df, next_basename)
    
    




