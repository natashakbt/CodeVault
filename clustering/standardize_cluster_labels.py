  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:06:52 2025

@author: natasha
"""

import umap
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from scipy.stats import chi2_contingency
from scipy.stats import zscore
import seaborn as sns
#from matplotlib.legend_handler import# WHAT WAS HERE? 
import shutil
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d  # for smoothing
from scipy.optimize import curve_fit
import piecewise_regression
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import scipy.stats as stats
import scikit_posthocs as sp
import pingouin as pg
from scipy.stats import tukey_hsd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import sem  # Standard error of the mean
import umap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde

# TODO: I THINK THIS CODE DOESN'T WORK AFTER THE LABELS HAVE BEEN STANDARDIZED ALREADY.
# MAKE IT RE-RUN-ABLE FRIENDLY

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
# Function to standardize MTM cluster labels
# Compare cosine similarity of average feature vectors between two test sessions
# Standardize the labels based on maximizing the sum of cosine similarity values
# ==============================================================================      

def standardize_labels(this_vector_df, next_basename, processed_basenames):
    
    processed_basenames.append(next_basename)
    
    # Build dataframe of average feature vector for each cluster (row)
    for cluster in cluster_range:

        filtered_df = df[(df['basename'] == next_basename) & (df['cluster_num'] == cluster)]
        feature_matrix = np.vstack(filtered_df['features'].values)  # Stack rows into a matrix
        next_avg_vector = np.mean(feature_matrix, axis=0) # Compute the average vector
        next_vector_df.at[next_vector_df[next_vector_df['clust_num'] == cluster].index[0], 'avg_vector'] = next_avg_vector.tolist()
    
    this_vectors = np.vstack(this_vector_df['avg_vector'].values)
    next_vectors = np.vstack(next_vector_df['avg_vector'].values)
    
    # Compute pairwise cosine similarity between every feature vector
    similarity_matrix = cosine_similarity(this_vectors, next_vectors)
    # Matrix of similarity 
    similarity_df = pd.DataFrame(similarity_matrix, 
                             index=this_vector_df['clust_num'], 
                             columns=next_vector_df['clust_num'])
    # Plot similarity matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_df, annot=True, cmap='coolwarm', cbar=True, fmt=".2f", 
                xticklabels=next_vector_df['clust_num'], yticklabels=this_vector_df['clust_num'])
    
    plt.title(f'{next_basename}')
    plt.xlabel('Next Vector')
    plt.ylabel('This Vector')

    plt.tight_layout()
    plt.show()
    
    # Re-order indices of cluster labels such that it produces the highest summed cosine simiarity value
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

    # Apply the optimal cluster label mapping
    for i in range(len(row_ind)):
        this_cluster = this_vector_df.iloc[row_ind[i]]['clust_num']
        next_cluster = next_vector_df.iloc[col_ind[i]]['clust_num']
        df.loc[(df['basename'] == next_basename) & (df['cluster_num'] == next_cluster), 'new_cluster_num'] = this_cluster
        print(f"{next_cluster} → {this_cluster}")

    # Update average vectors to be a combination of the two sessions
    for cluster in cluster_range:
        combined_df = pd.concat([
            df[(df['basename'] == b) & (df['cluster_num'] == cluster)] for b in processed_basenames
        ])
        
        if not combined_df.empty:
            feature_matrix = np.vstack(combined_df['features'].values)  
            this_avg_vector = np.mean(feature_matrix, axis=0).astype(object)
            this_vector_df.at[this_vector_df[this_vector_df['clust_num'] == cluster].index[0], 'avg_vector'] = this_avg_vector.tolist()

    return this_vector_df, processed_basenames

# %%% Plot waveforms per cluster label
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
# Setup color map
# ==============================================================================

# Colros for gapes and no movement is set
color_mapping = {
    -1.0: '#ff9900',  # Gapes Color for cluster -1
    -2.0: '#D3D3D3'   # No movement Color for cluster -2
}

# Generate unique colors for basenames
basename_list = df['basename'].unique()
basename_colors = plt.cm.viridis_r(np.linspace(0, 1, len(basename_list)))
basename_color_map = dict(zip(basename_list, basename_colors))



# ==============================================================================
# Plot waveforms divided cluster and basename
# ==============================================================================
cluster_basename_groups = df.groupby(['cluster_num', 'basename'])
for (cluster, basename), group in cluster_basename_groups:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use predefined colors for -1.0 and -2.0 clusters, otherwise assign colors by basename
    color = color_mapping.get(cluster, basename_color_map.get(basename, 'black'))
    
    for segment in group['segment_raw']:
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


# %%% Standardize Cluster Labels
# ==============================================================================
# Standardize Cluster Labels
# ==============================================================================

# Initialize important things
unique_basename = df['basename'].unique().tolist()
cluster_range = df['cluster_num'].unique()[df['cluster_num'].unique() >= 0].tolist()
cluster_range.sort()

df['new_cluster_num'] = np.nan
df.loc[df['cluster_num'] < 0, 'new_cluster_num'] = df['cluster_num']
df.loc[df['basename'] == unique_basename[0], 'new_cluster_num'] = df['cluster_num']


initialize_this_vector = {'clust_num': cluster_range, 'avg_vector': np.nan}
this_vector_df = pd.DataFrame(data=initialize_this_vector, dtype=object)

initialize_next_vector = {'clust_num': cluster_range, 'avg_vector': np.nan}
next_vector_df = pd.DataFrame(data=initialize_next_vector, dtype=object)

processed_basenames = [unique_basename[0]]

# Standardize first and second test sessions first
basename = unique_basename[0]
next_basename = unique_basename[1]
for cluster in cluster_range:
    filtered_df = df[(df['basename'] == basename) & (df['cluster_num'] == cluster)]
    feature_matrix = np.vstack(filtered_df['features'].values)  # Stack rows into a matrix
    this_avg_vector = np.mean(feature_matrix, axis=0).astype(object) # Compute the average vector
    this_vector_df.at[this_vector_df[this_vector_df['clust_num'] == cluster].index[0], 'avg_vector'] = this_avg_vector.tolist()
print(unique_basename[1])
this_vector_df, processed_basenames = standardize_labels(this_vector_df, next_basename, processed_basenames)


# Standardize all other test sessions sequentially
for basename in unique_basename[1:-1]:
    idx = unique_basename.index(basename)
    next_basename = unique_basename[idx+1]
    print(next_basename)
    this_vector_df, processed_basenames =  standardize_labels(this_vector_df, next_basename, processed_basenames)


# %% Check dataframe is good, then overwrite and save

# First testing that 'df' dataframe looks OK
unique_clusters_by_basename = df.groupby('basename')['new_cluster_num'].unique()

if any(pd.isna(unique_clusters_by_basename)): # Check if there are any rows with unassigned cluster numbers
    print("Warning: there are NaN values in new_cluster_num")
lengths = [len(unique_clusters) for unique_clusters in unique_clusters_by_basename] # Check that all test sessions have the same number of cluster labels
if len(set(lengths)) != 1:
    print("Warning: Number of cluster labels vary by test session", lengths)

# Rename cluster-related column names
df.rename(columns={'cluster_num': 'old_cluster_num'}, inplace=True)
df.rename(columns={'new_cluster_num': 'cluster_num'}, inplace=True)


# Overwrite and safe dataframe
df.to_pickle(file_path) # Overwrite and save dataset



# %% Plot overlapped waveforms combined per cluster label
# ==============================================================================
# Setup folder structure and clear any .png files in folders
# ============================================================================== 
# Create folder for saving plots
overlap_dir = os.path.join(dirname, 'combined_overlap_clusters')


for folder in [overlap_dir]:
    os.makedirs(folder, exist_ok=True)
    all_files = glob.glob(os.path.join(folder, '*.png'))
    for file in all_files:
        os.remove(file)

# Define a color mapping for cluster numbers
color_mapping = {
    -1: '#ff9900',      # Gapes Color for cluster -1
    -2: '#D3D3D3',      # No mvoement Color for cluster 0
     0: '#4285F4',     # Color for cluster 1
     1: '#88498F',    # Color for cluster 2
     2: '#0CBABA'        # Color for cluster 3
}    

# ==============================================================================
# Plot waveforms divided cluster and basename
# TWO DIFFERENT VERSIONS. DECIDING WHICH IS THE BEST
# ==============================================================================

cluster_basename_groups = df.groupby(['cluster_num'])
for (cluster), group in cluster_basename_groups:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color map defined above
    color = color_mapping.get(cluster, basename_color_map.get(cluster, 'black'))
    
    waveforms = np.vstack(group['segment_norm_interp'].values)  # Assuming each is a NumPy array of the same length
    avg_waveform = np.mean(waveforms, axis=0)  # Compute the mean waveform

    
    for row in group.iterrows():
        #max_amp = max(row[1]['segment_raw'])
        #scaling_factor = row[1]['raw_features'][4]
        #segment = (row[1]['segment_raw'])/max_amp * scaling_factor
        #ax.plot(segment, alpha=0.1, color=color)
        ax.plot(row[1]['segment_norm_interp'], alpha=0.1, color=color)
    
    ax.plot(avg_waveform, color='black', linestyle='dashed', linewidth=2, label="Average Waveform")
    
    ax.set_title(f'Cluster {cluster} Waveforms')
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    #ax.set_ylim(0, 20)
    # Save plot
    plot_filename = f'norm_cluster{cluster}.png'

    plot_path = os.path.join(overlap_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)



cluster_basename_groups = df.groupby(['cluster_num'])
for cluster, group in cluster_basename_groups:
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color map defined above
    color = color_mapping.get(cluster, basename_color_map.get(basename, 'black'))
    
    # List to store normalized waveforms
    normalized_waveforms = []
    max_length = 0  # Track max length of waveforms
    
    for _, row in group.iterrows():
        max_amp = max(row['segment_raw'])
        scaling_factor = row['raw_features'][4]
        segment = (row['segment_raw']) / max_amp * scaling_factor
        segment = (row['segment_raw'])
        max_length = max(max_length, len(segment))  # Update max length
        normalized_waveforms.append(segment)  # Collect for averaging
        
        ax.plot(segment, alpha=0.1, color=color)  # Individual waveforms
    
    # Pad waveforms to the same length
    #padded_waveforms = [np.pad(w, (0, max_length - len(w)), mode='constant') for w in normalized_waveforms]
    padded_waveforms = [ np.pad(w, ((max_length - len(w)) // 2, (max_length - len(w) + 1) // 2), mode='constant') for w in normalized_waveforms ]

    # Compute and plot the average waveform
    if padded_waveforms:
        avg_waveform = np.mean(padded_waveforms, axis=0)  # Compute mean across waveforms
        ax.plot(avg_waveform, 'k--', linewidth=2, label="Avg Waveform")  # Plot in black dotted line
    
    ax.set_title(f'Cluster {cluster} Waveforms')
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(0, 500)
    ax.legend()

    # Save plot
    plot_filename = f'cluster{cluster}.png'
    plot_path = os.path.join(overlap_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)



fig, ax = plt.subplots(figsize=(10, 6))
filtered_df = df[df['cluster_num'] >= 0]

for cluster, group in filtered_df.groupby(['cluster_num']):
    color = color_mapping.get(cluster, basename_color_map.get(cluster, 'black'))
    # Stack waveforms and compute mean & standard error
    waveforms = np.vstack(group['segment_norm_interp'].values)
    avg_waveform = np.mean(waveforms, axis=0)  
    #std_err = sem(waveforms, axis=0)  
    std_err = np.std(waveforms, axis=0)  # Compute standard deviation

    ax.plot(avg_waveform, color='k', label=f'Cluster {cluster}')
    ax.fill_between(range(len(avg_waveform)), avg_waveform - std_err, avg_waveform + std_err, color=color, alpha=0.3)

ax.set_title('Cluster Average Waveforms with Standard Error')
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.legend()
plot_path = os.path.join(overlap_dir, 'avg_waveforms.png')

#plt.savefig(plot_path)

plt.show()
plt.close(fig)





fig, ax = plt.subplots(figsize=(10, 6))
filtered_df = df[df['cluster_num'] >= 0]

# Find max waveform length
max_length = max(len(w) for w in filtered_df['segment_raw'])

for cluster, group in filtered_df.groupby(['cluster_num']):
    color = color_mapping.get(cluster, basename_color_map.get(cluster, 'black'))

    # Pad waveforms evenly at the beginning and end
    waveforms = [
        np.pad(w, ((max_length - len(w)) // 2, (max_length - len(w)) - (max_length - len(w)) // 2), mode='constant')
        for w in group['segment_raw']
    ]
    waveforms = np.vstack(waveforms)

    # Compute mean waveform
    avg_waveform = np.mean(waveforms, axis=0)

    ax.plot(avg_waveform, color=color, label=f'Cluster {cluster}')

ax.set_title('Cluster Average Waveforms')
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.legend()
plt.show()
plt.close(fig)

# %% UMAP

# ==============================================================================
# Plot UMAP of waveforms, clustered by GMM 
# ==============================================================================

# See if multimodal waveforms cluster with UMAP reduction
reducer = umap.UMAP()
filtered_df = df[df['cluster_num'] >= 0]
waveforms = filtered_df['segment_norm_interp'].tolist()
#scaled_waveforms = StandardScaler().fit_transform(waveforms)
embedding = reducer.fit_transform(waveforms) # UMAP embedding

# Fit the GMM with the optimal number of clustersz
optimal_gmm = GaussianMixture(n_components=3, random_state=42)
optimal_gmm.fit(embedding)
labels = optimal_gmm.predict(embedding)

# Add cluster number label to df dataframe
filtered_df['gmm_cluster'] = labels

# Create a list of colors for each point in the embedding based on its cluster_num
cluster_nums = filtered_df['gmm_cluster'].tolist()

# Create the scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_nums, cmap='hot', s=20, edgecolor='k', alpha=0.5)
plt.title('GMM on UMAP of Waveforms')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

# Optionally, add a legend or colorbar
plt.show()



# ==============================================================================
# Plot UMAP of waveforms, clustered by k-means 
# ==============================================================================
  
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
filtered_df['kmeans_cluster'] = kmeans.fit_predict(embedding)

# Prepare for Plotting
cluster_nums = filtered_df['kmeans_cluster'].tolist()

# Create Scatter Plot
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_nums, cmap='tab20', s=20, edgecolor='k', alpha=0.5)
plt.title('K-Means Clustering on UMAP of Waveforms')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

# Optionally, add a colorbar

plt.show()


# ==============================================================================
# Plot UMAP, color by standardized cluster label
# ==============================================================================
# Define a color mapping for cluster numbers
color_mapping = {
    -1: '#ff9900',      # Gapes Color for cluster -1
    -2: '#D3D3D3',      # No mvoement Color for cluster 0
     0: '#4285F4',     # Color for cluster 1
     1: '#88498F',    # Color for cluster 2
     2: '#0CBABA'        # Color for cluster 3
}    

# Create a list of colors for each point in the embedding based on its cluster_num
cluster_nums = filtered_df['cluster_num'].tolist()  # Assuming 'cluster_num' column exists
colors = [color_mapping[cluster] for cluster in cluster_nums]

# Create the scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=50, edgecolor='k', alpha=0.7)
plt.title('UMAP Scatter Plot of Waveforms')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

# Optionally, add a legend or colorbar
plt.show()


# ============================================= =================================
# Plotting just specific cluster_num
# ==============================================================================
prototypical_waveforms = []
center_points = []
cluster_cycle = [0, 1, 2]
reset_filtered_df = filtered_df.reset_index(drop=True)
for single_cluster_num in cluster_cycle:
    # Filter for a specific cluster
    cluster_0_df = reset_filtered_df[reset_filtered_df['cluster_num'] == single_cluster_num]
    cluster_0_indices = cluster_0_df.index
    
    # Filter the UMAP embedding array to only include those rows
    embedding_0 = embedding[cluster_0_indices]
    
    # Define color just for cluster 0
    color_0 = color_mapping[single_cluster_num]
    colors_0 = [color_0] * len(cluster_0_df)
    
    # Create scatter plot of waveforms in UMAP space
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding_0[:, 0], embedding_0[:, 1], c=colors_0, s=50, edgecolor='k', alpha=0.7)
    plt.title(f'UMAP Scatter Plot - Cluster {single_cluster_num} Only')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()
    
    '''
    # KDE density
    umap_df = pd.DataFrame({
        'UMAP1': embedding_0[:, 0],
        'UMAP2': embedding_0[:, 1]
    })
    
    plt.figure(figsize=(10, 8))
    sns.kdeplot(
        data=umap_df, x='UMAP1', y='UMAP2',
        fill=True, thresh=0.05, levels=100, cmap="Blues"
    )
    plt.title(f'2D KDE of Cluster {single_cluster_num} in UMAP Space')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()
    '''
    
    ### PLOT KDE WITH CENTER
    x = embedding_0[:, 0]
    y = embedding_0[:, 1]
    
    # Fit KDE
    kde = gaussian_kde(np.vstack([x, y]))
    
    # Evaluate KDE on a grid
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)
    
    # Find location of maximum density
    max_idx = np.unravel_index(np.argmax(density), density.shape)
    x_center = xx[max_idx]
    y_center = yy[max_idx]
    center_points.append([x_center, y_center])

    # Plot KDE
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, c=color_mapping[0], s=30, alpha=0.5, label='Points')
    plt.contourf(xx, yy, density, levels=100, cmap='Blues', alpha=0.6)
    plt.plot(x_center, y_center, 'ro', label='Density Peak (Center)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(f'Cluster {single_cluster_num} KDE')
    plt.legend()
    plt.show()
    

    ### FIND REAL WAVEFORM CLOSEST TO THE DENSITY CENTER
    # Coordinates of center of density (from previous step)
    center = np.array([x_center, y_center])
    # Compute distances to center
    distances = np.linalg.norm(embedding_0 - center, axis=1)
    
    # Index of the closest point
    closest_index = np.argmin(distances)
    
    # Coordinates of the closest point
    closest_point = embedding_0[closest_index]

    closest_row = cluster_0_df.iloc[closest_index]

    prototypical_waveforms.append({
        'cluster': single_cluster_num,
        'segment': closest_row['segment_norm_interp']
    })

#### PLOTTING PROTOTYPICAL WAVEOFORM (BASED ON KDE CENTER) FOR EACH CLUSTER
plt.figure(figsize=(10, 7))

for i, wf in enumerate(prototypical_waveforms):
    cluster_id = wf['cluster']
    segment = wf['segment']
    color = color_mapping[cluster_id]
    
    plt.plot(segment, color=color, 
             linewidth = 3.5,
             label=f'Cluster {cluster_id}')

plt.title('Prototypical Segments by Cluster')
plt.xlabel('Time (ms)')
plt.ylabel('Norm. Amplitude')
plt.show()




# Create the scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=50, edgecolor='k', alpha=0.7)
for point in center_points:
    x_center = point[0]
    y_center = point[1]
    plt.scatter(x_center, y_center, 
                c='red', s=300, 
                edgecolor='k', linewidth=2.5)
plt.title('UMAP Scatter Plot of Waveforms with KDE peak')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

# Optionally, add a legend or colorbar
plt.show()

# ==============================================================================
# 3D UMAP
# ==============================================================================
# Initialize UMAP reducer with 3 components for 3D embedding
reducer = umap.UMAP(n_components=3)

# Filter valid clusters
filtered_df = df[df['cluster_num'] >= 0]
waveforms = filtered_df['segment_norm_interp'].tolist()

# Compute 3D UMAP embedding
embedding = reducer.fit_transform(waveforms)

# Define color mapping for clusters
color_mapping = {
    -1: '#ff9900',  # Gapes Color for cluster -1
    -2: '#D3D3D3',  # No movement Color for cluster -2
     0: '#4285F4',  # Color for cluster 0
     1: '#88498F',  # Color for cluster 1
     2: '#0CBABA'   # Color for cluster 2
}

# Map cluster numbers to colors
cluster_nums = filtered_df['cluster_num'].tolist()
colors = [color_mapping[cluster] for cluster in cluster_nums]

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with cluster colors
sc = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                c=colors, s=50, edgecolor='k', alpha=0.7)

# Labels and title
ax.set_title('3D UMAP Scatter Plot of Waveforms')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')

plt.show()

   

# %% PLOTS TO SEE IF NEWLY STANDARDIZED CLUSTERS LOOK DIFFERENT - IN PROGRESS

# ==============================================================================
# Plot heatmap. Each column is a feature and each row is a waveform's feature vector
# ==============================================================================

# Just MTM
color_mapping = ['#4285F4','#88498F','#0CBABA']  

# All behaviors
#color_mapping = [ '#D3D3D3','#ff9900', '#4285F4','#88498F','#0CBABA']  


feature_names = [
    "duration",
    "left_interval",
    "right_interval",
    "max_freq",
    "amplitude_norm",
    "pca_0",
    "pca_1",
    "pca_2"
]

features_expanded = pd.DataFrame(df["features"].tolist(), index=df.index, columns=feature_names)
# Add cluster_num column at the front
features_expanded.insert(0, "cluster_num", df["cluster_num"])

# Sort the rows by cluster number
features_expanded = features_expanded.sort_values(by="cluster_num")

# Remove rows where cluster num is a negative value
features_expanded = features_expanded.loc[features_expanded['cluster_num'] >= 0]


plt.figure(figsize=(12, 8))
data1 = features_expanded.copy()
data1.loc[:, features_expanded.columns != 'cluster_num'] = float('nan')
ax = sns.heatmap(data1, cmap=color_mapping)
data2 = features_expanded.copy()
data2['cluster_num'] = float('nan')
sns.heatmap(data2, yticklabels=False, cmap='viridis', vmax=3)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
plt.xlabel("Features", fontsize=16)
plt.ylabel("MTM clusters", fontsize=16)

plt.show()
plt.close()



plot_data = features_expanded[["cluster_num", "pca_0", "pca_1"]]


# Calculate Euclidean norm of [pca_0, pca_1]
plot_data['pca_magnitude'] = np.linalg.norm(plot_data[['pca_0', 'pca_1']].values, axis=1)

# Sort by cluster_num, then by magnitude
plot_data = plot_data.sort_values(by=["cluster_num", "pca_magnitude"])

# Drop helper column before plotting
plot_data = plot_data[["cluster_num", "pca_0", "pca_1"]]


plt.figure(figsize=(12, 8))
data1 = plot_data.copy()
data1.loc[:, plot_data.columns != 'cluster_num'] = float('nan')
ax = sns.heatmap(data1, cmap=color_mapping,
                 square=False, linewidths=0, linecolor='none', 
                 cbar=True, rasterized=True)

data2 = plot_data.copy()
data2['cluster_num'] = float('nan')
sns.heatmap(data2, yticklabels=False, cmap='viridis', vmax=3)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
plt.xlabel("Features", fontsize=16)
plt.ylabel("MTM clusters", fontsize=16)

plt.show()
plt.close()


# Create padded version of plot_data
wide_plot_data = pd.DataFrame()
wide_plot_data["cluster_num"] = plot_data["cluster_num"]

# Duplicate PCA columns for width
wide_plot_data["pca_0_a"] = plot_data["pca_0"]
wide_plot_data["pca_0_b"] = plot_data["pca_0"]
wide_plot_data["pca_1_a"] = plot_data["pca_1"]
wide_plot_data["pca_1_b"] = plot_data["pca_1"]

# Heatmap 1: Show only cluster_num column
data1 = wide_plot_data.copy()
for col in data1.columns:
    if col != 'cluster_num':
        data1[col] = np.nan

# Heatmap 2: Show PCA columns only
data2 = wide_plot_data.copy()
data2["cluster_num"] = np.nan

# Plot
plt.figure(figsize=(10, 10))
ax = sns.heatmap(data1, cmap=color_mapping,
                 cbar=False, linewidths=0, linecolor='none', 
                 rasterized=True)

sns.heatmap(data2, cmap='viridis', vmax=3,
            cbar=True, yticklabels=False, linewidths=0, linecolor='none', 
            rasterized=True)

# Adjust labels manually
ax.set_xticks([0.5, 2, 4])  # indexes of columns: cluster_num, pca_0_a, pca_1_a
ax.set_xticklabels(["Cluster", "PC0", "PC1"], fontsize=22)
#ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
#plt.xlabel("Features", fontsize=16)
#plt.ylabel("MTM clusters", fontsize=16)
plt.tight_layout()
plt.show()


# ==============================================================================
# Heatmap of waveforms
# ==============================================================================
color_mapping = ['#4285F4','#88498F','#0CBABA']  

features_expanded = pd.DataFrame(df["segment_norm_interp"].tolist(), index=df.index)
# Add cluster_num column at the front
features_expanded.insert(0, "cluster_num", df["cluster_num"])

# Sort the rows by cluster number
features_expanded = features_expanded.sort_values(by="cluster_num")

# Remove rows where cluster num is a negative value
features_expanded = features_expanded.loc[features_expanded['cluster_num'] >= 0]


plt.figure(figsize=(12, 8))
data1 = features_expanded.copy()
data1.loc[:, features_expanded.columns != 'cluster_num'] = float('nan')
ax = sns.heatmap(data1, cmap=color_mapping)
data2 = features_expanded.copy()
data2['cluster_num'] = float('nan')
sns.heatmap(data2, yticklabels=False, cmap='Spectral', vmax=3)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
plt.xlabel("Features", fontsize=16)
plt.ylabel("MTM clusters", fontsize=16)

plt.show()
plt.close()

# ==============================================================================
# Clustermap by features - not actually clustering (when row_cluster = False)
# ==============================================================================

# Define a color mapping for cluster numbers
color_mapping = ['#4285F4','#88498F','#0CBABA']  

features_expanded = pd.DataFrame(df["features"].tolist(), index=df.index)
features_expanded.columns = feature_names
# Add cluster_num column at the front
features_expanded.insert(0, "cluster_num", df["cluster_num"])

# Sort the rows by cluster number
features_expanded = features_expanded.sort_values(by="cluster_num")

# Remove rows where cluster num is a negative value
features_expanded = features_expanded.loc[features_expanded['cluster_num'] >= 0]


plt.figure(figsize=(12, 8))
clust_label = features_expanded.pop("cluster_num")
lut = dict(zip(clust_label.unique(), color_mapping))
row_colors = clust_label.map(lut)
sns.clustermap(features_expanded, row_colors=row_colors, cmap="jet", row_cluster = False)
plt.show()
plt.close()


# ==============================================================================
# Clustermap for PCA0 and PCA1 only - clustering
# ==============================================================================
# Define a color mapping for cluster numbers
color_mapping = ['#4285F4','#88498F','#0CBABA']  

features_expanded = pd.DataFrame(df["features"].tolist(), index=df.index)
features_expanded.columns = feature_names
# Add cluster_num column at the front
features_expanded.insert(0, "cluster_num", df["cluster_num"])

# Sort the rows by cluster number
features_expanded = features_expanded.sort_values(by="cluster_num")

# Remove rows where cluster num is a negative value
features_expanded = features_expanded.loc[features_expanded['cluster_num'] >= 0]

pca_for_clustermap = features_expanded[['pca_0', 'pca_1', 'cluster_num']]

plt.figure(figsize=(12, 8))
clust_label = pca_for_clustermap.pop("cluster_num")
lut = dict(zip(clust_label.unique(), color_mapping))
row_colors = clust_label.map(lut)
sns.clustermap(pca_for_clustermap, row_colors=row_colors, cmap="mako")
plt.show()
plt.close()

# ==============================================================================
# Scatter plot of PCA0 by PCA1 - colors by cluster
# ==============================================================================
color_mapping = {
     0: '#4285F4',  # Color for cluster 0
     1: '#88498F',  # Color for cluster 1
     2: '#0CBABA'   # Color for cluster 2
}

colors = features_expanded['cluster_num'].map(color_mapping)

plt.scatter(
    features_expanded['pca_0'], features_expanded['pca_1'], 
    c=colors,
    s=5
    #alpha=0.7
)

plt.xlabel('PCA 0')
plt.ylabel('PCA 1')
plt.show()

# %% STATS TO SEE IF NEWLY STANDARDIZED CLUSTERS ARE DIFFERENT - IN PROGRESS

# ==============================================================================
# Three variations of stats to see if features are significantly different across clusters
# ==============================================================================
feature_names = [
    "duration",
    "left_interval",
    "right_interval",
    "max_freq",
    "amplitude_norm",
    "pca_0",
    "pca_1",
    "pca_2"
]

features_expanded = pd.DataFrame(df["features"].tolist(), index=df.index, columns=feature_names)
# Add cluster_num column at the front
features_expanded.insert(0, "cluster_num", df["cluster_num"])

# Sort the rows by cluster number
features_expanded = features_expanded.sort_values(by="cluster_num")

# Remove rows where cluster num is a negative value
features_expanded = features_expanded.loc[features_expanded['cluster_num'] >= 0]


# ANOVA and on a random n=50 sample of waveforms for each feature
feature_dict = {key: [] for key in feature_names}

for _ in range(1000):
    for feature in feature_names:
        group_zero = features_expanded[features_expanded["cluster_num"] == 0][feature].sample(n=50)
        group_one = features_expanded[features_expanded["cluster_num"] == 1][feature].sample(n=50)
        group_two = features_expanded[features_expanded["cluster_num"] == 2][feature].sample(n=50)

        f_statistic, p_value = stats.f_oneway(group_zero, group_one, group_two)
        feature_dict[feature].append(p_value)
        #if p_value < 0.05:   
        #    print(f'{feature}: {p_value}')
        #else:
        #    print(f'{feature} not significant ({p_value})')
# Plot the distribution of p-values for each feature
for feature, p_values in feature_dict.items():
    plt.figure(figsize=(12, 8))
    sns.histplot(p_values, bins=10, kde=True)
    plt.axvline(x=0.05, color="red", linestyle="--")
    plt.title(f"P-Values for {feature}")
    plt.xlabel("p-value")
    plt.ylabel("Count")
    plt.show()



# ANOVA and Tukey HSD post-hoc
# I think the data is not always normal - so not as good of a test
for feature in feature_names:
    group_zero = features_expanded[features_expanded["cluster_num"] == 0][feature]
    group_one = features_expanded[features_expanded["cluster_num"] == 1][feature]
    group_two = features_expanded[features_expanded["cluster_num"] == 2][feature]
    data_ano = pd.DataFrame({
        "score": pd.concat([group_zero, group_one, group_two]).values,
        "group": ["0"] * len(group_zero) + ["1"] * len(group_one) + ["2"] * len(group_two)
    })
    
    anova_results = pg.anova(dv='score', between="group", data=data_ano, detailed=True)
    print(f"feature: {feature}\n", anova_results.round(3))
    if anova_results["p-unc"][0] < 0.05 and anova_results["np2"][0] > 0.06:
        print("significant and medium-large effect size")
        res = tukey_hsd(group_zero, group_one, group_two)
        #res = pairwise_tukeyhsd(data_ano["score"], data_ano["group"]) # Does this work for Tukey-Kramer?
        print(res)
    elif anova_results["p-unc"][0] < 0.05 and anova_results["np2"][0] > 0.01:
        print("significant but small effect size")
    print("\n")




# KRUSKAL WALLIS WITH DUNN'S POST-HOC: FOR NON-NORMAL DATA
# Most correct?
for feature in feature_names:
    h_stat, p_value = stats.kruskal(features_expanded[features_expanded["cluster_num"] == 0][feature],
                                    features_expanded[features_expanded["cluster_num"] == 1][feature],
                                    features_expanded[features_expanded["cluster_num"] == 2][feature])
    
    # Calculate Eta-squared for effect size
    k = 3  # number of groups (clusters)
    N = len(features_expanded)  # total number of observations
    eta_squared = (h_stat - k + 1) / (N - k)
    
    
    print(feature)
    print(f'{h_stat:.2f}, {p_value:.4f}, {eta_squared:.4f}\n')
    
    if p_value < 0.05 :
        if eta_squared > 0.06:  # Only proceed if Kruskal-Wallis is significant
            print(f"{feature} p-value: {p_value}")
            print(f"H-stat {h_stat.round(3)}, effect size: {eta_squared.round(4)}")
            
            # Perform Dunn’s test with Bonferroni correction
            dunn_results = sp.posthoc_dunn(features_expanded, val_col=feature, group_col="cluster_num", p_adjust="bonferroni")
            print(dunn_results)

        elif eta_squared > 0.01:
            print(f'{feature} is signficiant ({p_value}) but small effect size ({eta_squared.round(4)})')
        else:
            print(f'{feature} is signficiant ({p_value}) but inconsequential effect size')
    else:
        print(f'{feature} not significant ({p_value})')
    print('\n')




# %% - ARCHIVE


'''


def standardize_labels(this_vector_df, next_basename):
    # Build dataframe of average feature vector for each cluster (row)
    for cluster in cluster_range:

        filtered_df = df[(df['basename'] == next_basename) & (df['cluster_num'] == cluster)]
        feature_matrix = np.vstack(filtered_df['features'].values)  # Stack rows into a matrix
        next_avg_vector = np.mean(feature_matrix, axis=0) # Compute the average vector
        next_vector_df.at[next_vector_df[next_vector_df['clust_num'] == cluster].index[0], 'avg_vector'] = next_avg_vector.tolist()
    
    this_vectors = np.vstack(this_vector_df['avg_vector'].values)
    next_vectors = np.vstack(next_vector_df['avg_vector'].values)
    
    # Compute pairwise cosine similarity between every feature vector
    similarity_matrix = cosine_similarity(this_vectors, next_vectors)
    # Matrix of similarity 
    similarity_df = pd.DataFrame(similarity_matrix, 
                             index=this_vector_df['clust_num'], 
                             columns=next_vector_df['clust_num'])
    # Plot similarity matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_df, annot=True, cmap='coolwarm', cbar=True, fmt=".2f", 
                xticklabels=next_vector_df['clust_num'], yticklabels=this_vector_df['clust_num'])
    
    plt.title(f'{next_basename}')
    plt.xlabel('Next Vector')
    plt.ylabel('This Vector')

    plt.tight_layout()
    plt.show()
    
    ## Standardize labels by matching greatest similarities
    # Flatten similarity matrix (to find maximum value)
    similarity_list = similarity_matrix.flatten().tolist()
    
    this_used = []
    next_used = []

    while len(this_used) < len(cluster_range):
        
        max_num = np.max(similarity_list)
        max_index = np.argwhere(similarity_matrix == max_num)[0] # Find matrix index of maximum number

        # Use matrix index to find the corresponding cluster numbers
        this_cluster = this_vector_df['clust_num'].iloc[max_index[0]]
        next_cluster = next_vector_df['clust_num'].iloc[max_index[1]]
        
        # Check that the clusters have not been used already
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
'''







