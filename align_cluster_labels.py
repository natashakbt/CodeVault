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
import matplotlib
import matplotlib.pyplot as plt
import glob
from scipy.stats import chi2_contingency
from scipy.stats import zscore
import seaborn as sns
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
from scipy.stats import sem 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from mtm_analysis_config import dirname


# ==============================================================================
# Load data and get setup
# ==============================================================================
file_path = os.path.join(dirname, 'clustering_df_update_with_laser.pkl')
df = pd.read_pickle(file_path)


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
# Function to standardize MTM cluster labels
# > Compare cosine similarity of average feature vectors between two test sessions
# > Standardize the labels based on maximizing the sum of cosine similarity values
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


# %% Standardize Cluster Labels
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
# ============================================================================== 
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





# %% Different clusering within UMAP


# See if multimodal waveforms cluster with UMAP reduction
reducer = umap.UMAP()
filtered_df = df[df['cluster_num'] >= 0]
waveforms = filtered_df['segment_norm_interp'].tolist()
#scaled_waveforms = StandardScaler().fit_transform(waveforms)
embedding = reducer.fit_transform(waveforms) # UMAP embedding
# Define a color mapping for cluster numbers
color_mapping = {
    -1: '#ff9900',      # Gapes Color for cluster -1
    -2: '#D3D3D3',      # No mvoement Color for cluster 0
     0: '#4285F4',     # Color for cluster 1
     1: '#88498F',    # Color for cluster 2
     2: '#0CBABA'        # Color for cluster 3
}    


# ==============================================================================
# Plot UMAP, color by standardized cluster label
# ==============================================================================

# Create a list of colors for each point in the embedding based on its cluster_num
cluster_nums = filtered_df['cluster_num'].tolist()  # Assuming 'cluster_num' column exists
colors = [color_mapping[cluster] for cluster in cluster_nums]

# Create the scatter plot
plt.figure(figsize=(10, 8))
#ax = fig.add_subplot(projection='3d')
plt.ion()
plt.scatter(embedding[:, 0], embedding[:, 1], c=colors,  edgecolor='k', alpha=0.7)
plt.title('UMAP Scatter Plot of Waveforms')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

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







# %%

def nonlinear_alpha_cmap(hex_color, n_levels=256, gamma=3.0):
    """
    Create a colormap that fades from fully transparent to solid cluster color
    with a nonlinear alpha curve (gamma > 1 = slower fade-in).
    """
    rgb = mcolors.to_rgb(hex_color)
    colors = []

    for i in range(n_levels):
        frac = i / (n_levels - 1)
        alpha = frac ** gamma  # nonlinear: fades in slowly
        colors.append((*rgb, alpha))
    
    return mcolors.ListedColormap(colors)



# Set up the plot
#plt.figure(figsize=(12, 10))
ax = plt.figure(figsize=(12, 10)).add_subplot(projection='3d')
# Reset index to align with embedding
reset_filtered_df = filtered_df.reset_index(drop=True)
cluster_cycle = [0, 1, 2]
center_points = []

# KDE grid resolution
grid_j = 80
'''
for cluster_num in cluster_cycle:
    cluster_df = reset_filtered_df[reset_filtered_df['cluster_num'] == cluster_num]
    cluster_indices = cluster_df.index
    embedding_cluster = embedding[cluster_indices]

    # Get x and y
    x = embedding_cluster[:, 0]
    y = embedding_cluster[:, 1]
    color = color_mapping[cluster_num]

    # Plot the scatter points
    plt.scatter(x, y, c=color, s=12, edgecolor=color, alpha=0.1, label=f'Cluster {cluster_num}')
'''
for cluster_num in cluster_cycle:
    cluster_df = reset_filtered_df[reset_filtered_df['cluster_num'] == cluster_num]
    cluster_indices = cluster_df.index
    embedding_cluster = embedding[cluster_indices]

    # Get x and y
    x = embedding_cluster[:, 0]
    y = embedding_cluster[:, 1]
    color = color_mapping[cluster_num]
    # KDE
    kde = gaussian_kde(np.vstack([x, y]))

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xx, yy = np.mgrid[xmin:xmax:grid_j*1j, ymin:ymax:grid_j*1j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)

    cmap = nonlinear_alpha_cmap(color_mapping[cluster_num], gamma=3)  # Try gamma = 3 to 5
    ax.plot_surface(xx, yy, density, cmap=cmap)
    
    ## Density peak dots
    # Find center
    max_idx = np.unravel_index(np.argmax(density), density.shape)
    x_center = xx[max_idx]
    y_center = yy[max_idx]
    center_points.append([x_center, y_center])
    # Plot red center dot
    plt.plot(x_center, y_center, density[max_idx], 'ro', markersize=25, markeredgecolor='k')
    
    

# Create circle marker legend handles with solid color
legend_handles = []
for cluster_num in cluster_cycle:
    color = color_mapping[cluster_num]
    handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                           markersize=10, label=f'Cluster {cluster_num+1}')
    legend_handles.append(handle)

# Optional: Add red circle for density peak
peak_handle = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                            markersize=10, label='Density Peaks')
legend_handles.append(peak_handle)

plt.legend(handles=legend_handles)

# Remove all grid lines
ax.grid(False)

# Make only the floor (xy plane at z=0) visible
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False

# Set floor color (z plane)
ax.zaxis.pane.set_facecolor((0.97, 0.97, 0.97))  # RGB


ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])


def nonlinear_alpha_cmap(hex_color, n_levels=256, gamma=3.0):
    """
    Create a colormap that fades from fully transparent to solid cluster color
    with a nonlinear alpha curve (gamma > 1 = slower fade-in).
    """
    rgb = mcolors.to_rgb(hex_color)
    colors = []

    for i in range(n_levels):
        frac = i / (n_levels - 1)
        alpha = frac ** gamma  # nonlinear: fades in slowly
        colors.append((*rgb, alpha))
    
    return mcolors.ListedColormap(colors)



# Set up the plot
#plt.figure(figsize=(12, 10))
ax = plt.figure(figsize=(12, 10)).add_subplot(projection='3d')
# Reset index to align with embedding
reset_filtered_df = filtered_df.reset_index(drop=True)
cluster_cycle = [0, 1, 2]
center_points = []

# KDE grid resolution
grid_j = 80
'''
for cluster_num in cluster_cycle:
    cluster_df = reset_filtered_df[reset_filtered_df['cluster_num'] == cluster_num]
    cluster_indices = cluster_df.index
    embedding_cluster = embedding[cluster_indices]

    # Get x and y
    x = embedding_cluster[:, 0]
    y = embedding_cluster[:, 1]
    color = color_mapping[cluster_num]

    # Plot the scatter points
    plt.scatter(x, y, c=color, s=12, edgecolor=color, alpha=0.1, label=f'Cluster {cluster_num}')
'''
for cluster_num in cluster_cycle:
    cluster_df = reset_filtered_df[reset_filtered_df['cluster_num'] == cluster_num]
    cluster_indices = cluster_df.index
    embedding_cluster = embedding[cluster_indices]

    # Get x and y
    x = embedding_cluster[:, 0]
    y = embedding_cluster[:, 1]
    color = color_mapping[cluster_num]
    # KDE
    kde = gaussian_kde(np.vstack([x, y]))

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xx, yy = np.mgrid[xmin:xmax:grid_j*1j, ymin:ymax:grid_j*1j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)

    cmap = nonlinear_alpha_cmap(color_mapping[cluster_num], gamma=3)  # Try gamma = 3 to 5
    ax.plot_surface(xx, yy, density, cmap=cmap)
    
    ## Density peak dots
    # Find center
    max_idx = np.unravel_index(np.argmax(density), density.shape)
    x_center = xx[max_idx]
    y_center = yy[max_idx]
    center_points.append([x_center, y_center])
    # Plot red center dot
    plt.plot(x_center, y_center, density[max_idx], 'ro', markersize=25, markeredgecolor='k')
    
    

# Create circle marker legend handles with solid color
legend_handles = []
for cluster_num in cluster_cycle:
    color = color_mapping[cluster_num]
    handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                           markersize=10, label=f'Cluster {cluster_num+1}')
    legend_handles.append(handle)

# Optional: Add red circle for density peak
peak_handle = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                            markersize=10, label='Density Peaks')
legend_handles.append(peak_handle)

#plt.legend(handles=legend_handles)

# Remove all grid lines
ax.grid(False)

# Make only the floor (xy plane at z=0) visible
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False

# Set floor color (z plane)
ax.zaxis.pane.set_facecolor((0.97, 0.97, 0.97))  # RGB


ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])


# Make axis lines thicker
ax.xaxis.line.set_linewidth(2.5)
ax.yaxis.line.set_linewidth(2.5)
ax.zaxis.line.set_linewidth(2.5)

# Increase tick label font sizes
ax.tick_params(axis='x', labelsize=14, width=2)
ax.tick_params(axis='y', labelsize=14, width=2)
ax.tick_params(axis='z', labelsize=14, width=2)

# Increase axis label font size
ax.set_xlabel('UMAP 1', fontsize=18, labelpad=12)
ax.set_ylabel('UMAP 2', fontsize=18, labelpad=12)
ax.set_zlabel('Density', fontsize=18, labelpad=12)



# Final touches
plt.title('')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
ax.set_zlabel('Density')
#plt.legend()
plt.tight_layout()
ax.view_init(elev=15, azim=-110)  # adjust numbers to taste
#plt.savefig("/home/natasha/Desktop/final_figures/3d_umap_kde_clusters.svg", format="svg")  # Save before show
#plt.savefig("/home/natasha/Desktop/final_figures/3d_umap_kde_clusters.png", format="png")  # Save before show
#plt.show()

from matplotlib import animation

# Function to update view angle
def rotate(angle):
    ax.view_init(elev=15, azim=angle)

# Make animation
rot_animation = animation.FuncAnimation(
    plt.gcf(), rotate, frames=np.arange(0, 360, 2), interval=100
)

# Save as mp4 (high quality)
rot_animation.save(
    "/home/natasha/Desktop/final_figures/3d_umap_kde_clusters.mp4",
    writer="ffmpeg", dpi=200
)

# Or save as GIF
rot_animation.save(
    "/home/natasha/Desktop/final_figures/3d_umap_kde_clusters.gif",
    writer="pillow", dpi=150
)


# %% 2D UMAP with KDE Density and Density Peaks


# Set up the plot
fig, ax = plt.subplots(figsize=(12, 10))

# Reset index to align with embedding
reset_filtered_df = filtered_df.reset_index(drop=True)
cluster_cycle = [0, 1, 2]
center_points = []

# KDE grid resolution
grid_j = 100

for cluster_num in cluster_cycle:
    cluster_df = reset_filtered_df[reset_filtered_df['cluster_num'] == cluster_num]
    cluster_indices = cluster_df.index
    embedding_cluster = embedding[cluster_indices]

    # Get x and y
    x = embedding_cluster[:, 0]
    y = embedding_cluster[:, 1]
    color = color_mapping[cluster_num]

    # Plot the scatter points
    ax.scatter(x, y, c=color, s=12, edgecolor=color, alpha=0.2, label=f'Cluster {cluster_num}')

for cluster_num in cluster_cycle:
    cluster_df = reset_filtered_df[reset_filtered_df['cluster_num'] == cluster_num]
    cluster_indices = cluster_df.index
    embedding_cluster = embedding[cluster_indices]

    # Get x and y
    x = embedding_cluster[:, 0]
    y = embedding_cluster[:, 1]
    color = color_mapping[cluster_num]

    # KDE
    kde = gaussian_kde(np.vstack([x, y]))

    # Expand plotting limits slightly to avoid clipping
    x_margin = (x.max() - x.min()) * 0.05
    y_margin = (y.max() - y.min()) * 0.05
    xmin, xmax = x.min() - x_margin, x.max() + x_margin
    ymin, ymax = y.min() - y_margin, y.max() + y_margin

    xx, yy = np.mgrid[xmin:xmax:grid_j*1j, ymin:ymax:grid_j*1j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)

    cmap = nonlinear_alpha_cmap(color_mapping[cluster_num], gamma=4)
    ax.contourf(xx, yy, density, levels=75, cmap=cmap)

    # Find center
    max_idx = np.unravel_index(np.argmax(density), density.shape)
    x_center = xx[max_idx]
    y_center = yy[max_idx]
    center_points.append([x_center, y_center])

    # Plot red center dot
    ax.plot(x_center, y_center, 'ro', markersize=20, markeredgecolor='k')

# Remove all axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

# Add two small reference lines in bottom-left corner
offset_x = (xmax - xmin) * 0.04  # 2% inset from left
offset_y = (ymax - ymin) * 0.05  # 2% inset from bottom
corner_x, corner_y = xmin + offset_x, ymin + offset_y

line_length_x = (xmax - xmin) * 0.10  # 10% of x-range
line_length_y = (ymax - ymin) * 0.10  # 10% of y-range

# Plot inset mini axes
ax.plot([corner_x, corner_x + line_length_x], [corner_y, corner_y], color='k', lw=4)  # x-axis
ax.plot([corner_x, corner_x], [corner_y, corner_y + line_length_y], color='k', lw=4)  # y-axis


# Add labels at the end of the mini axes
# Add labels in the middle of the mini axes
ax.text(corner_x + line_length_x / 2, corner_y - (ymax - ymin)*0.015, 
        'UMAP 1', fontsize=16, ha='center', va='top')
ax.text(corner_x - (xmax - xmin)*0.015, corner_y + line_length_y / 2, 
        'UMAP 2', fontsize=16, ha='right', va='center', rotation='vertical')

# Create circle marker legend handles with solid color
legend_handles = []
for cluster_num in cluster_cycle:
    color = color_mapping[cluster_num]
    handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                           markersize=10, label=f'Cluster {cluster_num+1}')
    legend_handles.append(handle)

# Optional: Add red circle for density peak
peak_handle = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                            markersize=10, label='Density Peaks')
legend_handles.append(peak_handle)

ax.legend(handles=legend_handles)

# Final touches
plt.title('UMAP with KDE Density and Cluster Centers')
plt.tight_layout()
plt.savefig("/home/natasha/Desktop/final_figures/umap_kde_clusters.svg", format="svg")
plt.savefig("/home/natasha/Desktop/final_figures/umap_kde_clusters.png", format="png")
plt.show()

# %%


#### PLOTTING PROTOTYPICAL WAVEOFORM (BASED ON KDE CENTER) FOR EACH CLUSTER
plt.figure(figsize=(10, 7))

for i, wf in enumerate(prototypical_waveforms):
    cluster_id = wf['cluster']
    segment = wf['segment']
    color = color_mapping[cluster_id]
    
    plt.plot(segment, color=color, 
             linewidth = 3.5,
             label=f'Cluster {cluster_id+1}')

plt.title('Prototypical Segments by Cluster')
plt.xlabel('Time (ms)')
plt.ylabel('Norm. Amplitude')
plt.legend()
plt.savefig("/home/natasha/Desktop/final_figures/prototypical_waveform.svg", format="svg")  # Save before show
plt.savefig("/home/natasha/Desktop/final_figures/prototypical_waveform.png", format="png")  # Save before show
plt.show()

# %% EACH ARCHETYPAL WAVEFORM INDIVIDUALLY




for i, wf in enumerate(prototypical_waveforms):
    plt.figure(figsize=(10, 7))
    cluster_id = wf['cluster']
    segment = wf['segment']
    color = color_mapping[cluster_id]
    
    plt.plot(segment, color=color, linewidth=15)

    # Remove everything but the traces
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.show()


# %% ARCHIVE - a bunch of different UMAP plotting and clustering code

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
# 3D UMAP (3 dimmensions of UMAP)
# ==============================================================================
# Initialize UMAP reducer with 3 components for 3D embedding
reducer = umap.UMAP(n_components=3)
# Filter valid clusters
filtered_df = df[df['cluster_num'] >= 0]
waveforms = filtered_df['segment_norm_interp'].tolist()

# Compute 3D UMAP embedding
embedding = reducer.fit_transform(waveforms)

embedding = embedding[::10]
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

colors = colors[::10]

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

   






