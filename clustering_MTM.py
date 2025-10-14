# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os
import umap
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import piecewise_regression
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
from mtm_analysis_config import dirname, iterations, fixed_cluster_num, color_mapping, extra_colors


# ==============================================================================
# Load data and get setup
# ==============================================================================
file_path = os.path.join(dirname, 'all_datasets_emg_pred.pkl') # all events from classifier predictions
df = pd.read_pickle(file_path)
#df = df[~df['laser']]

unique_basenames = df['basename'].unique()

# Make a dataframe of just mouth or tongue movement events
mtm_bool = df.event_type.str.contains('MTMs')
mtm_df = df.loc[mtm_bool]


# ==============================================================================
# Define and start UMAP process
# ==============================================================================
mtm_features = np.stack(mtm_df.features.values)

# UMAP dimmentionality reduction and feature scaling
reducer = umap.UMAP()
n_components_range = list(range(1, 15))  # Define a range of cluster numbers to test
scaled_mtm_features = StandardScaler().fit_transform(mtm_features) # Scale features
embedding = reducer.fit_transform(scaled_mtm_features) # UMAP embedding


# ==============================================================================
# Prep folder for saving UMAP clustering plots
# ==============================================================================
# Create directory 
clust_dir = os.path.join(dirname, 'clustering_results')
os.makedirs(clust_dir, exist_ok=True)
# Remove any png files in plots folder
png_files = glob.glob(os.path.join(clust_dir, '*.png'))
for file in png_files:
    os.remove(file)
    
# Create directory 
mahal_dir = os.path.join(clust_dir, 'mahalanobis_matrix')
os.makedirs(mahal_dir, exist_ok=True)
# Remove any png files in plots folder
png_files = glob.glob(os.path.join(mahal_dir, '*.png'))
for file in png_files:
    os.remove(file)


# ==============================================================================
# Define functions
# ==============================================================================

def calc_mahalanobis_distance_matrix(mtm_session_df):
    cluster_labels = np.unique(mtm_session_df.cluster_num)
    mahal_matrix = np.zeros((len(cluster_labels), len(cluster_labels)))
    full_features = mtm_session_df.scaled_features
    for i, clust_i in enumerate(cluster_labels):
        for j, clust_j in enumerate(cluster_labels):
            this_clust_idx = mtm_session_df.index[np.where(mtm_session_df.cluster_num == clust_i)[0]]
            this_cluster_data = np.stack(full_features.loc[this_clust_idx])
            if len(this_cluster_data) > 2:
                this_cluster_mean = np.mean(this_cluster_data, axis = 0)
                this_cluster_cov = np.cov(this_cluster_data, rowvar=False)
                inv_cov = np.linalg.pinv(this_cluster_cov)
                
                other_clust_idx = mtm_session_df.index[np.where(mtm_session_df.cluster_num == clust_j)[0]]
                other_cluster_data = np.stack(full_features.loc[other_clust_idx])
                mahal_list = [
                    mahalanobis(x, this_cluster_mean, inv_cov) \
                        for x in other_cluster_data
                        ]
                mahal_matrix[i,j] = np.mean(mahal_list)
            else:
                mahal_matrix[i,j] = np.nan
    
    return mahal_matrix



# %% # Run GMM on PCA of MTMs in individual sessions WITH MAHALANOBIS DISTANCE
# ==============================================================================
# Set color maps
# ==============================================================================

if not np.isnan(fixed_cluster_num):
    # Keep only keys >= 0 and sort by key
    custom_colors = [color_mapping[k] for k in sorted(color_mapping) if k >= 0]
    
    if len(custom_colors) >= fixed_cluster_num:
        cmap = ListedColormap(custom_colors)
    
    else:
        custom_colors.extend(extra_colors)
        if len(custom_colors) >= fixed_cluster_num:
            cmap = ListedColormap(custom_colors)
        else:
            print(f"Warning: The number of positive-key colors ({len(custom_colors)}) "
            f"does not match the set number of clusters ({fixed_cluster_num}).\n"
            "Using tab10 cmap instead.\n"
            "Consider updating 'color_mapping' in mtm_analysis_config."
            )
            tab10 = plt.get_cmap("Accent")
            for i in range(tab10.N):
                color = tab10(i)  # RGBA tuple
                if color not in custom_colors:
                    custom_colors.append(color)
            cmap = ListedColormap(custom_colors)

else:
    custom_colors = [color_mapping[k] for k in sorted(color_mapping) if k >= 0]
    
    custom_colors.extend(extra_colors)
        
    tab10 = plt.get_cmap("Accent")
    for i in range(tab10.N):
        color = tab10(i)  # RGBA tuple
        if color not in custom_colors:  # Compare RGBA directly
            custom_colors.append(color)


    cmap = ListedColormap(custom_colors)

# ==============================================================================
# Initialize data
# ==============================================================================
mtm_df['cluster_num'] = np.nan
mtm_df['scaled_features'] = np.nan

# Initialize lists to keep track of PCA results
optimal_cluster_list = []
session_size_list = []
pca_dimmensions = []
non_diag_elements = []
diag_elements = []


# ==============================================================================
# PCA with GMM on a session-by-session basis
# ==============================================================================
for session in tqdm(df.session_ind.unique()):
    #if session == 0:
    #    continue
    for i in range(iterations):
        # Filter data for the current session
        mtm_session_bool = mtm_df.session_ind == session
        mtm_session_df = mtm_df.loc[mtm_session_bool].copy()  # Make a copy to avoid SettingWithCopyWarning
        mtm_session_features = np.stack(mtm_session_df.features.values)
        scaled_mtm_session = StandardScaler().fit_transform(mtm_session_features)  # Scale features
        
        mtm_session_df['scaled_features'] = list(scaled_mtm_session) # ADDED FOR MAHAL DIST
        
        #TODO: NEEDED TO CHECK IF I WHITENED?
        pca = PCA(n_components=0.9)
        embedding = pca.fit_transform(scaled_mtm_session)
        pca_dimmensions.append(embedding.shape[1])
        
        embedding_umap = reducer.fit_transform(scaled_mtm_session)  # UMAP embedding of PCA - for plotting
        

        if np.isnan(fixed_cluster_num):
            # Determine the optimal number of clusters using BIC if not given a fixed cluster number
            bic_scores = []
            for n_components in n_components_range:
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(embedding)
                bic = gmm.bic(embedding)
                bic_scores.append(bic)
            # Find the optimal cluster number by looking for the elbow in the BIC values
            pw_fit = piecewise_regression.Fit(n_components_range, bic_scores, n_breakpoints=1)
            pw_results = pw_fit.get_results()
            if pw_results["converged"] == False:
                breakpoint1 = bic_scores.index(min(bic_scores))
            else:
                breakpoint1 = pw_results["estimates"]["breakpoint1"]["estimate"]
            optimal_n_components = round(breakpoint1)
            optimal_cluster_list.append(breakpoint1)
            
            # Store the optimal clusters number and the number of MTMs within a session
            session_size_list.append(len(mtm_session_df))

        else:
            optimal_n_components = fixed_cluster_num
            optimal_cluster_list.append(fixed_cluster_num)

        # Fit the GMM with the optimal cluster number
        optimal_gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
        optimal_gmm.fit(embedding)
        labels = optimal_gmm.predict(embedding)

        # Add cluster number label to df dataframe
        df.loc[(df.session_ind == session) & (df.event_type == 'MTMs'), 'cluster_num'] = labels
        
        
        if not np.isnan(fixed_cluster_num):
            # Calculate mahalanobis distance if cluster number is fixed
            mtm_session_df['cluster_num'] = labels 
            mahal_matrix = calc_mahalanobis_distance_matrix(mtm_session_df)
            
            # Extract diagonal and non-diagonal elements
            non_diag_mask = ~np.eye(mahal_matrix.shape[0], dtype=bool)
            non_diag_elements.extend(mahal_matrix[non_diag_mask].tolist())
            diag_elements.extend(np.diagonal(mahal_matrix).tolist())
            
        # Everything below is plotting
        if i == 0: # For speed, only create individual session plots for the first iteration
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot the UMAP projections with optimal GMM clusters, using the custom colormap
            scatter = ax.scatter(
                embedding_umap[:, 0], embedding_umap[:, 1],
                c=labels, cmap=cmap, s=45,
                linewidths=0.5, edgecolor = 'black'
            )
            ax.set_title(f'Session {session}: UMAP projection of PCA with GMM ({optimal_n_components} clusters)')
            
            # Remove all axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_ticks([])  # remove ticks
            
            # Add inset reference axes in bottom-left corner
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            
            offset_x = (xmax - xmin) * 0.04  # 4% inset from left
            offset_y = (ymax - ymin) * 0.05  # 5% inset from bottom
            corner_x, corner_y = xmin + offset_x, ymin + offset_y
            
            line_length_x = (xmax - xmin) * 0.10  # 10% of x-range
            line_length_y = (ymax - ymin) * 0.10  # 10% of y-range
            
            # Plot mini-axes
            ax.plot([corner_x, corner_x + line_length_x], [corner_y, corner_y], color='k', lw=4)  # x-axis
            ax.plot([corner_x, corner_x], [corner_y, corner_y + line_length_y], color='k', lw=4)  # y-axis
            
            # Labels for mini-axes (with slight adjustments)
            ax.text(corner_x + line_length_x / 2 + (xmax - xmin) * 0.02,   # shift right
                    corner_y - (ymax - ymin) * 0.015,
                    'UMAP 1', fontsize=16, ha='center', va='top')
            
            ax.text(corner_x - (xmax - xmin) * 0.015,
                    corner_y + line_length_y / 2 + (ymax - ymin) * 0.02,   # shift up
                    'UMAP 2', fontsize=16, ha='right', va='center', rotation='vertical')
            # Save outputs
            umap_session_path_png = os.path.join(clust_dir, f'session_{session}_umap-of-PCA.png')
            plt.savefig(umap_session_path_png)
            umap_session_path_svg = os.path.join(clust_dir, f'session_{session}_umap-of-PCA.svg')
            plt.savefig(umap_session_path_svg)
            plt.show()
            plt.clf()

            
            if np.isnan(fixed_cluster_num):
                # Plot the BIC values, linear regressions, and breakpoint with confidence intervals
                plt.figure(figsize=(10, 6))  # adjust width, height in inches

                pw_fit.plot_fit(color="black", linewidth=3)
                pw_fit.plot_data(color="gray", marker = 'o', s = 100)
                pw_fit.plot_breakpoints(color="red", linestyle = '--', linewidth=3.5)
                pw_fit.plot_breakpoint_confidence_intervals(color = "red")
                plt.gca().set_yticklabels([])
                plt.xlabel('Number of clusters')
                plt.ylabel('BIC Score (a.u.)')
                plt.title(f'Session {session}: BIC Scores. Elbow at {round(breakpoint1,3)}')
                bic_session_path_png = os.path.join(clust_dir, f'session_{session}_bic.png')
                plt.savefig(bic_session_path_png)
                bic_session_path_svg = os.path.join(clust_dir, f'session_{session}_bic.svg')
                plt.savefig(bic_session_path_svg)
                plt.clf()
            
            if not np.isnan(fixed_cluster_num):
                # Plot the mahalanobis matrix
                fig, ax = plt.subplots()
                im = ax.matshow(mahal_matrix, cmap='viridis')
                for (i, j), z in np.ndenumerate(mahal_matrix):
                    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
                ax.set_ylabel("this cluster")
                ax.set_xlabel("other cluster")
                plt.title(f"Session {session}")
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Mahalanobis Distance")
                mahal_session_path_png = os.path.join(mahal_dir, f'session_{session}_mahalanobis.png')
                plt.savefig(mahal_session_path_png)
                mahal_session_path_svg = os.path.join(mahal_dir, f'session_{session}_mahalanobis.svg')
                plt.savefig(mahal_session_path_svg)
                plt.clf()
      

# %% Save clustering stats results to be used for stats tests
# (NOT saving the clustering index label assignments, that's the next code section below)
# ==============================================================================      

if not np.isnan(fixed_cluster_num): # If cluster number is fixed number, then save mahalanobis distances
    # Data to save
    mahal_data = pd.DataFrame({
        'value': list(diag_elements) + list(non_diag_elements),
        'group': ['diag'] * len(diag_elements) + ['non_diag'] * len(non_diag_elements)
    })

    output_file_path = os.path.join(dirname, 'mahalanobis_data_with_laser.pkl')
    mahal_data.to_pickle(output_file_path)
    
    print(f"Mahalanobis DataFrame successfully saved to {output_file_path}")
    

if np.isnan(fixed_cluster_num): # If cluster number is variable, save stats related to what cluster numbers fit best across multiple iterations
    non_fixed_clust_data = pd.DataFrame({
        'optimal_cluster_list': optimal_cluster_list,
        'session_size_list': session_size_list,
        'pca_dimmensions': pca_dimmensions
    })

    output_file_path = os.path.join(dirname, 'non_fixed_clust_data_with_laser.pkl')
    non_fixed_clust_data.to_pickle(output_file_path)
    
    print(f"non_fixed_clust_data DataFrame successfully saved to {output_file_path}")




# %% # Save new dataframe with assigned cluster number index to each row (i.e. movement event)
# ==============================================================================

if not np.isnan(fixed_cluster_num):
    # Assign '0' to cluster_num for events where event_type is 'no movement'
    df.loc[df['event_type'] == 'no movement', 'cluster_num'] = -2
    
    # Assign '-1' to cluster_num for events where event_type is 'gape'
    df.loc[df['event_type'] == 'gape', 'cluster_num'] = -1
    
    
    ## Save the new dataframe into a pickle file
    output_file_path = os.path.join(dirname, 'clustering_df_update_with_laser.pkl')
    df.to_pickle(output_file_path)
    
    print(f"DataFrame successfully saved to {output_file_path}")
    
