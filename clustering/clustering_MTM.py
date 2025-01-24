# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
#file_path = os.path.join(dirname, 'mtm_clustering_df.pkl') # only labeled stuff?
file_path = os.path.join(dirname, 'all_datasets_emg_pred.pkl') #everything with predictions?
df = pd.read_pickle(file_path)
df = df.rename(columns={'pred_event_type': 'event_type'})

unique_basenames = df['basename'].unique()
basename_to_num = {name: idx for idx, name in enumerate(unique_basenames)}
df['session_ind'] = df['basename'].map(basename_to_num)

# Make a dataframe of just mouth or tongue movement events
#mtm_bool = df.event_type.str.contains('mouth or tongue movement')
mtm_bool = df.event_type.str.contains('MTMs')
mtm_df = df.loc[mtm_bool]



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
  
# ==============================================================================
# UMAP of all MTMs in all sessions
# ==============================================================================
# Array of every MTM event and their values for each of the 8 features
mtm_features = np.stack(mtm_df.features.values)

# UMAP dimmentionality reduction and feature scaling
reducer = umap.UMAP()
n_components_range = list(range(1, 15))  # Define a range of cluster numbers to test
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
umap_all_path = os.path.join(clust_dir, 'all_sessions_umap.png')
plt.savefig(umap_all_path)
plt.clf()

# Plot BIC values for a range of cluster sizes
plt.plot(n_components_range, bic_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('BIC')
plt.title('All Sessions: BIC Scores')
bic_all_path = os.path.join(clust_dir, 'all_sessions_bic.png')
plt.savefig(bic_all_path)
plt.clf()   


# ==============================================================================
# Important inputs for UMAP of individual sessions
# ==============================================================================
#fixed_cluster_num = np.nan
fixed_cluster_num = np.nan
iterations = 1 # Number of times to repeat UMAP reduction


# %% Define functions
# ==============================================================================
# Important inputs for UMAP of individual sessions
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
# Run GMM on PCA of MTMs in individual sessions WITH MAHALANOBIS DISTANCE
# ==============================================================================
# Create directory 
pca_dir = os.path.join(clust_dir, 'PCA_results')
os.makedirs(pca_dir, exist_ok=True)
# Remove any png files in plots folder
png_files = glob.glob(os.path.join(pca_dir, '*.png'))
for file in png_files:
    os.remove(file)
    
# Create directory 
mahal_dir = os.path.join(pca_dir, 'mahalanobis_matrix')
os.makedirs(mahal_dir, exist_ok=True)
# Remove any png files in plots folder
png_files = glob.glob(os.path.join(mahal_dir, '*.png'))
for file in png_files:
    os.remove(file)

# Initialize lists to keep track of UMAP results
optimal_cluster_list = []
session_size_list = []
pca_dimmensions = []
non_diag_elements = []
diag_elements = []


mtm_df['cluster_num'] = np.nan
mtm_df['scaled_features'] = np.nan

if fixed_cluster_num == 3:
    custom_colors = ['#4285F4', '#88498F', '#0CBABA']
    cmap = ListedColormap(custom_colors)
#elif fixed_cluster_num == 4:
else:
    custom_colors = ['#4285F4', '#88498F', '#08605F', '#0CBABA',  '#B0B0B0']
    cmap = ListedColormap(custom_colors)
#else:
#    cmap = 'inferno'

# UMAP with GMM on a session-by-session basis
for session in df.session_ind.unique():
    if session == 0:
        continue
    for i in range(iterations):
        # Filter data for the current session
        mtm_session_bool = mtm_df.session_ind == session
        mtm_session_df = mtm_df.loc[mtm_session_bool].copy()  # Make a copy to avoid SettingWithCopyWarning
        mtm_session_features = np.stack(mtm_session_df.features.values)
        scaled_mtm_session = StandardScaler().fit_transform(mtm_session_features)  # Scale features
        
        mtm_session_df['scaled_features'] = list(scaled_mtm_session) # ADDED FOR MAHAL DIST
        
        pca = PCA(n_components=0.9)
        embedding = pca.fit_transform(scaled_mtm_session)
        pca_dimmensions.append(embedding.shape[1])
        
        embedding_umap = reducer.fit_transform(scaled_mtm_session)  # UMAP embedding of PCA - for plotting
        
        # Determine the optimal number of clusters using BIC
        bic_scores = []
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(embedding)
            bic = gmm.bic(embedding)
            bic_scores.append(bic)
        
        # Find the optimal cluster number by looking for the elbow in the BIC values
        if np.isnan(fixed_cluster_num):
            pw_fit = piecewise_regression.Fit(n_components_range, bic_scores, n_breakpoints=1)
            pw_results = pw_fit.get_results()
            breakpoint1 = pw_results["estimates"]["breakpoint1"]["estimate"]
            optimal_n_components = round(breakpoint1)
        else:
            optimal_n_components = fixed_cluster_num
        # Store the optimal clusters number and the number of MTMs within a session
        optimal_cluster_list.append(breakpoint1)
        session_size_list.append(len(mtm_session_df))

        # Fit the GMM with the optimal number of clustersz
        optimal_gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
        optimal_gmm.fit(embedding)
        labels = optimal_gmm.predict(embedding)

        # Add cluster number label to df dataframe
        #df.loc[(df.session_ind == session) & (df.event_type == 'mouth or tongue movement'), 'cluster_num'] = labels
        
        mtm_session_df['cluster_num'] = labels 
        mahal_matrix = calc_mahalanobis_distance_matrix(mtm_session_df)
        
        # Extract non-diagonal elements
        non_diag_mask = ~np.eye(mahal_matrix.shape[0], dtype=bool)
        non_diag_elements.extend(mahal_matrix[non_diag_mask].tolist())
        
        # Extract non-diagonal elements
        diag_elements.extend(np.diagonal(mahal_matrix).tolist())
        # For speed, only create individual session plots for the first iteration
        if i == 0:
            
            # Plot the UMAP projections with optimal GMM clusters, using the custom colormap
            scatter = plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=labels, cmap=cmap, s=20)
            plt.title(f'Session {session}: UMAP projection of PCA with GMM ({optimal_n_components} clusters)')
            cbar = plt.colorbar(scatter)
            cbar.set_ticks([])  # Set specific tick positions
            umap_session_path = os.path.join(pca_dir, f'session_{session}_umap-of-PCA.png')
            plt.savefig(umap_session_path)
            plt.clf()

            # Plot the BIC values, linear regressions, and breakpoint with confidence intervals
            pw_fit.plot_fit(color="red", linewidth=1)
            pw_fit.plot_data(marker = 'o', s = 60)
            pw_fit.plot_breakpoints()
            pw_fit.plot_breakpoint_confidence_intervals()
            plt.xlabel('Number of clusters')
            plt.ylabel('BIC Score')
            plt.title(f'Session {session}: BIC Scores. Elbow at {round(breakpoint1,3)}')
            bic_session_path = os.path.join(pca_dir, f'session_{session}_bic.png')
            plt.savefig(bic_session_path)
            plt.clf()
            
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
            mahal_session_path = os.path.join(mahal_dir, f'session_{session}_mahalanobis.png')
            plt.savefig(mahal_session_path)
            plt.clf()
      
## Histogram of 
bin_edges = np.linspace(min(mahal_matrix.flatten()), max(mahal_matrix.flatten()), num=31)  # 20 bins

plt.hist(diag_elements, bins=bin_edges, 
         color='cornflowerblue', 
         edgecolor='black', 
         alpha=0.7,  # Transparency for the first histogram
         label='Diagonal Elements')  # Add a label

plt.hist(non_diag_elements, bins=bin_edges, 
         color='salmon', 
         edgecolor='black', 
         alpha=0.7,  # Transparency for the second histogram
         label='Non-Diagonal Elements')  # Add a label

plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title(f'Frequency Distribution ({iterations} iterations)')
plt.legend()  # Show the legend for clarity
plt.show()


## Scatter plot of optimal cluster size vs sessions size (i.e. number of MTMs)
# Define jitter amount
jitter_strength = 0.05  # Adjust the strength of jitter as needed

# Add jitter to the session size and optimal clusters
session_size_jittered = np.array(session_size_list) + np.random.normal(0, jitter_strength, len(session_size_list))
optimal_cluster_jittered = np.array(optimal_cluster_list) + np.random.normal(0, jitter_strength, len(optimal_cluster_list))

# Make scatter plot
plt.scatter(session_size_jittered, optimal_cluster_list, c='cornflowerblue', marker='o')
plt.xlabel('Session Size')
plt.ylabel('Optimal Number of Clusters')
plt.title(f'Optimal Cluster Number vs Session Size ({iterations} iterations)')
scatter_plot_path = os.path.join(pca_dir, 'optimal_clusters_vs_session_size.png')
plt.savefig(scatter_plot_path)
plt.show()
    
## Histogram of ditribution of optimal cluster sizes across all iterations
mode_result = stats.mode(optimal_cluster_list, keepdims=False)
print(f"The mode is: {mode_result[0]}")
    
#plt.hist(optimal_cluster_list, bins=len(n_components_range), 
plt.hist(optimal_cluster_list, bins=20,
         color='cornflowerblue', 
         edgecolor='black')
plt.axvline(x=mode_result[0]+0.3, 
            color='red', 
            linestyle='--', 
            linewidth=2)
plt.xlabel('Optimal Number of Clusters')
plt.ylabel('Frequency')
plt.title(f'Frequency of Optimal Cluster Number ({iterations} iterations)')
histogram_path = os.path.join(pca_dir, 'optimal_clusters_histogram.png')
plt.savefig(histogram_path)
plt.show()
    


# %% # Assign cluster numbers to events and save new dataframe
# ==============================================================================
# Assign cluster numbers to events and save new dataframe
# ==============================================================================
# Assign '0' to cluster_num for events where event_type is 'no movement'
df.loc[df['event_type'] == 'no movement', 'cluster_num'] = -2

# Assign '-1' to cluster_num for events where event_type is 'gape'
df.loc[df['event_type'] == 'gape', 'cluster_num'] = -1


## Save the new dataframe into a pickle file
output_file_path = os.path.join(dirname, 'clustering_df_update.pkl')
df.to_pickle(output_file_path)

print(f"DataFrame successfully saved to {output_file_path}")








# %% # Run GMM on UMAP of MTMs in individual sessions - OBSOLETE
# ==============================================================================
# Run GMM on UMAP of MTMs in individual sessions - OBSOLETE
# ==============================================================================
'''
# Create directory 
umap_dir = os.path.join(clust_dir, 'UMAP_results')
os.makedirs(umap_dir, exist_ok=True)
# Remove any png files in plots folder
png_files = glob.glob(os.path.join(umap_dir, '*.png'))
for file in png_files:
    os.remove(file)

# Initialize lists to keep track of UMAP results
optimal_cluster_list = []
session_size_list = []
pca_dimmensions = []

mtm_df['cluster_num'] = np.nan

if fixed_cluster_num == 3:
    custom_colors = ['#4285F4', '#88498F', '#0CBABA']
    cmap = ListedColormap(custom_colors)
elif fixed_cluster_num == 4:
    custom_colors = ['#4285F4', '#88498F', '#0CBABA', '#08605F']
    cmap = ListedColormap(custom_colors)
else:
    cmap = 'viridis'

# UMAP with GMM on a session-by-session basis
for session in df.session_ind.unique():
    if session == 0:
        continue
    for i in range(iterations):
        # Filter data for the current session
        mtm_session_bool = mtm_df.session_ind == session
        mtm_session_df = mtm_df.loc[mtm_session_bool].copy()  # Make a copy to avoid SettingWithCopyWarning
        mtm_session_features = np.stack(mtm_session_df.features.values)

        scaled_mtm_session = StandardScaler().fit_transform(mtm_session_features)  # Scale features
        embedding = reducer.fit_transform(scaled_mtm_session)  # UMAP embedding
        
        
        # Determine the optimal number of clusters using BIC
        bic_scores = []
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(embedding)
            bic = gmm.bic(embedding)
            bic_scores.append(bic)

        if np.isnan(fixed_cluster_num): # Find the number of components with the lowest BIC
            optimal_n_components = n_components_range[np.argmin(bic_scores)]
        else:
            optimal_n_components = fixed_cluster_num
        # Store the optimal clusters number and the number of MTMs within a session
        optimal_cluster_list.append(optimal_n_components)
        session_size_list.append(len(mtm_session_df))

        # Fit the GMM with the optimal number of clustersz
        optimal_gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
        optimal_gmm.fit(embedding)
        labels = optimal_gmm.predict(embedding)

        # Add cluster number label to df dataframe
        #df.loc[(df.session_ind == session) & (df.event_type == 'mouth or tongue movement'), 'cluster_num'] = labels

        # For speed, only create individual session plots for the first iteration
        if i == 0:
            # Another scatter plot where colors based on the 'yes' or 'no' in 'multimodal' column
            #colors = mtm_session_df['multimodal'].map({'no': 'cornflowerblue', 'yes': 'lightcoral'})
            #scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=20)
            # Create a custom legend
            #handles = [
            #    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cornflowerblue', markersize=10, label='No'),
            #    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=10, label='Yes')
            #]
            #plt.legend(handles=handles, title='Multimodal')
            #umap_session_path = os.path.join(umap_dir, f'session_{session}_umap.png')
            #plt.savefig(umap_session_path)
            #plt.clf()
            
            # Plot the UMAP projections with optimal GMM clusters, using the set colormap
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, s=20)
            plt.title(f'Session {session}: UMAP projection with GMM ({optimal_n_components} clusters)')
            cbar = plt.colorbar(scatter) 
            cbar.set_ticks([])  # Set specific tick positions
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
jitter_strength = 0.25  # Adjust the strength of jitter as needed

# Add jitter to the session size and optimal clusters
session_size_jittered = np.array(session_size_list) + np.random.normal(0, jitter_strength, len(session_size_list))
optimal_cluster_jittered = np.array(optimal_cluster_list) + np.random.normal(0, jitter_strength, len(optimal_cluster_list))

# Make scatter plot
plt.scatter(session_size_jittered, optimal_cluster_jittered, c='cornflowerblue', marker='o')
plt.xlabel('Session Size')
plt.ylabel('Optimal Number of Clusters')
plt.title(f'Optimal Cluster Number vs Session Size ({iterations} iterations)')
scatter_plot_path = os.path.join(umap_dir, 'optimal_clusters_vs_session_size.png')
plt.savefig(scatter_plot_path)
plt.show()
    
## Histogram of ditribution of optimal cluster sizes across all iterations
mode_result = stats.mode(optimal_cluster_list, keepdims=False)
print(f"The mode is: {mode_result[0]}")
    
plt.hist(optimal_cluster_list, bins=len(n_components_range), 
         color='cornflowerblue', 
         edgecolor='black')
plt.axvline(x=mode_result[0]+0.3, 
            color='red', 
            linestyle='--', 
            linewidth=2)
plt.xlabel('Optimal Number of Clusters')
plt.ylabel('Frequency')
plt.title(f'Frequency of Optimal Cluster Number ({iterations} iterations)')
histogram_path = os.path.join(umap_dir, 'optimal_clusters_histogram.png')
plt.savefig(histogram_path)
plt.show()
'''


