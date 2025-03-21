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
from scipy.optimize import linear_sum_assignment
import scipy.stats as stats
import scikit_posthocs as sp
import pingouin as pg
from scipy.stats import tukey_hsd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

    # Update average vectors to be a combination of 
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



# %% Plot waveforms combined per cluster label
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
# ==============================================================================
cluster_basename_groups = df.groupby(['cluster_num'])
for (cluster), group in cluster_basename_groups:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color map defined above
    color = color_mapping.get(cluster, basename_color_map.get(cluster, 'black'))
    for row in group.iterrows():
        max_amp = max(row[1]['segment_raw'])
        scaling_factor = row[1]['raw_features'][4]
        segment = (row[1]['segment_raw'])/max_amp * scaling_factor
        ax.plot(segment, alpha=0.1, color=color)
    
    ax.set_title(f'Cluster {cluster} Waveforms')
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-2, 15)
    # Save plot
    plot_filename = f'cluster{cluster}.png'

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
        
        max_length = max(max_length, len(segment))  # Update max length
        normalized_waveforms.append(segment)  # Collect for averaging
        
        ax.plot(segment, alpha=0.1, color=color)  # Individual waveforms
    
    # Pad waveforms to the same length
    padded_waveforms = [np.pad(w, (0, max_length - len(w)), mode='constant') for w in normalized_waveforms]

    # Compute and plot the average waveform
    if padded_waveforms:
        avg_waveform = np.mean(padded_waveforms, axis=0)  # Compute mean across waveforms
        ax.plot(avg_waveform, 'k--', linewidth=2, label="Avg Waveform")  # Plot in black dotted line
    
    ax.set_title(f'Cluster {cluster} Waveforms')
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(0, 20)
    ax.legend()

    # Save plot
    plot_filename = f'cluster{cluster}.png'
    plot_path = os.path.join(overlap_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)



# %%



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

data1 = features_expanded.copy()
data1.loc[:, features_expanded.columns != 'cluster_num'] = float('nan')
ax = sns.heatmap(data1, cmap=color_mapping)
data2 = features_expanded.copy()
data2['cluster_num'] = float('nan')
sns.heatmap(data2, yticklabels=False, cmap='viridis', vmax=3)


# ==============================================================================
# Three variations of stats to see if features are significantly different across clusters
# ==============================================================================


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
    if p_value < 0.05 :
        if eta_squared > 0.06:  # Only proceed if Kruskal-Wallis is significant
            print(f"{feature} p-value: {p_value.round(5)}, effect size: {eta_squared.round(4)}")
            
            # Perform Dunn’s test with Bonferroni correction
            dunn_results = sp.posthoc_dunn(features_expanded, val_col=feature, group_col="cluster_num", p_adjust="bonferroni")
            print(dunn_results.round(3))

        elif eta_squared > 0.01:
            print(f'{feature} is signficiant ({p_value.round(5)}) but small effect size ({eta_squared.round(4)})')
        else:
            print(f'{feature} is signficiant ({p_value.round(5)}) but inconsequential effect size')
    else:
        print(f'{feature} not significant ({p_value.round(5)})')
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







