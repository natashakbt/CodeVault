#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 13:51:22 2025

@author: natasha
"""

import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import glob
from scipy.stats import chi2_contingency, zscore, ttest_rel, chisquare, power_divergence, percentileofscore
import scipy.stats as stats
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
import shutil
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import piecewise_regression
import umap
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
from pickle import load

# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/clustering_data/'
file_path = os.path.join(dirname, 'clustering_df_update.pkl')
df = pd.read_pickle(file_path)

transition_file_path = os.path.join(dirname, 'scaled_mode_tau_cut.pkl')
transition_df = pd.read_pickle(transition_file_path)

# Remove any data for df that does not have an associated transition time in scaled_mode_tau
df['basename'] = df['basename'].str.lower() # All basenames to lowercase
transition_df['basename'] = transition_df['basename'].str.lower() # All basenames to lowercase
transition_df = transition_df.rename(columns={'taste': 'taste_num'}) # NEW changed column name.
tau_basenames = transition_df.basename.unique() # Find all basenames in transition_df
df = df.loc[df['basename'].isin(tau_basenames)] # Keep only basenames 
# Manually removed this specific data:
df = df[~((df['basename'] == 'km50_5tastes_emg_210911_104510_copy') & (df['taste'] == 1))]
df = df[~((df['basename'] == 'km50_5tastes_emg_210911_104510_copy') & (df['taste'] == 4))]


obj_path = os.path.join(dirname, 'pca_obj.pkl')
with open(obj_path, 'rb') as file:
    pca_obj = load(file)

# ==============================================================================
# Check PCA
# ==============================================================================

check_filtered_df = df[df['cluster_num'].isin([0, 1, 2])]
check_norm_interp = check_filtered_df['segment_norm_interp']
check_norm_interp = check_norm_interp.apply(pd.Series)

X_pca_check = pca_obj.transform(check_norm_interp)

X_pca_from_df = np.array([features[-3:] for features in check_filtered_df['raw_features']])

# Check if they're close?
np.allclose(X_pca_check, X_pca_from_df)


# =============================================================================
# 
# =============================================================================

cluster_range = [0, 1, 2]

i_dont_know = []

for i in cluster_range:    
    filtered_df = df[df['cluster_num'] == i]
    X_pca = np.array([features[-3:] for features in filtered_df['features']])
    mean_vector = np.mean(X_pca, axis=0)
    i_dont_know.append(mean_vector)
    
# i_dont_know_array = np.array(i_dont_know)

# =============================================================================
# 
# =============================================================================
from sklearn.linear_model import LinearRegression

# lr = LinearRegression()
# lr.fit(check_norm_interp, X_pca_from_df)

# lr_project = lr.predict(check_norm_interp)

X = check_norm_interp.copy()
X = np.concatenate([X, np.ones((len(X),1))], axis=1)
y = X_pca_from_df.copy()

B = np.linalg.pinv(X.T @ X) @ X.T @ y
# lr_project = check_norm_interp @ B[:-1] + B[-1]

lr_project = X@B

fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,10))
ax[0].imshow(X_pca_from_df, interpolation=None, aspect='auto',
             vmin = -3, vmax = 3)
ax[1].imshow(lr_project, interpolation=None, aspect='auto',
             vmin = -3, vmax = 3)

B_inv = np.linalg.pinv(B)


# y := means of the MTM clusters in feature space
#       shape = datapoints, PCAs
# X_inv_from_y = (y - B[-1]) @ B_inv
X_inv_from_y = y @ B_inv
X_inv_from_y = X_inv_from_y[:, :-1]

plt.imshow(X_inv_from_y, interpolation=None, aspect='auto',)
plt.plot(X_inv_from_y[:100].T)
# y = i_dont_know

# y := means of the MTM clusters in feature space
#       shape = datapoints, PCAs

inv_proj = []

for y in i_dont_know:
    X_inv_from_y = (y - B[-1])[None,:] @ B_inv
    X_inv_from_y = X_inv_from_y[:, :-1]
    inv_proj.append(X_inv_from_y)

inv_proj_array = np.squeeze(np.array(inv_proj))
plt.plot(inv_proj_array.T)
plt.title("Is this right??")
plt.show()


# =============================================================================
# Make new PCA
# =============================================================================
from sklearn.decomposition import PCA

pca_new = PCA(n_components=3)
pca_new.fit(check_norm_interp)
X_pca_new = pca_new.transform(check_norm_interp)


# ==============================================================================
# Reconstruct waveforms from PCA
# ==============================================================================

cluster_range = [0, 1, 2]
colors = {
    0: '#4285F4',
    1: '#88498F',
    2: '#0CBABA',
}

mean_features = []
mean_waveforms = []

for i in cluster_range:    
    filtered_df = df[df['cluster_num'] == i]
    X_pca = np.array([features[-3:] for features in filtered_df['features']])
    X_reconstructed = pca_obj.inverse_transform(X_pca)
    
    
    
    for waveform in X_reconstructed:
        plt.plot(waveform, color=colors[i], alpha=0.1)
    plt.title(f"Inverse-Transformed Waveforms\nCluster {i}")
    plt.show()
    
    mean_feature = np.mean(X_pca, axis=0)
    mean_features.append(mean_feature)
    
    prototypical_waveform = np.mean(X_reconstructed, axis=0)
    mean_waveforms.append(prototypical_waveform)
    
explained_variance_ratio = pca_obj.explained_variance_ratio_

  

for w in mean_features:
    waveform = pca_obj.inverse_transform(w)
    plt.plot(waveform)
plt.title("Reconstructing waveforms from mean PCA values")
plt.show()

for w in mean_waveforms:
    plt.plot(w)
plt.title("Mean of reconstructed waveforms")
plt.show()








