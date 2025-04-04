#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 10:39:29 2025

@author: natasha
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy import stats
from tqdm import tqdm
from sklearn.inspection import DecisionBoundaryDisplay

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

#features_expanded = features_expanded.drop('cluster_num', axis=1)

X = features_expanded.drop(columns=["cluster_num"])
y = features_expanded["cluster_num"]



# %% TRAINING SVM + PLOTTING CONFUSION MATRIX
# ==============================================================================
# Training SVM
# ==============================================================================

accuracy_scores = []
confusion_matrices = []
# Train the classifier 10x
for i in tqdm(range(10)):
    # Split the dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
     
    rbf_svc = svm.SVC(kernel='rbf') # Non-linear
    rbf_svc.fit(X_train, y_train)
    
    y_pred = rbf_svc.predict(X_test) 
    

    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
    
    matrix = confusion_matrix(y_test, y_pred, normalize='pred')
    confusion_matrices.append(matrix)  # Store the matrix
    
    plt.imshow(matrix)
    plt.xticks(ticks=np.arange(3), labels=[0, 1, 2])
    plt.yticks(ticks=np.arange(3), labels=[0, 1, 2])

    plt.xlabel("Predicted Cluster Labels")
    plt.ylabel("True Cluster Labels")
    cbar = plt.colorbar()
    cbar.set_label('Normalized Accuracy')
    
    plt.show()

average_matrix = np.mean(confusion_matrices, axis=0)

# Plot the average confusion matrix
plt.imshow(average_matrix, cmap='Blues')
plt.xticks(ticks=np.arange(3), labels=[0, 1, 2])
plt.yticks(ticks=np.arange(3), labels=[0, 1, 2])

plt.xlabel("Predicted Cluster Labels")
plt.ylabel("True Cluster Labels")
cbar = plt.colorbar()
cbar.set_label('Normalized Accuracy')

plt.title("Average Confusion Matrix")
plt.show()


# Perform a one-sample t-test
t_stat, p_value = stats.ttest_1samp(accuracy_scores, 0.3)
if p_value < 0.05:
    print(f'The mean accuracy is significantly above 0.3 (p-value: {p_value})')
else:
    print('The mean accuracy is not significantly different from chance!')



#%%
from sklearn.decomposition import PCA

def plot_training_data_with_decision_boundary(
    kernel, ax=None, long_title=True, support_vectors=True
):
    # Train the SVC
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    #y_2d = pca.fit_transform(y)
    
    clf = svm.SVC(kernel='rbf').fit(X_2d, y)


    # Settings for plotting
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X_2d, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
        shading='auto',
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    if support_vectors:
        # Plot bigger circles around samples that serve as support vectors
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    # Plot samples by color and add legend
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    if long_title:
        ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")
    else:
        ax.set_title(kernel)

    if ax is None:
        plt.show()

plot_training_data_with_decision_boundary("rbf")

# %% Plotting SVM






