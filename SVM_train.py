#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 13:21:58 2025

@author: natasha
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy import stats
from tqdm import tqdm


# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/clustering_data/'
file_path = os.path.join(dirname, 'clustering_df_update.pkl')
df = pd.read_pickle(file_path)


feature_names = [
    "duration",
    "left_interval",
    "right_interval",
    "max_freq",
    "amplitude_norm",
    "pca_0",
    "pca_1",
    "pca_2",
    
]


features_expanded = pd.DataFrame(df["features"].tolist(), index=df.index, columns=feature_names)
# Add cluster_num column at the front

features_expanded.insert(0, "session_ind", df["session_ind"])
features_expanded.insert(0, "animal_num", df["animal_num"])
features_expanded.insert(0, "cluster_num", df["cluster_num"])
# Sort the rows by cluster number
features_expanded = features_expanded.sort_values(by="cluster_num")
# Remove rows where cluster num is a negative value
features_expanded = features_expanded.loc[features_expanded['cluster_num'] >= 0]



# %% TRAINING SVM LEAVE-ONE-ANIMAL-OUT
# ==============================================================================
# Training SVM
# ==============================================================================


session_ind = features_expanded['session_ind'].unique()

sample_frac = 1  # 30% of the training data
accuracy_scores = []
confusion_matrices = []
# Train the classifier 10x
for session_i in tqdm(session_ind):
    session_df = features_expanded[features_expanded['session_ind'] == session_i]
    animal_num = session_df['animal_num'].iloc[0]
    train_df = features_expanded[features_expanded['animal_num'] != animal_num]
    if session_i in train_df['session_ind'].values:
        print(f"uh oh {session_i}")

    X_train_all = train_df.drop(columns = ['cluster_num', 'animal_num', 'session_ind'])
    y_train_all = train_df['cluster_num']
    
    X_test = session_df.drop(columns = ['cluster_num', 'animal_num', 'session_ind'])
    y_test = session_df['cluster_num']

    X_train = X_train_all.sample(frac=sample_frac, random_state=42)
    y_train = y_train_all.loc[X_train.index]
    #test_indices = X_test.index
    
    rbf_svc = svm.SVC(kernel='rbf', probability=True) # Non-linear
    rbf_svc.fit(X_train, y_train)
    
    y_pred = rbf_svc.predict(X_test) 
    y_proba = rbf_svc.predict_proba(X_test)

    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
    
    matrix = confusion_matrix(y_test, y_pred, normalize='pred')
    confusion_matrices.append(matrix)

# %% Plot + stats of SVM
# ==============================================================================
# Plot + stats of SVM
# ==============================================================================

# Building average confusion matrix and std
matrices_as_array = np.array(confusion_matrices)
average_matrix = matrices_as_array.mean(axis=0)
std_matrix = matrices_as_array.std(axis=0)

# Plot the average confusion matrix with black and white colormap
plt.figure(figsize=(10, 10))  # Adjust size as needed
plt.imshow(average_matrix, cmap='Greys_r')

# Set tick labels
plt.xticks(ticks=np.arange(3), labels=[0, 1, 2])
plt.yticks(ticks=np.arange(3), labels=[0, 1, 2])
   
# Axis labels
plt.xlabel("Predicted Cluster Labels")
plt.ylabel("True Cluster Labels")

# Add text annotations to each cell
for i in range(average_matrix.shape[0]):
    for j in range(average_matrix.shape[1]):
        mean_val = average_matrix[i, j]
        std_val = std_matrix[i, j]
        
        text = f"{mean_val:.2f}\n±{std_val:.2f}"
        #text = f"{mean_val:.2f}"
        text_color = 'black' if mean_val > 0.5 else 'white'
        
        plt.text(j, i, text, ha='center', va='center',
                 color=text_color, fontsize=40, fontweight='bold')


# Add title
plt.title("Average Confusion Matrix")

# Show the plot
plt.tight_layout()

png_pc0_plot = os.path.join('/home/natasha/Desktop/final_figures', 'svm_confusion_matrix.png')
svg_pc0_plot = os.path.join('/home/natasha/Desktop/final_figures', 'svm_confusion_matrix.svg')
plt.savefig(png_pc0_plot)
plt.savefig(svg_pc0_plot)
plt.show()
t_stat, p_value = stats.ttest_1samp(accuracy_scores, 0.3)
if p_value < 0.05:
    print(f'The mean accuracy is significantly above 0.3 (p-value: {p_value})')
else:
    print('The mean accuracy is not significantly different from chance!')

# %%



# Convert list of confusion matrices to array
matrices_as_array = np.array(confusion_matrices)  # shape: (n_matrices, n_classes, n_classes)

# Extract diagonal values (accuracy per class) from each matrix
# This will give a shape (n_matrices, n_classes)
diagonal_accuracies = np.array([np.diag(cm) for cm in matrices_as_array])

# Create a boxplot
plt.figure(figsize=(8, 12))
plt.boxplot([diagonal_accuracies[:, i] for i in range(diagonal_accuracies.shape[1])],
            labels=[1, 2, 3],
            boxprops=dict(linewidth=2.5),
            whiskerprops=dict(linewidth=2.5),
            capprops=dict(linewidth=2.5),
            medianprops=dict(linewidth=2.5, color='blue'))


plt.xlabel("Cluster Label")
plt.axhline(y=0.3, color='red', linestyle='--', linewidth=3)

# Remove top and right spines
ax = plt.gca()  # get current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.ylim(0, 1)  # since accuracies are proportions

# Make axes (spines) thicker
for spine in ax.spines.values():
    spine.set_linewidth(2)  # increase number for thicker lines

# Make tick labels bigger and bolder
ax.tick_params(axis='both', which='major', labelsize=32, width=2)  # labelsize = font size, width = tick line thickness
plt.xlabel("Cluster label", fontsize=37, labelpad=10)
plt.ylabel("SVM accuracy", fontsize=37, labelpad=10)
plt.tight_layout()
plt.show()



# diagonal_accuracies: shape (n_matrices, n_classes)
n_classes = diagonal_accuracies.shape[1]

for i in range(n_classes):
    t_stat, p_value = stats.ttest_1samp(diagonal_accuracies[:, i], 0.3)
    mean_val = np.mean(diagonal_accuracies[:, i])
    if p_value < 0.05:
        print(f"Class {i+1}: mean={mean_val:.3f} is significantly above 0.3 (p={p_value:.4f})")
    else:
        print(f"Class {i+1}: mean={mean_val:.3f} is NOT significantly above 0.3 (p={p_value:.4f})")
