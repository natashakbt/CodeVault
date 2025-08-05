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
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
import seaborn as sns
import scikit_posthocs as sp
from itertools import combinations
from scipy.stats import ks_2samp

# ==============================================================================
# Load data and get setup
# ==============================================================================
'''
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

'''

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


all_high_conf_waveforms = []

accuracy_scores = []
confusion_matrices = []
# Train the classifier 10x
for i in tqdm(range(10)):
    # Split the dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y)
    
    test_indices = X_test.index
    
    rbf_svc = svm.SVC(kernel='rbf', probability=True) # Non-linear
    rbf_svc.fit(X_train, y_train)
    
    y_pred = rbf_svc.predict(X_test) 
    y_proba = rbf_svc.predict_proba(X_test)

    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
    
    matrix = confusion_matrix(y_test, y_pred, normalize='pred')
    confusion_matrices.append(matrix)
    
    if i == 1:
        plt.imshow(matrix)
        plt.xticks(ticks=np.arange(3), labels=[0, 1, 2])
        plt.yticks(ticks=np.arange(3), labels=[0, 1, 2])
    
        plt.xlabel("Predicted Cluster Labels")
        plt.ylabel("True Cluster Labels")
        cbar = plt.colorbar()
        cbar.set_label('Normalized Accuracy')
        
        plt.show()
    
    # Pulling out the most confident (>90) waveforms. For plotting below
    confidences = np.max(y_proba, axis=1)
    high_conf_mask = confidences >= 0.9
    
    # Get original indices for high-confidence predictions
    high_conf_indices = test_indices[high_conf_mask]
    
    # Pull the waveforms from df
    waveforms = df.loc[high_conf_indices, 'segment_norm_interp']
    waveforms = df.loc[high_conf_indices, 'segment_raw']
    
    waveform_data = pd.DataFrame({
        'waveform': waveforms,
        'true_label': y_test.loc[high_conf_indices].values,
        'pred_label': y_pred[high_conf_mask],
        'confidence': confidences[high_conf_mask]
    })
    all_high_conf_waveforms.append(waveform_data)

# ==============================================================================
# Plotting waveforms
# ==============================================================================

final_waveforms_df = pd.concat(all_high_conf_waveforms, ignore_index=True)
final_waveforms_df = final_waveforms_df[
    final_waveforms_df['true_label'] == final_waveforms_df['pred_label']
]

mean_waveforms = []
# Group waveforms by true_label
for label in sorted(final_waveforms_df['true_label'].unique()):
    group = final_waveforms_df[final_waveforms_df['true_label'] == label]
    print(len(group))
    plt.figure(figsize=(8, 4))
    for waveform in group['waveform']:
        plt.plot(waveform, alpha=0.2, color='gray')
    
    # Optionally: Add mean waveform
    mean_waveform = np.mean(np.stack(group['waveform'].values), axis=0)
    mean_waveforms.append(mean_waveform)
    
   #plt.plot(mean_waveform, color='black', linewidth=2, label='Mean waveform')
    
    plt.title(f"Overlayed Waveforms for Cluster {label}")
    plt.tight_layout()
    plt.show()

for w in mean_waveforms:
    plt.plot(w)
plt.title("Mean of 90% confidence waveforms")
plt.show()




# TAKE TOP 10 CONFIDENCE WAVEFORMS IN EACH CLUSTER AND PLOT
top_waveforms_by_cluster = []

for label in sorted(final_waveforms_df['pred_label'].unique()):
    top10 = (
        final_waveforms_df[final_waveforms_df['pred_label'] == label]
        .sort_values(by='confidence', ascending=False)
        .head(10)
    )
    top_waveforms_by_cluster.append(top10)

# Combine into one DataFrame
top_waveforms_df = pd.concat(top_waveforms_by_cluster)



# Loop through each predicted cluster
for label in sorted(final_waveforms_df['pred_label'].unique()):
    # Get top 10 waveforms for this cluster by confidence
    top10 = (
        final_waveforms_df[final_waveforms_df['pred_label'] == label]
        .sort_values(by='confidence', ascending=False)
        .head(10)
    )

    plt.figure(figsize=(8, 4))
    
    for i, waveform in enumerate(top10['waveform']):
        plt.plot(waveform, alpha=0.8, label=f"Waveform {i+1}", color='k')
    
    plt.title(f"Top 10 Most Confident Waveforms (Predicted Cluster {label})")

    plt.tight_layout()
    plt.show()

# ==============================================================================
# Confusion matrix + stat of SVM accuracy
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
        
        #text = f"{mean_val:.2f}\n±{std_val:.2f}"
        text = f"{mean_val:.2f}"
        text_color = 'black' if mean_val > 0.6 else 'white'
        
        plt.text(j, i, text, ha='center', va='center',
                 color=text_color, fontsize=40, fontweight='bold')


# Add title
plt.title("Average Confusion Matrix")

# Show the plot
plt.tight_layout()

# Save the plot
plt.savefig("/home/natasha/Desktop/final_figures/my_plot.svg", format="svg")
plt.show()
# Perform a one-sample t-test
t_stat, p_value = stats.ttest_1samp(accuracy_scores, 0.3)
if p_value < 0.05:
    print(f'The mean accuracy is significantly above 0.3 (p-value: {p_value})')
else:
    print('The mean accuracy is not significantly different from chance!')


# %% MEASURING WAVEFORM METRICS
# ==============================================================================
# Measuring waveform metrics
# Negative gradient, positive gradient, amplitude, width (50%), area under the curve
# symmetry (pearson's r), skew
# ==============================================================================

waveform_metrics_df = df
new_columns = ['amplitude', 'area', 'width', 
               'pos_grad', 'neg_grad', 'symmetry', 'skew']
waveform_metrics_df[new_columns] = np.nan

for index, row in waveform_metrics_df.iterrows():
    waveform= row['segment_raw']
    max_val = waveform.max()
    
    waveform_metrics_df.at[index, 'area'] = np.trapz(waveform)
    waveform_metrics_df.at[index, 'amplitude'] = max_val
    
    half_max = max_val / 2.0
    above_half = waveform >= half_max
    indices = np.where(above_half)[0]
    if len(indices) < 2:
        width = np.nan
    else:
        width = indices[-1] - indices[0]
    waveform_metrics_df.at[index, 'width'] = width
    
    mirrored_waveform = waveform[::-1]
    sym_stat = stats.pearsonr(waveform, mirrored_waveform)
    waveform_metrics_df.at[index, 'symmetry'] = sym_stat[0]
    
    waveform_metrics_df.at[index, 'skew'] = stats.skew(waveform)

    peak_index = np.argmax(waveform)
    slopes = np.gradient(waveform)
    
    rising_slopes = slopes[:peak_index + 1]
    avg_rising_slope = np.mean(rising_slopes)
    waveform_metrics_df.at[index, 'pos_grad'] = avg_rising_slope
        
    falling_slopes = slopes[peak_index:]
    avg_falling_slope = np.mean(falling_slopes)
    waveform_metrics_df.at[index, 'neg_grad'] = avg_falling_slope

# ==============================================================================
# Plots of metrics + stats
# ==============================================================================
subset_df = waveform_metrics_df[waveform_metrics_df['cluster_num'].isin([0, 1, 2])]


# Loop through each metric and plot a violin plot
for metric in new_columns:
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=subset_df, x='cluster_num', y=metric)
    plt.title(f'{metric}')
    plt.xlabel('Cluster Number')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()
    
 # Loop through each metric and plot a box plot   
for metric in new_columns:
    plt.figure(figsize=(6, 4))
    sns.boxenplot(data=subset_df, x='cluster_num', y=metric)
    plt.title(f'{metric}')
    plt.xlabel('Cluster Number')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

# Kolmogorov-Smirnov test
ks_results = []
for metric in new_columns:
    # Group values by true_label
    print(f"\n🔍 {metric}")
    metric_results = {'metric': metric}
    groups = {label: subset_df[subset_df['cluster_num'] == label][metric] for label in [0, 1, 2]}
    
    for (label1, label2) in combinations(groups.keys(), 2):
        stat, p = ks_2samp(groups[label1], groups[label2])
        key = f"{label1}_vs_{label2}"
        metric_results[f"ks_stat_{key}"] = stat
        metric_results[f"ks_p_{key}"] = p

        print(f"KS test {key}: stat = {stat:.3f}, p = {p:.3f}")
        if p < 0.05:
            if stat < 0.05:
                print("→ Neglibile effect")
            elif stat < 0.2:
                print("→ Small effect")
            elif stat < 0.3:
                print("→ Medium effect")
            else:
                print("→ Large effect")
        else:
            print("not significant")
    
    ks_results.append(metric_results)
    
# Kruskal-wallis test
results = []
for metric in new_columns:
    # Group values by true_label
    groups = [subset_df[subset_df['cluster_num'] == label][metric] for label in [0, 1, 2]]
    
    # Kruskal-Wallis test
    H, p = stats.kruskal(*groups)
    n = len(subset_df)
    k = len(groups)
    epsilon_squared = (H - k + 1) / (n - k)
    results.append({'metric': metric, 
                    'kruskal_H': H,
                    'kruskal_p': p,
                    'epsilon_squared': epsilon_squared
                    })

    print(f"\n🔍 {metric}")
    print(f"Kruskal-Wallis p-value: {p:.4f}")
    #print(f"Epsilon-squared (effect size): {epsilon_squared:.4f}")
    
    if epsilon_squared < 0.01:
        print("negligible effect size")
    elif epsilon_squared < 0.04:
        print("Weak effect size")
    elif epsilon_squared < 0.16:
        print("moderate effect")
    # If significant, do post-hoc Dunn test
    if p < 0.05:
        print("→ Running post-hoc Dunn's test:")
        posthoc = sp.posthoc_dunn(subset_df, val_col=metric, group_col='cluster_num')
        #print(posthoc)
        significant_pairs = posthoc[posthoc < 0.05]
        print(significant_pairs)
    else:
        print("not significant")

for r in results:
    metric = r['metric']
    H = r['epsilon_squared']
    print(f'{metric}: {H:.3f}')







