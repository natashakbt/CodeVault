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
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator

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

#features_expanded = features_expanded.drop('cluster_num', axis=1)

X = features_expanded.drop(columns=["cluster_num"])
y = features_expanded["cluster_num"]


# %% PC0
# ==============================================================================
# PC0
# ==============================================================================
# Kruskal-Wallis test
groups = [group['pca_0'].values for name, group in features_expanded.groupby('cluster_num')]
kruskal_stat, kruskal_p = stats.kruskal(*groups)

print(f"Kruskal-Wallis H={kruskal_stat:.3f}, p={kruskal_p:.3e}")


if kruskal_p < 0.05:
    dunn_stats = sp.posthoc_dunn(
        features_expanded,
        val_col='pca_0',
        group_col='cluster_num',
    )
    
    print("\nDunn's post-hoc test p-values:")
    print(dunn_stats)

#TODO HOW TO REPORT EFFECT SIZE FOR DUNN'S?

color_mapping = {
    0: '#4285F4',
    1: '#88498F',
    2: '#0CBABA'
}

plt.figure(figsize=(10,7))
ax = sns.boxplot(
    data=features_expanded,
    x='cluster_num', y='pca_0',
    hue='cluster_num',
    linewidth=3,
    legend=False,
    fill=True,
    palette = color_mapping,
    dodge=False
)
# --- Add significance bars ---
pairs = [(0,1), (0,2), (1,2)]   # all pairwise comparisons
annot = Annotator(ax, pairs, data=features_expanded,
                  x='cluster_num', y='pca_0', hue='cluster_num')

annot.configure(text_format='star', loc='inside')
annot.set_pvalues([0.0001, 0.0001, 0.0001])  # manually set p-values for ****
annot.annotate()

plt.xlabel('Cluster Number')
plt.ylabel('PC0')
png_pc0_plot = os.path.join('/home/natasha/Desktop/final_figures', 'pc0_boxplot.png')
svg_pc0_plot = os.path.join('/home/natasha/Desktop/final_figures', 'pc0_boxplot.svg')
plt.savefig(png_pc0_plot)
plt.savefig(svg_pc0_plot)
plt.show()


# %% TRAINING SVM LEAVE-ONE-ANIMAL-OUT + PLOTTING CONFUSION MATRIX - NEW
# ==============================================================================
# Training SVM
# ==============================================================================


session_ind = features_expanded['session_ind'].unique()

sample_frac = 0.3  # 30% of the training data
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
plt.show()
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

'''
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
'''
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




# %% MEASURING WAVEFORM METRICS - CLUSTER SPECIFIC- IF THEY CHANGE BY 'before' VS 'after' EVENT_POSITION
# ==============================================================================
# Setup dataframe with event position (fixed time 800ms after taste delivery)
# ==============================================================================
waveform_metrics_df['event_position'] = np.nan

for idx, row in waveform_metrics_df.iterrows():
    segment_bounds = row['segment_bounds']

    event_position = "before" if segment_bounds[1] < 2800 else "after"
    waveform_metrics_df.at[idx, 'event_position'] = event_position


cluster_num_i_care_about = [0, 1, 2]
# ==============================================================================
# Plots by event position
# ==============================================================================
for cluster_num in cluster_num_i_care_about:
    subset_df = waveform_metrics_df[waveform_metrics_df['cluster_num'].isin([cluster_num])]
  
    # Loop through each metric and plot a violin plot
    for metric in new_columns:
        plt.figure(figsize=(6, 4))
        sns.violinplot(data=subset_df, x='event_position', y=metric)
        plt.title(f'{metric}')
        plt.xlabel('Cluster Number')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.show()
        
     # Loop through each metric and plot a box plot   
    for metric in new_columns:
        plt.figure(figsize=(6, 4))
        sns.boxenplot(data=subset_df, x='event_position', y=metric)
        plt.title(f'{metric}')
        plt.xlabel('Cluster Number')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.show()
        
# ==============================================================================
# Stats by event position
# ==============================================================================
for cluster_num in cluster_num_i_care_about:
    print("______")
    print(f"Analyzing cluster {cluster_num}")
    print("-----")
    subset_df = waveform_metrics_df[waveform_metrics_df['cluster_num'].isin([cluster_num])]
    for metric in new_columns:
        before_df = subset_df[subset_df['event_position']=='before'][metric]
        after_df = subset_df[subset_df['event_position']=='after'][metric]
        
        before_res = stats.normaltest(before_df)
        after_res = stats.normaltest(after_df)
        
        if before_res.pvalue > 0.05 and after_res.pvalue > 0.05:
            stats.ttest_ind(before_df, after_df)
            print("ttest!")
        else:
            U1, p = mannwhitneyu(before_df, after_df)
            if p < 0.05:
                # Test effect size
                n1 = len(before_df)
                n2 = len(after_df)
                N = n1 + n2
                
                # Mean and SD of U
                mean_U = n1 * n2 / 2
                sd_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                
                # Z-score
                Z = (U1 - mean_U) / sd_U
                
                # Effect size r
                r = Z / np.sqrt(N)
                
                if r > 0.3:
                    print(metric, ':', p)
                    print(f"effect size: {r}")
                else:
                    print(f"{metric} signficant, but weak effect size")
            else:
                print(f"{metric} insignificant")

        


# %% ARCHIVE OLD CODE
'''
# %% TRAINING SVM + PLOTTING CONFUSION MATRIX - OLD
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

'''
