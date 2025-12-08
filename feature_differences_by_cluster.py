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
from scipy import stats
from tqdm import tqdm
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
from mtm_analysis_config import dirname, feature_names, color_mapping
from scipy.signal import find_peaks


# ==============================================================================
# Load data and get setup
# ==============================================================================

file_path = os.path.join(dirname, 'clustering_df_update.pkl')
df = pd.read_pickle(file_path)

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

# For saving stats
feature_results = pd.DataFrame(columns=['feature', 'H', 'p', 'epsilon', 'effect_size'])


# %% PC0
# ==============================================================================
# PC0 - Mann Whitney U test
# ==============================================================================
pc0_cluster1 = features_expanded[features_expanded["cluster_num"] == 0]['pca_0']
pc0_cluster2 = features_expanded[features_expanded["cluster_num"] == 1]['pca_0']
pc0_cluster3 = features_expanded[features_expanded["cluster_num"] == 2]['pca_0']

# Define cluster pairs
pairs = [
    ('Cluster 0 vs 1', pc0_cluster1, pc0_cluster2),
    ('Cluster 0 vs 2', pc0_cluster1, pc0_cluster3),
    ('Cluster 1 vs 2', pc0_cluster2, pc0_cluster3)
]


for name, group1, group2 in pairs:
    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    print(f"{name}: U={u_stat}, p={p_value:.4f}")
    '''
color_mapping = {
    0: '#4285F4',
    1: '#88498F',
    2: '#0CBABA'
}
'''
# ==============================================================================
# PC0 - plotting
# ==============================================================================
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


# %%
# ==============================================================================
# Three variations of stats to see if features are significantly different across clusters
# ==============================================================================

features_expanded = pd.DataFrame(df["features"].tolist(), index=df.index, columns=feature_names)
features_expanded.insert(0, "cluster_num", df["cluster_num"]) # Add cluster_num column at the front
features_expanded = features_expanded.sort_values(by="cluster_num") # Sort the rows by cluster number
features_expanded = features_expanded.loc[features_expanded['cluster_num'] >= 0] # Remove rows where cluster num is a negative value


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



# KRUSKAL WALLIS FOR NON-NORMAL DATA
for feature in feature_names:
    h_stat, p_value = stats.kruskal(features_expanded[features_expanded["cluster_num"] == 0][feature],
                                    features_expanded[features_expanded["cluster_num"] == 1][feature],
                                    features_expanded[features_expanded["cluster_num"] == 2][feature])
    
    # Calculate Eta-squared for effect size
    k = 3  # number of groups (clusters)
    N = len(features_expanded)  # total number of observations
    epsilon_squared = (h_stat - k + 1) / (N - k)
    
    
    print(feature)
    print(f'{h_stat:.2f}, {p_value:.4f}, {epsilon_squared:.4f}\n')
    
    if p_value < 0.05 :
        if epsilon_squared > 0.16:  # Only proceed if Kruskal-Wallis is significant
            print(f"{feature} p-value: {p_value}")
            print(f"H-stat {h_stat.round(3)}, effect size: {epsilon_squared.round(4)}")
            effect_size = 'relatively strong'
        elif epsilon_squared > 0.04:  # Only proceed if Kruskal-Wallis is significant
            print(f"{feature} p-value: {p_value}")
            print(f"H-stat {h_stat.round(3)}, effect size: {epsilon_squared.round(4)}")
            effect_size = 'moderate'
        elif epsilon_squared > 0.01:
            print(f'{feature} is signficiant ({p_value}) but weak effect size ({epsilon_squared.round(4)})')
            effect_size = 'weak'
            
        else:
            print(f'{feature} is signficiant ({p_value}) but inconsequential effect size')
            effect_size = 'negligible'
    else:
        print(f'{feature} not significant ({p_value})')
        effect_size = 'n/a'
    print('\n')
    feature_results.loc[len(feature_results)] = [feature, h_stat, p_value, epsilon_squared, effect_size]




# %% MEASURING WAVEFORM METRICS
# ==============================================================================
# Measuring waveform metrics
# Negative gradient, positive gradient, amplitude, width (at 50% max amplitude), area under the curve
# symmetry (pearson's r), skew
# ==============================================================================

waveform_metrics_df = df
new_columns = ['amplitude', 'area', 'width', 
               'pos_grad', 'neg_grad', 'symmetry', 'skew', 'bimod_test']
waveform_metrics_df[new_columns] = np.nan

for index, row in tqdm(waveform_metrics_df.iterrows()):
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

    # Test bimodality
    peaks, _ = find_peaks(waveform, prominence=np.max(waveform)*0.1)  # adjust threshold as needed
    
    if len(peaks) == 0:
        print('blah')
        continue
    
    first_peak_idx = peaks[0]
    seg_after_peak = waveform[first_peak_idx:]
    
    derivative = np.gradient(seg_after_peak)
    positive_deriv_sum = np.sum(derivative[derivative > 0])

    waveform_metrics_df.at[index, 'bimod_test'] = positive_deriv_sum




# ==============================================================================
# Plots of metrics + stats
# ==============================================================================
subset_df = waveform_metrics_df[waveform_metrics_df['cluster_num'].isin([0, 1, 2])]

# Kruskal-wallis test
results = []
for metric in tqdm(new_columns):
    # Group values by true_label
    groups = [subset_df[subset_df['cluster_num'] == label][metric] for label in [0, 1, 2]]
    
    # Kruskal-Wallis test
    H, p_value = stats.kruskal(*groups)
    n = len(subset_df)
    k = len(groups)
    epsilon_squared = (H - k + 1) / (n - k)
    results.append({'metric': metric, 
                    'kruskal_H': H,
                    'kruskal_p': p_value,
                    'epsilon_squared': epsilon_squared
                    })

    print(f"\n🔍 {metric}")
    print(f"Kruskal-Wallis p-value: {p_value:.4f}")
    #print(f"Epsilon-squared (effect size): {epsilon_squared:.4f}")
    
    if epsilon_squared < 0.01:
        print("negligible effect size")
    elif epsilon_squared < 0.04:
        print("Weak effect size")
    elif epsilon_squared < 0.16:
        print("moderate effect")
    # If significant, do post-hoc Dunn test
    if p_value < 0.05:
        print("→ Running post-hoc Dunn's test:")
        posthoc = sp.posthoc_dunn(subset_df, val_col=metric, group_col='cluster_num')
        #print(posthoc)
        significant_pairs = posthoc[posthoc < 0.05]
        print(significant_pairs)
    else:
        print("not significant")

    if p_value < 0.05 :
        if epsilon_squared > 0.16:  # Only proceed if Kruskal-Wallis is significant
            print(f"{metric} p-value: {p_value}")
            print(f"H-stat {H.round(3)}, effect size: {epsilon_squared.round(4)}")
            effect_size = 'relatively strong'
        elif epsilon_squared > 0.04:  # Only proceed if Kruskal-Wallis is significant
            print(f"{metric} p-value: {p_value}")
            print(f"H-stat {H.round(3)}, effect size: {epsilon_squared.round(4)}")
            effect_size = 'moderate'
        elif epsilon_squared > 0.01:
            print(f'{metric} is signficiant ({p_value}) but weak effect size ({epsilon_squared.round(4)})')
            effect_size = 'weak'
            
        else:
            print(f'{metric} is signficiant ({p_value}) but inconsequential effect size')
            effect_size = 'negligible'
    else:
        print(f'{metric} not significant ({p_value})')
        effect_size = 'n/a'
    print('\n')
    feature_results.loc[len(feature_results)] = [metric, H, p_value, epsilon_squared, effect_size]


# %%


# Add a tiny offset to avoid zeros
#subset_df['bimod_test_log'] = subset_df['bimod_test'] + 1e-6

plt.figure(figsize=(10,7))
ax = sns.boxplot(
    data=subset_df,
    x='cluster_num', y='bimod_test',
    hue='cluster_num',
    linewidth=3,
    palette=color_mapping
)

ax.set_yscale('log')
#ax.set_ylim(1e-6, subset_df['bimod_test'].max()*1.2)
plt.xlabel('Cluster')
plt.ylabel('Positive derivative sum (log scale)')
plt.show()

# %% Save statistical results
feature_results.to_csv('/home/natasha/Desktop/clustering_data/feature_results.csv', index=False)


# %% ARCHIVE OLD CODE
'''
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

'''    


