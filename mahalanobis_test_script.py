#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 15:12:48 2025

@author: natasha
"""

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# -------------------------------
# Load data
# -------------------------------
file_path = '/home/natasha/Desktop/clustering_data/mahalanobis_data.pkl'
mahal_data = pd.read_pickle(file_path)

diag_elements = mahal_data[mahal_data['group'] == 'diag']['value']
non_diag_elements = mahal_data[mahal_data['group'] == 'non_diag']['value']

# -------------------------------
# Stats: KS
# -------------------------------

ks_stats = stats.kstest(diag_elements, non_diag_elements)

if ks_stats.pvalue < 0.05:
    print("Diagonal distances statistically different from non-diagonal")
    print(f"pvalue: {ks_stats.pvalue}, KS Statistic (D): {ks_stats.statistic}")
else:
    print("Warning: diagonal distances NOT statistically different from non-diagonal")

# -------------------------------
# Plot - distribution histogram
# -------------------------------
# Define bins
bin_edges = np.linspace(mahal_data['value'].min(), mahal_data['value'].max(), num=31)
plt.figure(figsize=(10, 8))
# Plot
plt.hist(diag_elements, bins=bin_edges, 
         color='dimgray', edgecolor='black', alpha=1, label='Diagonal Elements')
plt.hist(non_diag_elements, bins=bin_edges, 
         color='moccasin', edgecolor='black', alpha=0.7, label='Non-Diagonal Elements')

plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Frequency Distribution')
plt.legend()
plt.savefig("/home/natasha/Desktop/final_figures/mahalanobis_distance.svg", format="svg")  # Save before show
plt.savefig("/home/natasha/Desktop/final_figures/mahalanobis_distance.png", format="png")  # Save before show
plt.show()


# -------------------------------
# Plot - box plots
# -------------------------------
# Combine data
data_to_plot = [diag_elements, non_diag_elements]
labels = ['Diagonal', 'Non-Diagonal']

plt.figure(figsize=(6, 10))  # shorter width to make boxes closer
box = plt.boxplot(data_to_plot, labels=labels, patch_artist=True,
                  widths=0.8,  # narrower boxes to squish them closer
                  boxprops=dict(facecolor='lightgray', color='black', linewidth=3),
                  medianprops=dict(color='red', linewidth=4),
                  whiskerprops=dict(color='black', linewidth=3),
                  capprops=dict(color='black', linewidth=3),
                  flierprops=dict(marker='o', markerfacecolor='gray', markersize=5, linestyle='none')
                 )

plt.ylabel('Values')
plt.xlabel('Mahalanobis Distances')


# --- Add significance bar ---
y_max = max(max(diag_elements), max(non_diag_elements))
bar_height = y_max + 0.05 * y_max  # a little above the tallest box
plt.plot([1, 2], [bar_height, bar_height], color='black', lw=2)  # horizontal bar
plt.text(1.5, bar_height + 0.005*y_max, '***', ha='center', va='bottom', fontsize=24)
plt.ylim(top=bar_height + 0.08*y_max)

plt.savefig("/home/natasha/Desktop/final_figures/mahalanobis_distance_boxplot_sig.svg", format="svg")
plt.savefig("/home/natasha/Desktop/final_figures/mahalanobis_distance_boxplot_sig.png", format="png")
plt.show()



