#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 15:12:48 2025

@author: natasha
"""

import pickle
from scipy import stats
import pandas as pd

file_path = '/home/natasha/Desktop/clustering_data/mahalanobis_data.pkl'
 # all events from classifier predictions
mahal_data = pd.read_pickle(file_path)
# Load the saved data

diag_elements = mahal_data['diag_elements']
non_diag_elements = mahal_data['non_diag_elements']

# Run KS test
ks_stats = stats.kstest(diag_elements, non_diag_elements)

if ks_stats.pvalue < 0.05:
    print("Diagonal distances statistically different from non-diagonal")
    print(f"pvalue: {ks_stats.pvalue}, KS Statistic (D): {ks_stats.statistic}")
else:
    print("Warning: diagonal distances NOT statistically different from non-diagonal")