#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 15:12:48 2025

@author: natasha
"""

import numpy as np
import pickle
from scipy import stats
import pandas as pd
from scipy.stats import chisquare

file_path = '/home/natasha/Desktop/clustering_data/mahalanobis_data.pkl'
 # all events from classifier predictions
mahal_data = pd.read_pickle(file_path)
# Load the saved data

diag_elements = mahal_data[mahal_data['group'] == 'diag']['value']
non_diag_elements = mahal_data[mahal_data['group'] == 'non_diag']['value']

# Stats: KS
ks_stats = stats.kstest(diag_elements, non_diag_elements)

if ks_stats.pvalue < 0.05:
    print("Diagonal distances statistically different from non-diagonal")
    print(f"pvalue: {ks_stats.pvalue}, KS Statistic (D): {ks_stats.statistic}")
else:
    print("Warning: diagonal distances NOT statistically different from non-diagonal")







'''
# Stats: Chi-squared
max_val = max(max(diag_elements), max(non_diag_elements))
bins = np.arange(0, max_val+10, 10)
    
diag_binned, _ = np.histogram(diag_elements)
non_diag_binned, _ = np.histogram(non_diag_elements)
    
chi2_stat, p_val = chisquare(f_obs=diag_binned, f_exp=non_diag_binned)

'''