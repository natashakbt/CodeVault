#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 11:15:46 2025

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
# load data
# ==============================================================================
file_path = os.path.join(dirname, 'scored_df_with_vid_labels.pkl') # all events from classifier predictions
df = pd.read_pickle(file_path)

final_figures_dir = os.path.join(dirname, "final_figures")