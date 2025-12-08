#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:20:21 2025

@author: natasha
"""

import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import glob
from scipy.stats import chi2_contingency
from scipy.stats import zscore
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
import shutil
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d  # for smoothing
from scipy.optimize import curve_fit
import piecewise_regression
from scipy.stats import ttest_rel
import scipy.stats as stats
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import matplotlib as mpl



# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/christina_data/'
file_path = os.path.join(dirname, 'christina_all_datasets.pkl')
df = pd.read_pickle(file_path)

df = df.rename(columns={'pred': 'cluster_num'})