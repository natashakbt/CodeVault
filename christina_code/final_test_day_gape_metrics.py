#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 13:26:33 2025

@author: natasha
"""

#import numpy as np
import pandas as pd
import os
#import math
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.patches as mpatches
import glob
#import matplotlib.colors as mcolors
from matplotlib import gridspec
import seaborn as sns
from scipy.stats import kruskal
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================================================================
# Load data and get setup
# ==============================================================================
dirname = '/home/natasha/Desktop/christina_data/'
file_path = os.path.join(dirname, 'gape_metrics_by_trial.pkl')
bout_df = pd.read_pickle(file_path)


# ==============================================================================
# SET PARAMETERS
# ==============================================================================
filter_licl_con_list = [['0.15M_LiCl', '0.15M_NaCl'], ['0.6M_LiCl', '0.6M_NaCl']] # Needs to be list of paired lists
var_to_plot_list = ['first_gape_bout_start', 'first_gape_bout_duration']
taste_to_plot = 'highqhcl'


for var_to_plot in var_to_plot_list:
    if not var_to_plot in bout_df.columns:
        print(f"⚠ Warning: var_to_plot set to '{var_to_plot}' is not valid")
        print(f"Variable should correspond to df column name: {bout_df.columns}")



# ==============================================================================
# %% Bar plot - getting setup
# ==============================================================================
# Setup folders for saving figures
# ==============================================================================
# Create subdirectory based on concentration
sub_dir = os.path.join(dirname, 'bar_plot')
os.makedirs(sub_dir, exist_ok=True)


# ==============================================================================
# Setup plotting dataframes
# ==============================================================================
# --- Filter datasets ---

max_train = (
    bout_df[bout_df['exp_day_type'] == 'Train']
    .groupby('animal_num')['exp_day_num']
    .max()
    .rename('max_train_num')
)

df2 = bout_df.merge(max_train, on='animal_num', how='left')

df2 = df2[
    (df2['num_of_cta'] == 4) &
    (df2['taste_name'] == 'highqhcl')
]



df3 = df2[
    ['basename',
     'first_gape_bout_start',
     'first_gape_bout_duration',
     'total_gape_time_1s',
     'licl_conc',
     'max_train_num',
     'trial_num']
]




for licl_pair in filter_licl_con_list:
    for var_to_plot in var_to_plot_list:
        df_my_plot = df3[df3['licl_conc'].isin(licl_pair)].copy()
        df_my_plot = df_my_plot[df_my_plot[var_to_plot].notna()]
    
        plt.figure(figsize=(6, 5))


        sns.stripplot(
            data=df_my_plot,
            x='max_train_num',
            y=var_to_plot,
            hue='licl_conc',
            dodge=True,        # separates the paired groups
            jitter=True,
            size=6
        )
    
        plt.title(f"highqhcl gapes on final test day: {licl_pair}")
        plt.ylabel(f"{var_to_plot}")
        plt.xlabel("Max Training Day")
        plt.legend(title="LiCl Conc", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plot_dir_path_png = os.path.join(sub_dir, f"{licl_pair}_highqhcl_gapes_by_training.png")
        plt.savefig(plot_dir_path_png, bbox_inches = "tight")

            
        plt.show()







