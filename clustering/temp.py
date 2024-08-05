# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import umap

# Load data
file_path = '/home/natasha/Desktop/clustering_data/mtm_clustering_df.pkl'
df = pd.read_pickle(file_path)

# Make a dataframe of just mouth or tongue movement events
mtm_bool = df.event_type.str.contains('mouth or tongue movement')
mtm_df = df.loc[mtm_bool]

# Says what number session
mtm_df.session_ind

# Array of ??
mtm_features = np.stack(mtm_df.features.values)


