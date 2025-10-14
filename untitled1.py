#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 15:40:25 2025

@author: natasha
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy import stats
from tqdm import tqdm
from mtm_analysis_config import dirname
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==============================================================================
# Load data and get setup
# ==============================================================================
file_path = os.path.join(dirname, 'clustering_df_update.pkl')
df = pd.read_pickle(file_path)



# %%

# Flip only for cluster 0
mask = df['cluster_num'] == 0

# Use .apply with np.flip for the selected rows
df.loc[mask, 'segment_norm_interp'] = df.loc[mask, 'segment_norm_interp'].apply(np.flip)


# %%

segment_df = pd.DataFrame(df['segment_norm_interp'].tolist())

# Create another dataframe with cluster_num
pca_df = df[['cluster_num', 'session_ind', 'animal_num']].copy()

# Optional: reset index if needed
segment_df.reset_index(drop=True, inplace=True)
pca_df.reset_index(drop=True, inplace=True)


#pca_dimmensions = []

pca = PCA(n_components=3)
embedding = pca.fit_transform(segment_df)
#pca_dimmensions.append(embedding.shape[1])

pca_components_df = pd.DataFrame(
    embedding,
    columns=['PC1', 'PC2', 'PC3']
)

# Combine cluster_num with PCA components
pca_df = pd.concat([pca_df.reset_index(drop=True), pca_components_df.reset_index(drop=True)], axis=1)
pca_df = pca_df[pca_df['cluster_num'] >= 0].reset_index(drop=True)
print(pca_df.head())



# %%    


X = pca_df[['PC1', 'PC2', 'PC3']]  # features
y = pca_df['cluster_num']           # labels


iterations = 10
test_size = 0.2
accuracy_scores = []
confusion_matrices = []
for i in range(iterations):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
    
    rbf_svc = svm.SVC(kernel='rbf', probability=True) # Non-linear
    rbf_svc.fit(X_train, y_train)
    
    y_pred = rbf_svc.predict(X_test) 
    y_proba = rbf_svc.predict_proba(X_test)

    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
    
    matrix = confusion_matrix(y_test, y_pred, normalize='pred')
    confusion_matrices.append(matrix)

# Average accuracy across iterations
mean_acc = np.mean(accuracy_scores)
print(f'Mean accuracy over {iterations} iterations: {mean_acc:.3f}')



# %%

session_ind = pca_df['session_ind'].unique()

sample_frac = 1  # for speed, can use a fraction of the training data to train. Minimum suggested is 0.3 (30%)
accuracy_scores = []
confusion_matrices = []
# Train the classifier 10x
for session_i in tqdm(session_ind):
    #X = pca_df[['PC1', 'PC2', 'PC3']]  # features
    #y = pca_df['cluster_num']           # labels

    session_df = pca_df[pca_df['session_ind'] == session_i]
    animal_num = session_df['animal_num'].iloc[0]
    train_df = pca_df[pca_df['animal_num'] != animal_num]
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

for (animal, session), group in pca_df.groupby(['animal_num', 'session_ind']):
    print(animal, session)
    
