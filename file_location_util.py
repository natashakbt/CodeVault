#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 11:29:58 2025

@author: natasha

Read a directory path from a file_location text file.
Directory path should be where .pkl file from EMG classifier is stored
and will be the basis for where figures are generated
"""
import os

def load_dirname(config_file="file_location.txt"):
    with open(config_file) as f:
        dirname = f.readline().strip()
    
    # Ensure trailing slash
    if not dirname.endswith(os.sep):
        dirname += os.sep

    # Check that directory exists
    if not os.path.isdir(dirname):
        raise FileNotFoundError(f"❌ Directory does not exist: {dirname}")
        
    final_figures_dir = os.path.join(dirname, "final_figures")
    os.makedirs(final_figures_dir, exist_ok=True)

    return dirname
