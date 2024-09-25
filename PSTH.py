#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jun  3 14:53:40 2024

@author: kmaigler
"""
#this code reads in spike arrays from an h5 file and generates a binned repsonse
#then plots a PSTH
# =============================================================================
# Import stuff
# =============================================================================
import numpy as np
import tables
import pylab as plt
import easygui
import os
from scipy.ndimage.filters import gaussian_filter1d

# =============================================================================
# Functions
# ============================================================================= 
#take in spike array and return binned responses 
def build_bin_resp_array(spike_array, wanted_units, pre_stim = 2000, window_size = 250, step_size = 25):
    """
    return:
        binned_taste_resp [trials, units, bins]
    """
    x = np.arange(0, spike_array.shape[-1], step_size)

    binned_taste_resps = [1000.0*np.mean(spike_array[:, wanted_units, s:s+window_size], axis = 2) for s in x]
    
    return np.moveaxis(np.array(binned_taste_resps), 0, -1)
# =============================================================================
# Load in data and plot
# ============================================================================= 
dir_folder = easygui.diropenbox(msg = 'Choose where the h5 file is...')
os.chdir(dir_folder)

#look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files
#open file				
hf5 = tables.open_file(hdf5_name, 'r+')
#get spike trains
trains_dig_in = hf5.list_nodes('/spike_trains')
all_spikes = np.asarray([spikes.spike_array[:] for spikes in trains_dig_in])
num_units = all_spikes.shape[2]
#bin response
response = [build_bin_resp_array(trains_dig_in[i].spike_array[:], np.arange(num_units),) \
        for i in range(len(trains_dig_in))]
#use the h5 file name to get your animal name and date    
date = hdf5_name.split('_')[2]
animal_name = hdf5_name.split('_', 1)[0].replace('.', '').upper()
newname = '_'.join([animal_name, date])
#   
print(response[0].shape[1])
print('number of units is %i'%num_units)
print('from session: %s'%newname)

######plotting a psth##########################
#this assumes your prestim array is 2000 ****
#define time you want to plot according to taste delivery = 0
timerange = easygui.multenterbox(msg = 'What time would you like to plot your PSTH?', 
                                  fields = ['start', 'end', 'step'],
                                  values = [-500, 1500, 25])
startx = int(timerange[0])
stopx = int(timerange[1])
stepx = int(timerange[2])
xx = range(startx, stopx, stepx)     
tbin = np.array(range(-2000, 5000, 25))# ****this assumes your prestim array is 2000 ****
tbin1 = int(np.where(tbin==startx)[0])
tbin2 = int(np.where(tbin==stopx)[0])
#get identites of tastes/digins
tastes = easygui.multenterbox(msg = 'Put in the taste identities of the digital inputs', 
                                  fields = [train._v_name for train in trains_dig_in],
                                  values = ['QHCl', 'CA', 'Water', 'NaCl', 'Sucrose'])
for unit in range(response[0].shape[1]):
    fig, axes = plt.subplots(figsize = (5,4), dpi=500)
    for tt in range(len(tastes)):
        smoothunit = gaussian_filter1d((np.mean(response[tt][:, unit, tbin1:tbin2], axis = 0)), sigma = 1.5)
        axes.plot(xx, smoothunit, linewidth=4, label = tastes[tt])
        axes.set_ylabel('Firing rate (Hz)')
        axes.set_xlabel('time from taste delivery (ms)')
        axes.set_title('%s'%newname +' '+'unit# %i'%unit)
        axes.legend(loc = 'best')